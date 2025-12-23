"""
Monkey-Patch for DeepSeek OCR Processor to Support Pre-Computed Embeddings

This module patches vLLM's DeepseekOCRMultiModalProcessor to handle
ImageEmbeddingItems (pre-computed embeddings) in addition to raw PIL images.

Import this module BEFORE creating the LLM instance:
    import embedding_patch  # Apply patches
    from vllm import LLM
    llm = LLM(model="deepseek-ai/DeepSeek-OCR", ...)

The patch is transparent and maintains backward compatibility with raw images.
"""

import math
import torch
import logging
from typing import Mapping
from transformers import BatchFeature

logger = logging.getLogger(__name__)


def create_patched_call_hf_processor(original_method):
    """
    Create a patched version of _call_hf_processor that handles embeddings

    This wrapper checks for ImageEmbeddingItems and processes them specially,
    while delegating raw PIL images to the original processor.
    """
    def patched_call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """
        Patched processor that handles both images and embeddings
        """
        from vllm.multimodal.parse import ImageEmbeddingItems

        # Check if we're dealing with pre-computed embeddings
        images = mm_data.get('images', mm_data.get('image'))

        if images is not None and isinstance(images, ImageEmbeddingItems):
            # Handle pre-computed embeddings
            logger.info("[EmbeddingPatch] Detected ImageEmbeddingItems, processing embeddings")
            return _process_embeddings(self, prompt, images, mm_kwargs, tok_kwargs)

        # For raw PIL images, use the original processor
        logger.debug("[EmbeddingPatch] Processing raw PIL images")
        return original_method(self, prompt, mm_data, mm_kwargs, tok_kwargs)

    return patched_call_hf_processor


def _process_embeddings(
    processor_self,
    prompt: str,
    image_embeds,  # ImageEmbeddingItems
    mm_kwargs: Mapping[str, object],
    tok_kwargs: Mapping[str, object],
) -> BatchFeature:
    """
    Process pre-computed image embeddings

    Args:
        processor_self: The processor instance (self)
        prompt: Text prompt with <image> tokens
        image_embeds: ImageEmbeddingItems with pre-computed embeddings
        mm_kwargs: Multimodal kwargs
        tok_kwargs: Tokenizer kwargs

    Returns:
        BatchFeature with tokenized input and embeddings
    """
    from vllm.transformers_utils.processors.deepseek_ocr import BASE_SIZE, IMAGE_SIZE

    # Get tokenizer
    tokenizer = processor_self.info.get_tokenizer()

    # Get image token
    hf_processor = processor_self.info.get_hf_processor()
    image_token = hf_processor.image_token
    image_token_id = tokenizer.vocab[image_token]

    # Count images in prompt
    num_images = prompt.count(image_token)

    if num_images != len(image_embeds):
        raise ValueError(
            f"Prompt contains {num_images} <image> tokens but "
            f"got {len(image_embeds)} image embeddings"
        )

    # Get number of tokens per embedding
    first_embed = image_embeds.get(0)
    num_tokens_per_image = first_embed.shape[0]  # Should be 111

    logger.info(f"[EmbeddingPatch] Processing {num_images} embeddings, "
                f"{num_tokens_per_image} tokens each")

    # Tokenize prompt with image placeholders
    text_splits = prompt.split(image_token)

    all_input_ids = []
    for idx, text_part in enumerate(text_splits):
        # Tokenize text part
        if idx == 0:
            # First part: add BOS
            text_ids = tokenizer.encode(text_part, add_special_tokens=True)
        else:
            # Other parts: no BOS
            text_ids = tokenizer.encode(text_part, add_special_tokens=False)

        all_input_ids.extend(text_ids)

        # Add image tokens (except after the last text split)
        if idx < len(text_splits) - 1:
            all_input_ids.extend([image_token_id] * num_tokens_per_image)

    # Remove final EOS if present (vLLM adds it)
    if all_input_ids and all_input_ids[-1] == tokenizer.eos_token_id:
        all_input_ids = all_input_ids[:-1]

    # Convert to tensor
    input_ids = torch.tensor([all_input_ids], dtype=torch.long)

    # Create spatial crop info (single tile for embeddings)
    images_spatial_crop = torch.tensor(
        [[1, 1]] * num_images,
        dtype=torch.long
    )

    # Create dummy pixel_values and images_crop
    pixel_values = torch.zeros((num_images, 3, BASE_SIZE, BASE_SIZE))
    images_crop = torch.zeros((0, 3, IMAGE_SIZE, IMAGE_SIZE))

    # Stack embeddings
    if isinstance(image_embeds.data, list):
        embeddings_batch = torch.stack(image_embeds.data, dim=0)
    else:
        embeddings_batch = image_embeds.data

    # Move to CPU for vLLM V1 serialization (will be moved back to GPU by vLLM)
    embeddings_batch = embeddings_batch.cpu()

    logger.info(f"[EmbeddingPatch] Created BatchFeature: "
                f"input_ids={input_ids.shape}, "
                f"embeddings={embeddings_batch.shape}")

    return BatchFeature(
        data=dict(
            input_ids=input_ids,
            # Put embeddings in pixel_values (vLLM will pass this through)
            # Shape: [1, 111, 1280] for embeddings vs [N, 3, H, W] for images
            pixel_values=embeddings_batch,  # The actual embeddings!
            images_crop=images_crop,  # Empty for single-tile
            images_spatial_crop=images_spatial_crop,  # [1, 1] for single-tile
        ),
        tensor_type="pt",
    )


def patch_deepseek_ocr_processor():
    """
    Apply monkey-patch to DeepseekOCRMultiModalProcessor

    This patches both:
    1. _call_hf_processor to handle ImageEmbeddingItems
    2. _get_prompt_updates to skip size calculation for embeddings
    """
    try:
        from vllm.model_executor.models.deepseek_ocr import DeepseekOCRMultiModalProcessor
        from vllm.multimodal.parse import ImageEmbeddingItems
        from vllm.multimodal.processing import PromptReplacement

        # Save original methods
        original_call_hf = DeepseekOCRMultiModalProcessor._call_hf_processor
        original_get_prompt_updates = DeepseekOCRMultiModalProcessor._get_prompt_updates

        # Apply _call_hf_processor patch
        DeepseekOCRMultiModalProcessor._call_hf_processor = create_patched_call_hf_processor(
            original_call_hf
        )

        # Apply _get_prompt_updates patch
        def patched_get_prompt_updates(
            self,
            mm_items,
            hf_processor_mm_kwargs,
            out_mm_kwargs,
        ):
            """Patched _get_prompt_updates that handles ImageEmbeddingItems"""
            from vllm.multimodal.parse import ImageEmbeddingItems, ImageProcessorItems

            hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
            image_token_id = hf_processor.image_token_id
            assert isinstance(image_token_id, int)

            def get_replacement_deepseek_vl2(item_idx: int):
                images = mm_items.get_items(
                    "image", (ImageEmbeddingItems, ImageProcessorItems)
                )

                # Check if we're dealing with embeddings (2D tensors) or images (3D tensors)
                first_item = images.get(item_idx)

                if isinstance(first_item, torch.Tensor):
                    # Check tensor dimensions to distinguish embeddings from images
                    if first_item.dim() == 2:
                        # 2D tensor = embeddings [seq_len, hidden_dim]
                        num_image_tokens = first_item.shape[0]  # seq_len = 111
                        logger.info(f"[EmbeddingPatch] Image {item_idx}: {num_image_tokens} tokens (detected from 2D tensor shape)")
                    elif first_item.dim() == 3:
                        # 3D tensor = image [C, H, W], calculate token count
                        from vllm.transformers_utils.processors.deepseek_ocr import CROP_MODE
                        size = images.get_image_size(item_idx)

                        num_image_tokens = self.info.get_num_image_tokens(
                            image_width=size.width,
                            image_height=size.height,
                            cropping=CROP_MODE,
                        )
                        logger.debug(f"[EmbeddingPatch] Image {item_idx}: {num_image_tokens} tokens (calculated from image size)")
                    else:
                        # Unexpected dimensions, use default
                        num_image_tokens = 111
                        logger.warning(f"[EmbeddingPatch] Image {item_idx}: unexpected tensor dim={first_item.dim()}, using default 111 tokens")
                elif isinstance(images, ImageEmbeddingItems):
                    # Direct ImageEmbeddingItems
                    num_image_tokens = images.get_feature_size(item_idx)
                    logger.debug(f"[EmbeddingPatch] Image {item_idx}: {num_image_tokens} tokens (from ImageEmbeddingItems)")
                else:
                    # PIL Image or other format
                    from vllm.transformers_utils.processors.deepseek_ocr import CROP_MODE
                    size = images.get_image_size(item_idx)

                    num_image_tokens = self.info.get_num_image_tokens(
                        image_width=size.width,
                        image_height=size.height,
                        cropping=CROP_MODE,
                    )
                    logger.debug(f"[EmbeddingPatch] Image {item_idx}: {num_image_tokens} tokens (from PIL image)")

                return [image_token_id] * num_image_tokens

            return [
                PromptReplacement(
                    modality="image",
                    target=[image_token_id],
                    replacement=get_replacement_deepseek_vl2,
                )
            ]

        DeepseekOCRMultiModalProcessor._get_prompt_updates = patched_get_prompt_updates

        logger.info("="*60)
        logger.info("[EmbeddingPatch] Successfully patched DeepseekOCRMultiModalProcessor")
        logger.info("[EmbeddingPatch] - Patched _call_hf_processor")
        logger.info("[EmbeddingPatch] - Patched _get_prompt_updates")
        logger.info("[EmbeddingPatch] Model now supports ImageEmbeddingItems!")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"[EmbeddingPatch] Failed to patch processor: {e}")
        raise


def patch_deepseek_ocr_model():
    """
    Apply monkey-patch to DeepseekOCRForCausalLM to handle pre-computed embeddings

    This patches embed_multimodal to check for image_embeds and bypass vision encoder.
    """
    try:
        from vllm.model_executor.models.deepseek_ocr import DeepseekOCRForCausalLM

        # Save original method
        original_embed = DeepseekOCRForCausalLM.embed_multimodal

        def patched_embed_multimodal(self, **kwargs):
            """Patched embed_multimodal that handles pre-computed embeddings"""
            logger.info(f"[EmbeddingPatch] embed_multimodal called with keys: {list(kwargs.keys())}")

            pixel_values = kwargs.get("pixel_values")

            # Check if pixel_values contains embeddings (3D) or images (4D)
            if pixel_values is not None and isinstance(pixel_values, torch.Tensor):
                logger.info(f"[EmbeddingPatch] pixel_values shape: {pixel_values.shape}, dim: {pixel_values.dim()}")

                if pixel_values.dim() == 3:
                    # Shape: [batch, seq_len, hidden_dim] = embeddings
                    logger.info(f"[EmbeddingPatch] Detected pre-computed embeddings in pixel_values")

                    # Convert to list format (NestedTensors)
                    embeddings_list = [pixel_values[i] for i in range(pixel_values.shape[0])]
                    return embeddings_list

                elif pixel_values.dim() == 4:
                    # Shape: [batch, C, H, W] = images, use vision encoder
                    logger.info(f"[EmbeddingPatch] Detected PIL images in pixel_values, using vision encoder")
                    return original_embed(self, **kwargs)

            # Fallback: no pixel_values or unexpected format
            logger.info(f"[EmbeddingPatch] Using vision encoder (fallback)")
            return original_embed(self, **kwargs)

        # Apply patch
        DeepseekOCRForCausalLM.embed_multimodal = patched_embed_multimodal

        logger.info("[EmbeddingPatch] Successfully patched DeepseekOCRForCausalLM")

    except Exception as e:
        logger.error(f"[EmbeddingPatch] Failed to patch model: {e}")
        raise


# Auto-apply patches when module is imported
logger.info("[EmbeddingPatch] Applying patches...")
patch_deepseek_ocr_processor()
patch_deepseek_ocr_model()
logger.info("[EmbeddingPatch] All patches applied successfully!")
