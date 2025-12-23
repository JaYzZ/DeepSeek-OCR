#!/usr/bin/env python3
"""
vLLM-based Decoder for Qwen3-VL with Pre-computed Visual Embeddings

This decoder uses vLLM for efficient production inference, accepting pre-computed
visual embeddings from OCR encoders or other vision encoders.

**Production-Ready Features:**
- Accepts pre-computed visual embeddings (bypass vision encoder)
- Batch processing with multiple images
- Proper MRoPE handling via grid_thw
- vLLM's fast inference with PagedAttention
- Support for Qwen3-VL deepstack features (multi-level visual features)

**Deepstack Features:**
Qwen3-VL uses deepstack features (3 intermediate layers from vision encoder) for better
visual understanding. vLLM expects these features concatenated along the hidden dimension:

    Format: [final_features | deepstack_0 | deepstack_1 | deepstack_2]
    Shape: [seq_len, hidden_dim * (1 + num_deepstack_levels)]
    Example: [111, 1280 * 4] = [111, 5120] for OCR with 3 deepstack levels

**Usage:**
    from OCRVL.decoder import Qwen3VLDecoder
    from OCRInfer.encoder import DPSKOCREncoder

    # Encode images with deepstack
    encoder = DPSKOCREncoder()
    ocr_features, deepstack = encoder.encode_images_with_deepstack(images)

    # Option 1: Let decoder concatenate deepstack automatically
    decoder = Qwen3VLDecoder()
    texts = decoder.decode(
        visual_embeddings=ocr_features,  # List of [seq_len, dim] tensors
        deepstack_features=deepstack,    # List of list of [seq_len, dim] tensors
        grid_thw=[[1, 10, 10]] * len(images),
        prompts=["Transcribe:" for _ in images],
    )

    # Option 2: Manually concatenate deepstack before passing
    import torch
    concatenated = [
        torch.cat([feat] + ds_levels, dim=-1)
        for feat, ds_levels in zip(ocr_features, deepstack)
    ]
    texts = decoder.decode(
        visual_embeddings=concatenated,  # Already concatenated
        grid_thw=[[1, 10, 10]] * len(images),
        prompts=["Transcribe:" for _ in images],
    )
"""

import logging
from typing import List, Optional, Union

import torch
from OCRInfer.utils.model_paths import resolve_model_path
from vllm import LLM, SamplingParams

# Apply runtime fix for vLLM dict embedding bug
from .vllm_embedding_fix import apply_vllm_embedding_fix
apply_vllm_embedding_fix()

logger = logging.getLogger(__name__)


class Qwen3VLDecoder:
    """
    vLLM-based decoder for Qwen3-VL accepting pre-computed visual embeddings
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize vLLM decoder for Qwen3-VL

        Args:
            model_path: HuggingFace model path or local path
            device: Device for inference (parsed from tensor_parallel_size)
            dtype: Model dtype (bfloat16 recommended)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        logger.info("=" * 70)
        logger.info("Initializing Qwen3VLDecoder (vLLM)")
        logger.info("=" * 70)
        logger.info(f"  Model: {model_path}")
        logger.info(f"  GPU Memory: {gpu_memory_utilization}")
        logger.info(f"  Max Length: {max_model_len}")
        logger.info(f"  Dtype: {dtype}")
        logger.info(f"  Tensor Parallel: {tensor_parallel_size}")

        self.model_path = resolve_model_path(model_path)
        self.device = device
        self.dtype = dtype

        if self.model_path != model_path:
            logger.info(f"  Using local model mirror: {self.model_path}")

        # Initialize vLLM engine
        logger.info("\nInitializing vLLM engine...")
        self.llm = LLM(
            model=str(self.model_path),
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 100},  # Support many image chunks
            enable_mm_embeds=True,  # Enable pre-computed embeddings
        )

        # Get tokenizer for constructing prompts
        self.tokenizer = self.llm.get_tokenizer()

        # Get config for special token IDs
        self.config = self.llm.llm_engine.model_config.hf_config
        self.vision_start_id = self.config.vision_start_token_id
        self.image_pad_id = self.config.image_token_id
        self.vision_end_id = self.config.vision_end_token_id

        logger.info("=" * 70)
        logger.info("✓ Qwen3VLDecoder Ready (vLLM)")
        logger.info("=" * 70)

    @staticmethod
    def concatenate_deepstack(
        visual_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        deepstack_features: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
        Concatenate deepstack features with visual embeddings for vLLM.

        vLLM's Qwen3-VL expects deepstack features concatenated along the hidden dimension:
        [final_features | deepstack_0 | deepstack_1 | deepstack_2]

        Args:
            visual_embeddings: Final visual embeddings
                - Single: [seq_len, dim]
                - Batch: List of [seq_len, dim]
            deepstack_features: Deepstack features per image
                - Format: [[ds0, ds1, ds2], [ds0, ds1, ds2], ...]
                - Each ds: [seq_len, dim]

        Returns:
            List of concatenated embeddings, each [seq_len, dim * (1 + num_levels)]

        Example:
            >>> final = [torch.randn(111, 1280), torch.randn(111, 1280)]
            >>> deepstack = [[torch.randn(111, 1280) for _ in range(3)] for _ in range(2)]
            >>> concat = Qwen3VLDecoder.concatenate_deepstack(final, deepstack)
            >>> concat[0].shape
            torch.Size([111, 5120])  # 1280 * 4
        """
        # Normalize to list
        if isinstance(visual_embeddings, torch.Tensor):
            if visual_embeddings.ndim == 2:
                embeddings_list = [visual_embeddings]
            elif visual_embeddings.ndim == 3:
                embeddings_list = [visual_embeddings[i] for i in range(visual_embeddings.shape[0])]
            else:
                raise ValueError(f"visual_embeddings must be 2D or 3D, got {visual_embeddings.ndim}D")
        else:
            embeddings_list = list(visual_embeddings)

        if len(embeddings_list) != len(deepstack_features):
            raise ValueError(
                f"Mismatch: {len(embeddings_list)} embeddings but {len(deepstack_features)} deepstack sets"
            )

        # Concatenate each image's features with its deepstack levels
        concatenated = []
        for final_feat, ds_levels in zip(embeddings_list, deepstack_features):
            # Validate deepstack levels
            if not isinstance(ds_levels, (list, tuple)) or len(ds_levels) != 3:
                raise ValueError(f"Expected 3 deepstack levels per image, got {len(ds_levels)}")

            # Concatenate: [final | ds0 | ds1 | ds2]
            concat_feat = torch.cat([final_feat] + list(ds_levels), dim=-1)
            concatenated.append(concat_feat)

        return concatenated

    def decode(
        self,
        visual_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        grid_thw: Optional[Union[List[int], List[List[int]]]] = None,
        prompts: Union[str, List[str]] = "Transcribe all text:",
        deepstack_features: Optional[List[List[torch.Tensor]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **generation_kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from pre-computed visual embeddings using vLLM

        Args:
            visual_embeddings: Pre-computed visual embeddings
                - Single: torch.Tensor of shape [seq_len, dim] or [seq_len, dim*(1+deepstack)]
                - Batch: List of tensors, each [seq_len, dim] or [seq_len, dim*(1+deepstack)]
                - Example: [111, 1280] for OCR features without deepstack
                - Example: [111, 5120] for OCR features with 3 deepstack levels concatenated
            grid_thw: MRoPE grid (temporal, height, width) for each image
                - Single grid: [t, h, w]  → broadcast to all images
                - Per-image grids: [[t,h,w], [t,h,w], ...]
                - If None: auto-infer from token count (111 tokens → [1,10,10])
                - Example: [1, 10, 10] for OCR (10×10 grid + separators)
            prompts: Text prompt(s) for generation
                - Single string → same prompt for all images
                - List of strings → one per image
            deepstack_features: Optional deepstack features for Qwen3-VL
                - Format: [[ds0, ds1, ds2], [ds0, ds1, ds2], ...] for each image
                - Each ds: [seq_len, dim] tensor (same seq_len as visual_embeddings)
                - If provided, will be concatenated with visual_embeddings automatically
                - If None: visual_embeddings must already include deepstack (concatenated)
            max_tokens: Maximum tokens to generate per image
            temperature: Sampling temperature (0.0 = greedy decoding)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text (str if single input, List[str] if batch)

        Example:
            # With automatic deepstack concatenation
            >>> decoder = Qwen3VLDecoder()
            >>> final_feats = [torch.randn(111, 1280)]
            >>> deepstack = [[torch.randn(111, 1280) for _ in range(3)]]
            >>> text = decoder.decode(final_feats, deepstack_features=deepstack)

            # With pre-concatenated embeddings
            >>> concat_feats = [torch.randn(111, 5120)]  # Already concatenated
            >>> text = decoder.decode(concat_feats)
        """
        from .grid_thw_utils import get_ocr_grid_thw

        # Normalize embeddings to list
        if isinstance(visual_embeddings, torch.Tensor):
            if visual_embeddings.ndim == 2:
                embeddings_list = [visual_embeddings]
            elif visual_embeddings.ndim == 3:
                embeddings_list = [visual_embeddings[i] for i in range(visual_embeddings.shape[0])]
            else:
                raise ValueError(f"visual_embeddings must be 2D or 3D, got {visual_embeddings.ndim}D")
        else:
            embeddings_list = list(visual_embeddings)

        # Handle deepstack features if provided
        if deepstack_features is not None:
            logger.info("Concatenating deepstack features with visual embeddings...")
            embeddings_list = self.concatenate_deepstack(embeddings_list, deepstack_features)
            logger.info(f"✓ Concatenated embeddings shape: {embeddings_list[0].shape}")

        batch_size = len(embeddings_list)

        # Auto-infer or normalize grid_thw
        if grid_thw is None:
            # Auto-infer from token count
            grid_thw_list = [get_ocr_grid_thw(emb) for emb in embeddings_list]
            logger.info(f"Auto-inferred grid_thw from token counts: {grid_thw_list[0]}")
        elif isinstance(grid_thw[0], int):
            # Single grid for all images
            grid_thw_list = [grid_thw] * batch_size
        else:
            grid_thw_list = grid_thw
            if len(grid_thw_list) != batch_size:
                raise ValueError(
                    f"grid_thw count ({len(grid_thw_list)}) != embedding count ({batch_size})"
                )

        # Normalize prompts
        if isinstance(prompts, str):
            prompts_list = [prompts] * batch_size
        else:
            prompts_list = list(prompts)
            if len(prompts_list) != batch_size:
                raise ValueError(
                    f"Prompt count ({len(prompts_list)}) != embedding count ({batch_size})"
                )

        # Build vLLM inputs with custom embedding items
        vllm_inputs = []
        for idx, (embedding, grid, prompt_text) in enumerate(zip(embeddings_list, grid_thw_list, prompts_list)):
            # Get number of visual tokens
            num_visual_tokens = embedding.shape[0]

            # Build input_ids with vision placeholder tokens
            # Format: <|vision_start|> <|image_pad|> ... <|image_pad|> <|vision_end|> <prompt>
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            # Construct full input
            input_ids = (
                [self.vision_start_id]
                + [self.image_pad_id] * num_visual_tokens
                + [self.vision_end_id]
                + prompt_ids
            )

            # Prepare vLLM multimodal input with pre-computed embeddings
            # For Qwen3-VL: Pass embeddings as dict via multi_modal_data
            # vLLM expects DictEmbeddingItems with fields: image_embeds, image_grid_thw
            vllm_input = {
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "image": {
                        "image_embeds": embedding.cpu(),  # [seq_len, dim]
                        "image_grid_thw": torch.tensor([grid], dtype=torch.long).cpu(),  # [1, 3]
                    }
                },
            }
            vllm_inputs.append(vllm_input)

        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p if temperature > 0 else 1.0,
            **generation_kwargs,
        )

        # Generate with vLLM
        outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params)

        # Extract generated text
        results = [output.outputs[0].text for output in outputs]

        # Return single string if single input, else list
        return results[0] if len(results) == 1 and isinstance(prompts, str) else results

    def __repr__(self) -> str:
        return f"Qwen3VLDecoder(model={self.model_path}, backend=vLLM)"
