#!/usr/bin/env python3
"""
vLLM Decoder Using MultiModal ImageEmbeddingItems API

This decoder uses vLLM's proper multimodal API with ImageEmbeddingItems to pass
pre-computed visual embeddings. This goes through vLLM's internal multimodal
processing pipeline, unlike prompt_embeds which bypasses it.

Key difference from vllm_v1_decoder.py:
- vllm_v1_decoder uses prompt_embeds (bypasses multimodal pipeline) ❌
- This decoder uses ImageEmbeddingItems (proper multimodal pipeline) ✅

Architecture:
  Visual Tokens [110, 1280] (from lightweight encoder)
       ↓
  Append view_separator → [111, 1280]
       ↓
  ImageEmbeddingItems wrapper
       ↓
  vLLM MultiModal Pipeline
       ↓
  LLM.generate({"image": ImageEmbeddingItems(...)})
       ↓
  Generated Text

Usage:
    from vllm_embedding_decoder import VLLMEmbeddingDecoder

    decoder = VLLMEmbeddingDecoder(model_path="deepseek-ai/DeepSeek-OCR")
    text = decoder.decode(
        visual_embeddings=vistoks,  # [110, 1280]
        prompt="Transcribe all text:",
        max_tokens=2048
    )
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import torch
from OCRInfer.utils.model_paths import resolve_model_path


# CRITICAL: Apply embedding patches BEFORE importing vLLM
from . import embedding_patch  # noqa: F401  # This patches vLLM to support ImageEmbeddingItems

# Import vLLM components
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor
from vllm.multimodal.parse import ImageEmbeddingItems

logger = logging.getLogger(__name__)


class VLLMEmbeddingDecoder:
    """
    Decoder using vLLM with ImageEmbeddingItems for pre-computed embeddings

    This decoder properly integrates with vLLM's multimodal pipeline,
    unlike prompt_embeds which bypasses it entirely.
    """

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        device: str = "cuda",
    ):
        """
        Initialize vLLM decoder with ImageEmbeddingItems support

        Args:
            model_path: HuggingFace model path
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            dtype: Model dtype
            device: Device for processing
        """
        logger.info("="*60)
        logger.info("Initializing VLLMEmbeddingDecoder (ImageEmbeddingItems)")
        logger.info("="*60)
        logger.info(f"  Model: {model_path}")
        logger.info(f"  GPU Memory: {gpu_memory_utilization}")
        logger.info(f"  Max Length: {max_model_len}")
        logger.info(f"  Dtype: {dtype}")

        self.model_path = resolve_model_path(model_path)
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        if self.model_path != model_path:
            logger.info(f"  Using local model mirror: {self.model_path}")

        # Load view_separator (needed to match vLLM's internal format)
        logger.info("\n[Step 1] Loading view_separator...")
        self._load_view_separator()

        # Initialize vLLM engine
        logger.info("\n[Step 2] Initializing vLLM engine...")
        self.llm = LLM(
            model=self.model_path,
            enable_prefix_caching=False,  # Visual tokens not cacheable
            mm_processor_cache_gb=0,  # Disable processor cache
            logits_processors=[NGramPerReqLogitsProcessor],  # Prevent repetition
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True,
            dtype=dtype
        )

        logger.info("="*60)
        logger.info("✓ VLLMEmbeddingDecoder Ready")
        logger.info("="*60)

    def _load_view_separator(self):
        """
        Load view_separator token from model

        The lightweight encoder outputs [110, 1280], but vLLM expects [111, 1280]
        with view_separator appended for single-tile images.
        """
        try:
            from . import transformers_patch  # noqa: F401
        except ImportError:
            pass

        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=self.dtype,
            device_map=self.device
        )

        if hasattr(model.model, 'view_seperator'):  # Note: typo in original model
            self.view_separator = model.model.view_seperator.to(
                device=self.device, dtype=self.dtype
            )
            logger.info(f"  ✓ View separator loaded: {self.view_separator.shape}")
        else:
            raise RuntimeError("Cannot find view_seperator in model")

        del model
        torch.cuda.empty_cache()

    def decode(
        self,
        visual_embeddings: torch.Tensor,
        prompt: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        ngram_size: int = 30,
        window_size: int = 90,
    ) -> str:
        """
        Decode visual embeddings to text using vLLM with ImageEmbeddingItems

        Args:
            visual_embeddings: Visual embeddings [seq_len, hidden_dim] or [1, seq_len, hidden_dim]
            prompt: Text prompt (e.g., "Transcribe all text:")
                   Will be formatted as: <image>\n{prompt}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            ngram_size: N-gram size for repetition blocking
            window_size: Window size for n-gram blocking

        Returns:
            Generated text
        """
        # Format full prompt with <image> token
        if prompt:
            full_prompt = f"<image>\n{prompt}"
        else:
            full_prompt = "<image>\nFree OCR."

        logger.info(f"Decoding with vLLM + ImageEmbeddingItems")
        logger.info(f"  Prompt: {repr(full_prompt)}")
        logger.info(f"  Visual embeddings (input): {visual_embeddings.shape}")

        # Normalize shape: [seq_len, hidden_dim]
        if visual_embeddings.dim() == 3:
            visual_embeddings = visual_embeddings.squeeze(0)

        visual_embeddings = visual_embeddings.to(device=self.device, dtype=self.dtype)

        # Handle different embedding shapes for backward compatibility
        # New encoder outputs [111, 1280] with view_separator included
        # Old encoder outputs [110, 1280] without view_separator
        if visual_embeddings.shape[0] == 110:
            # Old format: append view_separator
            visual_embeddings = torch.cat([
                visual_embeddings,
                self.view_separator.unsqueeze(0)
            ], dim=0)
            logger.info(f"  ✓ Appended view_separator: [110, 1280] -> [111, 1280]")
        elif visual_embeddings.shape[0] == 111:
            # New format: view_separator already included
            logger.info(f"  ✓ Using embeddings with view_separator already included: [111, 1280]")

        logger.info(f"  Visual embeddings (final): {visual_embeddings.shape}")

        # Wrap in ImageEmbeddingItems (proper multimodal API)
        image_embeds = ImageEmbeddingItems(data=[visual_embeddings])
        logger.info(f"  ✓ Wrapped in ImageEmbeddingItems")

        # Create sampling params with n-gram blocking
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0.0 else 0.9,
            extra_args=dict(
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            ),
            skip_special_tokens=False,
        )

        # Generate with vLLM using MultiModalDataDict
        logger.info(f"  Generating with vLLM...")

        try:
            model_input = {
                "prompt": full_prompt,
                "multi_modal_data": {
                    "image": image_embeds  # Pass ImageEmbeddingItems
                }
            }

            outputs = self.llm.generate(model_input, sampling_params=sampling_params)
            text = outputs[0].outputs[0].text
            logger.info(f"  ✓ Generated {len(text)} characters")

            return text

        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise

    def decode_batch(
        self,
        visual_embeddings_list: list[torch.Tensor],
        prompts: Optional[list[str]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        ngram_size: int = 30,
        window_size: int = 90,
    ) -> list[str]:
        """
        Decode multiple visual embeddings to text using vLLM with BATCHED processing

        This is the production-ready method for high-throughput decoding.
        vLLM will process multiple requests efficiently using continuous batching.

        Args:
            visual_embeddings_list: List of visual embeddings [seq_len, hidden_dim]
            prompts: List of text prompts (e.g., ["Transcribe all text:", ...])
                    If None, uses default prompt for all
            max_tokens: Maximum tokens to generate per request
            temperature: Sampling temperature (0.0 = greedy)
            ngram_size: N-gram size for repetition blocking
            window_size: Window size for n-gram blocking

        Returns:
            List of generated texts
        """
        if len(visual_embeddings_list) == 0:
            return []

        # Default prompts if not provided
        if prompts is None:
            prompts = [""] * len(visual_embeddings_list)

        assert len(visual_embeddings_list) == len(prompts), \
            f"Mismatch: {len(visual_embeddings_list)} embeddings but {len(prompts)} prompts"

        logger.info(f"Batch decoding {len(visual_embeddings_list)} items with vLLM")

        # Prepare all inputs
        model_inputs = []
        for idx, (visual_embeddings, prompt) in enumerate(zip(visual_embeddings_list, prompts)):
            # Format full prompt with <image> token
            if prompt:
                full_prompt = f"<image>\n{prompt}"
            else:
                full_prompt = "<image>\nFree OCR."

            # Normalize shape: [seq_len, hidden_dim]
            if visual_embeddings.dim() == 3:
                visual_embeddings = visual_embeddings.squeeze(0)

            visual_embeddings = visual_embeddings.to(device=self.device, dtype=self.dtype)

            # Handle different embedding shapes for backward compatibility
            # New encoder outputs [111, 1280] with view_separator included
            # Old encoder outputs [110, 1280] without view_separator
            if visual_embeddings.shape[0] == 110:
                # Old format: append view_separator
                visual_embeddings = torch.cat([
                    visual_embeddings,
                    self.view_separator.unsqueeze(0)
                ], dim=0)
                logger.debug(f"  Appended view_separator: [110, 1280] -> [111, 1280]")
            elif visual_embeddings.shape[0] == 111:
                # New format: view_separator already included
                logger.debug(f"  Using embeddings with view_separator: [111, 1280]")
            else:
                # Unexpected shape - log warning but continue
                logger.warning(f"  ⚠ Unexpected embedding shape: {visual_embeddings.shape}, expected [110, 1280] or [111, 1280]")

            # Wrap in ImageEmbeddingItems (proper multimodal API)
            image_embeds = ImageEmbeddingItems(data=[visual_embeddings])

            model_input = {
                "prompt": full_prompt,
                "multi_modal_data": {
                    "image": image_embeds  # Pass ImageEmbeddingItems
                }
            }
            model_inputs.append(model_input)

        # Create sampling params with n-gram blocking
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0 if temperature == 0.0 else 0.9,
            extra_args=dict(
                ngram_size=ngram_size,
                window_size=window_size,
                whitelist_token_ids={128821, 128822},  # <td>, </td>
            ),
            skip_special_tokens=False,
        )

        # Generate with vLLM using BATCHED processing
        logger.info(f"  Generating {len(model_inputs)} texts with vLLM continuous batching...")

        try:
            # vLLM handles batching internally with continuous batching
            outputs = self.llm.generate(model_inputs, sampling_params=sampling_params)

            # Extract texts
            texts = [output.outputs[0].text for output in outputs]
            logger.info(f"  ✓ Generated {len(texts)} texts (avg {sum(len(t) for t in texts) / len(texts):.0f} chars)")

            return texts

        except Exception as e:
            logger.error(f"vLLM batch generation failed: {e}")
            raise


def main():
    """Usage example"""
    print("\nVLLMEmbeddingDecoder - Uses ImageEmbeddingItems for proper multimodal processing")
    print()
    print("Usage:")
    print("  from vllm_embedding_decoder import VLLMEmbeddingDecoder")
    print()
    print("  decoder = VLLMEmbeddingDecoder(model_path='deepseek-ai/DeepSeek-OCR')")
    print("  text = decoder.decode(")
    print("      visual_embeddings=vistoks,  # [110, 1280]")
    print("      prompt='Transcribe all text:',")
    print("      max_tokens=2048")
    print("  )")


if __name__ == "__main__":
    main()
