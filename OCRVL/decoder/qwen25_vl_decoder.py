#!/usr/bin/env python3
"""
vLLM-based Decoder for Qwen2.5-VL with Pre-computed Visual Features

This decoder uses vLLM for efficient production inference, accepting pre-computed
visual embeddings from any source (OCR encoder, native encoder, cached embeddings).

Key features:
- Accepts visual features from ANY source (OCR, native, custom, cached)
- Proper MRoPE (grid_thw) handling for position embeddings
- vLLM-optimized for high-throughput inference
- Qwen2.5-VL uses final-layer features only (no deepstack)

Architecture:
  Visual Features [seq_len, dim] + grid_thw → vLLM (image_embeds) → Generated Text

Visual tokens can come from:
  - DeepSeek-OCR encoder (111×1280, grid=[1,10,10])
  - Qwen2.5-VL native encoder (variable length)
  - Any custom vision encoder
  - Pre-computed/cached embeddings

Usage:
    from OCRVL.decoder import Qwen25VLDecoder

    decoder = Qwen25VLDecoder()
    text = decoder.decode(
        visual_features=visual_tokens,   # [seq_len, dim] or list of tensors
        grid_thw=[1, 10, 10],            # MRoPE grid (temporal, height, width)
        prompt="Transcribe all text:",
        max_tokens=2048
    )
"""

import logging
from typing import List, Optional, Union

import torch
from OCRInfer.utils.model_paths import resolve_model_path
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class Qwen25VLDecoder:
    """
    vLLM-based decoder for Qwen2.5-VL with pre-computed visual features

    This decoder properly integrates with vLLM's multimodal pipeline using the
    `image_embeds` parameter to bypass vision encoder and accept pre-computed
    visual embeddings from any source.
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
    ):
        """
        Initialize vLLM decoder for Qwen2.5-VL

        Args:
            model_path: HuggingFace model path or local path
            device: Device for inference (parsed from tensor_parallel_size)
            dtype: Model dtype (bfloat16 recommended)
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum sequence length
            tensor_parallel_size: Number of GPUs for tensor parallelism
        """
        logger.info("=" * 70)
        logger.info("Initializing Qwen25VLDecoder (vLLM)")
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
        logger.info("✓ Qwen25VLDecoder Ready (vLLM)")
        logger.info("=" * 70)

    def decode(
        self,
        visual_features: Union[torch.Tensor, List[torch.Tensor]],
        prompt: Union[str, List[str]] = "Transcribe all text:",
        grid_thw: Optional[Union[torch.Tensor, List[int], List[List[int]]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **generation_kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from visual features using vLLM

        Args:
            visual_features: Pre-computed visual tokens from any encoder
                - Single: [seq_len, dim] or [N, seq_len, dim]
                - Batch: List of [seq_len, dim]
                - Examples: [111, 1280] (OCR), [400, 2048] (Qwen native)
            prompt: Text prompt(s) for generation
            grid_thw: MRoPE grid (temporal, height, width) for visual features
                - Single: [t, h, w] or torch.Tensor([t, h, w])
                - Batch: [[t,h,w], ...] or torch.Tensor([[t,h,w], ...])
                - Default: [1, 10, 10] for OCR features (111 tokens = 10×10 + separators)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text (single string or list of strings)
        """
        # Normalize visual features to list
        if isinstance(visual_features, torch.Tensor):
            if visual_features.ndim == 2:
                visual_features_list = [visual_features]
            elif visual_features.ndim == 3:
                visual_features_list = [visual_features[i] for i in range(visual_features.shape[0])]
            else:
                raise ValueError(f"visual_features must be 2D or 3D, got {visual_features.ndim}D")
        else:
            visual_features_list = list(visual_features)

        batch_size = len(visual_features_list)

        # Normalize prompts
        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        else:
            prompts = list(prompt)
            if len(prompts) != batch_size:
                raise ValueError(
                    f"Prompt count ({len(prompts)}) != feature count ({batch_size})"
                )

        # Normalize grid_thw (critical for MRoPE!)
        if grid_thw is None:
            # Default OCR grid: 10×10 visual tokens + separators = 111 total
            grid_thw_list = [[1, 10, 10]] * batch_size
        elif isinstance(grid_thw, torch.Tensor):
            if grid_thw.ndim == 1:
                grid_thw_list = [grid_thw.tolist()] * batch_size
            else:
                grid_thw_list = grid_thw.tolist()
        elif isinstance(grid_thw, list):
            if isinstance(grid_thw[0], int):
                # Single grid for all: [t, h, w]
                grid_thw_list = [grid_thw] * batch_size
            else:
                # Per-image grid: [[t,h,w], ...]
                grid_thw_list = grid_thw
        else:
            raise ValueError(f"Unsupported grid_thw type: {type(grid_thw)}")

        # Build vLLM inputs for batch
        vllm_inputs = []
        for idx, (prompt_text, visual_feat) in enumerate(zip(prompts, visual_features_list)):
            # Get number of visual tokens
            num_visual_tokens = visual_feat.shape[0]

            # Build input_ids with vision placeholder tokens
            # Format: <|vision_start|> <|image_pad|> ... <|image_pad|> <|vision_end|> <prompt>
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            # Construct full input: vision_start + image_pads + vision_end + prompt
            input_ids = (
                [self.vision_start_id]
                + [self.image_pad_id] * num_visual_tokens
                + [self.vision_end_id]
                + prompt_ids
            )

            # Prepare vLLM multimodal input with pre-computed embeddings
            # NOTE: Pass as list of 2D tensors to be recognized as embeddings
            # vLLM checks for list[Tensor(2D)] or Tensor(3D) in is_embeddings()
            vllm_input = {
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "image": [visual_feat.cpu()],  # List of 2D tensors (embeddings)
                },
                "mm_processor_kwargs": {
                    "image_grid_thw": torch.tensor([grid_thw_list[idx]], dtype=torch.long),  # Tensor is hashable
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
        return results[0] if len(results) == 1 and isinstance(prompt, str) else results

    def __repr__(self) -> str:
        return f"Qwen25VLDecoder(model={self.model_path}, backend=vLLM)"
