#!/usr/bin/env python3
"""vLLM decoder for Qwen2.5-VL with precomputed visual features."""

import logging
from typing import List, Optional, Union

import torch
from OCRInfer.utils.model_paths import resolve_model_path
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class Qwen25VLDecoder:
    """vLLM decoder for Qwen2.5-VL with precomputed visual features."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
    ):
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

        logger.info("\nInitializing vLLM engine...")
        self.llm = LLM(
            model=str(self.model_path),
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            limit_mm_per_prompt={"image": 100},
            enable_mm_embeds=True,
        )

        self.tokenizer = self.llm.get_tokenizer()
        self.config = self.llm.llm_engine.model_config.hf_config
        self.vision_start_id = self.config.vision_start_token_id
        self.image_pad_id = self.config.image_token_id
        self.vision_end_id = self.config.vision_end_token_id

        logger.info("=" * 70)
        logger.info("Qwen25VLDecoder ready")
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

        if isinstance(prompt, str):
            prompts = [prompt] * batch_size
        else:
            prompts = list(prompt)
            if len(prompts) != batch_size:
                raise ValueError(f"Prompt count ({len(prompts)}) != feature count ({batch_size})")

        if grid_thw is None:
            grid_thw_list = [[1, 10, 10]] * batch_size
        elif isinstance(grid_thw, torch.Tensor):
            if grid_thw.ndim == 1:
                grid_thw_list = [grid_thw.tolist()] * batch_size
            else:
                grid_thw_list = grid_thw.tolist()
        elif isinstance(grid_thw, list):
            if isinstance(grid_thw[0], int):
                grid_thw_list = [grid_thw] * batch_size
            else:
                grid_thw_list = grid_thw
        else:
            raise ValueError(f"Unsupported grid_thw type: {type(grid_thw)}")

        vllm_inputs = []
        for idx, (prompt_text, visual_feat) in enumerate(zip(prompts, visual_features_list)):
            num_visual_tokens = visual_feat.shape[0]
            prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            input_ids = (
                [self.vision_start_id]
                + [self.image_pad_id] * num_visual_tokens
                + [self.vision_end_id]
                + prompt_ids
            )
            vllm_inputs.append(
                {
                    "prompt_token_ids": input_ids,
                    "multi_modal_data": {"image": [visual_feat.cpu()]},
                    "mm_processor_kwargs": {
                        "image_grid_thw": torch.tensor([grid_thw_list[idx]], dtype=torch.long),
                    },
                }
            )

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p if temperature > 0 else 1.0,
            **generation_kwargs,
        )
        outputs = self.llm.generate(vllm_inputs, sampling_params=sampling_params)
        results = [output.outputs[0].text for output in outputs]
        return results[0] if len(results) == 1 and isinstance(prompt, str) else results

    def __repr__(self) -> str:
        return f"Qwen25VLDecoder(model={self.model_path}, backend=vLLM)"
