#!/usr/bin/env python3
"""vLLM decoder for Qwen3-VL with precomputed visual embeddings."""

import logging
from typing import List, Optional, Union

import torch
from OCRInfer.utils.model_paths import resolve_model_path
from vllm import LLM, SamplingParams

from .vllm_embedding_fix import apply_vllm_embedding_fix

apply_vllm_embedding_fix()

logger = logging.getLogger(__name__)


class Qwen3VLDecoder:
    """vLLM decoder for Qwen3-VL accepting precomputed visual embeddings."""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        tensor_parallel_size: int = 1,
    ):
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
        logger.info("Qwen3VLDecoder ready")
        logger.info("=" * 70)

    @staticmethod
    def concatenate_deepstack(
        visual_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        deepstack_features: List[List[torch.Tensor]],
    ) -> List[torch.Tensor]:
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

        concatenated = []
        for final_feat, ds_levels in zip(embeddings_list, deepstack_features):
            if not isinstance(ds_levels, (list, tuple)) or len(ds_levels) != 3:
                raise ValueError(f"Expected 3 deepstack levels per image, got {len(ds_levels)}")
            concatenated.append(torch.cat([final_feat] + list(ds_levels), dim=-1))

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
        from .grid_thw_utils import get_ocr_grid_thw

        if isinstance(visual_embeddings, torch.Tensor):
            if visual_embeddings.ndim == 2:
                embeddings_list = [visual_embeddings]
            elif visual_embeddings.ndim == 3:
                embeddings_list = [visual_embeddings[i] for i in range(visual_embeddings.shape[0])]
            else:
                raise ValueError(f"visual_embeddings must be 2D or 3D, got {visual_embeddings.ndim}D")
        else:
            embeddings_list = list(visual_embeddings)

        if deepstack_features is not None:
            logger.info("Concatenating deepstack features with visual embeddings...")
            embeddings_list = self.concatenate_deepstack(embeddings_list, deepstack_features)
            logger.info(f"Concatenated embeddings shape: {embeddings_list[0].shape}")

        batch_size = len(embeddings_list)

        if grid_thw is None:
            grid_thw_list = [get_ocr_grid_thw(embedding) for embedding in embeddings_list]
            logger.info(f"Auto-inferred grid_thw from token counts: {grid_thw_list[0]}")
        elif isinstance(grid_thw[0], int):
            grid_thw_list = [grid_thw] * batch_size
        else:
            grid_thw_list = grid_thw
            if len(grid_thw_list) != batch_size:
                raise ValueError(
                    f"grid_thw count ({len(grid_thw_list)}) != embedding count ({batch_size})"
                )

        if isinstance(prompts, str):
            prompts_list = [prompts] * batch_size
        else:
            prompts_list = list(prompts)
            if len(prompts_list) != batch_size:
                raise ValueError(
                    f"Prompt count ({len(prompts_list)}) != embedding count ({batch_size})"
                )

        vllm_inputs = []
        for embedding, grid, prompt_text in zip(embeddings_list, grid_thw_list, prompts_list):
            num_visual_tokens = embedding.shape[0]
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
                    "multi_modal_data": {
                        "image": {
                            "image_embeds": embedding.cpu(),
                            "image_grid_thw": torch.tensor([grid], dtype=torch.long).cpu(),
                        }
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
        return results[0] if len(results) == 1 and isinstance(prompts, str) else results

    def __repr__(self) -> str:
        return f"Qwen3VLDecoder(model={self.model_path}, backend=vLLM)"
