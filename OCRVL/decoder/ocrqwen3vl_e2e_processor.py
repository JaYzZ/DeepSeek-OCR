"""
OCRQwen3VL End-to-End vLLM Processor

This processor loads the full OCRQwen3VL model directly in vLLM for end-to-end inference.
Unlike OCRVLProcessor which uses base Qwen3VL + separate connectors, this processor
leverages the fully migrated end-to-end OCRQwen3VL model with vLLM's native multimodal support.

Key differences from OCRVLProcessor:
- OCRVLProcessor: Base Qwen3VL + pre-computed OCR embeddings + connector injection
- OCRQwen3VLProcessor: Full OCRQwen3VL model + native vLLM multimodal inputs

Usage:
    ```python
    processor = OCRQwen3VLProcessor(
        checkpoint_path="OCRVL/checkpoints/OCR-Qwen3-VL-2B",
    )

    # Simple inference with image + text prompt
    output = processor.generate(
        images=[image1, image2],
        prompts="What do you see?",
    )

    # Batch inference
    outputs = processor.generate(
        images=[[img1, img2], [img3]],  # Multi-image per prompt
        prompts=["Compare these images", "Describe this"],
    )
    ```
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


class OCRQwen3VLProcessor:
    """
    End-to-end vLLM processor for OCRQwen3VL model.

    This processor loads the full OCRQwen3VL model (with integrated DPSK vision tower)
    directly in vLLM and uses native multimodal input format for efficient inference.

    The end-to-end model handles:
    - DPSK vision encoding internally (via DPSKVisionTowerAdapter)
    - OCR connectors (integrated into the model)
    - Text generation

    vLLM handles:
    - Continuous batching
    - Multimodal input processing
    - Efficient memory management
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        enable_lora: bool = False,
        lora_path: Optional[Union[str, Path]] = None,
        limit_mm_per_prompt: Optional[dict] = None,
    ):
        """
        Initialize OCRQwen3VL end-to-end vLLM processor.

        Args:
            checkpoint_path: Path to OCRQwen3VL checkpoint directory
            gpu_memory_utilization: vLLM GPU memory utilization (0-1)
            max_model_len: Maximum model sequence length
            dtype: Data type for model ("bfloat16", "float16", etc.)
            trust_remote_code: Whether to trust remote code when loading
            enable_lora: Enable LoRA adapters
            lora_path: Optional path to LoRA adapters directory.
                     If None, looks for checkpoint/lora_adapters/
                     Example: "checkpoints/llamafactory/qwen3vl-2b/lora/alignment"
            limit_mm_per_prompt: Limit multimodal inputs per prompt (e.g., {"image": 10})
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.dtype = dtype

        logger.info("=" * 80)
        logger.info("OCRQwen3VL End-to-End vLLM Processor Initialization")
        logger.info("=" * 80)
        logger.info(f"  Checkpoint: {self.checkpoint_path}")
        logger.info(f"  GPU memory: {gpu_memory_utilization}")
        logger.info(f"  Max length: {max_model_len}")
        logger.info(f"  Dtype: {dtype}")
        logger.info("")

        # Check if checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Detect or validate LoRA adapters
        if enable_lora:
            if lora_path is not None:
                # User specified a LoRA path
                self.lora_dir = Path(lora_path)
                if not self.lora_dir.exists():
                    logger.warning(f"  LoRA path not found: {self.lora_dir}")
                    logger.warning(f"  Continuing without LoRA (merged checkpoint?)")
                    self.has_lora = False
                else:
                    self.has_lora = True
            else:
                # Auto-detect LoRA in checkpoint subdirectory
                self.lora_dir = self.checkpoint_path / "lora_adapters"
                self.has_lora = self.lora_dir.exists()

            if self.has_lora:
                logger.info(f"  LoRA adapters: {self.lora_dir}")
                # Check for adapter_config.json
                adapter_config = self.lora_dir / "adapter_config.json"
                if adapter_config.exists():
                    import json
                    with open(adapter_config) as f:
                        cfg = json.load(f)
                    logger.info(f"  LoRA rank (r): {cfg.get('r', 'N/A')}")
                    logger.info(f"  LoRA alpha: {cfg.get('lora_alpha', 'N/A')}")
                    logger.info(f"  Target modules: {len(cfg.get('target_modules', []))} modules")
        else:
            self.lora_dir = None
            self.has_lora = False

        # Initialize tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.checkpoint_path),
            trust_remote_code=trust_remote_code,
        )
        logger.info("  ✓ Tokenizer loaded")

        # Initialize vLLM with full OCRQwen3VL model
        logger.info("Initializing vLLM with OCRQwen3VL model...")

        vllm_kwargs = {
            "model": str(self.checkpoint_path),
            "trust_remote_code": trust_remote_code,
            "dtype": dtype,
            "seed": None,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "disable_log_stats": True,
            "enable_prefix_caching": False,  # Vision tokens not cacheable
        }

        # Set multimodal limits
        if limit_mm_per_prompt:
            vllm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt
        else:
            # Default: support up to 20 images per prompt
            vllm_kwargs["limit_mm_per_prompt"] = {"image": 20}

        # Enable LoRA if requested
        if self.has_lora:
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64

        # Initialize vLLM
        self.llm = LLM(**vllm_kwargs)

        # Pre-load LoRA adapters if present
        if self.has_lora:
            from vllm.lora.request import LoRARequest
            self.lora_request_name = "ocrqwen3vl_lora"
            self.lora_int_id = 1

            lora_request = LoRARequest(
                lora_name=self.lora_request_name,
                lora_int_id=self.lora_int_id,
                lora_path=str(self.lora_dir),
            )
            self.llm.llm_engine.add_lora(lora_request)
            logger.info(f"  ✓ LoRA adapters loaded: {self.lora_dir.name}")
        else:
            self.lora_request_name = None
            self.lora_int_id = None

        logger.info("  ✓ vLLM initialized with OCRQwen3VL model")
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ OCRQwen3VL End-to-End vLLM Processor Ready")
        logger.info("=" * 80)
        logger.info("")

    def generate(
        self,
        images: Union[Image.Image, List[Image.Image], List[List[Image.Image]]],
        prompts: Union[str, List[str]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop_tokens: Optional[List[str]] = None,
        **generation_kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from images using OCRQwen3VL with vLLM.

        Args:
            images: Input images
                - Single image: PIL.Image
                - Multiple images for one prompt: [PIL.Image, ...]
                - Batch with multiple images each: [[PIL.Image, ...], ...]
            prompts: Text prompt(s)
                - str: Single prompt (applied to all images or image list)
                - List[str]: One prompt per image batch
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter
            stop_tokens: Optional stop token strings
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text (str if single prompt, List[str] if batch)
        """
        # Normalize inputs
        is_single_prompt = isinstance(prompts, str)
        is_single_image = isinstance(images, Image.Image)

        if is_single_image:
            # Single image, single prompt
            image_batches = [[images]]
            prompts_list = [prompts] if is_single_prompt else prompts
        elif isinstance(images[0], Image.Image):
            # List of images for single prompt
            if is_single_prompt:
                image_batches = [images]
                prompts_list = [prompts]
            else:
                # Batch: one image per prompt
                image_batches = [[img] for img in images]
                prompts_list = prompts
        else:
            # List of lists (already batched)
            image_batches = images
            prompts_list = [prompts] if is_single_prompt else prompts

        # Ensure we have matching counts
        if len(prompts_list) != len(image_batches):
            raise ValueError(
                f"Number of prompts ({len(prompts_list)}) must match "
                f"number of image batches ({len(image_batches)})"
            )

        # Build vLLM inputs using native multimodal format
        vllm_inputs = []
        for img_batch, prompt_text in zip(image_batches, prompts_list):
            # Build messages format (same as official Qwen3VL)
            messages = [
                {
                    "role": "user",
                    "content": (
                        [{"type": "image", "image": img} for img in img_batch]
                        + [{"type": "text", "text": prompt_text}]
                    ),
                }
            ]

            # Apply chat template to get prompt string
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Build vLLM TextPrompt format
            vllm_input = {
                "prompt": prompt,
                "multi_modal_data": {"image": img_batch},
            }
            vllm_inputs.append(vllm_input)

        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p if temperature > 0 else 1.0,
            stop=stop_tokens,
            **generation_kwargs,
        )

        # Prepare LoRA request if enabled
        lora_request = None
        if self.has_lora:
            from vllm.lora.request import LoRARequest
            lora_request = LoRARequest(
                lora_name=self.lora_request_name,
                lora_int_id=self.lora_int_id,
                lora_path=str(self.lora_dir),
            )

        # Generate with vLLM
        total_images = sum(len(batch) for batch in image_batches)
        logger.info(f"Generating for {total_images} images across {len(vllm_inputs)} prompts...")

        outputs = self.llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        # Extract results
        results = [output.outputs[0].text for output in outputs]

        return results[0] if is_single_prompt else results

    def chat(
        self,
        messages: List[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **generation_kwargs,
    ) -> str:
        """
        Chat interface compatible with OpenAI-style messages format.

        Args:
            messages: List of message dicts with 'role' and 'content'
                Content can be a string or list with {"type": "image", "image": ...}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated response text

        Example:
            ```python
            response = processor.chat([
                {"role": "user", "content": [
                    {"type": "image", "image": img1},
                    {"type": "text", "text": "What's in this image?"}
                ]},
                {"role": "assistant", "content": "A cat."},
                {"role": "user", "content": "What color is it?"},
            ])
            ```
        """
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Extract images from messages
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        images.append(item["image"])

        # Build vLLM input
        vllm_input = {
            "prompt": prompt,
            "multi_modal_data": {"image": images} if images else None,
        }

        # Sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **generation_kwargs,
        )

        # Generate
        outputs = self.llm.generate([vllm_input], sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def shutdown(self):
        """Gracefully shutdown vLLM resources."""
        try:
            if hasattr(self, "llm") and self.llm is not None:
                engine = getattr(self.llm, "llm_engine", None)
                if engine is not None and hasattr(engine, "shutdown"):
                    try:
                        engine.shutdown()
                    except Exception as e:
                        logger.info(f"Info: vLLM engine shutdown returned: {e}")
        finally:
            try:
                del self.llm
            except Exception:
                pass
            import gc
            import torch
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.llm = None

    def __repr__(self) -> str:
        return f"OCRQwen3VLProcessor(checkpoint={self.checkpoint_path.name}, backend=vLLM, mode=end-to-end)"
