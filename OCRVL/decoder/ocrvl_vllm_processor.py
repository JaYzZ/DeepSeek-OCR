"""
OCRVL vLLM Processor

Direct vLLM integration for OCRVL without HuggingFace export overhead.
Loads base Qwen3-VL model in vLLM and applies OCRVL connectors on-the-fly.
"""

import gc
import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn as nn
from OCRInfer.utils.model_paths import resolve_model_path
from Qwen.decoder.grid_thw_utils import get_ocr_grid_thw
from Qwen.decoder.vllm_embedding_fix import apply_vllm_embedding_fix
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

logger = logging.getLogger(__name__)


class OCRVLProcessor:
    """
    OCRVL vLLM Processor for efficient inference with direct connector loading.

    Bypasses HuggingFace export by:
    1. Loading base Qwen3-VL model in vLLM
    2. Loading OCRVL connectors separately
    3. Applying connectors to OCR features before vLLM generation

    This avoids:
    - Creating HF checkpoint files
    - Tokenizer reconfiguration
    - Model config modifications

    Example:
        ```python
        processor = OCRVLProcessor(checkpoint_path="checkpoints/step_1000")

        # Single image
        output = processor.generate(
            visual_embeddings=ocr_features,
            deepstack_features=deepstack,
            prompts="Transcribe all text:"
        )

        # Multiple images for one prompt (multi-image)
        output = processor.generate(
            visual_embeddings=[img1_feat, img2_feat],
            deepstack_features=[img1_ds, img2_ds],
            prompts="Answer:"  # Single prompt for both images
        )
        ```
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        base_model: str = "Qwen/Qwen3-VL-2B-Instruct",
        connector_file: str = "connectors.pt",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 8192,
        enable_lora: Optional[bool] = None,
    ):
        """
        Initialize OCRVL vLLM processor.

        Args:
            checkpoint_path: Path to OCRVL checkpoint directory
            base_model: HuggingFace model ID for Qwen3-VL
            connector_file: Filename for connector weights
            device: Device for connectors
            dtype: Data type for connectors
            gpu_memory_utilization: vLLM GPU memory utilization
            max_model_len: Maximum model sequence length
            enable_lora: Enable LoRA adapters if present (auto-detect if None)
        """
        # Fix vLLM embedding handling for multi-modal
        apply_vllm_embedding_fix()

        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.dtype = dtype

        # Log configuration
        logger.info("=" * 80)
        logger.info("OCRVL vLLM Processor Initialization")
        logger.info("=" * 80)
        logger.info(f"  Checkpoint: {self.checkpoint_path}")
        logger.info(f"  Connectors: {connector_file}")
        logger.info(f"  Base model: {base_model}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Dtype: {dtype}")
        logger.info("")

        # Detect and configure LoRA
        self._detect_lora(enable_lora)

        # Load OCRVL connectors
        self._load_connectors(connector_file)

        # Initialize vLLM with base Qwen3-VL (and LoRA if present)
        self._init_vllm(base_model, gpu_memory_utilization, max_model_len)

        logger.info("=" * 80)
        logger.info("✓ OCRVL vLLM Processor Ready")
        logger.info("=" * 80)
        logger.info("")

    def _detect_lora(self, enable_lora: Optional[bool]):
        """Detect and configure LoRA adapters if present."""
        lora_dir = self.checkpoint_path / "lora_adapters"
        lora_config = lora_dir / "adapter_config.json"
        lora_weights = lora_dir / "adapter_model.safetensors"

        has_lora = lora_dir.exists() and lora_config.exists() and lora_weights.exists()

        if enable_lora is None:
            # Auto-detect
            self.use_lora = has_lora
        else:
            # User specified
            self.use_lora = enable_lora
            if self.use_lora and not has_lora:
                raise FileNotFoundError(
                    f"LoRA enabled but adapters not found at {lora_dir}"
                )

        if self.use_lora:
            self.lora_path = lora_dir
            logger.info("LoRA Configuration:")
            logger.info(f"  LoRA adapters: {self.lora_path}")
            logger.info(f"  Status: Enabled")

            # Load LoRA config for logging
            with open(lora_config) as f:
                lora_cfg = json.load(f)
            logger.info(f"  LoRA rank (r): {lora_cfg.get('r', 'N/A')}")
            logger.info(f"  LoRA alpha: {lora_cfg.get('lora_alpha', 'N/A')}")
            logger.info(f"  Target modules: {', '.join(lora_cfg.get('target_modules', []))}")
            logger.info("")
        else:
            self.lora_path = None
            if has_lora:
                logger.info("LoRA adapters found but not enabled (use enable_lora=True to enable)")
                logger.info("")

    def _load_connectors(self, connector_file: str):
        """Load OCRVL connector weights."""
        logger.info("Loading OCRVL connectors...")

        connector_path = self.checkpoint_path / connector_file
        if not connector_path.exists():
            raise FileNotFoundError(f"Connector file not found: {connector_path}")

        # Load connector state dict
        state_dict = torch.load(connector_path, map_location=self.device, weights_only=True)

        # Parse connector architecture from nested structure
        # Format: {'ocr_connector': OrderedDict({'0.weight': ..., '0.bias': ..., '2.weight': ..., '2.bias': ...})}
        ocr_state = state_dict['ocr_connector']
        first_linear_weight = ocr_state['0.weight']
        target_dim = first_linear_weight.shape[0]
        input_dim = first_linear_weight.shape[1]

        logger.info(f"  Target LLM dimension: {target_dim}")
        logger.info(f"  OCR input dimension: {input_dim}")

        # Build main OCR connector
        # Structure: Linear(input_dim → target_dim) + GELU + Linear(target_dim → target_dim)
        self.ocr_connector = nn.Sequential(
            nn.Linear(input_dim, target_dim),
            nn.GELU(),
            nn.Linear(target_dim, target_dim)
        ).to(device=self.device, dtype=self.dtype)

        # Load weights
        self.ocr_connector.load_state_dict(ocr_state)
        self.ocr_connector.eval()
        logger.info(f"  ✓ Loaded ocr_connector: {input_dim} → {target_dim}")

        # Deepstack connectors
        self.deepstack_connectors = {}
        if 'deepstack_connectors' in state_dict:
            deepstack_dict = state_dict['deepstack_connectors']
            for dim_key in deepstack_dict.keys():
                connector_state = deepstack_dict[dim_key]

                # Get input dim from first linear layer
                ds_input_dim = connector_state['0.weight'].shape[1]

                # Build connector
                connector = nn.Sequential(
                    nn.Linear(ds_input_dim, target_dim),
                    nn.GELU(),
                    nn.Linear(target_dim, target_dim)
                ).to(device=self.device, dtype=self.dtype)

                # Load weights
                connector.load_state_dict(connector_state)
                connector.eval()
                self.deepstack_connectors[str(dim_key)] = connector
                logger.info(f"  ✓ Loaded deepstack connector [{dim_key}]: {ds_input_dim} → {target_dim}")

        # Store connectors as a simple namespace for easy access
        class Connectors:
            pass
        self.connectors = Connectors()
        self.connectors.ocr_connector = self.ocr_connector
        self.connectors.deepstack_connectors = self.deepstack_connectors

        logger.info("")

    def _init_vllm(self, base_model: str, gpu_memory_utilization: float, max_model_len: int):
        """Initialize vLLM engine with base Qwen3-VL and optional LoRA."""
        logger.info("Initializing vLLM engine...")

        # Resolve model path (handle HF or local mirror)
        self.base_model_path = resolve_model_path(base_model)

        # Initialize tokenizer for building inputs
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.base_model_path),
            trust_remote_code=True,
        )

        # Get special token IDs for vision
        self.vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        # Build vLLM configuration
        vllm_kwargs = {
            "model": str(self.base_model_path),
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "seed": None,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "disable_log_stats": True,
            # Multi-modal settings
            "limit_mm_per_prompt": {"image": 100},  # Support up to 100 images per prompt
            "enable_mm_embeds": True,  # Enable pre-computed embeddings
            # Continuous batching settings
            "enable_prefix_caching": False,  # Vision tokens not cacheable (important for OCRVL)
        }

        # Add LoRA configuration if enabled
        if self.use_lora:
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64  # Conservative upper bound
            logger.info(f"  LoRA enabled in vLLM (max_rank=64)")

        # Initialize vLLM
        self.llm = LLM(**vllm_kwargs)

        # If LoRA is enabled, preload the adapters into vLLM engine
        if self.use_lora:
            self.lora_request_name = "ocrvl_lora"
            self.lora_int_id = 1

            # CRITICAL: Pre-load LoRA adapters into vLLM engine
            # This must be done before generation - passing LoRARequest to generate() alone isn't enough
            lora_request = LoRARequest(
                lora_name=self.lora_request_name,
                lora_int_id=self.lora_int_id,
                lora_path=str(self.lora_path),
            )
            self.llm.llm_engine.add_lora(lora_request)
            logger.info(f"  ✓ Pre-loaded LoRA adapters into vLLM engine: {self.lora_path.name}")
        else:
            self.lora_request_name = None
            self.lora_int_id = None

    def apply_connectors(
        self,
        visual_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        deepstack_features: Optional[List[List[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """
        Apply OCRVL connectors to transform OCR features to LLM embeddings.

        Args:
            visual_embeddings: Final-layer OCR features
                - Single: [seq_len, 1280]
                - Batch: List of [seq_len, 1280]
            deepstack_features: Deepstack features (3 levels per image)
                - Format: [[ds0, ds1, ds2], ...] where each ds is [seq_len, 1024]

        Returns:
            List of concatenated embeddings, each [seq_len, target_dim * 4]
            Format: [final_proj | ds0_proj | ds1_proj | ds2_proj]
        """
        # Normalize to list
        if isinstance(visual_embeddings, torch.Tensor):
            if visual_embeddings.ndim == 2:
                embeddings_list = [visual_embeddings]
            elif visual_embeddings.ndim == 3:
                embeddings_list = list(visual_embeddings)
            else:
                raise ValueError(f"visual_embeddings must be 2D or 3D, got {visual_embeddings.ndim}D")
        else:
            embeddings_list = list(visual_embeddings)

        batch_size = len(embeddings_list)

        # Move embeddings to connector device
        embeddings_list = [emb.to(device=self.device, dtype=self.dtype) for emb in embeddings_list]

        # Apply final connector
        with torch.no_grad():
            final_projected = [self.connectors.ocr_connector(emb) for emb in embeddings_list]

        # Handle deepstack if provided
        if deepstack_features is not None:
            if len(deepstack_features) != batch_size:
                raise ValueError(
                    f"Deepstack count ({len(deepstack_features)}) != embedding count ({batch_size})"
                )

            # Apply deepstack connectors and concatenate
            deepstack_connector = self.connectors.deepstack_connectors['1024']

            concatenated = []
            for final_proj, ds_levels in zip(final_projected, deepstack_features):
                # Validate and project deepstack levels
                if not isinstance(ds_levels, (list, tuple)) or len(ds_levels) != 3:
                    raise ValueError(f"Expected 3 deepstack levels, got {len(ds_levels)}")

                # Move to device and project
                ds_levels = [ds.to(device=self.device, dtype=self.dtype) for ds in ds_levels]
                with torch.no_grad():
                    ds_projected = [deepstack_connector(ds) for ds in ds_levels]

                # Concatenate: [final | ds0 | ds1 | ds2]
                concatenated.append(
                    torch.cat([final_proj] + ds_projected, dim=-1)
                )

            return concatenated
        else:
            # No deepstack, return only final projections
            return final_projected

    def generate(
        self,
        visual_embeddings: Union[torch.Tensor, List[torch.Tensor]],
        grid_thw: Optional[Union[List[int], List[List[int]]]] = None,
        prompts: Union[str, List[str]] = "Transcribe all text:",
        deepstack_features: Optional[Union[List[List[torch.Tensor]], List[List[List[torch.Tensor]]]]] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        **generation_kwargs,
    ) -> Union[str, List[str]]:
        """
        Generate text from OCR visual embeddings using vLLM with OCRVL connectors.

        Args:
            visual_embeddings: OCR visual features from DPSKOCREncoder
                - Single image: [seq_len, 1280]
                - Multiple images for one prompt: List of [seq_len, 1280] (with prompts as str)
                - Batch (one image per prompt): List of [seq_len, 1280] (with prompts as List[str])
            grid_thw: MRoPE grid (temporal, height, width) for each image
                - Format: [[t, h, w], ...] or [t, h, w] for single/broadcast
                - For multi-image single prompt: [[t,h,w], [t,h,w], ...]
                - Default: Auto-inferred from token count
            prompts: Text prompt(s) for generation
                - str: Single prompt (possibly with multiple images)
                - List[str]: Batch of prompts (one per visual_embedding)
            deepstack_features: Deepstack features (3 levels per image)
                - Format: [[ds0, ds1, ds2], ...] matching visual_embeddings structure
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            top_p: Nucleus sampling parameter

        Returns:
            Generated text (str if single prompt, List[str] if batch)
        """
        # Determine if this is multi-image for single prompt or batch
        is_single_prompt = isinstance(prompts, str)
        is_tensor_input = isinstance(visual_embeddings, torch.Tensor)

        if is_tensor_input:
            # Single image, single prompt
            visual_emb_batch = [[visual_embeddings]]
            deepstack_batch = [[deepstack_features]] if deepstack_features else None
            num_prompts = 1
        elif is_single_prompt and isinstance(visual_embeddings, list):
            # Multiple images, single prompt
            visual_emb_batch = [visual_embeddings]
            deepstack_batch = [deepstack_features] if deepstack_features else None
            num_prompts = 1
        else:
            # Batch mode
            # Detect if it's a batch of multi-image prompts (list of lists) or single-image prompts (list of tensors)
            if visual_embeddings and isinstance(visual_embeddings[0], list):
                # Batch of multi-image prompts: [[img1, img2], [img3, img4], ...]
                visual_emb_batch = visual_embeddings
                deepstack_batch = deepstack_features if deepstack_features else None
            else:
                # Batch of single-image prompts: [img1, img2, img3, ...]
                visual_emb_batch = [[emb] for emb in visual_embeddings]
                if deepstack_features:
                    deepstack_batch = [[ds] for ds in deepstack_features]
                else:
                    deepstack_batch = None
            num_prompts = len(visual_embeddings)

        # Flatten all embeddings for connector application
        all_embeddings = [emb for prompt_embs in visual_emb_batch for emb in prompt_embs]
        all_deepstack = None
        if deepstack_batch:
            all_deepstack = [ds for prompt_ds in deepstack_batch for ds in prompt_ds]

        # Apply connectors to all images
        logger.info("Applying OCRVL connectors...")
        all_llm_embeddings = self.apply_connectors(all_embeddings, all_deepstack)
        logger.info(f"✓ Projected {len(all_llm_embeddings)} image embeddings")

        # Group back by prompt
        llm_embeddings_grouped = []
        idx = 0
        for prompt_embs in visual_emb_batch:
            count = len(prompt_embs)
            llm_embeddings_grouped.append(all_llm_embeddings[idx:idx+count])
            idx += count

        # Handle grid_thw
        if grid_thw is None:
            # Auto-infer for each image
            grid_thw_grouped = []
            for prompt_embs in visual_emb_batch:
                grids = [get_ocr_grid_thw(emb) for emb in prompt_embs]
                grid_thw_grouped.append(grids)
            logger.info(f"Auto-inferred grid_thw for first prompt: {grid_thw_grouped[0]}")
        else:
            # Normalize provided grid_thw
            if isinstance(grid_thw[0], int):
                # Single grid - broadcast to all
                grid_thw_grouped = [[grid_thw] * len(prompt_embs) for prompt_embs in visual_emb_batch]
            elif is_single_prompt:
                # List of grids for single prompt
                grid_thw_grouped = [grid_thw]
            else:
                # List of grids, one per prompt
                grid_thw_grouped = [[g] for g in grid_thw]

        # Normalize prompts
        prompts_list = [prompts] * num_prompts if is_single_prompt else list(prompts)

        # Build vLLM inputs
        vllm_inputs = []
        for prompt_embeddings, prompt_grids, prompt_text in zip(llm_embeddings_grouped, grid_thw_grouped, prompts_list):
            # Build input_ids with chat format: <|im_start|>user\n[vision]<|im_end|>\n[prompt_text]
            # Training format: <|im_start|>user\n<|vision_start|>...<|vision_end|>...<|im_end|>\n<|im_start|>assistant\n[caption]

            # Qwen3-VL chat token IDs
            user_start_ids = [151644, 872, 198]  # <|im_start|>user\n
            user_end_ids = [151645, 198]         # <|im_end|>\n
            assistant_start_ids = [151644, 77091, 198]  # <|im_start|>assistant\n

            # Parse prompt_text to handle special tokens correctly
            if prompt_text.startswith('<|im_start|>'):
                # Prompt contains chat template tokens - parse them manually
                # Expected: '<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n'
                prompt_ids = []
                if '<|im_start|>user\n<|im_end|>\n<|im_start|>assistant\n' in prompt_text:
                    # Pure vision mode: just assistant start
                    prompt_ids = assistant_start_ids
                else:
                    # Fallback: try to tokenize
                    prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            else:
                # Regular text prompt - tokenize normally
                prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

            # Qwen3-VL official format: All images in ONE user block
            # <|im_start|>user<img1><img2>...<|im_end|>[prompt_text]
            # Training uses same format with rendered text as one of the images

            # Build vision blocks for all images (concatenated in ONE user block)
            vision_token_ids = []
            for emb in prompt_embeddings:
                num_tokens = emb.shape[0]  # Should be 100 for each image
                vision_token_ids.extend([
                    self.vision_start_id,
                    *([self.image_pad_id] * num_tokens),
                    self.vision_end_id,
                ])

            # Wrap all vision tokens in ONE user block, then append prompt
            # Format: <|im_start|>user\n[all_vision_tokens]<|im_end|>\n[prompt_text]
            input_ids = user_start_ids + vision_token_ids + user_end_ids + prompt_ids

            # Concatenate embeddings and grids
            combined_embedding = torch.cat(prompt_embeddings, dim=0)  # [total_seq, dim]
            combined_grids = torch.tensor(prompt_grids, dtype=torch.long)  # [num_images, 3]

            # Prepare vLLM input with pre-computed embeddings
            vllm_input = {
                "prompt_token_ids": input_ids,
                "multi_modal_data": {
                    "image": {
                        "image_embeds": combined_embedding.cpu(),  # [total_seq_len, dim]
                        "image_grid_thw": combined_grids.cpu(),  # [num_images, 3]
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

        # Prepare LoRA request if enabled
        lora_request = None
        if self.use_lora:
            lora_request = LoRARequest(
                lora_name=self.lora_request_name,
                lora_int_id=self.lora_int_id,
                lora_path=str(self.lora_path),
            )
            logger.info(f"  Applying LoRA: {self.lora_path.name}")

        # Generate with vLLM
        total_images = sum(len(embs) for embs in llm_embeddings_grouped)
        logger.info(f"Generating text for {total_images} images across {num_prompts} prompts...")
        outputs = self.llm.generate(
            vllm_inputs,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        # Extract results
        results = [output.outputs[0].text for output in outputs]

        return results[0] if is_single_prompt else results

    def __repr__(self) -> str:
        return f"OCRVLProcessor(checkpoint={self.checkpoint_path.name}, backend=vLLM)"

    # --- Lifecycle ---------------------------------------------------------
    def shutdown(self):
        """Gracefully shutdown vLLM resources to avoid lingering processes.

        Some long-running vLLM worker threads can keep the Python process
        alive after generation. This method mirrors the cleanup used by
        upstream tools: shut down the engine, drop references, and free CUDA
        memory so shard processes can exit cleanly.
        """
        try:
            if hasattr(self, "llm") and self.llm is not None:
                engine = getattr(self.llm, "llm_engine", None)
                if engine is not None and hasattr(engine, "shutdown"):
                    try:
                        engine.shutdown()
                    except Exception as e:  # non-fatal
                        logger.info(f"Info: vLLM engine shutdown returned: {e}")
        finally:
            # Best-effort reference drops and CUDA cleanup
            try:
                del self.llm
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            self.llm = None
