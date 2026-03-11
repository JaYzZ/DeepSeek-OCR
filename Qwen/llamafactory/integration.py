"""
LlamaFactory integration for Qwen3VL training with latent supervision.

This module patches the training pipeline to support:
1. Latent injection: Inject pre-encoded features at <latent> positions
2. Latent supervision: Load supervision targets for REPA loss computation
3. Thinking loss: Combine CE loss with REPA loss on latent predictions

To enable: Set environment variable QWEN3VL_LATENT_SUPERVISION=1
"""

import functools
import gc
import logging
import os
import sys
import time
import traceback
from typing import Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions

from transformers import AutoTokenizer, TrainerCallback, Trainer
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from peft import PeftModelForCausalLM
from llamafactory.data import SFTDataCollatorWith4DAttentionMask, loader as loader_module
from llamafactory.data.converter import SharegptDatasetConverter
from llamafactory.data.mm_plugin import Qwen3VLPlugin
from llamafactory.data.processor.supervised import PackedSupervisedDatasetProcessor, SupervisedDatasetProcessor
from llamafactory.data.template import (
    FunctionFormatter,
    ReasoningTemplate,
    StringFormatter,
    ToolFormatter,
    register_template,
)
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

from Qwen.llamafactory.vae_callback import VAESaveCallback
from Qwen.llamafactory.transparent_eval_callback import QwenTransparentEvalCallback
from Qwen.llamafactory.curriculum_callback import QwenCurriculumCallback

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    try:
        return (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


class QwenLossLoggingCallback(TrainerCallback):
    """Log loss spec + loss breakdown at Trainer logging cadence (rank 0 only).

    We intentionally keep the presentation as a single human-readable line:
      loss_spec=..., use_latent_vae=..., Loss: CE=..., pre_think_mse=..., Total=...

    Avoid logging from model.forward() because it is called many times per step.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._last_logged_step: int | None = None
        self._swanlab: Any = None
        self._swanlab_checked = False
        self._swanlab_enabled: bool | None = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        rank_0 = _is_rank0()
        if not rank_0:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        if self._last_logged_step == step:
            return

        # Mirror Trainer semantics.
        if step == 0 and not getattr(args, "logging_first_step", False):
            return
        strategy = getattr(args, "logging_strategy", None)
        logging_steps = int(getattr(args, "logging_steps", 0) or 0)
        if strategy == "steps" and logging_steps > 0 and (step % logging_steps) != 0:
            return

        info = getattr(self._model, "_qwen3vl_last_loss_info", None)
        if isinstance(info, dict):
            loss_spec = info.get("loss_spec")
            use_latent_vae = bool(info.get("use_latent_vae"))
            ce = info.get("ce")
            thinking = info.get("thinking")
            vae = info.get("vae")
            pred_embed = info.get("pred_embed_ce_loss", info.get("pred_embed"))
            total = info.get("total")
            step_metrics: dict[str, float] = {}

            parts: list[str] = []
            if ce is not None:
                parts.append(f"CE={ce:.4f}")
                step_metrics["qwen3vl/loss/ce"] = float(ce)
            if thinking is not None:
                # thinking loss name is embedded in loss_spec (e.g. pre_think_mse / ot / mse).
                # Prefer showing the explicit spec string the user configured.
                if isinstance(loss_spec, str) and loss_spec:
                    for name in [x.strip() for x in loss_spec.replace("+", " ").split()]:
                        if name in ("ot", "mse", "repa", "nce", "pre_think_mse"):
                            parts.append(f"{name}={thinking:.4f}")
                else:
                    parts.append(f"thinking={thinking:.4f}")
                step_metrics["qwen3vl/loss/thinking"] = float(thinking)
            if vae is not None:
                parts.append(f"vae={vae:.4f}")
                step_metrics["qwen3vl/loss/vae"] = float(vae)
            if use_latent_vae:
                if pred_embed is not None:
                    parts.append(f"pred_embed_ce_loss={pred_embed:.4f}")
                    step_metrics["qwen3vl/loss/pred_embed_ce_loss"] = float(pred_embed)
                else:
                    parts.append("pred_embed_ce_loss=NA")
            if total is not None:
                parts.append(f"Total={total:.4f}")
                step_metrics["qwen3vl/loss/total"] = float(total)

            if isinstance(logs, dict):
                if "grad_norm" in logs and logs["grad_norm"] is not None:
                    step_metrics["qwen3vl/train/grad_norm"] = float(logs["grad_norm"])
                if "learning_rate" in logs and logs["learning_rate"] is not None:
                    step_metrics["qwen3vl/train/learning_rate"] = float(logs["learning_rate"])
                if "epoch" in logs and logs["epoch"] is not None:
                    step_metrics["qwen3vl/train/epoch"] = float(logs["epoch"])
                if "loss" in logs and logs["loss"] is not None:
                    step_metrics["qwen3vl/train/loss"] = float(logs["loss"])

            loss_str = f"Loss: {', '.join(parts)}" if parts else "Loss: N/A"
            logger.info(
                f"\n[Qwen3VL Latent] step={step} loss_spec={loss_spec}, use_latent_vae={use_latent_vae}, {loss_str}"
            )
            self._log_to_swanlab(step, step_metrics, args)
        self._last_logged_step = step

    def _swanlab_is_enabled(self, args: Any) -> bool:
        if self._swanlab_enabled is not None:
            return self._swanlab_enabled

        env_flag = os.environ.get("USE_SWANLAB")
        if env_flag is not None:
            self._swanlab_enabled = env_flag.strip().lower() in {"1", "true", "yes", "on"}
            return self._swanlab_enabled

        use_swanlab = getattr(args, "use_swanlab", None)
        if use_swanlab is not None:
            if isinstance(use_swanlab, str):
                self._swanlab_enabled = use_swanlab.strip().lower() in {"1", "true", "yes", "on"}
            else:
                self._swanlab_enabled = bool(use_swanlab)
            return self._swanlab_enabled

        report_to = getattr(args, "report_to", None)
        if isinstance(report_to, str):
            targets = [report_to]
        elif isinstance(report_to, (list, tuple, set)):
            targets = [str(x) for x in report_to]
        else:
            targets = []
        self._swanlab_enabled = any(x.strip().lower() == "swanlab" for x in targets)
        return self._swanlab_enabled

    def _log_to_swanlab(self, step: int, metrics: dict[str, float], args: Any) -> None:
        if not metrics:
            return
        if not self._swanlab_is_enabled(args):
            return

        if not self._swanlab_checked:
            self._swanlab_checked = True
            try:
                import swanlab  # type: ignore

                self._swanlab = swanlab
            except Exception:
                self._swanlab = None

        if self._swanlab is None:
            return

        try:
            self._swanlab.log(metrics, step=step)
        except Exception as e:
            logger.debug(f"[Qwen3VL Latent] swanlab.log failed: {e}")


def _get_loss_spec() -> str:
    """Get loss spec from env var with default.

    Single place to define the default loss type.
    Default: ot+pre_think_mse
    """
    return os.environ.get("QWEN3VL_LOSS_TYPE", "ot+pre_think_mse").lower()


class LatentVAE(nn.Module):
    """Latent VAE module that maps LLM hidden states to latent distributions.

    Similar to ReGuLaR's LatentPolicy:
    - Takes LLM hidden states as input
    - Outputs mean and std for latent distributions
    - Enables sampling of latent reasoning states
    """

    def __init__(self, hidden_size: int, intermediate_size: int = 512, deterministic: bool = False):
        super().__init__()
        self.deterministic = deterministic
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.trainable = True  # Mark as trainable

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.LayerNorm(intermediate_size),
        )

        self.mean = nn.Linear(intermediate_size, hidden_size)
        if not deterministic:
            self.log_std = nn.Linear(intermediate_size, hidden_size)

        # Ensure all parameters are trainable
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Normal:
        """Map hidden states to latent distribution.

        Args:
            x: Hidden states [batch, seq_len, hidden_size]
            temperature: Temperature for sampling (higher = more random)

        Returns:
            Normal distribution over latent space
        """
        x = self.fc(x)
        mean = self.mean(x)
        if self.deterministic:
            return torch.distributions.Normal(mean, torch.ones_like(mean) * 1e-9)
        log_std = self.log_std(x)
        std = log_std.exp() * temperature
        return torch.distributions.Normal(mean, std)

    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample latent from the distribution.

        Args:
            x: Hidden states [batch, seq_len, hidden_size]
            temperature: Temperature for sampling

        Returns:
            Sampled latent embeddings [batch, seq_len, hidden_size]
        """
        dist = self.forward(x, temperature)
        return dist.rsample()

    def get_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Get mean of the latent distribution (for deterministic use).

        Args:
            x: Hidden states [batch, seq_len, hidden_size]

        Returns:
            Mean latent embeddings [batch, seq_len, hidden_size]
        """
        dist = self.forward(x, temperature=1.0)
        return dist.mean

# Ensure logger outputs even if logging not yet configured
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid duplicate logs


def _register_qwen3vl_latent_template(logger) -> None:
    """Register the qwen3vl_latent template for LlamaFactory."""
    try:
        # Create multimodal plugin instance with all required tokens
        mm_plugin = Qwen3VLPlugin(
            image_token="<|image_pad|>",
            video_token=None,  # Qwen3VL doesn't use video_token
            audio_token=None,  # Qwen3VL doesn't use audio_token
        )

        # Register the template
        register_template(
            name="qwen3vl_latent",
            format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
            format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
            format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
            format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
            format_observation=StringFormatter(
                slots=["<|im_start|>user\n\n{{content}}\n<|im_end|>\n<|im_start|>assistant\n"]
            ),
            format_tools=ToolFormatter(tool_format="qwen"),
            stop_words=["<|im_end|>"],
            replace_eos=True,
            mm_plugin=mm_plugin,
            template_class=ReasoningTemplate,
        )
        logger.debug("[Qwen3VL Latent] ✓ Registered qwen3vl_latent template with Qwen3VLPlugin")
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Template registration failed: {e}")
        raise


def _patch_qwen2vl_image_processor(logger) -> None:
    """No-op: grid_thw patch not needed, vision model handles pre-merge values correctly."""
    pass


def _patch_tokenizer_for_special_tokens(logger) -> None:
    """Patch tokenizer to add <latent> and <think_sep> special tokens."""
    orig_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

    @classmethod  # type: ignore[misc]
    @functools.wraps(orig_tokenizer_from_pretrained)
    def wrapped_tokenizer_from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """Wrapped tokenizer that adds thinking special tokens."""
        tokenizer = orig_tokenizer_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if tokenizer is None:
            return tokenizer

        # Only add tokens for Qwen3-VL models.
        model_hint = str(pretrained_model_name_or_path)
        name_hint = getattr(tokenizer, "name_or_path", "")
        if "Qwen3-VL" not in model_hint and "Qwen3-VL" not in name_hint and "Qwen3-VL" not in str(type(tokenizer)):
            return tokenizer

        # Special tokens to add (must be single tokens)
        special_tokens = [
            "<latent>",
            "<think_sep>",
        ]

        # Check if tokens exist and get their IDs
        added_count = 0
        for token in special_tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) > 1:
                # Token is not single in current vocab - add it as special token
                num_added = tokenizer.add_special_tokens(
                    {"additional_special_tokens": [token]},
                    replace_additional_special_tokens=False
                )
                added_count += num_added
                if _is_rank0():
                    logger.info(f"[Qwen3VL Latent] Added token to tokenizer: {token}")

        # Set up token ID environment variables
        os.environ["QWEN3VL_THINKING_START_ID"] = "151667"
        os.environ["QWEN3VL_THINKING_END_ID"] = "151668"
        if _is_rank0():
            logger.info("[Qwen3VL Latent] Using Qwen3VL native thinking tokens: start=151667, end=151668")

        # Store token IDs
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id >= 0 and token_id != tokenizer.unk_token_id:
                if token == "<latent>":
                    os.environ["QWEN3VL_LATENT_TOKEN_ID"] = str(token_id)
                    if _is_rank0():
                        logger.info(f"[Qwen3VL Latent] {token} = ID {token_id} (env: QWEN3VL_LATENT_TOKEN_ID)")
                elif token == "<think_sep>":
                    os.environ["QWEN3VL_THINKING_SEP_ID"] = str(token_id)
                    if _is_rank0():
                        logger.info(f"[Qwen3VL Latent] {token} = ID {token_id} (env: QWEN3VL_THINKING_SEP_ID)")
            else:
                if _is_rank0():
                    logger.info(f"[Qwen3VL Latent] Token {token} not found in vocab - will be added by LlamaFactory")

        return tokenizer

    AutoTokenizer.from_pretrained = wrapped_tokenizer_from_pretrained
    logger.debug("[Qwen3VL Latent] Patched AutoTokenizer for special tokens")


def _patch_once() -> None:
    """Apply patches once per process."""
    # Check if latent supervision is enabled (default: 1 for latent thinking training)
    if os.environ.get("QWEN3VL_LATENT_SUPERVISION", "1") != "1":
        return

    if _is_rank0():
        logger.info(f"[Qwen3VL Latent] Starting latent supervision integration...")

    # PID-based guard
    pid = str(os.getpid())
    if os.environ.get("QWEN3VL_LATENT_PATCHED_PID", "") == pid:
        return
    os.environ["QWEN3VL_LATENT_PATCHED_PID"] = pid

    if _is_rank0():
        logger.info("[Qwen3VL Latent] Starting latent supervision integration...")

    # Patch 0: Patch tokenizer for special tokens (<latent>, <think_sep>)
    try:
        _patch_tokenizer_for_special_tokens(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch tokenizer for special tokens: {e}")

    # Grid_thw handling: vision model manages pre-merge values correctly
    _patch_qwen2vl_image_processor(logger)

    # Register Qwen3VL latent template (before any other patches)
    try:
        _register_qwen3vl_latent_template(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to register template: {e}")

    # Patch 1: Add thinking_projection module to model
    try:
        _patch_model_for_thinking_projection(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch model for thinking_projection: {e}")

    # Patch 2: Patch dataset converter to preserve latent fields
    try:
        _patch_dataset_converter(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch dataset converter: {e}")

    # Patch 3: Patch dataset preprocessing to preserve latent columns
    try:
        _patch_dataset_preprocessing(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch dataset preprocessing: {e}")

    # Patch 3b: Disable dataset-level packing when pack-after-injection is enabled
    try:
        _patch_packed_dataset_processor(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch packed dataset processor: {e}")

    # Patch 4: Patch data collator to load latent supervision
    try:
        _patch_data_collator(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch data collator: {e}")

    # Patch 5: Patch forward pass to inject latents and compute thinking loss
    try:
        _patch_model_forward(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch model forward: {e}")

    # Patch 6: Register transparent eval callback
    try:
        _patch_trainer_callback(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch trainer callback: {e}")

    # Patch 7: Patch generate() to filter out latent supervision kwargs
    try:
        _patch_model_generate(logger)
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch model generate: {e}")

    if _is_rank0():
        logger.info("[Qwen3VL Latent] ✓ Latent supervision integration complete")


def _patch_model_for_thinking_projection(logger) -> None:
    """Add LatentVAE module to Qwen3VL model.

    This patches the model's _load_from_state_dict to ensure VAE is created
    before checkpoint loading, so VAE weights can be properly restored.
    """
    # Check if VAE is enabled
    loss_spec = _get_loss_spec()
    if "vae" not in loss_spec:
        return

    # Store original _load_from_state_dict
    original_load = Qwen3VLForConditionalGeneration._load_from_state_dict

    @functools.wraps(original_load)
    def patched_load_from_state_dict(self, state_dict, *args, **kwargs):
        # Check if VAE weights exist in state dict
        vae_keys = [k for k in state_dict.keys() if k.startswith('latent_vae.')]
        if vae_keys:
            # Create VAE before loading so weights can be restored
            hidden_size = self.config.hidden_size
            vae = LatentVAE(hidden_size=hidden_size, intermediate_size=int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512")), deterministic=False)
            self.register_module('latent_vae', vae)
            logger.info(f"[Qwen3VL Latent] Pre-created LatentVAE for checkpoint loading: {len(vae_keys)} keys")

        # Call original load
        return original_load(self, state_dict, *args, **kwargs)

    # Apply patch
    Qwen3VLForConditionalGeneration._load_from_state_dict = patched_load_from_state_dict

    logger.info("[Qwen3VL Latent] Patched _load_from_state_dict for VAE checkpoint loading")


def _ensure_vae_trainable(model) -> None:
    """Ensure VAE module parameters are trainable.

    This should be called after model is prepared for training to ensure
    VAE parameters are included in trainable params.
    """
    if hasattr(model, 'latent_vae') and model.latent_vae is not None:
        for name, param in model.named_parameters():
            if 'latent_vae' in name:
                param.requires_grad = True
        logger.debug("[Qwen3VL Latent] Enabled VAE parameters as trainable")


# Export function for external use
def ensure_vae_in_model(model) -> None:
    """Ensure VAE is part of model and its parameters are trainable.

    Call this after model is loaded to ensure VAE is included in
    the model's trainable parameters for the optimizer.
    """
    _ensure_vae_trainable(model)


def save_vae_checkpoint(model, output_dir: str) -> None:
    """Save VAE weights separately from the model checkpoint.

    This saves only the latent_vae module weights to a separate file.

    Args:
        model: The model with latent_vae module
        output_dir: Directory to save VAE checkpoint
    """
    if not hasattr(model, 'latent_vae') or model.latent_vae is None:
        logger.warning("[Qwen3VL Latent] No VAE to save")
        return

    import safetensors.torch
    import os

    # Canonical layout: <checkpoint_dir>/vae.safetensors
    os.makedirs(output_dir, exist_ok=True)

    # Get VAE state dict
    vae_state_dict = model.latent_vae.state_dict()

    # Save VAE weights
    vae_path = os.path.join(output_dir, "vae.safetensors")
    safetensors.torch.save_file(vae_state_dict, vae_path)
    logger.info(f"[Qwen3VL Latent] Saved VAE checkpoint to {vae_path}")


def load_vae_checkpoint(model, checkpoint_path: str) -> None:
    """Load VAE weights from a separate checkpoint file.

    Args:
        model: The model to load VAE weights into
        checkpoint_path: Path to VAE checkpoint file
    """
    if not hasattr(model, 'latent_vae') or model.latent_vae is None:
        logger.warning("[Qwen3VL Latent] No VAE to load into")
        return

    import safetensors.torch
    import os

    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "vae.safetensors")

    if not os.path.exists(checkpoint_path):
        logger.warning(f"[Qwen3VL Latent] VAE checkpoint not found: {checkpoint_path}")
        return

    vae_state_dict = safetensors.torch.load_file(checkpoint_path)
    model.latent_vae.load_state_dict(vae_state_dict, strict=True)
    logger.info(f"[Qwen3VL Latent] Loaded VAE checkpoint from {checkpoint_path}")


def _patch_dataset_converter(logger) -> None:
    """Patch SharegptDatasetConverter to preserve latent supervision fields.

    The standard converter only extracts: messages, images, videos, audios, tools, system
    We need to also preserve: latent_ground_truth, latent_supervision, num_latent_steps
    """
    # Store original __call__
    original_call = SharegptDatasetConverter.__call__

    @functools.wraps(original_call)
    def wrapped_call(self, example: dict[str, Any]) -> dict[str, Any]:
        """Wrapped converter that preserves latent supervision fields."""
        # Call original converter
        result = original_call(self, example)

        # Preserve latent supervision fields if present
        latent_fields = [
            'latent_ground_truth',
            'latent_supervision',
            'num_latent_steps',
            'cot',
            'latent_seq_lens',
            'cot_chunk_token_ids',
        ]
        for field in latent_fields:
            if field in example:
                result[field] = example[field]

        return result

    # Apply patch
    SharegptDatasetConverter.__call__ = wrapped_call
    logger.debug("[Qwen3VL Latent] ✓ Patched SharegptDatasetConverter.__call__ to preserve latent fields")


def _patch_dataset_preprocessing(logger) -> None:
    """Patch supervised preprocessing to preserve latent metadata fields.

    LlamaFactory's supervised processors emit only model tensors by default.
    We keep CHIMERA latent fields aligned with valid samples so the collator
    receives them directly (no path-based reconstruction fallback).
    """
    original_preprocess = SupervisedDatasetProcessor.preprocess_dataset
    latent_fields = [
        "latent_ground_truth",
        "latent_supervision",
        "num_latent_steps",
        "cot",
        "latent_seq_lens",
        "cot_chunk_token_ids",
    ]

    @functools.wraps(original_preprocess)
    def wrapped_preprocess(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        model_inputs = original_preprocess(self, examples)

        if not any(field in examples for field in latent_fields):
            return model_inputs

        prompts = examples.get("_prompt", [])
        responses = examples.get("_response", [])
        kept_indices: list[int] = []
        for i in range(len(prompts)):
            if len(prompts[i]) % 2 != 1 or len(responses[i]) != 1:
                continue
            kept_indices.append(i)

        output_len = len(model_inputs.get("input_ids", []))
        if output_len != len(kept_indices):
            raise ValueError(
                "[Qwen3VL Latent] preprocess alignment mismatch: "
                f"model_inputs={output_len} vs kept_indices={len(kept_indices)}."
            )

        for field in latent_fields:
            if field in examples:
                values = examples[field]
                model_inputs[field] = [values[i] if i < len(values) else None for i in kept_indices]

        return model_inputs

    SupervisedDatasetProcessor.preprocess_dataset = wrapped_preprocess
    logger.debug("[Qwen3VL Latent] ✓ Patched SupervisedDatasetProcessor to preserve latent fields")

    # Force re-tokenization cache refresh for latent runs.
    # Without this, HF datasets.map may reuse older cached artifacts that were
    # created before latent fields were preserved in preprocess output.
    original_get_preprocessed_dataset = loader_module._get_preprocessed_dataset

    @functools.wraps(original_get_preprocessed_dataset)
    def wrapped_get_preprocessed_dataset(
        dataset,
        data_args,
        training_args,
        stage,
        template,
        tokenizer,
        processor=None,
        is_eval: bool = False,
    ):
        latent_on = os.environ.get("QWEN3VL_LATENT_SUPERVISION", "1") == "1"
        if not latent_on:
            return original_get_preprocessed_dataset(
                dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval
            )

        prev_overwrite = getattr(data_args, "overwrite_cache", False)
        data_args.overwrite_cache = True
        try:
            return original_get_preprocessed_dataset(
                dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval
            )
        finally:
            data_args.overwrite_cache = prev_overwrite

    loader_module._get_preprocessed_dataset = wrapped_get_preprocessed_dataset
    logger.debug("[Qwen3VL Latent] ✓ Patched loader._get_preprocessed_dataset to refresh cache for latent runs")


def _patch_packed_dataset_processor(logger) -> None:
    """Disable dataset-level packing so we can pack after injection."""
    original_preprocess = PackedSupervisedDatasetProcessor.preprocess_dataset

    @functools.wraps(original_preprocess)
    def wrapped_preprocess(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not hasattr(wrapped_preprocess, "_logged_skip"):
            logger.debug("[Qwen3VL Latent] pack-after-injection: skipping dataset-level packing")
            wrapped_preprocess._logged_skip = True
        return SupervisedDatasetProcessor.preprocess_dataset(self, examples)

    PackedSupervisedDatasetProcessor.preprocess_dataset = wrapped_preprocess
    logger.debug("[Qwen3VL Latent] ✓ Patched PackedSupervisedDatasetProcessor for pack-after-injection")


def _get_cutoff_len_from_collator(collator, logger) -> int:
    env_cutoff = os.environ.get("QWEN3VL_CUTOFF_LEN", None)
    if env_cutoff is not None:
        try:
            cutoff = int(env_cutoff)
            if cutoff > 0:
                return cutoff
        except ValueError:
            logger.warning(f"[Qwen3VL Latent] Invalid QWEN3VL_CUTOFF_LEN={env_cutoff!r}, falling back to collator/tokenizer settings")

    cutoff = getattr(collator, "max_length", None)
    if cutoff is not None and cutoff > 0:
        return int(cutoff)

    cutoff = getattr(getattr(collator, "tokenizer", None), "model_max_length", None)
    if cutoff is None or cutoff <= 0 or cutoff > 100000:
        raise ValueError(
            "[Qwen3VL Latent] pack-after-injection requires a valid cutoff_len. "
            "Export QWEN3VL_CUTOFF_LEN (e.g., 2048)."
        )
    return int(cutoff)


def _latent_seq_len(feat: torch.Tensor) -> int:
    if feat.dim() == 1:
        return 1
    if feat.dim() == 2:
        return int(feat.shape[0])
    raise ValueError(f"Unexpected latent tensor dim: {feat.dim()}")


def _flatten_latent_tensors(nested_list: list) -> list[torch.Tensor]:
    """Recursively flatten nested list of tensors to a flat list.

    Handles arbitrary nesting levels:
    - [tensor] -> [tensor]
    - [[tensor]] -> [tensor]
    - [[[tensor]]] -> [tensor]

    Args:
        nested_list: Potentially nested list of tensors

    Returns:
        Flat list of tensors
    """
    if isinstance(nested_list, torch.Tensor):
        return [nested_list]
    elif isinstance(nested_list, list):
        result = []
        for item in nested_list:
            result.extend(_flatten_latent_tensors(item))
        return result
    return []


def _flatten_latent_values(nested_list: Any) -> list[Any]:
    """Recursively flatten latent containers while preserving leaf values."""
    if isinstance(nested_list, (torch.Tensor, str)):
        return [nested_list]
    if isinstance(nested_list, list):
        result: list[Any] = []
        for item in nested_list:
            result.extend(_flatten_latent_values(item))
        return result
    return []


# Marker CE is ALWAYS ON - thinking tokens always have their token IDs as labels
# (no env var needed, no function needed - just use the token ID directly)


def _latent_step_ce_enabled() -> bool:
    """Whether CE loss on expanded latent positions is enabled.

    Controlled by QWEN3VL_LATENT_STEP_CE_LOSS (curriculum toggle, 1/0).
    Defaults to enabled.
    """
    return os.environ.get("QWEN3VL_LATENT_STEP_CE_LOSS", "1").strip() == "1"


def _resample_token_sequence(token_ids: list[int], target_len: int) -> list[int]:
    """Resample an ordered token sequence to the requested length, preserving order."""
    if target_len <= 0:
        return []
    if not token_ids:
        return []
    if len(token_ids) == target_len:
        return list(token_ids)
    indices = torch.linspace(0, len(token_ids) - 1, target_len).long().tolist()
    return [token_ids[i] for i in indices]


def _normalize_cot_chunk_token_ids(value: Any) -> Optional[list[list[int]]]:
    """Normalize precomputed per-step chunk token ids from dataset metadata."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return None

    normalized: list[list[int]] = []
    for chunk in value:
        if hasattr(chunk, "tolist"):
            chunk = chunk.tolist()
        if not isinstance(chunk, list):
            return None
        normalized.append([int(tok) for tok in chunk])
    return normalized


def _normalize_int_list(value: Any) -> Optional[list[int]]:
    """Normalize a dataset field that should be a flat list of ints."""
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    return [int(x) for x in value]


def _build_latent_ce_targets(
    *,
    exp_len: int,
    boundary_label: int,
    latent_label: int,
    use_latent_token: bool,
    use_cot_token: bool,
    cot_step_token_ids: Optional[list[int]],
    ignore_index: int,
) -> tuple[int, torch.Tensor]:
    """Build CE targets for one expanded latent block and its preceding scaffold token.

    Behavior:
    - The chunk subsequence is sampled only from the original CoT chunk tokens.
    - Structural markers (<think_sep>, </think>) remain outside the chunk sequence.
    - The scaffold token before a latent block predicts the first chunk token.
    - The final latent position predicts the structural boundary token.
    - If latent-step CE is configured to use CoT subsequences, preceding positions learn
      an evenly distributed ordered subsequence of the original CoT step tokens.
    """
    if exp_len <= 0:
        return ignore_index, torch.empty(0, dtype=torch.long)

    if use_latent_token:
        block_tokens = [latent_label] * exp_len
    elif use_cot_token:
        block_tokens = _resample_token_sequence(cot_step_token_ids or [], exp_len)
        if len(block_tokens) < exp_len:
            pad_value = ignore_index
            if block_tokens:
                pad_value = block_tokens[-1]
            elif latent_label != ignore_index:
                pad_value = latent_label
            block_tokens.extend([pad_value] * (exp_len - len(block_tokens)))
    else:
        block_tokens = [ignore_index] * exp_len

    first_target = block_tokens[0] if block_tokens else boundary_label

    latent_labels = []
    if exp_len > 1:
        latent_labels.extend(block_tokens[1:])
    latent_labels.append(boundary_label)

    return first_target, torch.tensor(latent_labels, dtype=torch.long)


def _expand_sample_for_latent_injection(
    input_ids: list[int],
    labels: list[int],
    latent_token_id: int,
    thinking_start_id: int,
    thinking_end_id: int,
    latent_ground_truth: Optional[list[torch.Tensor]],
    latent_lengths: Optional[list[int]],
    ignore_index: int,
    cot_step_token_ids: Optional[list[list[int]]] = None,
) -> tuple[list[int], list[int]]:
    """Expand <latent> tokens into placeholder sequences to match latent lengths.

    Args:
        cot_step_token_ids: Ordered CoT token subsequences (one list per latent step).
            Used when latent-step CE is configured to learn CoT tokens.
    """
    # Convert to tensor for O(1) indexing (faster than list.index() which is O(n))
    ids_tensor = torch.tensor(input_ids, dtype=torch.long)

    # Find thinking bounds using torch.where (O(1) overall)
    thinking_mask = (ids_tensor == thinking_start_id) | (ids_tensor == thinking_end_id)
    thinking_positions = torch.nonzero(thinking_mask).flatten()

    if len(thinking_positions) < 2:
        return input_ids, labels

    start_pos = thinking_positions[0].item()
    end_pos = thinking_positions[-1].item()

    if start_pos >= end_pos or ids_tensor[start_pos] != thinking_start_id:
        return input_ids, labels

    # Find latent tokens in thinking section (vectorized)
    thinking_section = ids_tensor[start_pos + 1:end_pos]
    latent_mask = (thinking_section == latent_token_id)
    latent_indices = torch.nonzero(latent_mask).flatten() + start_pos + 1

    if len(latent_indices) == 0:
        return input_ids, labels

    latent_indices = latent_indices.tolist()

    if latent_lengths is not None:
        expansion_lengths = [int(x) for x in latent_lengths if int(x) > 0]
    elif latent_ground_truth:
        expansion_lengths = [_latent_seq_len(feat) for feat in latent_ground_truth]
    else:
        return input_ids, labels

    # SKIP on mismatch: Each <latent> token must have exactly one corresponding ground truth tensor
    if len(latent_indices) != len(expansion_lengths):
        logger.warning(
            f"[Qwen3VL Latent] Skipping sample: found {len(latent_indices)} <latent> tokens "
            f"but {len(expansion_lengths)} latent expansions. "
            "Each <latent> token must have exactly one corresponding ground truth tensor."
        )
        return input_ids, labels  # Skip this sample, return original

    # Build new sequences using torch.cat 
    result_segments = []

    # Add tokens before first latent
    result_segments.append(ids_tensor[:latent_indices[0]])

    for i, (latent_idx, exp_len) in enumerate(zip(latent_indices, expansion_lengths)):
        # Add repeated latent tokens
        result_segments.append(torch.full((exp_len,), latent_token_id, dtype=torch.long))

        # Add tokens between latents
        if i < len(latent_indices) - 1:
            next_idx = latent_indices[i + 1]
            result_segments.append(ids_tensor[latent_idx + 1:next_idx])
        else:
            # Last latent - add remaining tokens
            result_segments.append(ids_tensor[latent_idx + 1:])

    # Concatenate all at once (much faster than list.extend in loop)
    new_input_ids_tensor = torch.cat(result_segments).tolist()

    # Build labels the same way
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    # Marker CE: Ensure thinking tokens (<think>, </think>) always have CE loss
    # This is separate from Latent Step CE - Marker CE is ALWAYS ON
    # Get thinking token IDs
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))

    # Before first latent
    new_labels_tensor = labels_tensor[:latent_indices[0]]

    # Check latent-step CE controls:
    # - CE_LOSS gate controls whether latent-position CE is active at all.
    # - CE_TOKEN controls latent token vs CoT token labels when CE is active.
    ce_enabled = _latent_step_ce_enabled()
    use_latent_token = ce_enabled and os.environ.get("QWEN3VL_LATENT_STEP_CE_TOKEN", "0") == "1"
    use_cot_token = ce_enabled and not use_latent_token
    latent_label = latent_token_id if use_latent_token else ignore_index

    block_first_targets: list[int] = []
    block_label_tensors: list[torch.Tensor] = []
    for i, (latent_idx, exp_len) in enumerate(zip(latent_indices, expansion_lengths)):
        step_tokens = None
        if cot_step_token_ids is not None and i < len(cot_step_token_ids):
            step_tokens = cot_step_token_ids[i]
        boundary_label = int(labels_tensor[latent_idx].item()) if latent_idx < labels_tensor.shape[0] else ignore_index
        first_target, latent_block_labels = _build_latent_ce_targets(
            exp_len=exp_len,
            boundary_label=boundary_label,
            latent_label=latent_label,
            use_latent_token=use_latent_token,
            use_cot_token=use_cot_token,
            cot_step_token_ids=step_tokens,
            ignore_index=ignore_index,
        )
        block_first_targets.append(first_target)
        block_label_tensors.append(latent_block_labels)

    # The scaffold token before the first latent block predicts the first sampled chunk token.
    if new_labels_tensor.numel() > 0 and block_first_targets:
        new_labels_tensor = new_labels_tensor.clone()
        new_labels_tensor[-1] = block_first_targets[0]

    for i, (latent_idx, _exp_len) in enumerate(zip(latent_indices, expansion_lengths)):
        new_labels_tensor = torch.cat([
            new_labels_tensor,
            block_label_tensors[i],
        ])

        # Content between latents (keep original labels)
        if i < len(latent_indices) - 1:
            next_idx = latent_indices[i + 1]
            inter_labels = labels_tensor[latent_idx + 1:next_idx].clone()
            if inter_labels.numel() > 0:
                # The structural separator token predicts the first sampled token of the next chunk.
                inter_labels[-1] = block_first_targets[i + 1]
            new_labels_tensor = torch.cat([new_labels_tensor, inter_labels])
        else:
            # After last latent - keep remaining labels
            new_labels_tensor = torch.cat([new_labels_tensor, labels_tensor[latent_idx + 1:]])

    # Convert to tensor (already a tensor from torch.cat, use clone to avoid warning)
    new_labels_tensor = new_labels_tensor.clone()

    # Marker CE: Standard next-token prediction already handles thinking tokens correctly
    # - Labels for thinking token positions already contain next-token targets
    # - We only modify labels at <latent> positions (via latent_label)
    # - Thinking tokens naturally keep their original labels for standard CE loss

    return new_input_ids_tensor, new_labels_tensor.tolist()


def _pack_features_after_injection(
    batch: List[dict],
    latent_fields_list: List[dict],
    collator,
    logger,
) -> tuple[List[dict], List[dict]]:
    """Pack sequences after expanding latent placeholders to match injected lengths."""
    cutoff_len = _get_cutoff_len_from_collator(collator, logger)
    latent_token_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))
    ignore_index = getattr(collator, "label_pad_token_id", -100)
    pad_token_id = collator.tokenizer.pad_token_id
    block_diag_attn = getattr(collator, "block_diag_attn", False)

    if not hasattr(_pack_features_after_injection, "_logged"):
        logger.debug(
            f"[Qwen3VL Latent] pack-after-injection enabled: cutoff_len={cutoff_len}, "
            f"block_diag_attn={block_diag_attn}"
        )
        _pack_features_after_injection._logged = True

    expanded_features: List[dict] = []
    expanded_latent_fields: List[dict] = []
    lengths: List[int] = []
    # OPTIMIZED: No longer need length2indexes with first-fit decreasing algorithm
    dropped = 0
    max_seen_len = 0

    for idx, feature in enumerate(batch):
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        latent_fields = latent_fields_list[idx] if idx < len(latent_fields_list) else {}

        # Keep latent payload as file paths inside dataloader workers to avoid
        # passing large tensors through multiprocessing shared memory.
        latent_gt = latent_fields.get("latent_ground_truth") or []
        latent_gt_lengths = _normalize_int_list(latent_fields.get("latent_seq_lens"))
        if latent_gt_lengths is not None and latent_gt and len(latent_gt_lengths) != len(latent_gt):
            logger.warning(
                "[Qwen3VL Latent] Ignoring latent_seq_lens due to length mismatch: %s vs %s",
                len(latent_gt_lengths),
                len(latent_gt),
            )
            latent_gt_lengths = None
        if latent_gt and latent_gt_lengths is None:
            if isinstance(latent_gt[0], torch.Tensor):
                latent_gt_lengths = [_latent_seq_len(t) for t in latent_gt]
            else:
                latent_gt_lengths = _load_latent_seq_lens(latent_gt)

        latent_sup = latent_fields.get("latent_supervision") or []

        cot_step_token_ids = _normalize_cot_chunk_token_ids(latent_fields.get("cot_chunk_token_ids"))
        if cot_step_token_ids is not None and latent_gt:
            num_latent_steps = len(latent_gt)
            if len(cot_step_token_ids) != num_latent_steps:
                logger.warning(
                    "[Qwen3VL Latent] Ignoring cot_chunk_token_ids due to step mismatch: %s vs %s",
                    len(cot_step_token_ids),
                    num_latent_steps,
                )
                cot_step_token_ids = None

        new_input_ids, new_labels = _expand_sample_for_latent_injection(
            input_ids=input_ids,
            labels=labels,
            latent_token_id=latent_token_id,
            thinking_start_id=thinking_start_id,
            thinking_end_id=thinking_end_id,
            latent_ground_truth=latent_gt,
            latent_lengths=latent_gt_lengths,
            ignore_index=ignore_index,
            cot_step_token_ids=cot_step_token_ids,
        )

        length = len(new_input_ids)
        if length > max_seen_len:
            max_seen_len = length
        if length > cutoff_len:
            logger.warning(
                f"[Qwen3VL Latent] Dropped example: expanded length {length} > cutoff_len {cutoff_len}."
            )
            dropped += 1
            continue

        new_feature = dict(feature)
        new_feature["input_ids"] = new_input_ids
        new_feature["labels"] = new_labels
        new_feature["attention_mask"] = [1] * len(new_input_ids)

        # OPTIMIZED: No longer need length2indexes with first-fit decreasing
        lengths.append(length)
        expanded_features.append(new_feature)
        expanded_latent_fields.append(latent_fields)

    if not lengths:
        logger.warning("[Qwen3VL Latent] pack-after-injection: no valid samples after expansion.")
        # Clear latent fields since original batch is not expanded - continuing with
        # original input_ids would cause STRICT mismatch in latent injection
        empty_latent_fields = [{"latent_ground_truth": [], "latent_supervision": [], "num_latent_steps": 0} for _ in batch]
        return batch, empty_latent_fields

    # Aggregate drop stats (log occasionally, only when debug enabled)
    debug_enabled = os.environ.get("QWEN3VL_DEBUG_FORWARD", "0") == "1"
    stats = getattr(_pack_features_after_injection, "_stats", None)
    if stats is None:
        stats = {"total": 0, "dropped": 0, "last_log": 0, "max_len": 0}
    stats["total"] += len(batch)
    stats["dropped"] += dropped
    stats["max_len"] = max(stats["max_len"], max_seen_len)
    if debug_enabled and stats["total"] - stats["last_log"] >= 1000:
        drop_rate = (stats["dropped"] / max(1, stats["total"])) * 100.0
        logger.info(
            "[Qwen3VL Latent] pack-after-injection drop stats: "
            f"dropped={stats['dropped']}/{stats['total']} ({drop_rate:.2f}%), "
            f"max_seen_len={stats['max_len']}, cutoff_len={cutoff_len}"
        )
        stats["last_log"] = stats["total"]
    _pack_features_after_injection._stats = stats

    # OPTIMIZED: First-fit decreasing bin packing (O(n log n) instead of O(n²))
    # Sort by length decreasing, then assign to bins using first-fit
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    knapsacks = []
    current_knapsack = []
    current_sum = 0

    for idx in sorted_indices:
        length = lengths[idx]
        if length > cutoff_len:
            # Drop this sample
            dropped += 1
            stats['dropped'] += 1
            continue

        if current_sum + length <= cutoff_len:
            # Add to current knapsack
            current_knapsack.append(idx)
            current_sum += length
        else:
            # Start new knapsack
            if current_knapsack:
                knapsacks.append(current_knapsack)
            current_knapsack = [idx]
            current_sum = length

    # Don't forget the last knapsack
    if current_knapsack:
        knapsacks.append(current_knapsack)
    packed_features: List[dict] = []
    packed_latent_fields: List[dict] = []

    for knapsack in knapsacks:
        # Pre-compute total lengths for pre-allocation
        # OPTIMIZED: knapsack contains indices, not lengths
        total_len = sum(lengths[idx] for idx in knapsack)

        # Use list comprehension for faster initialization
        packed_input_ids: list[int] = [0] * total_len
        packed_labels: list[int] = [0] * total_len
        packed_attention_mask: list[int] = [0] * total_len

        # Use lists for mutable accumulation (extend faster than +=)
        packed_images: list = []
        packed_videos: list = []
        packed_audios: list = []
        packed_latent_gt: list = []
        packed_latent_sup: list = []
        packed_num_latent_steps: list = []

        # Track current position for in-place assignment (faster than extend)
        current_pos = 0
        # OPTIMIZED: knapsack contains indices directly
        for seg_idx, index in enumerate(knapsack):
            feature = expanded_features[index]
            feat_input_ids = feature["input_ids"]
            feat_labels = feature["labels"]
            feat_len = len(feat_input_ids)

            # In-place assignment (faster than +=)
            packed_input_ids[current_pos:current_pos + feat_len] = feat_input_ids
            packed_labels[current_pos:current_pos + feat_len] = feat_labels

            # Vectorized attention mask creation
            attn_val = seg_idx + 1 if block_diag_attn else 1
            packed_attention_mask[current_pos:current_pos + feat_len] = [attn_val] * feat_len

            current_pos += feat_len

            # Extend lists (slightly faster than +=)
            images = feature.get("images")
            if images:
                packed_images.extend(images)
            videos = feature.get("videos")
            if videos:
                packed_videos.extend(videos)
            audios = feature.get("audios")
            if audios:
                packed_audios.extend(audios)

            latent_fields = expanded_latent_fields[index] if index < len(expanded_latent_fields) else {}
            latent_gt = latent_fields.get("latent_ground_truth")
            if latent_gt:
                # Keep as nested list [[sample1_tensors], [sample2_tensors], ...]
                # NOT flattened, so injection can access per-sample
                packed_latent_gt.append(latent_gt)
            latent_sup = latent_fields.get("latent_supervision")
            if latent_sup:
                # Keep as nested list [[sample1_tensors], [sample2_tensors], ...]
                packed_latent_sup.append(latent_sup)
            num_steps = latent_fields.get("num_latent_steps")
            if num_steps is not None:
                if isinstance(num_steps, list):
                    packed_num_latent_steps.extend(num_steps)
                else:
                    packed_num_latent_steps.append(num_steps)

        # Trim to actual size
        packed_input_ids = packed_input_ids[:current_pos]
        packed_labels = packed_labels[:current_pos]
        packed_attention_mask = packed_attention_mask[:current_pos]

        # Padding (vectorized)
        if len(packed_input_ids) < cutoff_len + 1:
            pad_length = cutoff_len - len(packed_input_ids) + 1
            packed_input_ids.extend([pad_token_id] * pad_length)
            packed_labels.extend([ignore_index] * pad_length)
            packed_attention_mask.extend([0] * pad_length)

        if len(packed_input_ids) != cutoff_len + 1:
            raise ValueError(
                "[Qwen3VL Latent] pack-after-injection: packed length mismatch "
                f"{len(packed_input_ids)} != cutoff_len+1 ({cutoff_len + 1})."
            )

        packed_features.append(
            {
                "input_ids": packed_input_ids,
                "attention_mask": packed_attention_mask,
                "labels": packed_labels,
                "images": packed_images or None,
                "videos": packed_videos or None,
                "audios": packed_audios or None,
            }
        )
        packed_latent_fields.append(
            {
                "latent_ground_truth": packed_latent_gt,
                "latent_supervision": packed_latent_sup,
                "num_latent_steps": packed_num_latent_steps,
            }
        )

    return packed_features, packed_latent_fields


def _patch_data_collator(logger) -> None:
    """Patch data collator to load latent supervision from disk.

    This modifies the data collation to:
    1. Load latent_ground_truth tensors for injection (thinking features)
    2. Load latent_supervision tensors for OT loss (original image features)
    3. Compute latent_positions mask from input_ids

    Data format:
    - latent_ground_truth: [thinking_0.latent.pt, thinking_1.latent.pt, ...]
    - latent_supervision: [original_superv_0.latent.pt, original_superv_1.latent.pt, ...]
      Both lists have the same length (one per thinking chunk)
    """
    # Store original __call__ method
    original_call = SFTDataCollatorWith4DAttentionMask.__call__

    @functools.wraps(original_call)
    def wrapped_call(self, batch: List[dict]) -> dict:
        """Wrapped collator that loads latent supervision."""
        total_start = time.time()

        # Snapshot image paths per sample for debug (original collator mutates batch)
        batch_images_per_sample = []
        if batch and isinstance(batch[0], dict):
            for item in batch:
                images = item.get("images", None) or []
                batch_images_per_sample.append(list(images))

        # Extract latent fields BEFORE calling original collator
        # (tokenizer can't handle these fields)
        latent_fields_list = []
        has_latent = False

        if batch and isinstance(batch[0], dict):
            for item in batch:
                latent_item = {}
                # Keep non-token fields out of the tokenizer/collator path.
                # `cot` is optional and is used only for optional CE labeling of latent positions.
                for key in [
                    'latent_ground_truth',
                    'latent_supervision',
                    'num_latent_steps',
                    'cot',
                    'latent_seq_lens',
                    'cot_chunk_token_ids',
                ]:
                    if key in item:
                        latent_item[key] = item.pop(key)  # Remove from batch item
                        has_latent = True

                # Strictness: if sample still contains <latent> token ids, metadata must exist.
                # Avoid implicit image-path fallback to keep dataflow deterministic.
                input_ids = item.get("input_ids", [])
                if (
                    isinstance(input_ids, list)
                    and int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669")) in input_ids
                    and not latent_item.get("latent_ground_truth")
                ):
                    raise ValueError(
                        "[Qwen3VL Latent] Missing latent_ground_truth in preprocessed sample "
                        "that contains <latent> token(s). Ensure dataset_info columns are preserved."
                    )

                latent_fields_list.append(latent_item)

            if has_latent and not hasattr(wrapped_call, '_logged_extract'):
                logger.debug(f"[Qwen3VL Latent] Extracted latent fields from batch (tokenizer won't see them)")
                wrapped_call._logged_extract = True

        # Pack-after-injection path (always enabled when latent supervision is active)
        pack_start = time.time()
        try:
            batch, latent_fields_list = _pack_features_after_injection(
                batch=batch,
                latent_fields_list=latent_fields_list,
                collator=self,
                logger=logger,
            )
            has_latent = any(bool(item) for item in latent_fields_list)
            batch_images_per_sample = [
                list(item.get("images", None) or []) for item in batch
            ]
        except Exception as e:
            logger.warning(f"[Qwen3VL Latent] pack-after-injection failed: {e}")
            raise
        pack_time = time.time() - pack_start

        # Call original collator (tokenizer only sees standard fields)
        collator_start = time.time()
        result = original_call(self, batch)
        collator_time = time.time() - collator_start

        # LOG: Show actual result shapes (debug level, only first few steps)
        if 'input_ids' in result:
            if not hasattr(wrapped_call, '_collator_log_count'):
                wrapped_call._collator_log_count = 0
            if wrapped_call._collator_log_count < 3:
                logger.debug(f"[Qwen3VL COLLATOR RESULT] input_ids={result['input_ids'].shape}, total_tokens={result['input_ids'].shape[0] * result['input_ids'].shape[1]}, images={len(result.get('pixel_values', [])) if isinstance(result.get('pixel_values'), list) else (result.get('pixel_values').shape[0] if torch.is_tensor(result.get('pixel_values')) else 'N/A')}")
                wrapped_call._collator_log_count += 1

        # Debug: validate pixel_values length matches image_grid_thw product (only first few steps)
        if not hasattr(wrapped_call, '_pixel_val_count'):
            wrapped_call._pixel_val_count = 0
        # Only validate first few steps to avoid overhead
        if wrapped_call._pixel_val_count < 3:
            try:
                pixel_values = result.get("pixel_values", None)
                image_grid_thw = result.get("image_grid_thw", None)
                if pixel_values is not None and image_grid_thw is not None:
                    pv_len = pixel_values.shape[0] if torch.is_tensor(pixel_values) else len(pixel_values)
                    if torch.is_tensor(image_grid_thw):
                        grid_prod = int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item())
                        num_grids = int(image_grid_thw.shape[0])
                        if grid_prod != pv_len:
                            # Build a detailed error to pinpoint the offending batch
                            total_images = sum(len(x) for x in batch_images_per_sample)
                            pv_shape = getattr(pixel_values, "shape", None)
                            pv_dtype = getattr(pixel_values, "dtype", None)
                            proc = getattr(self, "processor", None)
                            ip = getattr(proc, "image_processor", None) if proc is not None else None
                            msg_lines = [
                                "[Qwen3VL Latent] pixel_values/image_grid_thw mismatch in collator",
                                f"  pixel_values_len: {pv_len}",
                                f"  pixel_values_shape: {pv_shape}",
                                f"  pixel_values_dtype: {pv_dtype}",
                                f"  image_grid_thw_sum: {grid_prod}",
                                f"  image_grid_thw_count: {num_grids}",
                                f"  batch_images_total: {total_images}",
                                f"  image_grid_thw: {image_grid_thw.tolist()}",
                                f"  processor: {type(proc)}",
                                f"  image_processor: {type(ip)}",
                            ]
                            # Add per-sample image paths (truncate if too long)
                            for i, paths in enumerate(batch_images_per_sample):
                                if not paths:
                                    msg_lines.append(f"  sample[{i}] images: []")
                                    continue
                                preview = paths if len(paths) <= 2 else paths[:2] + ["..."]
                                msg_lines.append(f"  sample[{i}] images({len(paths)}): {preview}")
                            raise ValueError("\n".join(msg_lines))
                wrapped_call._pixel_val_count += 1
            except Exception as e:
                # Re-raise to surface exact debug info
                raise

        # Add latent supervision handling for thinking datasets
        if has_latent:
            try:
                if not hasattr(wrapped_call, '_logged_process'):
                    logger.debug(f"[Qwen3VL Latent] Processing batch with latent supervision...")
                    wrapped_call._logged_process = True
                result = _add_latent_supervision_to_batch(result, latent_fields_list, logger)
            except Exception as e:
                logger.warning(f"[Qwen3VL Latent] Failed to add latent supervision: {e}")
                logger.warning(traceback.format_exc())

        total_time = time.time() - total_start
        # Log timing every 10 steps to avoid spam
        if not hasattr(wrapped_call, '_timing_count'):
            wrapped_call._timing_count = 0
        wrapped_call._timing_count += 1
        if wrapped_call._timing_count <= 10 or wrapped_call._timing_count % 100 == 0:
            logger.debug(f"[Qwen3VL Latent TIMING] Step {wrapped_call._timing_count}: pack={pack_time:.3f}s, collator={collator_time:.3f}s, total={total_time:.3f}s")

        return result

    # Apply patch
    SFTDataCollatorWith4DAttentionMask.__call__ = wrapped_call
    logger.debug("[Qwen3VL Latent] ✓ Patched SFTDataCollatorWith4DAttentionMask.__call__")


def _patch_model_forward(logger) -> None:
    """Patch model forward pass to inject latents and compute thinking loss.

    This patches the model's forward method to:
    1. Inject latent_ground_truth at <latent> positions (no projection needed)
    2. Compute thinking loss using latent_supervision directly on LLM hidden states
    3. Combine CE loss with thinking loss

    APPROACH (following RoT implementation, adapted for LlamaFactory PEFT):
    - Calls PEFT-wrapped forward with output_hidden_states=True
    - Extracts last_hidden_states from outputs.hidden_states[-1]
    - Preserves PEFT/LoRA optimization by using the full model forward
    """
    # Store original PEFT forward
    original_forward = PeftModelForCausalLM.forward

    # Capture token IDs from environment for use in patched_forward closure
    latent_token_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))

    @functools.wraps(original_forward)
    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        latent_ground_truth=None,
        latent_ground_truth_paths=None,
        latent_ground_truth_packed=None,
        latent_supervision=None,
        latent_supervision_paths=None,
        latent_positions=None,
        thinking_loss_weight=None,
        **kwargs
    ):
        """Patched forward with latent injection and thinking loss."""
        # Check debug flag first
        debug_enabled = os.environ.get("QWEN3VL_DEBUG_FORWARD", "0") == "1"

        # DEBUG: Log what was passed as direct parameters
        if debug_enabled:
            print(f"[PARAM_DEBUG] latent_ground_truth: type={type(latent_ground_truth)}, is_none={latent_ground_truth is None}", flush=True, file=sys.stderr)
            if latent_ground_truth is not None:
                try:
                    print(f"[PARAM_DEBUG] latent_ground_truth len: {len(latent_ground_truth)}", flush=True, file=sys.stderr)
                    inner_lens = [len(x) if x else 0 for x in latent_ground_truth]
                    print(f"[PARAM_DEBUG] latent_ground_truth inner lens: {inner_lens}", flush=True, file=sys.stderr)
                except Exception as e:
                    print(f"[PARAM_DEBUG] error getting inner lens: {e}", flush=True, file=sys.stderr)

        # DEBUG: Log state before injection
        if debug_enabled and latent_ground_truth is not None:
            try:
                before_inject_inner = [len(x) if x else 0 for x in latent_ground_truth]
                print(f"[INJECT_DEBUG] Before injection: inner lens = {before_inject_inner}", flush=True, file=sys.stderr)
            except Exception as e:
                print(f"[INJECT_DEBUG] error before injection: {e}", flush=True, file=sys.stderr)
            print(f"[PARAM_DEBUG] latent_supervision: type={type(latent_supervision)}, is_none={latent_supervision is None}", flush=True, file=sys.stderr)
            print(f"[PARAM_DEBUG] latent_positions: type={type(latent_positions)}, is_none={latent_positions is None}", flush=True, file=sys.stderr)

        # DEBUG: Log all kwargs keys to trace what's being passed
        debug_kwargs_keys = list(kwargs.keys()) if kwargs else []
        if debug_enabled and kwargs:
            print(f"[KWARGS_DEBUG] kwargs keys: {debug_kwargs_keys}", flush=True, file=sys.stderr)
            for k in ['latent_ground_truth', 'latent_supervision', 'latent_positions']:
                v = kwargs.get(k)
                if v is not None:
                    print(f"[KWARGS_DEBUG] {k} in kwargs: len={len(v) if hasattr(v, '__len__') else 'no len'}", flush=True, file=sys.stderr)
        if debug_enabled:
            pv_shape = getattr(pixel_values, 'shape', None)
            in_ids_shape = getattr(input_ids, 'shape', None) if input_ids is not None else None
            print(f"[PATCHED_FORWARD] Called! pixel_values={pv_shape}, input_ids={in_ids_shape}", flush=True, file=sys.stderr)

            # DEBUG: Log input shapes (use print to ensure visibility)
            print(f"[DEBUG FORWARD] START input_ids={input_ids.shape if input_ids is not None else None}, "
                  f"pixel_values={pixel_values.shape if pixel_values is not None else None}, "
                  f"image_grid_thw={image_grid_thw if image_grid_thw is not None else None}, "
                  f"latent_ground_truth={len(latent_ground_truth) if latent_ground_truth else None}", flush=True, file=sys.stderr)

        # Check if VAE is in loss spec
        loss_spec = _get_loss_spec()

        # Materialize latent tensors on the training process to avoid large IPC payloads
        # from dataloader workers.
        latent_ground_truth = _materialize_latent_batch(
            latent_batch=latent_ground_truth,
            latent_paths_batch=latent_ground_truth_paths,
            strict=True,
        )
        latent_supervision = _materialize_latent_batch(
            latent_batch=latent_supervision,
            latent_paths_batch=latent_supervision_paths,
            strict=False,
        )

        # Invariant: when latent positions exist, path-based payloads must be
        # materialized into real tensors before loss computation.
        if latent_positions is not None and isinstance(latent_positions, torch.Tensor):
            sample_has_latent = latent_positions.any(dim=1).tolist() if latent_positions.ndim >= 2 else [bool(latent_positions.any().item())]
            gt_tensor_counts = [
                sum(1 for x in sample if isinstance(x, torch.Tensor))
                for sample in (latent_ground_truth or [])
            ]
            if any(sample_has_latent):
                for i, has_lat in enumerate(sample_has_latent):
                    if not has_lat:
                        continue
                    gt_cnt = gt_tensor_counts[i] if i < len(gt_tensor_counts) else 0
                    if gt_cnt == 0:
                        msg = (
                            f"[Qwen3VL Latent] sample {i} has latent_positions but no materialized "
                            "latent_ground_truth tensors after loading."
                        )
                        logger.warning(msg)

        if latent_ground_truth_packed is None and latent_ground_truth:
            packed_chunks = []
            for sample_gt in latent_ground_truth:
                if sample_gt:
                    packed_chunks.append(torch.cat(sample_gt, dim=0))
            if packed_chunks:
                latent_ground_truth_packed = torch.cat(packed_chunks, dim=0)

        # DEBUG: Log what's received
        if debug_enabled:
            print(f"[DEBUG] latent_ground_truth={type(latent_ground_truth)}, len={len(latent_ground_truth) if latent_ground_truth else 'None/0'}", flush=True, file=sys.stderr)
            print(f"[DEBUG] latent_supervision={type(latent_supervision)}, len={len(latent_supervision) if latent_supervision else 'None/0'}", flush=True, file=sys.stderr)
            print(f"[DEBUG] latent_positions={type(latent_positions)}, shape={latent_positions.shape if latent_positions is not None else 'None'}", flush=True, file=sys.stderr)

        use_latent_vae = "vae" in loss_spec

        # Lazy initialization of LatentVAE on first forward pass
        if use_latent_vae:
            # Check if VAE already exists (from checkpoint load) or needs to be created
            if not hasattr(self, 'latent_vae') or self.latent_vae is None:
                # Get hidden size from model config and create VAE with default config
                hidden_size = self.config.hidden_size
                # Create VAE with config from env var
                vae = LatentVAE(hidden_size=hidden_size, intermediate_size=int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512")), deterministic=False)
                # Register as module so it gets saved/loaded with checkpoint
                self.register_module('latent_vae', vae)
                logger.info(f"[Qwen3VL Latent] Created LatentVAE: hidden_size={hidden_size}, intermediate_size={os.environ.get('QWEN3VL_VAE_INTERMEDIATE_SIZE', '512')}")

            # Ensure VAE parameters are trainable (in case model was loaded with frozen params)
            _ensure_vae_trainable(self)

            # If model was loaded from checkpoint, VAE weights should already be in the state dict
            # and will be applied automatically by PyTorch's load_state_dict

        # Inject latent_ground_truth features if provided
        if latent_ground_truth is not None and latent_positions is not None:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            inputs_embeds = _inject_latent_features_inplace(
                inputs_embeds=inputs_embeds,
                latent_supervision=latent_ground_truth,
                latent_positions=latent_positions,
                latent_paths=latent_ground_truth_paths,
            )
            if attention_mask is not None and attention_mask.shape[1] != inputs_embeds.shape[1]:
                raise ValueError("[Qwen3VL Latent] pack-after-injection: attention_mask length mismatch.")
            if labels is not None and labels.shape[1] != inputs_embeds.shape[1]:
                raise ValueError("[Qwen3VL Latent] pack-after-injection: labels length mismatch.")
            if position_ids is not None and position_ids.shape[-1] != inputs_embeds.shape[1]:
                raise ValueError("[Qwen3VL Latent] pack-after-injection: position_ids length mismatch.")

        # Pop output_hidden_states from kwargs to avoid duplicate keyword argument
        kwargs.pop('output_hidden_states', None)

        # Check if we need thinking loss
        has_latent_positions = False
        if latent_positions is not None:
            if isinstance(latent_positions, torch.Tensor):
                has_latent_positions = bool(latent_positions.any().item())
            else:
                logger.warning(
                    "[Qwen3VL Latent] Expected latent_positions tensor, got %s",
                    type(latent_positions),
                )
        need_hidden_states = latent_supervision is not None and has_latent_positions

        # Debug logging for latent stats
        if debug_enabled:
            sup_len = 0
            gt_len = 0
            pos_none = latent_positions is None
            if latent_supervision:
                sup_len = len(latent_supervision)
            if latent_ground_truth:
                gt_len = len(latent_ground_truth)
        if debug_enabled:
            print(f"[LATENT_STATS] gt_len={gt_len}, sup_len={sup_len}, pos_none={pos_none}, has_pos={has_latent_positions}, need_hs={need_hidden_states}", flush=True, file=sys.stderr)

        # Call PEFT-wrapped original forward (preserves LoRA optimization)
        # NOTE: output_hidden_states triggers the HF output recorder, which is very slow for long context.
        # We capture only the final hidden state via a forward hook to avoid per-layer recording.
        last_hidden_states = None
        hook_handle = None
        use_hook = bool(need_hidden_states) and os.environ.get("QWEN3VL_HIDDEN_STATES_HOOK", "1") != "0"
        if use_hook:
            try:
                text_model = self.model.language_model

                def _capture_last_hidden(_module, _inputs, output):
                    nonlocal last_hidden_states
                    # Qwen3VLTextModel returns BaseModelOutputWithPast or tuple
                    if hasattr(output, "last_hidden_state"):
                        last_hidden_states = output.last_hidden_state
                    elif isinstance(output, (tuple, list)) and len(output) > 0:
                        last_hidden_states = output[0]
                    else:
                        last_hidden_states = output

                hook_handle = text_model.register_forward_hook(_capture_last_hidden)
            except Exception as e:
                logger.warning(f"[Qwen3VL Latent] Hidden-state hook registration failed, falling back to output_hidden_states: {e}")
                use_hook = False

        # Cast pixel_values to bfloat16 if present (fix for TransparentEvalCallback dtype mismatch)
        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        # Cast inputs_embeds to bfloat16 if they come from vision encoder in float32
        if inputs_embeds is not None and inputs_embeds.dtype != torch.bfloat16:
            inputs_embeds = inputs_embeds.to(torch.bfloat16)

        # DEBUG: Log before model forward
        if debug_enabled:
            print(f"[DEBUG FORWARD] Before model forward:", flush=True, file=sys.stderr)
            print(f"  input_ids: {input_ids.shape if input_ids is not None else None}", flush=True, file=sys.stderr)
            print(f"  inputs_embeds: {inputs_embeds.shape if inputs_embeds is not None else None}", flush=True, file=sys.stderr)
            print(f"  pixel_values: {pixel_values.shape if pixel_values is not None else None}", flush=True, file=sys.stderr)
            print(f"  image_grid_thw: {image_grid_thw}", flush=True, file=sys.stderr)
            print(f"  labels: {labels.shape if labels is not None else None}", flush=True, file=sys.stderr)

        # Only skip pixel_values when latent_ground_truth is present and non-empty
        # OR when latent tokens exist in input_ids (latent injection training)
        has_latent = False
        latent_len = 0

        # Quick check: is latent_ground_truth being passed?
        gt_len = 0
        sup_len = 0
        if latent_ground_truth is not None:
            gt_len = len(latent_ground_truth)
        if latent_supervision is not None:
            sup_len = len(latent_supervision)
        if gt_len == 0 and sup_len == 0 and debug_enabled:
            print(f"[LATENT_STATS] gt=0, sup=0 - latent data NOT passed to forward!", flush=True, file=sys.stderr)

        if latent_ground_truth is not None:
            for item in latent_ground_truth:
                if item is not None and len(item) > 0:
                    has_latent = True
                    latent_len += 1

        # Fallback: Check if input_ids contain latent tokens (for latent injection training)
        # If latent tokens exist, skip pixel_values even if latent_ground_truth wasn't passed
        latent_token_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
        if not has_latent and input_ids is not None:
            has_latent = (input_ids == latent_token_id).any().item()

        if debug_enabled:
            print(f"[LATENT_CHECK] latent_ground_truth={type(latent_ground_truth)}, len={len(latent_ground_truth) if latent_ground_truth else 0}, has_latent={has_latent}, latent_len={latent_len}", flush=True, file=sys.stderr)

        if has_latent:
            # Has valid latent ground truth - use latent injection
            pixel_values_to_model = None
            image_grid_thw_to_model = None
        else:
            # No latent ground truth - use normal pixel_values
            pixel_values_to_model = pixel_values
            image_grid_thw_to_model = image_grid_thw

        if debug_enabled:
            latent_present = latent_ground_truth is not None and len(latent_ground_truth) > 0
            print(f"[PATCHED_FORWARD] latent_present={latent_present}, latent_len={len(latent_ground_truth) if latent_ground_truth else 0}, pv={getattr(pixel_values_to_model, 'shape', None)}", flush=True, file=sys.stderr)

        outputs = original_forward(
            self,
            input_ids=None if inputs_embeds is not None else input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            pixel_values=pixel_values_to_model,
            image_grid_thw=image_grid_thw_to_model,
            output_hidden_states=need_hidden_states and not use_hook,
            **kwargs
        )

        if hook_handle is not None:
            hook_handle.remove()

        # Compute thinking loss using latent_supervision (direct on LLM hidden states)
        thinking_loss = None

        if need_hidden_states:
            # Extract last layer hidden states from hook or outputs.hidden_states[-1]
            if last_hidden_states is None:
                hidden_states_out = getattr(outputs, "hidden_states", None)
                if hidden_states_out is not None:
                    last_hidden_states = hidden_states_out[-1]
                else:
                    raise RuntimeError(
                        "[Qwen3VL Latent] Failed to capture last_hidden_states. "
                        "Set QWEN3VL_HIDDEN_STATES_HOOK=0 to fall back to output_hidden_states."
                    )

            # IMPORTANT: Delete intermediate layer outputs to free memory!
            if getattr(outputs, "hidden_states", None) is not None:
                del outputs.hidden_states
                outputs.hidden_states = None

            # Debug: Log what's being passed to _compute_thinking_loss
            if debug_enabled:
                gt_lens = [len(x) if x else 0 for x in latent_ground_truth] if latent_ground_truth else []
                sup_lens = [len(x) if x else 0 for x in latent_supervision] if latent_supervision else []
                print(f"[FORWARD_DEBUG] Before loss: gt_lens={gt_lens}, sup_lens={sup_lens}, latent_positions.any={latent_positions.any().item() if latent_positions is not None else 'None'}", flush=True, file=sys.stderr)

            thinking_loss, ot_stats = _compute_thinking_loss(
                hidden_states=last_hidden_states,
                latent_ground_truth=latent_ground_truth,
                latent_supervision=latent_supervision,
                latent_positions=latent_positions,
            )

            # Apply loss weight if specified
            if thinking_loss is not None and thinking_loss_weight is not None:
                weight = float(thinking_loss_weight)
                thinking_loss = thinking_loss * weight

        # Compute LatentVAE loss if enabled
        vae_loss = None
        latent_vae_loss = None  # Initialize to avoid unbound variable
        pred_embed_forward_loss = None  # Initialize for logging
        sampled_latents = None
        batch_indices = None
        seq_indices = None
        if use_latent_vae and need_hidden_states and latent_supervision is not None and last_hidden_states is not None:
            # VAE is already on the correct device as part of FSDP-wrapped model
            # Don't move it explicitly to avoid FSDP shard issues
            vae = self.latent_vae

            # Get hidden states at latent positions (shifted by 1 for autoregressive)
            # VAE uses latent_ground_truth (same as pre_think_mse), not latent_supervision (which is for OT)
            # Also returns sampled_latents for pred_embed_forward (second forward with sampled latent embeddings,
            # not sampled CoT token IDs).
            nll_loss, entropy, last_hidden_states, sampled_latents, batch_indices, seq_indices = _compute_vae_loss(
                vae=vae,
                hidden_states=last_hidden_states,
                latent_positions=latent_positions,
                latent_supervision_packed=latent_ground_truth_packed,
            )
            if nll_loss is not None and entropy is not None:
                # Match ReGuLaR: total loss = nll_loss + entropy_weight * entropy
                vae_loss_weight = float(os.environ.get("QWEN3VL_THINKING_LOSS_WEIGHT", "1.0"))
                entropy_weight = 0.01
                vae_loss = nll_loss * vae_loss_weight + entropy_weight * entropy
                latent_vae_loss = nll_loss  # For logging
                logger.debug(f"[Qwen3VL Latent] VAE loss: nll={nll_loss.item():.4f}, entropy={entropy.item():.4f}")
            else:
                # Log if VAE loss is None after computation
                rank_0 = not dist.is_initialized() or dist.get_rank() == 0
                if rank_0:
                    # Check lengths - VAE now uses latent_ground_truth
                    gt_lens = [len(s) if s else 0 for s in latent_ground_truth] if latent_ground_truth else []
                    pos_sum = latent_positions.sum().item() if latent_positions is not None else 0
                    logger.info(f"[Qwen3VL Latent] VAE returned None: gt_lens={gt_lens}, pos_sum={pos_sum}")

        # FSDP safety: if any rank has sampled latents, all ranks must run pred-embed second forward.
        if use_latent_vae and inputs_embeds is not None and labels is not None:
            local_pred_ready = (
                sampled_latents is not None
                and batch_indices is not None
                and seq_indices is not None
                and int(sampled_latents.numel()) > 0
            )
            global_pred_ready = local_pred_ready
            if dist.is_initialized():
                pred_ready_tensor = torch.tensor(
                    [1 if local_pred_ready else 0],
                    device=inputs_embeds.device,
                    dtype=torch.int32,
                )
                dist.all_reduce(pred_ready_tensor, op=dist.ReduceOp.MAX)
                global_pred_ready = int(pred_ready_tensor.item()) > 0

            if global_pred_ready:
                if local_pred_ready:
                    sampled_latents_for_forward = sampled_latents
                    batch_indices_for_forward = batch_indices
                    seq_indices_for_forward = seq_indices
                else:
                    sampled_latents_for_forward = inputs_embeds.new_empty((0, int(inputs_embeds.shape[-1])))
                    batch_indices_for_forward = torch.empty(0, dtype=torch.long, device=inputs_embeds.device)
                    seq_indices_for_forward = torch.empty(0, dtype=torch.long, device=inputs_embeds.device)

                pred_embed_forward_loss = _compute_pred_embed_forward_loss(
                    model=self.model,
                    original_inputs_embeds=inputs_embeds,
                    sampled_latents=sampled_latents_for_forward,
                    batch_indices=batch_indices_for_forward,
                    seq_indices=seq_indices_for_forward,
                    labels=labels,
                    attention_mask=attention_mask,
                    # Keep pred-embed forward text-only to avoid rank-dependent vision branches.
                    pixel_values=None,
                    image_grid_thw=None,
                )

        # Combine losses (OT is default, CE is optional during eval)
        ce_loss = outputs.loss if hasattr(outputs, 'loss') else None
        pred_embed_weight = float(os.environ.get("QWEN3VL_PRED_EMBED_FORWARD_WEIGHT", "1.0"))

        # Combine CE loss and thinking loss properly
        # Also include VAE loss and pred_embed_forward_loss (like ReGuLaR)
        if thinking_loss is not None:
            loss = thinking_loss
            if ce_loss is not None:
                loss = ce_loss + thinking_loss
            if vae_loss is not None:
                loss = loss + vae_loss
            if pred_embed_forward_loss is not None:
                # Add pred_embed_forward_loss with weight (like ReGuLaR)
                loss = loss + pred_embed_forward_loss * pred_embed_weight
        elif vae_loss is not None:
            # VAE only mode
            if ce_loss is not None:
                loss = ce_loss + vae_loss
            else:
                loss = vae_loss
            if pred_embed_forward_loss is not None:
                loss = loss + pred_embed_forward_loss * pred_embed_weight
        elif pred_embed_forward_loss is not None:
            # pred_embed_forward only
            loss = pred_embed_forward_loss * pred_embed_weight
        elif ce_loss is not None:
            # Fallback to CE loss only if thinking loss is not available
            loss = ce_loss
        else:
            loss = None

        # Debug: Log gradient norms on hidden states (only first few times)
        grad_debug_enabled = os.environ.get("QWEN3VL_LOG_HIDDEN_STATES_GRADIENT", "0") == "1"
        if grad_debug_enabled and last_hidden_states is not None and last_hidden_states.requires_grad:
            rank_0 = _is_rank0()
            if rank_0 and not hasattr(patched_forward, '_grad_log_count'):
                patched_forward._grad_log_count = 0
            if rank_0 and patched_forward._grad_log_count < 3:
                # Save hidden_states for gradient computation after backward
                def grad_hook(module, grad_input, grad_output):
                    grad_norm = grad_output[0].norm().item() if grad_output[0] is not None else 0
                    logger.info(f"[Qwen3VL Latent] hidden_states grad norm: {grad_norm:.6f}")
                last_hidden_states.register_hook(grad_hook)
                patched_forward._grad_log_count += 1

        # Log loss breakdown (always log for monitoring)
        if ce_loss is not None:
            ce_val = ce_loss.item()
        else:
            ce_val = 0.0

        # Get loss type from env var for logging
        loss_type = _get_loss_spec()
        # Parse loss types (e.g., "vae+ot+pre_think_mse" -> ["vae", "ot", "pre_think_mse"])
        loss_names = [l.strip() for l in loss_type.replace("+", " ").split()]

        # Get thinking loss value
        if thinking_loss is not None:
            thinking_val = thinking_loss.item()
        else:
            thinking_val = 0.0

        # Log all losses in one line on rank 0
        rank_0 = _is_rank0()

        if rank_0:
            # Stash for TrainerCallback logging.
            vae_val = float(vae_loss.item()) if vae_loss is not None else None
            pred_embed_val = float(pred_embed_forward_loss.item()) if pred_embed_forward_loss is not None else None
            setattr(
                self,
                "_qwen3vl_last_loss_info",
                {
                    "loss_spec": loss_type,
                    "use_latent_vae": "vae" in loss_type,
                    "ce": float(ce_val),
                    "thinking": float(thinking_val) if thinking_loss is not None else None,
                    "vae": vae_val,
                    "pred_embed_ce_loss": pred_embed_val,
                    "total": float(loss.item()) if loss is not None else None,
                },
            )

        # Update outputs
        if loss is not None:
            outputs.loss = loss

        return outputs

    # Apply patch to PEFT wrapper only (trainer uses PEFT-wrapped model)
    PeftModelForCausalLM.forward = patched_forward
    logger.debug("[Qwen3VL Latent] ✓ Patched PeftModelForCausalLM.forward")


def _ensure_vae_created(model) -> None:
    """Create VAE module on model if it doesn't exist and VAE loss is enabled.

    This must be called BEFORE create_optimizer so VAE params are available
    when the optimizer is created.
    """
    loss_spec = _get_loss_spec()
    if "vae" not in loss_spec:
        return

    if hasattr(model, 'latent_vae') and model.latent_vae is not None:
        return  # Already created

    # Get hidden size from model config
    if hasattr(model, 'config'):
        hidden_size = getattr(model.config, 'hidden_size', None)
        if hidden_size is None:
            hidden_size = getattr(model.config, 'text_config', {}).get('hidden_size', 2048)
    else:
        hidden_size = 2048

    # Create VAE
    vae = LatentVAE(hidden_size=hidden_size, intermediate_size=int(os.environ.get("QWEN3VL_VAE_INTERMEDIATE_SIZE", "512")), deterministic=False)
    model.register_module('latent_vae', vae)
    logger.debug(f"[Qwen3VL Latent] Pre-created LatentVAE: hidden_size={hidden_size}, intermediate_size={os.environ.get('QWEN3VL_VAE_INTERMEDIATE_SIZE', '512')}")


def _ensure_vae_in_optimizer(trainer, logger) -> None:
    """Ensure VAE parameters are in optimizer after optimizer is created."""
    loss_spec = _get_loss_spec()
    if "vae" not in loss_spec:
        return

    if not hasattr(trainer, 'optimizer') or trainer.optimizer is None:
        return

    # Find VAE parameters in model
    vae_params = []
    for name, param in trainer.model.named_parameters():
        if 'latent_vae' in name and param.requires_grad:
            vae_params.append((name, param))

    if not vae_params:
        return

    # Check if VAE params are already in optimizer
    existing_param_ids = {id(p) for group in trainer.optimizer.param_groups for p in group.get("params", [])}
    missing_params = [(n, p) for n, p in vae_params if id(p) not in existing_param_ids]

    if missing_params:
        trainer.optimizer.add_param_group({"params": [p for n, p in missing_params]})
        logger.info(f"[Qwen3VL Latent] Added {len(missing_params)} VAE params to optimizer")


# Trainer callback patching - auto-add VAE, TransparentEval, and Curriculum callbacks
def _patch_trainer_callback(logger) -> None:
    """Patch CustomSeq2SeqTrainer to auto-add callbacks and ensure VAE is trainable.

    Adds:
    - VAESaveCallback: Saves VAE weights separately when QWEN3VL_LOSS_TYPE includes "vae"
    - QwenTransparentEvalCallback: Runs transparent eval during training
    - QwenCurriculumCallback: Handles curriculum learning when QWEN3VL_CURRICULUM_ENABLE=1

    Also patches Trainer.create_optimizer to ensure VAE parameters are added to optimizer.
    """
    # Patch Trainer.create_optimizer to ensure VAE parameters are in optimizer
    try:
        original_create_optimizer = Trainer.create_optimizer

        @functools.wraps(original_create_optimizer)
        def patched_create_optimizer(self):
            """Ensure VAE parameters are trainable and added to optimizer."""
            # First call original create_optimizer
            optimizer = original_create_optimizer(self)

            # Then ensure VAE params are in optimizer
            loss_spec = _get_loss_spec()
            if "vae" not in loss_spec:
                return optimizer

            # Find VAE parameters in model
            vae_params = []
            for name, param in self.model.named_parameters():
                if 'latent_vae' in name and param.requires_grad:
                    vae_params.append((name, param))

            if not vae_params:
                return optimizer

            # Check if VAE params are already in optimizer
            existing_param_ids = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
            missing_params = [(n, p) for n, p in vae_params if id(p) not in existing_param_ids]

            if missing_params:
                # Add VAE params to optimizer
                optimizer.add_param_group({"params": [p for n, p in missing_params]})
                logger.info(f"[Qwen3VL Latent] Added {len(missing_params)} VAE params to optimizer")

            return optimizer

        Trainer.create_optimizer = patched_create_optimizer
        logger.debug("[Qwen3VL Latent] Patched Trainer.create_optimizer for VAE")
    except Exception as e:
        logger.warning(f"[Qwen3VL Latent] Failed to patch Trainer.create_optimizer: {e}")

    # Store original __init__ method
    original_init = CustomSeq2SeqTrainer.__init__

    @functools.wraps(original_init)
    def patched_init(self, model=None, args=None, callbacks=None, **kwargs):
        # Get training_args early (before calling original_init)
        training_args = args

        # Pre-create VAE if VAE loss is enabled (must be done BEFORE create_optimizer)
        loss_spec = _get_loss_spec()
        if "vae" in loss_spec and model is not None:
            # Create VAE now so it's available for create_optimizer
            _ensure_vae_created(model)

        # Call original __init__ (this calls create_optimizer)
        result = original_init(self, model=model, args=args, callbacks=callbacks, **kwargs)

        # After optimizer is created, ensure VAE params are in optimizer
        if "vae" in loss_spec:
            _ensure_vae_in_optimizer(self, logger)

            # Add VAE save callback
            if training_args is not None:
                save_steps = getattr(training_args, 'save_steps', 500)
                output_dir = getattr(training_args, 'output_dir', '.')
                vae_callback = VAESaveCallback(output_dir=output_dir, save_steps=save_steps)
                self.add_callback(vae_callback)
                if _is_rank0():
                    logger.info(f"[Qwen3VL Latent] Auto-registered VAESaveCallback (save_steps={save_steps})")

        # Log loss spec + loss breakdown at Trainer logging_steps cadence (rank 0 only).
        # This replaces forward() logging to avoid slowing training.
        if not hasattr(self, "_qwen3vl_loss_log_callbacks_added"):
            # Pass the actual trainer model so we can read stashed forward() info
            # regardless of how HF passes kwargs to callbacks.
            self.add_callback(QwenLossLoggingCallback(self.model))
            self._qwen3vl_loss_log_callbacks_added = True

        # Add TransparentEvalCallback only when evaluation is enabled.
        #
        # This training pipeline runs "transparent eval" as a post-training backfill
        # step (see Qwen/scripts/train_qwen3vl_r1onevision.sh). When do_eval=false,
        # registering this callback is unnecessary and can confuse debugging.
        if model is not None and training_args is not None and getattr(training_args, "do_eval", False):
            tokenizer = kwargs.get('tokenizer')
            processor = kwargs.get('processor')
            transparent_callback = QwenTransparentEvalCallback(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                synced_gpus=True,
            )
            self.add_callback(transparent_callback)
            if _is_rank0():
                logger.info("[Qwen3VL Latent] Auto-registered QwenTransparentEvalCallback")
        elif model is not None and training_args is not None and not getattr(training_args, "do_eval", False):
            if _is_rank0():
                logger.info("[Qwen3VL Latent] Skipping QwenTransparentEvalCallback (do_eval=false)")

        # 3) Add CurriculumCallback if enabled
        if os.environ.get("QWEN3VL_CURRICULUM_ENABLE", "0") == "1":
            curriculum_callback = QwenCurriculumCallback()
            self.add_callback(curriculum_callback)
            if _is_rank0():
                logger.info("[Qwen3VL Latent] Auto-registered QwenCurriculumCallback")

        return result

    # Apply the patch
    CustomSeq2SeqTrainer.__init__ = patched_init
    if _is_rank0():
        logger.info("[Qwen3VL Latent] Patched CustomSeq2SeqTrainer to auto-add callbacks")


def _generate_with_dynamic_thinking_mode(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    max_new_tokens: int = 512,
    **kwargs
) -> torch.Tensor:
    """Generate with dynamic thinking mode activation.

    KEY BEHAVIOR:
    - Starts with standard token generation
    - When model GENERATES <think> token, switches to continuous AR mode
    - Between <think> and </think>: Autoregressive hidden state generation (NO tokenization)
    - Naturally detects </think> by projecting hidden states to logits
    - After </think>: Standard token-based generation for answer

    This allows transparent thinking mode activation during inference without requiring
    <think> token in the input prompt.

    Process:
    1. Generate tokens normally until <think> appears in output
    2. When <think> detected: Switch to thinking mode (AR in hidden state space)
    3. Each step: Check if model wants to output </think> (via lm_head projection)
    4. When </think> detected: Exit thinking mode, continue with standard token generation
    5. max_new_tokens controls total budget for prefix + thinking + answer

    Args:
        model: Qwen3VL model
        tokenizer: Tokenizer
        input_ids: Input token IDs [batch, seq_len]
        pixel_values: Vision inputs (if any)
        image_grid_thw: Vision grid info (if any)
        max_new_tokens: Maximum total tokens (thinking + answer, default: 512)

    Returns:
        Generated token IDs [batch, total_seq_len]
    """
    device = next(model.parameters()).device
    batch_size = input_ids.shape[0]

    # Get special token IDs
    think_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))  # <think>
    think_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))      # </think>

    # Handle batch processing (for now, process first sample only)
    if batch_size > 1:
        logging.debug(f"[Thinking Mode] Batch size {batch_size} > 1, processing first sample only")
        input_ids = input_ids[0:1]
        if pixel_values is not None:
            if pixel_values.dim() == 5:
                pixel_values = pixel_values[0:1]
            elif pixel_values.dim() == 4:
                pixel_values = pixel_values[0:1]
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw[0:1]

    # Get embedding function
    embed_fn = model.get_input_embeddings()
    lm_model = model.model if hasattr(model, 'model') else model
    lm_head = model.lm_head if hasattr(model, 'lm_head') else None

    # Prepare initial inputs_embeds with vision - use official Qwen3VL forward method
    if pixel_values is not None and hasattr(model, 'visual'):
        with torch.no_grad():
            # Use official forward pass - let model handle vision processing internally
            # This properly merges vision and text embeddings
            outputs = model.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                output_hidden_states=True,
                return_dict=True
            )
            # Extract last hidden states which include merged vision+text
            inputs_embeds = outputs.hidden_states[-1]
    else:
        inputs_embeds = embed_fn(input_ids)

    # =========================================================================
    # Unified state machine: DISCRETE ↔ CONTINUOUS AR
    #
    # States:
    #   DISCRETE   - Standard token-by-token generation (embed → forward → lm_head → token)
    #   CONTINUOUS - Hidden state autoregression (hidden → forward → hidden), no tokenization
    #
    # Transitions:
    #   DISCRETE   → detect <think>  → CONTINUOUS
    #   CONTINUOUS → detect </think> → DISCRETE
    #
    # Supports multiple thinking sequences within a single generation.
    # KV cache is preserved across all state transitions.
    # =========================================================================
    generated_tokens = []
    kv_cache = None
    tokens_used = 0
    latent_step_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    max_thinking_steps = int(os.environ.get("QWEN3VL_MAX_THINKING_STEPS", str(max_new_tokens // 2)))

    # State: 'discrete' or 'continuous'
    state = 'discrete'
    current_hidden = None      # Used in continuous AR mode
    next_input_embeds = inputs_embeds  # First step processes full input embeddings
    thinking_steps = 0         # Counter for current thinking block

    logging.debug("[Thinking Mode] Starting unified state machine generation")

    with torch.no_grad():
        while tokens_used < max_new_tokens:
            # --- Forward pass ---
            # Get position_ids for RoPE - use zeros placeholder since we're using inputs_embeds
            # This tells Qwen3VL to compute positions internally
            position_ids = torch.zeros((1, next_input_embeds.shape[1]), dtype=torch.long, device=device) if state == 'discrete' else None

            if state == 'discrete':
                outputs = lm_model(
                    inputs_embeds=next_input_embeds,
                    attention_mask=torch.ones_like(next_input_embeds[:, :, 0], dtype=torch.long, device=device),
                    position_ids=position_ids,
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )
            else:  # continuous
                # In continuous mode, use position_ids=None to let model handle it
                outputs = lm_model(
                    inputs_embeds=current_hidden,
                    attention_mask=torch.ones_like(current_hidden[:, :, 0], dtype=torch.long, device=device),
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True
                )

            kv_cache = outputs.past_key_values

            # Extract last hidden state
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1][:, -1:, :]
            else:
                last_hidden = outputs.last_hidden_state[:, -1:, :]

            # Project to logits to determine next token
            if lm_head is None:
                break
            logits = lm_head(last_hidden)
            next_token = logits.argmax(dim=-1).item()
            tokens_used += 1

            # --- State transitions ---
            if state == 'discrete':
                if next_token == think_start_id:
                    # DISCRETE → CONTINUOUS: enter thinking mode
                    logging.info(f"[Thinking Mode] <think> at token {tokens_used}, entering continuous AR")
                    generated_tokens.append(next_token)
                    thinking_steps = 0
                    state = 'continuous'
                    # Feed <think> embedding to get first thinking hidden state
                    think_emb = embed_fn(torch.tensor([[think_start_id]], dtype=torch.long, device=device))
                    outputs = lm_model(
                        inputs_embeds=think_emb,
                        attention_mask=torch.ones_like(think_emb[:, :, 0], dtype=torch.long, device=device),
                        past_key_values=kv_cache,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    kv_cache = outputs.past_key_values
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        current_hidden = outputs.hidden_states[-1][:, -1:, :]
                    else:
                        current_hidden = outputs.last_hidden_state[:, -1:, :]
                    continue

                # Regular discrete token
                generated_tokens.append(next_token)
                if next_token == tokenizer.eos_token_id:
                    logging.debug(f"[Thinking Mode] EOS at token {tokens_used}")
                    break

                # Prepare next discrete input
                next_input_embeds = embed_fn(torch.tensor([[next_token]], dtype=torch.long, device=device))

            else:  # state == 'continuous'
                if next_token == think_end_id or thinking_steps >= max_thinking_steps:
                    # CONTINUOUS → DISCRETE: exit thinking mode
                    forced = thinking_steps >= max_thinking_steps and next_token != think_end_id
                    exit_reason = f"forced at {max_thinking_steps} cap" if forced else f"</think> detected"
                    logging.info(f"[Thinking Mode] Exiting continuous AR after {thinking_steps} steps ({exit_reason})")
                    generated_tokens.append(next_token)
                    state = 'discrete'
                    # Feed </think> embedding to update KV cache for discrete mode
                    end_think_emb = embed_fn(torch.tensor([[think_end_id]], dtype=torch.long, device=device))
                    outputs = lm_model(
                        inputs_embeds=end_think_emb,
                        attention_mask=torch.ones_like(end_think_emb[:, :, 0], dtype=torch.long, device=device),
                        past_key_values=kv_cache,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True
                    )
                    kv_cache = outputs.past_key_values
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                        last_hidden = outputs.hidden_states[-1][:, -1:, :]
                    else:
                        last_hidden = outputs.last_hidden_state[:, -1:, :]
                    # Get first answer token from </think> hidden state
                    logits = lm_head(last_hidden)
                    first_answer_token = logits.argmax(dim=-1).item()
                    tokens_used += 1
                    generated_tokens.append(first_answer_token)
                    if first_answer_token == tokenizer.eos_token_id:
                        break
                    next_input_embeds = embed_fn(torch.tensor([[first_answer_token]], dtype=torch.long, device=device))
                    current_hidden = None
                    continue

                # Continuous AR: feed hidden state forward, log placeholder
                thinking_steps += 1
                generated_tokens.append(latent_step_id)
                current_hidden = last_hidden

    # Return complete sequence: input + generated tokens
    full_output = torch.cat([
        input_ids,
        torch.tensor([generated_tokens], dtype=torch.long, device=device)
    ], dim=1)

    logging.debug(f"[Thinking Mode] Generated {len(generated_tokens)} tokens, {tokens_used} steps used")

    return full_output


def _is_vllm_environment() -> bool:
    """Detect if running in vLLM environment.

    vLLM has its own generation engine and doesn't use transformers' generate(),
    so thinking mode patches should be skipped.

    Can be controlled via VLLM_INFERENCE environment variable:
    - VLLM_INFERENCE=1: Force vLLM mode (skip patches)
    - VLLM_INFERENCE=0: Force transformers mode (apply patches)
    - Not set: Auto-detect based on imported modules
    """
    # Check environment variable override
    vllm_env = os.environ.get('VLLM_INFERENCE', '').strip()
    if vllm_env == '1':
        return True
    if vllm_env == '0':
        return False

    # Auto-detect: Check if vLLM modules are imported
    vllm_modules = [name for name in sys.modules if 'vllm' in name.lower()]
    return len(vllm_modules) > 0


def _patch_model_generate(logger) -> None:
    """Patch model.generate() to support dynamic thinking mode and filter latent kwargs.

    DYNAMIC THINKING MODE GENERATION:
    - Phase 1 (Discrete AR): Standard token generation until <think> token appears
    - Phase 2 (Continuous AR): When <think> detected, switch to continuous latent AR
      * Last hidden state directly becomes next input (no embed_fn, no tokenization)
      * Pure autoregression in hidden state space: hidden_t → lm_model → hidden_{t+1}
      * LM head only used to monitor for </think> exit signal
      * True continuous thinking without intermediate tokenization
    - Phase 3 (Discrete AR): When </think> detected, switch back to standard token generation
    
    This enables transparent thinking mode without pre-configured prompts in the input.

    VLLM COMPATIBILITY:
    - When running in vLLM environment, thinking mode patches are skipped
    - Model generates thinking tokens as regular tokens (functional but not optimal)
    - For optimal performance, use transformers-based inference

    Also filters out latent supervision kwargs and fixes max_length issues.
    """
    # Note: vLLM environment detected message for info
    if _is_vllm_environment():
        if _is_rank0() and os.environ.get("QWEN3VL_LOG_VLLM_HINT", "0") == "1":
            logger.info("[Qwen3VL Latent] vLLM environment detected")
            logger.info("[Qwen3VL Latent] For vLLM inference, use ThinkingLLM wrapper: from Qwen.vllm_thinking_mode import ThinkingLLM")

    # Store original generate method
    original_generate = Qwen3VLForConditionalGeneration.generate

    @functools.wraps(original_generate)
    def patched_generate(self, input_ids=None, inputs_embeds=None, **kwargs):
        """Enhanced generate with dynamic thinking mode support (ALWAYS ENABLED)."""
        # Remove latent supervision kwargs (only valid for forward())
        latent_kwargs = {
            'latent_positions',
            'latent_ground_truth',
            'latent_ground_truth_paths',
            'latent_ground_truth_packed',
            'latent_supervision',
            'latent_supervision_paths',
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in latent_kwargs}

        # Fix generation config to use max_new_tokens instead of max_length
        generation_config = filtered_kwargs.get('generation_config')
        max_new_tokens = filtered_kwargs.get('max_new_tokens')
        if max_new_tokens is None:
            max_new_tokens_str = os.environ.get("QWEN3VL_MAX_NEW_TOKENS", "8192")
            try:
                max_new_tokens = int(max_new_tokens_str)
            except ValueError:
                max_new_tokens = 8192

        if max_new_tokens and 'max_length' in filtered_kwargs:
            filtered_kwargs.pop('max_length', None)

        if generation_config is not None:
            if max_new_tokens:
                generation_config.max_new_tokens = max_new_tokens
                generation_config.max_length = None

        # THINKING MODE: Dynamic activation (monitors generated tokens)
        # Model will generate normally until <think> appears, then switch to continuous AR
        if input_ids is not None:
            # Get tokenizer
            tokenizer = getattr(self, 'tokenizer', None)
            if tokenizer is None:
                try:
                    model_path = getattr(self.config, '_name_or_path', None)
                    if model_path:
                        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                except Exception as e:
                    logger.warning(f"[Qwen3VL Latent] Failed to load tokenizer for dynamic thinking mode: {e}")

            if tokenizer is not None:
                # Use dynamic thinking mode (activates when model generates <think>)
                return _generate_with_dynamic_thinking_mode(
                    model=self,
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    pixel_values=filtered_kwargs.get('pixel_values'),
                    image_grid_thw=filtered_kwargs.get('image_grid_thw'),
                    max_new_tokens=max_new_tokens,
                    **{k: v for k, v in filtered_kwargs.items() if k not in {'pixel_values', 'image_grid_thw', 'max_new_tokens'}}
                )

        # Standard generation (fallback for inputs_embeds or no tokenizer)
        return original_generate(self, input_ids=input_ids, inputs_embeds=inputs_embeds, **filtered_kwargs)

    # Apply patch
    Qwen3VLForConditionalGeneration.generate = patched_generate
    logger.debug("[Qwen3VL Latent] ✓ Patched Qwen3VLForConditionalGeneration.generate with thinking mode support")


# ============================================================================
# Helper Functions
# ============================================================================

def _is_qwen3vl_model(model) -> bool:
    """Check if model is Qwen3VL."""
    try:
        config = model.config
        return hasattr(config, 'model_type') and config.model_type in ['qwen3_vl']
    except Exception:
        return False


def _inject_latent_features(
    inputs_embeds: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> torch.Tensor:
    """Inject latent_ground_truth features at marked positions.

    Replaces <latent> tokens with actual latent token sequences, similar to
    how <image> tokens are replaced with image patch tokens in VLMs.

    Args:
        inputs_embeds: Input embeddings [batch, seq_len, hidden_dim]
        latent_supervision: List of latent_ground_truth tensors for each sample
        latent_positions: Boolean mask indicating where <latent> tokens are

    Latent features are already at LLM hidden dimension, so no projection needed.

    Injection strategy:
    - Each <latent> token is replaced by its corresponding latent sequence
    - If latent is [seq_len, hidden_dim], it replaces 1 token with seq_len tokens
    - Input sequence is expanded accordingly
    """
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    hidden_dim = inputs_embeds.shape[-1]

    # Process each sample in the batch
    new_embeds_list = []
    new_masks_list = []  # Track which samples were modified

    for b in range(batch_size):
        sample_embeds = inputs_embeds[b]  # [seq_len, hidden_dim]
        latent_mask = latent_positions[b]

        if not latent_mask.any():
            # No latent tokens to inject, keep original
            new_embeds_list.append(sample_embeds)
            new_masks_list.append(False)
            continue

        supervision_raw = latent_supervision[b] if b < len(latent_supervision) else []
        # Flatten nested list structure: [[tensor]] -> [tensor]
        supervision = []
        for item in supervision_raw:
            if isinstance(item, torch.Tensor):
                supervision.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        supervision.append(sub)
        if len(supervision) == 0:
            new_embeds_list.append(sample_embeds)
            new_masks_list.append(False)
            continue

        # Get indices of <latent> tokens
        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        # Build new sequence by replacing <latent> with latent sequences
        new_tokens = []
        seq_idx = 0

        for latent_idx in latent_indices:
            # Add all tokens before this <latent>
            new_tokens.append(sample_embeds[seq_idx:latent_idx])

            # Get the corresponding latent features
            sup_idx = len([idx for idx in latent_indices if idx < latent_idx])
            if sup_idx < len(supervision):
                feat = supervision[sup_idx]

                # Move to correct device/dtype
                if isinstance(feat, torch.Tensor):
                    feat = feat.to(device=device, dtype=dtype)

                # Handle dimensions
                if feat.dim() == 1:
                    # Single token: [hidden_dim]
                    if feat.shape[0] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[0]} != LLM hidden dim {hidden_dim}"
                        )
                    new_tokens.append(feat.unsqueeze(0))
                elif feat.dim() == 2:
                    # Sequence: [seq_len, hidden_dim] - replace with full sequence
                    if feat.shape[-1] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[-1]} != LLM hidden dim {hidden_dim}"
                        )
                    new_tokens.append(feat)  # [seq_len, hidden_dim]
                else:
                    raise ValueError(f"Unexpected latent dim: {feat.dim()}")

            # Move seq_idx past this <latent> token
            seq_idx = latent_idx + 1

        # Add remaining tokens after the last <latent>
        new_tokens.append(sample_embeds[seq_idx:])

        # Concatenate all tokens
        new_embeds = torch.cat(new_tokens, dim=0)  # [new_seq_len, hidden_dim]
        new_embeds_list.append(new_embeds)
        new_masks_list.append(True)

    # Check if any samples were modified
    if not any(new_masks_list):
        return inputs_embeds

    # Pad sequences to the same length (max new sequence length)
    max_len = max(emb.shape[0] for emb in new_embeds_list)

    padded_embeds = []
    for emb in new_embeds_list:
        if emb.shape[0] < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - emb.shape[0], hidden_dim, device=device, dtype=dtype)
            emb = torch.cat([emb, padding], dim=0)
        padded_embeds.append(emb)

    # Stack into batch
    result = torch.stack(padded_embeds, dim=0)  # [batch, max_seq_len, hidden_dim]
    return result


def _inject_latent_features_inplace(
    inputs_embeds: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
    latent_paths: Optional[List[List[str]]] = None,
) -> torch.Tensor:
    """Inject latent features in-place when sequences are pre-expanded.

    NOTE: With gradient checkpointing enabled, we must clone inputs_embeds first
    to avoid in-place modification errors in the autograd graph.
    """
    batch_size = inputs_embeds.shape[0]
    device = inputs_embeds.device
    dtype = inputs_embeds.dtype
    hidden_dim = inputs_embeds.shape[-1]

    # Clone to avoid in-place modification issues with gradient checkpointing
    inputs_embeds = inputs_embeds.clone()

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        # Data is already flattened in _add_latent_supervision_to_batch
        supervision = latent_supervision[b] if b < len(latent_supervision) else []

        if len(supervision) == 0:
            continue

        flat_feats = []
        for feat in supervision:
            if isinstance(feat, torch.Tensor):
                feat = feat.to(device=device, dtype=dtype)
                if feat.dim() == 1:
                    if feat.shape[0] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[0]} != LLM hidden dim {hidden_dim}"
                        )
                    flat_feats.append(feat.unsqueeze(0))
                elif feat.dim() == 2:
                    if feat.shape[-1] != hidden_dim:
                        raise ValueError(
                            f"Latent feature dim {feat.shape[-1]} != LLM hidden dim {hidden_dim}"
                        )
                    flat_feats.append(feat)
                else:
                    raise ValueError(f"Unexpected latent dim: {feat.dim()}")

        if not flat_feats:
            continue

        flat_tokens = torch.cat(flat_feats, dim=0)  # [total_latent_len, hidden_dim]
        latent_indices = latent_mask.nonzero(as_tuple=False).squeeze(dim=-1)

        # STRICT invariant: expanded latent positions must equal materialized latent tokens.
        # If mismatch, skip this sample's latent injection (logs warning once).
        if flat_tokens.shape[0] != latent_indices.shape[0]:
            if not hasattr(_inject_latent_features_inplace, '_logged_mismatch'):
                logger.warning(
                    f"[Qwen3VL Latent] STRICT mismatch at injection: sample={b}, "
                    f"latent_positions={int(latent_indices.shape[0])}, latent_tokens={int(flat_tokens.shape[0])}. "
                    f"Skipping latent injection for this sample."
                )
                _inject_latent_features_inplace._logged_mismatch = True
            # Skip this sample - keep original embeddings
            continue

        inputs_embeds[b, latent_indices] = flat_tokens

    return inputs_embeds


def _match_sequence_length(
    pred: torch.Tensor,
    target: torch.Tensor,
    strategy: str = "truncate",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match sequence lengths between prediction and target tensors.

    Args:
        pred: Prediction tensor [T_pred, D]
        target: Target tensor [T_target, D]
        strategy: Matching strategy
            - 'truncate': Truncate both to min length (default, fastest)
            - 'repeat': Repeat shorter to match longer
            - 'interpolate': Linearly interpolate shorter to match longer

    Returns:
        Matched (pred, target) tensors with same sequence length
    """
    t_pred, t_target = pred.shape[0], target.shape[0]
    dim = pred.shape[-1]

    if t_pred == t_target:
        return pred, target

    if strategy == "truncate":
        min_len = min(t_pred, t_target)
        return pred[:min_len], target[:min_len]

    elif strategy == "repeat":
        max_len = max(t_pred, t_target)

        if t_pred < max_len:
            # Repeat prediction
            repeat_factor = (max_len + t_pred - 1) // t_pred
            pred = pred.repeat(repeat_factor, 1)[:max_len]

        if t_target < max_len:
            # Repeat target
            repeat_factor = (max_len + t_target - 1) // t_target
            target = target.repeat(repeat_factor, 1)[:max_len]

        return pred, target

    elif strategy == "interpolate":
        max_len = max(t_pred, t_target)

        if t_pred < max_len:
            # Interpolate prediction using nearest neighbor + repeat
            indices = torch.linspace(0, t_pred - 1, max_len, device=pred.device).long()
            pred = pred[indices]

        if t_target < max_len:
            # Interpolate target
            indices = torch.linspace(0, t_target - 1, max_len, device=target.device).long()
            target = target[indices]

        return pred, target

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _compute_pre_thinking_mse_loss(
    hidden_states: torch.Tensor,
    latent_ground_truth: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> Optional[torch.Tensor]:
    """Compute MSE loss on hidden states shifted back by 1 vs ground truth targets.

    Autoregressive shift: hidden state at position i predicts position i+1.
    So to predict latent targets at positions [p0, p1, ..., p_{N-1}], we use
    hidden states at [p0-1, p1-1, ..., p_{N-1}-1], i.e. from <think> through
    the second-to-last latent position.

    Args:
        hidden_states: Hidden states [batch, seq_len, hidden_dim]
        latent_ground_truth: Ground truth tensors per sample (for injection), each a list of [num_tokens, hidden_dim]
        latent_positions: Boolean mask [batch, seq_len] marking latent positions

    Returns:
        MSE loss scalar or None if no valid positions
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    pre_thinking_losses = []

    for b in range(batch_size):
        # Get latent positions from boolean mask and shift back by 1
        # This gives positions from <think> to second-to-last latent position
        positions = latent_positions[b].nonzero(as_tuple=False).squeeze(-1)
        if positions.numel() == 0:
            continue

        shifted_positions = positions - 1
        # Clamp to valid range (shouldn't trigger — <think> is never at position 0)
        shifted_positions = shifted_positions.clamp(min=0)

        # Extract hidden states at shifted positions: [num_latent_positions, hidden_dim]
        pred_hidden = hidden_states[b, shifted_positions, :]

        # Data is already flattened in _add_latent_supervision_to_batch
        sample_latent = latent_ground_truth[b] if b < len(latent_ground_truth) else []

        if len(sample_latent) == 0:
            continue

        # Verify all elements are tensors
        if not all(isinstance(t, torch.Tensor) for t in sample_latent):
            types = [type(t) for t in sample_latent]
            print(f"[PRE_THINK_MSE_DEBUG] batch={b}, sample_latent types: {types}", flush=True, file=sys.stderr)
            continue

        # Concatenate supervision targets: [total_tokens, hidden_dim]
        target_latent = torch.cat(sample_latent, dim=0)
        target_latent = target_latent.to(device=pred_hidden.device, dtype=pred_hidden.dtype)

        # Lengths should match by construction (expansion guarantees this)
        # Defensive pad/truncate for edge cases
        pred_len = pred_hidden.shape[0]
        target_len = target_latent.shape[0]

        if target_len == 0:
            continue

        if pred_len > target_len:
            pred_hidden = pred_hidden[:target_len, :]
        elif pred_len < target_len:
            pad = torch.zeros(target_len - pred_len, hidden_dim, device=pred_hidden.device, dtype=pred_hidden.dtype)
            pred_hidden = torch.cat([pred_hidden, pad], dim=0)

        mse_loss = torch.nn.functional.mse_loss(
            pred_hidden,     # [target_len, hidden_dim]
            target_latent,   # [target_len, hidden_dim]
            reduction='mean'
        )

        pre_thinking_losses.append(mse_loss)

    if len(pre_thinking_losses) > 0:
        return torch.stack(pre_thinking_losses).mean()
    else:
        return None


def _compute_thinking_loss(
    hidden_states: torch.Tensor,
    latent_ground_truth: List[List[torch.Tensor]],
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> Optional[torch.Tensor]:
    """Compute thinking loss on LLM hidden states at latent positions.

    Supports flexible loss combination via QWEN3VL_LOSS_TYPE env var:

    Syntax:
    - Single loss: "ot", "mse", "repa", "nce"
    - Equal weights: "ot+mse", "repa+nce+ot"
    - Custom weights: "ot:0.7+mse:0.3", "repa:0.5+nce:0.3+ot:0.2"

    Available loss types:
    - 'repa': Negative cosine similarity (for ground truth supervision)
    - 'nce': InfoNCE contrastive loss (aligned/negative pairs)
    - 'ot': EMO optimal transport (uses latent_supervision)
    - 'mse': Mean squared error (uses latent_supervision)
    - 'pre_think_mse': MSE between latent_ground_truth and prev token hidden states

    Examples:
        export QWEN3VL_LOSS_TYPE="ot"                    # Single loss
        export QWEN3VL_LOSS_TYPE="ot+mse"                # Equal weights (0.5, 0.5)
        export QWEN3VL_LOSS_TYPE="ot:0.7+mse:0.3"        # Custom weights
    """
    loss_spec = _get_loss_spec()

    # Parse loss specification
    loss_configs = _parse_loss_spec(loss_spec)

    # Debug logging (first call only)
    if not hasattr(_compute_thinking_loss, '_logged'):
        if _is_rank0():
            logger.debug(f"[Qwen3VL Latent] Computing thinking loss with spec: {loss_spec}, configs: {loss_configs}")
            logger.debug(f"[Qwen3VL Latent] hidden_states shape: {hidden_states.shape}")
            logger.debug(f"[Qwen3VL Latent] latent_ground_truth length: {len(latent_ground_truth) if latent_ground_truth else 0}")
            logger.debug(f"[Qwen3VL Latent] latent_supervision length: {len(latent_supervision) if latent_supervision else 0}")
            logger.debug(f"[Qwen3VL Latent] latent_positions shape: {latent_positions.shape}, any: {latent_positions.any().item()}")
        _compute_thinking_loss._logged = True

    # Debug flag for detailed logging
    debug_enabled = os.environ.get("QWEN3VL_DEBUG_FORWARD", "0") == "1"

    # Debug: Log latent data details
    if debug_enabled:
        rank_0 = _is_rank0()
        if rank_0:
            for i, (gt, sup) in enumerate(zip(latent_ground_truth, latent_supervision)):
                gt_len = len(gt) if gt else 0
                sup_len = len(sup) if sup else 0
                gt_inner = [t.shape if t is not None else None for t in gt] if gt else []
                sup_inner = [t.shape if t is not None else None for t in sup] if sup else []
                print(f"[LOSS_DEBUG] sample {i}: gt_len={gt_len}, sup_len={sup_len}, gt_shapes={gt_inner}, sup_shapes={sup_inner}", flush=True, file=sys.stderr)

    # Compute each loss and combine
    total_loss = 0.0
    total_weight = 0.0
    loss_values = {}
    ot_stats = None  # Store OT statistics for logging

    for loss_name, weight in loss_configs:
        if loss_name == "none":
            # Curriculum warmup: skip latent supervision loss
            continue
        elif loss_name == "repa":
            loss = _compute_repa_loss(hidden_states, latent_supervision, latent_positions)
        elif loss_name == "nce":
            loss = _compute_contrastive_loss(hidden_states, latent_supervision, latent_positions)
        elif loss_name == "ot":
            loss, ot_stats = _compute_ot_loss(hidden_states, latent_supervision, latent_positions)
        elif loss_name == "mse":
            loss = _compute_mse_loss(hidden_states, latent_supervision, latent_positions)
        elif loss_name == "pre_think_mse":
            # pre_think_mse uses latent_ground_truth (for injection) vs prev token hidden states
            loss = _compute_pre_thinking_mse_loss(hidden_states, latent_ground_truth, latent_positions)
        elif loss_name == "vae":
            # VAE loss requires access to the model - handled separately in forward
            # This is a placeholder - actual VAE loss computed in patched_forward
            loss = None  # Will be computed separately
        else:
            raise ValueError(f"Unknown loss type: {loss_name}. Must be 'repa', 'nce', 'ot', 'mse', 'pre_think_mse', 'vae', or 'none'")

        # Debug logging (first call only)
        if not hasattr(_compute_thinking_loss, '_logged_loss'):
            if _is_rank0():
                logger.debug(f"[Qwen3VL Latent] {loss_name} loss result: {loss}, weight: {weight}")
            _compute_thinking_loss._logged_loss = True

        if loss is not None:
            loss_values[loss_name] = loss.item() if isinstance(loss, torch.Tensor) else loss
            total_loss = total_loss + weight * loss
            total_weight = total_weight + weight

    if total_weight > 0:
        # Normalize by total weight
        total_loss = total_loss / total_weight
        return total_loss, ot_stats

    return None, None


def _parse_loss_spec(loss_spec: str) -> List[tuple]:
    """Parse loss specification string into list of (loss_name, weight) tuples.

    Syntax examples:
    - "ot" → [("ot", 1.0)]
    - "ot+mse" → [("ot", 0.5), ("mse", 0.5)]
    - "ot:0.7+mse:0.3" → [("ot", 0.7), ("mse", 0.3)]
    - "repa:0.5+nce:0.3+ot:0.2" → [("repa", 0.5), ("nce", 0.3), ("ot", 0.2)]

    Args:
        loss_spec: Loss specification string

    Returns:
        List of (loss_name, weight) tuples

    Raises:
        ValueError: If loss specification is invalid
    """
    if not loss_spec:
        raise ValueError("Loss specification cannot be empty")

    # Split by '+' to get individual loss specs
    loss_parts = loss_spec.split('+')

    loss_configs = []
    for part in loss_parts:
        part = part.strip()
        if not part:
            continue

        # Check if weight is specified
        if ':' in part:
            # Split by ':' to get loss name and weight
            loss_name, weight_str = part.split(':', 1)
            loss_name = loss_name.strip()
            weight_str = weight_str.strip()

            try:
                weight = float(weight_str)
            except ValueError:
                raise ValueError(f"Invalid weight '{weight_str}' for loss '{loss_name}'. Must be a number.")

            if weight < 0:
                raise ValueError(f"Weight for loss '{loss_name}' must be non-negative, got {weight}")
        else:
            # No weight specified, will use equal weighting
            loss_name = part.strip()
            weight = 1.0

        loss_configs.append((loss_name, weight))

    if not loss_configs:
        raise ValueError(f"No valid loss specifications found in '{loss_spec}'")

    # If all weights are 1.0, normalize to sum to 1
    all_weights = [w for _, w in loss_configs]
    if all(w == 1.0 for w in all_weights):
        num_losses = len(loss_configs)
        loss_configs = [(name, 1.0 / num_losses) for name, _ in loss_configs]

    return loss_configs


def _compute_repa_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> Optional[torch.Tensor]:
    """Compute REPA loss (direct negative cosine similarity) on LLM hidden states.

    Use this when you have ground truth supervision targets.

    Sequence matching: Uses QWEN3VL_MATCH_STRATEGY env var (default: truncate)
    - 'truncate': Truncate both to min length
    - 'repeat': Repeat shorter to match longer
    - 'interpolate': Interpolate shorter to match longer
    """
    batch_size = hidden_states.shape[0]
    thinking_losses = []

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        # Extract hidden states at latent positions (already at LLM hidden dim)
        sample_hidden = hidden_states[b][latent_mask]  # [num_latents, hidden_dim]

        # Get supervision targets (also at LLM hidden dim)
        supervision_raw = latent_supervision[b] if b < len(latent_supervision) else []
        # Flatten nested list structure: [[tensor]] -> [tensor]
        supervision = []
        for item in supervision_raw:
            if isinstance(item, torch.Tensor):
                supervision.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        supervision.append(sub)
        if len(supervision) == 0:
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                # Mean-pool if spatial: [T, hidden_dim] → [hidden_dim]
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor.mean(dim=0))
                else:
                    supervision_latents.append(sup_tensor)

        if len(supervision_latents) == 0:
            continue

        supervision_tensor = torch.stack(supervision_latents, dim=0)
        supervision_tensor = supervision_tensor.to(sample_hidden.device, sample_hidden.dtype)

        # Verify dimensions match
        if sample_hidden.shape[-1] != supervision_tensor.shape[-1]:
            raise ValueError(
                f"Hidden dim {sample_hidden.shape[-1]} != supervision dim {supervision_tensor.shape[-1]}. "
                f"Both must be at LLM hidden dimension for direct supervision."
            )

        # Match sequence lengths
        strategy = os.environ.get("QWEN3VL_MATCH_STRATEGY", "truncate").lower()
        sample_hidden, supervision_tensor = _match_sequence_length(
            sample_hidden, supervision_tensor, strategy=strategy
        )

        # REPA loss: negative cosine similarity (direct on hidden states)
        pred_norm = F.normalize(sample_hidden, dim=-1)
        superv_norm = F.normalize(supervision_tensor, dim=-1)

        sample_loss = -torch.mean((pred_norm * superv_norm).sum(dim=-1))
        thinking_losses.append(sample_loss)

    if thinking_losses:
        return torch.stack(thinking_losses).mean()
    return None


def _compute_contrastive_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
    temperature: float = 0.07,
) -> Optional[torch.Tensor]:
    """Compute InfoNCE-style contrastive loss on LLM hidden states.

    For varied sequence lengths, uses set-level pooling to create fixed-size
    representations per sample while preserving information.

    Pooling strategy: Mean + Max (captures both average and salient features)

    Args:
        hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
        latent_supervision: Supervision targets for each sample
        latent_positions: Boolean mask indicating latent positions
        temperature: Temperature parameter for softmax (default: 0.07)

    Returns:
        Contrastive loss or None if no valid pairs
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Collect all valid samples
    preds_list = []
    targets_list = []

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        # Extract hidden states at latent positions
        sample_hidden = hidden_states[b][latent_mask]  # [num_latents, hidden_dim]

        # Get supervision targets
        supervision_raw = latent_supervision[b] if b < len(latent_supervision) else []
        # Flatten nested list structure: [[tensor]] -> [tensor]
        supervision = []
        for item in supervision_raw:
            if isinstance(item, torch.Tensor):
                supervision.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        supervision.append(sub)
        if len(supervision) == 0:
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor.mean(dim=0))
                else:
                    supervision_latents.append(sup_tensor)

        if len(supervision_latents) == 0:
            continue

        supervision_tensor = torch.stack(supervision_latents, dim=0)  # [num_supervision, hidden_dim]

        # Set-level pooling: concatenate mean and max pooling
        # This captures both average features and salient features
        pred_mean = sample_hidden.mean(dim=0)  # [hidden_dim]
        pred_max = sample_hidden.max(dim=0)[0]  # [hidden_dim]

        target_mean = supervision_tensor.mean(dim=0)  # [hidden_dim]
        target_max = supervision_tensor.max(dim=0)[0]  # [hidden_dim]

        # Concatenate mean and max for richer representation
        pred_pooled = torch.cat([pred_mean, pred_max], dim=0)  # [2 * hidden_dim]
        target_pooled = torch.cat([target_mean, target_max], dim=0)  # [2 * hidden_dim]

        preds_list.append(pred_pooled)
        targets_list.append(target_pooled)

    if len(preds_list) < 2:
        return None  # Need at least 2 samples for contrastive loss

    # Stack into tensors
    preds = torch.stack(preds_list, dim=0)  # [N, 2 * hidden_dim]
    targets = torch.stack(targets_list, dim=0)  # [N, 2 * hidden_dim]

    # Ensure same dtype
    targets = targets.to(dtype=preds.dtype)

    # L2 normalize
    preds_norm = F.normalize(preds, dim=-1)
    targets_norm = F.normalize(targets, dim=-1)

    # Compute similarity matrix: N x N
    # sim[i, j] = cos(preds[i], targets[j])
    sim_matrix = torch.mm(preds_norm, targets_norm.t()) / temperature  # [N, N]

    # InfoNCE loss: for each row i, positive is at diagonal (i, i)
    # loss = -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
    labels = torch.arange(len(preds_list), device=device)

    loss = F.cross_entropy(sim_matrix, labels)

    return loss


def _compute_ot_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
    lm_head: nn.Module = None,
) -> Optional[torch.Tensor]:
    """Compute proper OT loss that NATIVELY handles varied sequence lengths.

    KEY INSIGHT: OT works with distributions of DIFFERENT sizes - no matching needed!

    For two distributions:
    - Q (predictions): [N, D]  - any size
    - P (targets): [M, D]     - any size (can be different!)

    Cost matrix: C[i,j] = 1 - cos(Q[i], P[j])  # [N, M] - rectangular!

    DEMD with uniform Q, P:
        DEMD = Q^T @ C @ P
             = (1/N) * Σ_i (1/M) * Σ_j C[i,j]
             = (1/(N*M)) * Σ_i Σ_j (1 - cos(Q[i], P[j]))

    This naturally handles varied lengths without truncation/padding!

    NOTE: No positional bias needed! ViT-encoded latents already contain spatial
    information in their feature representations. Pure semantic OT naturally respects
    the 2D structure through cosine similarity.

    Args:
        hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
        latent_supervision: Supervision targets for each sample
        latent_positions: Boolean mask indicating latent positions
        lm_head: Optional language model head for vocabulary projection

    Returns:
        OT loss or None if no valid pairs
    """
    device = hidden_states.device
    dtype = hidden_states.dtype
    # Optional token sampling for OT approximation (default: 16)
    # Set QWEN3VL_OT_SAMPLE_K=0 or "none" to disable sampling (full OT).
    sample_k_raw = os.environ.get("QWEN3VL_OT_SAMPLE_K", "16")
    sample_k = None
    if isinstance(sample_k_raw, str):
        if sample_k_raw.strip().lower() in ("none", "null", "off", "disable", "disabled"):
            sample_k = None
        else:
            try:
                sample_k = int(sample_k_raw)
            except ValueError:
                sample_k = 16
    elif isinstance(sample_k_raw, int):
        sample_k = sample_k_raw

    # Debug logging (first call only)
    if not hasattr(_compute_ot_loss, '_logged'):
        if _is_rank0():
            logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: hidden_states shape={hidden_states.shape}")
            logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: latent_supervision length={len(latent_supervision)}")
            logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: latent_positions shape={latent_positions.shape}, any={latent_positions.any()}")
        _compute_ot_loss._logged = True

    # Compute OT loss per-sample and average
    ot_losses = []

    # Statistics collection for detailed logging
    stats = {
        "num_valid_samples": 0,
        "total_pred_tokens": 0,
        "total_target_tokens": 0,
        "cost_stats": [],  # (min, max, mean, std) per sample
    }

    for b in range(hidden_states.shape[0]):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            # Debug: log skipped samples
            if not hasattr(_compute_ot_loss, '_logged_skip'):
                if _is_rank0():
                    logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: skipping batch {b}, no latent positions")
                _compute_ot_loss._logged_skip = True
            continue

        # Extract hidden states at latent positions (predictions)
        sample_hidden = hidden_states[b][latent_mask]  # [N, hidden_dim]

        # Get supervision targets
        supervision_raw = latent_supervision[b] if b < len(latent_supervision) else []
        # Flatten nested list structure: [[tensor]] -> [tensor]
        supervision = []
        for item in supervision_raw:
            if isinstance(item, torch.Tensor):
                supervision.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        supervision.append(sub)
        if len(supervision) == 0:
            # Debug: log missing supervision
            if not hasattr(_compute_ot_loss, '_logged_no_supervision'):
                if _is_rank0():
                    logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: batch {b} has no supervision targets (len={len(supervision)})")
                _compute_ot_loss._logged_no_supervision = True
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor)
                else:
                    supervision_latents.append(sup_tensor.unsqueeze(0))

        if len(supervision_latents) == 0:
            if not hasattr(_compute_ot_loss, '_logged_no_tensors'):
                if _is_rank0():
                    logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: batch {b} has no valid supervision tensors")
                _compute_ot_loss._logged_no_tensors = True
            continue

        # Concatenate to allow variable-length supervision tensors
        superv_flat = torch.cat(supervision_latents, dim=0)  # [M, hidden_dim]

        # NO length matching for OT - let N and M be naturally different!
        N = sample_hidden.shape[0]  # Number of predicted tokens
        M = superv_flat.shape[0]    # Number of target tokens (can differ!)

        # Move to device
        pred_tokens = sample_hidden.to(device=device, dtype=dtype)          # [N, D]
        target_tokens = superv_flat.to(device=device, dtype=dtype)          # [M, D]

        # Optional sampling to reduce OT cost
        if sample_k is not None and sample_k > 0:
            if N > sample_k:
                idx = torch.randperm(N, device=device)[:sample_k]
                pred_tokens = pred_tokens[idx]
                N = pred_tokens.shape[0]
            if M > sample_k:
                idx = torch.randperm(M, device=device)[:sample_k]
                target_tokens = target_tokens[idx]
                M = target_tokens.shape[0]

        # NOTE: N and M can be DIFFERENT! Cost matrix will be [N, M] rectangular.

        # Compute semantic cost matrix (RECTANGULAR: [N, M])
        # C[i,j] = 1 - cos(pred[i], target[j])
        if lm_head is not None:
            # True EMO: Project to vocabulary space
            E = lm_head.weight.data  # [vocab_size, hidden_dim]
            E = E / torch.linalg.vector_norm(E, ord=2, dim=1, keepdim=True)
            E = E.to(device=device, dtype=dtype)

            pred_logits = pred_tokens @ E.t()      # [N, vocab_size]
            target_logits = target_tokens @ E.t()  # [M, vocab_size]

            Q_θ = F.softmax(pred_logits, dim=-1)      # [N, vocab_size]
            P = F.softmax(target_logits, dim=-1)      # [M, vocab_size]

            pred_repr = Q_θ @ E      # [N, hidden_dim]
            target_repr = P @ E      # [M, hidden_dim]

            # Semantic cost: [N, M] (rectangular!)
            pred_repr_norm = F.normalize(pred_repr, dim=-1)
            target_repr_norm = F.normalize(target_repr, dim=-1)
            semantic_sim = torch.mm(pred_repr_norm, target_repr_norm.t())  # [N, M]
            cost_matrix = 1.0 - semantic_sim
        else:
            # Direct hidden space (default, more efficient)
            pred_norm = F.normalize(pred_tokens, dim=-1)  # [N, D]
            target_norm = F.normalize(target_tokens, dim=-1)  # [M, D]
            semantic_sim = torch.mm(pred_norm, target_norm.t())  # [N, M]
            cost_matrix = 1.0 - semantic_sim

        # DEMD with uniform distributions over DIFFERENT sizes:
        # Q: uniform over N tokens → [1/N, ..., 1/N]  (size N)
        # P: uniform over M tokens → [1/M, ..., 1/M]  (size M)
        #
        # DEMD = Q^T @ C @ P
        #      = Σ_i (Q[i] * Σ_j (P[j] * C[i,j]))
        #      = (1/N) * Σ_i (1/M) * Σ_j C[i,j]
        #      = (1/(N*M)) * Σ_i Σ_j C[i,j]
        #      = mean(cost_matrix)

        sample_ot_loss = cost_matrix.mean()  # Average over N×M rectangular matrix
        ot_losses.append(sample_ot_loss)

        # Collect statistics
        stats["num_valid_samples"] += 1
        stats["total_pred_tokens"] += N
        stats["total_target_tokens"] += M
        stats["cost_stats"].append((
            cost_matrix.min().item(),
            cost_matrix.max().item(),
            cost_matrix.mean().item(),
            cost_matrix.std().item() if cost_matrix.numel() > 1 else 0.0,
        ))

    # Debug: log final result
    if not hasattr(_compute_ot_loss, '_logged_result'):
        if _is_rank0():
            logger.debug(f"[Qwen3VL Latent] _compute_ot_loss: ot_losses length={len(ot_losses)}, returning={ot_losses is not None and len(ot_losses) > 0}")
        _compute_ot_loss._logged_result = True

    if ot_losses:
        # Compute aggregated statistics
        loss_value = torch.stack(ot_losses).mean()
        if stats["cost_stats"]:
            cost_mins = [s[0] for s in stats["cost_stats"]]
            cost_maxs = [s[1] for s in stats["cost_stats"]]
            cost_means = [s[2] for s in stats["cost_stats"]]
            cost_stds = [s[3] for s in stats["cost_stats"]]
            stats["aggregated"] = {
                "cost_min": min(cost_mins),
                "cost_max": max(cost_maxs),
                "cost_mean": sum(cost_means) / len(cost_means),
                "cost_std": sum(cost_stds) / len(cost_stds),
                "avg_pred_tokens": stats["total_pred_tokens"] / max(1, stats["num_valid_samples"]),
                "avg_target_tokens": stats["total_target_tokens"] / max(1, stats["num_valid_samples"]),
            }
        return loss_value, stats
    return None, None


def _compute_mse_loss(
    hidden_states: torch.Tensor,
    latent_supervision: List[List[torch.Tensor]],
    latent_positions: torch.BoolTensor,
) -> Optional[torch.Tensor]:
    """Compute Mean Squared Error (MSE) loss for autoregressive latent prediction.

    Autoregressive alignment: Each hidden state predicts the NEXT token's feature.
    - Hidden states at [N, M-2] are predictions
    - Targets at [N+1, M-1] are injected latent features

    Where:
    - N = position of <|latent_begin|> token
    - M = position of <|latent_end|> token
    - Positions [N+1, M-1] = latent thinking tokens (<latent>)

    MSE Loss:
        MSE = (1/N) Σ_i ||pred_i - target_i||^2

    Sequence matching: Uses QWEN3VL_MATCH_STRATEGY env var (default: truncate)
    - 'truncate': Truncate both to min length
    - 'repeat': Repeat shorter to match longer
    - 'interpolate': Interpolate shorter to match longer

    Args:
        hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
        latent_supervision: Supervision targets for each sample
        latent_positions: Boolean mask indicating latent positions [N+1, M-1]

    Returns:
        MSE loss or None if no valid pairs
    """
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    dtype = hidden_states.dtype

    mse_losses = []

    for b in range(batch_size):
        latent_mask = latent_positions[b]
        if not latent_mask.any():
            continue

        # AUTOREGRESSIVE: Extract hidden states at [N, M-2] (one BEFORE each latent)
        # to predict latent features at [N+1, M-1]
        latent_indices = torch.nonzero(latent_mask).squeeze(-1)  # [N+1, M-1]
        if latent_indices.numel() == 0:
            continue

        # Shift indices by -1 to get prediction positions [N, M-2]
        # Handle edge case where first latent is at position 0
        pred_indices = latent_indices - 1
        pred_indices = torch.clamp(pred_indices, min=0)  # Ensure valid indices

        sample_hidden = hidden_states[b][pred_indices]  # [num_latents, hidden_dim]

        # Get supervision targets
        supervision_raw = latent_supervision[b] if b < len(latent_supervision) else []
        # Flatten nested list structure: [[tensor]] -> [tensor]
        supervision = []
        for item in supervision_raw:
            if isinstance(item, torch.Tensor):
                supervision.append(item)
            elif isinstance(item, list):
                for sub in item:
                    if isinstance(sub, torch.Tensor):
                        supervision.append(sub)
        if len(supervision) == 0:
            continue

        supervision_latents = []
        for sup_tensor in supervision:
            if isinstance(sup_tensor, torch.Tensor):
                # Mean-pool if spatial: [T, hidden_dim] → [hidden_dim]
                if sup_tensor.dim() == 2:
                    supervision_latents.append(sup_tensor.mean(dim=0))
                else:
                    supervision_latents.append(sup_tensor)

        if len(supervision_latents) == 0:
            continue

        supervision_tensor = torch.stack(supervision_latents, dim=0)
        supervision_tensor = supervision_tensor.to(device=device, dtype=dtype)

        # Verify dimensions match
        if sample_hidden.shape[-1] != supervision_tensor.shape[-1]:
            raise ValueError(
                f"Hidden dim {sample_hidden.shape[-1]} != supervision dim {supervision_tensor.shape[-1]}. "
                f"Both must be at LLM hidden dimension for MSE loss."
            )

        # Match sequence lengths
        strategy = os.environ.get("QWEN3VL_MATCH_STRATEGY", "truncate").lower()
        sample_hidden, supervision_tensor = _match_sequence_length(
            sample_hidden, supervision_tensor, strategy=strategy
        )

        # Compute MSE loss for this sample
        # MSE = mean((pred - target)^2)
        sample_loss = F.mse_loss(
            sample_hidden,
            supervision_tensor,
            reduction='mean'
        )
        mse_losses.append(sample_loss)

    if mse_losses:
        return torch.stack(mse_losses).mean()
    return None


def _add_latent_supervision_to_batch(
    collated: dict,
    batch: List[dict],
    logger,
) -> dict:
    """Add latent supervision to collated batch.

    This loads latent tensors from file paths and computes latent_positions.

    Separates:
    - latent_ground_truth: Thinking features (for injection at <latent>)
    - latent_supervision: Original image features (for OT loss reference)

    Both lists have the same length (one entry per thinking chunk).
    """
    # Get special token IDs
    latent_token_id = int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669"))
    thinking_start_id = int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667"))
    thinking_end_id = int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668"))
    debug_enabled = os.environ.get("QWEN3VL_DEBUG_FORWARD", "0") == "1"

    # Debug: Log token IDs on first call (rank 0 only to avoid spam in distributed)
    if not hasattr(_add_latent_supervision_to_batch, '_logged_ids'):
        if _is_rank0():
            logger.debug(f"[Qwen3VL Latent] Token IDs: latent={latent_token_id}, think_start={thinking_start_id}, think_end={thinking_end_id}")
        _add_latent_supervision_to_batch._logged_ids = True

    # Compute latent positions from input_ids first (to avoid loading latents when no positions)
    latent_positions = None
    if 'input_ids' in collated:
        latent_positions = _find_latent_positions(
            input_ids=collated['input_ids'],
            latent_token_id=latent_token_id,
            thinking_start_id=thinking_start_id,
            thinking_end_id=thinking_end_id,
        )
        collated['latent_positions'] = latent_positions

    # Debug: Log batch and latent_positions info
    if debug_enabled:
        rank_0 = _is_rank0()
        if rank_0 and debug_enabled:
            print(f"[COLLATOR_DEBUG] latent_positions.shape={latent_positions.shape if latent_positions is not None else 'None'}, batch_len={len(batch)}", flush=True, file=sys.stderr)
            for i, item in enumerate(batch):
                print(f"[COLLATOR_DEBUG] batch[{i}]: type={type(item)}, keys={item.keys() if isinstance(item, dict) else 'not dict'}", flush=True, file=sys.stderr)
                if isinstance(item, dict):
                    for k in ['latent_ground_truth', 'latent_supervision']:
                        v = item.get(k)
                        print(f"[COLLATOR_DEBUG] batch[{i}][{k}]={type(v)}, bool={bool(v) if v is not None else 'None'}, len={len(v) if isinstance(v, (list, tuple)) else 'N/A'}", flush=True, file=sys.stderr)

    # Handle latents (only attach metadata if this sample has latent positions).
    # Keep path lists in collated output; materialize tensors in main process forward.
    latent_ground_truth = []
    latent_supervision = []
    latent_ground_truth_paths = []
    latent_supervision_paths = []

    for idx, item in enumerate(batch):
        has_positions = False
        if latent_positions is not None and idx < latent_positions.shape[0]:
            if isinstance(latent_positions, torch.Tensor):
                has_positions = bool(latent_positions[idx].any().item())

        if not has_positions:
            latent_ground_truth.append([])
            latent_supervision.append([])
            latent_ground_truth_paths.append([])
            latent_supervision_paths.append([])
            continue

        gt_values = _flatten_latent_values(item.get("latent_ground_truth"))
        sup_values = _flatten_latent_values(item.get("latent_supervision"))

        gt_tensors = [x for x in gt_values if isinstance(x, torch.Tensor)]
        gt_paths = [x for x in gt_values if isinstance(x, str)]
        sup_tensors = [x for x in sup_values if isinstance(x, torch.Tensor)]
        sup_paths = [x for x in sup_values if isinstance(x, str)]

        latent_ground_truth.append(gt_tensors)
        latent_supervision.append(sup_tensors)
        latent_ground_truth_paths.append(gt_paths)
        latent_supervision_paths.append(sup_paths)

    collated['latent_ground_truth'] = latent_ground_truth
    collated['latent_supervision'] = latent_supervision
    collated['latent_ground_truth_paths'] = latent_ground_truth_paths
    collated['latent_supervision_paths'] = latent_supervision_paths

    # Pre-pack latent ground truth for VAE path to avoid per-step Python loops in forward.
    packed_ground_truth_chunks: list[torch.Tensor] = []
    for sample_gt in latent_ground_truth:
        if sample_gt and all(isinstance(t, torch.Tensor) for t in sample_gt):
            packed_ground_truth_chunks.append(torch.cat(sample_gt, dim=0))
    if packed_ground_truth_chunks:
        collated['latent_ground_truth_packed'] = torch.cat(packed_ground_truth_chunks, dim=0)

    # NOTE: All sequence-related tensors are already correctly expanded:
    # - input_ids: expanded in _expand_sample_for_latent_injection (pack)
    # - labels: expanded in _expand_sample_for_latent_injection (pack)
    # - attention_mask: created as [1]*len(input_ids) in pack (line 561)
    # - position_ids: computed by collator's get_rope_index for expanded sequence
    #
    # No additional expansion is needed in this function.

    return collated


def _compute_vae_loss(
    vae: LatentVAE,
    hidden_states: torch.Tensor,
    latent_positions: torch.BoolTensor,
    latent_supervision_packed: Optional[torch.Tensor],
) -> tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Compute VAE loss for latent generation (vectorized, following ReGuLaR).

    This function:
    1. Extracts hidden states at latent positions (shifted by 1 for autoregressive)
    2. Passes through VAE to get latent distributions
    3. Computes NLL loss (vectorized across all samples)

    Args:
        vae: LatentVAE module
        hidden_states: LLM hidden states [batch, seq_len, hidden_dim]
        latent_positions: Boolean mask for latent positions
        latent_supervision_packed: Flattened latent targets from collator

    Returns:
        (nll_loss, entropy, hidden_states, sampled_latents, batch_indices, seq_indices_original)
    """
    device = hidden_states.device

    # Vectorized extraction: shift positions by 1 for autoregressive
    # latent_positions: [batch, seq_len] -> shift to get prev token hidden states
    shifted_positions = torch.zeros_like(latent_positions)
    shifted_positions[:, 1:] = latent_positions[:, :-1]

    # Extract all hidden states at once using boolean mask
    # pred_hidden: [total_latent_tokens, hidden_dim]
    pred_hidden = hidden_states[shifted_positions]  # [batch, seq, hidden] -> [batch * seq, hidden] for True positions

    # Filter out padding (non-latent positions become zeros after masking, but we need to get actual values)
    # Use nonzero to get actual latent positions
    latent_indices = latent_positions.nonzero(as_tuple=False)  # [N, 2] (batch_idx, seq_idx)

    if latent_indices.shape[0] == 0:
        return None, None, hidden_states, None, None, None

    # Get hidden states at shifted latent positions (for autoregressive prediction)
    batch_indices = latent_indices[:, 0]
    seq_indices_shifted = latent_indices[:, 1] - 1  # Shift by 1 for autoregressive
    seq_indices_shifted = seq_indices_shifted.clamp(min=0)  # Handle position 0

    # Also keep original positions for pred_embed_forward (to place sampled latents correctly)
    seq_indices_original = latent_indices[:, 1]

    pred_hidden = hidden_states[batch_indices, seq_indices_shifted, :]  # [N, hidden_dim]

    if latent_supervision_packed is None or not isinstance(latent_supervision_packed, torch.Tensor) or latent_supervision_packed.numel() == 0:
        return None, None, hidden_states, None, None, None

    target_latent = latent_supervision_packed.to(device=device, dtype=hidden_states.dtype)

    # Match lengths (in case of mismatch)
    n_pred = pred_hidden.shape[0]
    n_target = target_latent.shape[0]

    if n_target == 0:
        return None, None, hidden_states, None, None, None

    if n_pred > n_target:
        pred_hidden = pred_hidden[:n_target, :]
        batch_indices = batch_indices[:n_target]
        seq_indices_original = seq_indices_original[:n_target]
    elif n_pred < n_target:
        target_latent = target_latent[:n_pred, :]

    # Vectorized VAE forward and loss computation (following ReGuLaR)
    # No per-sample loop - process all at once
    embeds_std = 0.03
    vae_dist = vae(pred_hidden, temperature=1.0)
    # Compute NLL loss directly (no sampling needed for loss)
    nll_loss = -vae_dist.log_prob(target_latent / embeds_std).mean()

    # Entropy regularization (like ReGuLaR) to prevent std from growing unbounded
    entropy = vae_dist.entropy().mean()

    # Sample continuous latent embeddings for second forward (pred_embed_forward, like ReGuLaR).
    # This is separate from sampled CoT token IDs used by latent-step CE labeling.
    sampled_latents = vae_dist.rsample()

    return nll_loss, entropy, hidden_states, sampled_latents, batch_indices, seq_indices_original


def _compute_pred_embed_forward_loss(
    model,
    original_inputs_embeds: torch.Tensor,
    sampled_latents: torch.Tensor,
    batch_indices: torch.Tensor,
    seq_indices: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    pixel_values,
    image_grid_thw,
) -> Optional[torch.Tensor]:
    """Compute pred_embed_forward_loss (ReGuLaR style).

    This runs a second forward pass with sampled latents instead of ground truth,
    and reuses the same CE supervision mask/targets as the primary forward.

    Following ReGuLaR's approach:
    - Input: original sequence with sampled latent embeddings injected at latent positions
    - Labels: identical to the first forward, including structural-token CE behavior
      controlled by the existing label rewrite / latent-step CE settings
    """
    if model is None or original_inputs_embeds is None or labels is None:
        return None

    local_assign_count = 0
    if sampled_latents is not None and batch_indices is not None and seq_indices is not None:
        local_assign_count = min(
            int(sampled_latents.shape[0]),
            int(batch_indices.shape[0]),
            int(seq_indices.shape[0]),
        )
    local_has_work = local_assign_count > 0

    # FSDP safety: if any rank has work, all ranks run the second forward path.
    if dist.is_initialized():
        has_work = torch.tensor(
            [1 if local_has_work else 0],
            device=original_inputs_embeds.device,
            dtype=torch.int32,
        )
        dist.all_reduce(has_work, op=dist.ReduceOp.MAX)
        if int(has_work.item()) == 0:
            return None
    elif not local_has_work:
        return None

    # Detach inputs to prevent gradient interference.
    inputs_embeds_sampled = original_inputs_embeds.detach().clone()
    inputs_embeds_with_sampled = inputs_embeds_sampled.clone()
    if local_has_work:
        if (
            local_assign_count != int(sampled_latents.shape[0])
            or local_assign_count != int(batch_indices.shape[0])
            or local_assign_count != int(seq_indices.shape[0])
        ):
            logger.warning(
                "[Qwen3VL Latent] pred_embed_forward size mismatch: sampled=%s batch_idx=%s seq_idx=%s using=%s",
                int(sampled_latents.shape[0]),
                int(batch_indices.shape[0]),
                int(seq_indices.shape[0]),
                local_assign_count,
            )

        # Vectorized latent injection: no per-sample Python loop.
        b_idx = batch_indices[:local_assign_count]
        s_idx = seq_indices[:local_assign_count]
        sampled_cast = sampled_latents[:local_assign_count].to(dtype=inputs_embeds_sampled.dtype)
        inputs_embeds_with_sampled[b_idx, s_idx] = sampled_cast

    # Keep the same supervision mask/targets as first forward.
    # Note: labels already encode QWEN3VL_LATENT_STEP_CE_LOSS /
    # QWEN3VL_LATENT_STEP_CE_TOKEN behavior in _expand_sample_for_latent_injection().
    # This guarantees pred_embed_forward and main CE optimize the same token sequence
    # under the same env-var controls.
    labels_sampled = labels.detach().clone()

    # Build shifted labels first, then keep the suffix that starts at the earliest
    # supervised position on any rank. Counting supervised tokens is insufficient
    # because latent positions stay masked inside the thinking span.
    shift_labels = F.pad(labels_sampled, (0, 1), value=-100)[..., 1:].contiguous()
    if shift_labels.numel() > 0:
        valid_mask = shift_labels != -100
        seq_len = int(shift_labels.shape[1])
        supervised_any = valid_mask.any(dim=1)
        first_valid = torch.where(
            supervised_any,
            valid_mask.float().argmax(dim=1),
            torch.full((shift_labels.shape[0],), seq_len, device=shift_labels.device, dtype=torch.long),
        )
        local_keep = int((seq_len - first_valid).max().item())
    else:
        local_keep = 0
    if dist.is_initialized():
        keep_tensor = torch.tensor([local_keep], device=labels_sampled.device, dtype=torch.int32)
        dist.all_reduce(keep_tensor, op=dist.ReduceOp.MAX)
        global_keep = int(keep_tensor.item())
    else:
        global_keep = local_keep
    if global_keep <= 0:
        return None

    target = shift_labels[:, -global_keep:].contiguous()
    try:
        second_outputs = model(
            input_ids=None,
            inputs_embeds=inputs_embeds_with_sampled,
            attention_mask=attention_mask,
            labels=None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            logits_to_keep=global_keep,
        )
        logits = second_outputs.logits
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)).float(),
            target.view(-1),
            ignore_index=-100,
            reduction="mean",
        )
        if dist.is_initialized() and not local_has_work:
            return loss * 0.0
        return loss
    except Exception as e:
        if dist.is_initialized():
            raise
        logger.warning(f"[Qwen3VL Latent] pred_embed_forward unavailable: {type(model).__name__} forward failed: {e}")
    return None


# Module-level cache for latent tensors (to avoid reference issues after function wrapping)
# Using OrderedDict for O(1) LRU operations
from collections import OrderedDict
_LATENT_TENSOR_CACHE: OrderedDict = OrderedDict()
# Centralized user-facing knob:
# - Tensor cache size = QWEN3VL_LATENT_CACHE_SIZE
# - Length cache size = auto-scaled multiple of tensor cache (tiny entries).
_LATENT_CACHE_SIZE = int(os.environ.get("QWEN3VL_LATENT_CACHE_SIZE", "20000"))
_LATENT_TENSOR_MAX_CACHE_SIZE = _LATENT_CACHE_SIZE
_LATENT_SEQ_LEN_CACHE: OrderedDict = OrderedDict()
_LATENT_SEQ_LEN_MAX_CACHE_SIZE = max(_LATENT_CACHE_SIZE * 5, 50000)


def _load_latent_tensors(paths: List[str]) -> List[torch.Tensor]:
    """Load latent tensors from file paths with LRU caching.

    Caches loaded tensors to avoid repeated disk I/O.
    Ensures tensors are cloned to avoid memory-mapping issues.

    Handles tensor payloads saved in the supported cache formats:
    1. Dict with 'l_features' key (from feature cache): {'l_features': Tensor, 'grid_thw': Tensor}
    2. Dict with 'latent' key (from adaptive renderer cache): {'latent': Tensor, 'grid_thw': Tensor}
    3. Direct tensor payload
    """
    # Check cache first - use module-level cache variable
    cache = _LATENT_TENSOR_CACHE
    cached_tensors = []
    remaining_paths = []

    for i, path in enumerate(paths):
        if isinstance(path, str) and path in cache:
            cached_tensors.append((i, cache[path]))
        else:
            remaining_paths.append((i, path))

    # Load uncached paths in parallel using thread pool
    # Thread pool helps with NFS I/O latency (I/O bound, not CPU bound)
    # Use same number of workers as dataloader (default: 4, matches config)
    num_io_workers = getattr(_load_latent_tensors, '_io_workers', 4)
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        # Avoid nested parallelism in dataloader workers.
        num_io_workers = 1

    def load_single_tensor(path: str):
        """Load a single tensor from file."""
        if isinstance(path, torch.Tensor):
            return path

        max_retries = 2
        for retry in range(max_retries):
            try:
                latent = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
                break  # Success
            except OSError as e:
                if "Too many open files" in str(e) or "Errno 24" in str(e):
                    gc.collect()
                    if retry < max_retries - 1:
                        continue
                raise
            except Exception:
                try:
                    latent = torch.load(path, map_location='cpu', mmap=True)
                    break
                except OSError as e:
                    if "Too many open files" in str(e) or "Errno 24" in str(e):
                        gc.collect()
                        if retry < max_retries - 1:
                            continue
                    raise
                raise

        # Extract latent features if it's a dict (cached format)
        if isinstance(latent, dict):
            if 'l_features' in latent:
                tensor = latent['l_features'].clone()
            elif 'latent' in latent:
                tensor = latent['latent'].clone()
            else:
                return None
        elif isinstance(latent, torch.Tensor):
            tensor = latent.clone()
        else:
            return None

        del latent
        return tensor

    # Load uncached files in parallel
    new_tensors = []
    if remaining_paths:
        with ThreadPoolExecutor(max_workers=num_io_workers) as executor:
            futures = {executor.submit(load_single_tensor, path): (i, path) for i, path in remaining_paths}

            for future in as_completed(futures):
                i, path = futures[future]
                try:
                    tensor = future.result()
                    if tensor is not None:
                        cache[path] = tensor
                        new_tensors.append((i, tensor))
                except Exception as e:
                    logger.warning(f"[Qwen3VL Latent] Failed to load {path}: {e}")

    # Build result aligned to input order.
    # Keep strict index alignment here to avoid implicit re-indexing drift between:
    # - pack-after-injection length expansion (worker-side)
    # - forward-time latent materialization (main process)
    ordered: list[Optional[torch.Tensor]] = [None] * len(paths)
    for i, tensor in cached_tensors:
        if 0 <= i < len(ordered):
            ordered[i] = tensor
    for i, tensor in new_tensors:
        if 0 <= i < len(ordered):
            ordered[i] = tensor

    missing = [i for i, t in enumerate(ordered) if t is None]
    if missing:
        preview = [paths[i] for i in missing[:3] if i < len(paths)]
        logger.warning(
            "[Qwen3VL Latent] Missing %s/%s latent tensors during load (examples: %s). "
            "This can cause latent_positions/latent_tokens drift.",
            len(missing),
            len(paths),
            preview,
        )

    # Return only successfully loaded tensors while preserving relative order.
    return [t for t in ordered if isinstance(t, torch.Tensor)]


def _load_latent_seq_lens(paths: List[str]) -> List[int]:
    """Load only latent sequence lengths from file paths with a lightweight cache.

    This avoids materializing large latent tensors inside dataloader workers.
    """
    lens_cache = _LATENT_SEQ_LEN_CACHE
    result: dict[int, int] = {}
    uncached: List[tuple[int, str]] = []

    for idx, path in enumerate(paths):
        if isinstance(path, str) and path in lens_cache:
            result[idx] = int(lens_cache[path])
        else:
            uncached.append((idx, path))

    if uncached:
        for idx, path in uncached:
            try:
                latent = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            except Exception:
                latent = torch.load(path, map_location="cpu", mmap=True)

            tensor: Optional[torch.Tensor] = None
            if isinstance(latent, dict):
                if "l_features" in latent:
                    tensor = latent["l_features"]
                elif "latent" in latent:
                    tensor = latent["latent"]
            elif isinstance(latent, torch.Tensor):
                tensor = latent

            seq_len = int(_latent_seq_len(tensor)) if tensor is not None else 0
            if isinstance(path, str):
                lens_cache[path] = seq_len
                if len(lens_cache) > _LATENT_SEQ_LEN_MAX_CACHE_SIZE:
                    lens_cache.popitem(last=False)
            result[idx] = seq_len
            del latent

    # Preserve one-to-one alignment with input paths, including zero lengths.
    return [int(result.get(i, 0)) for i in range(len(paths))]


def _materialize_latent_batch(
    latent_batch: Optional[List[Any]],
    latent_paths_batch: Optional[List[Any]],
    strict: bool = False,
) -> Optional[List[List[torch.Tensor]]]:
    """Materialize latent tensors on the training process from optional path lists."""
    if latent_batch is None and latent_paths_batch is None:
        return None

    num_samples = max(len(latent_batch or []), len(latent_paths_batch or []))
    materialized: List[List[torch.Tensor]] = []
    for idx in range(num_samples):
        batch_values = _flatten_latent_values(
            latent_batch[idx] if latent_batch is not None and idx < len(latent_batch) else None
        )
        path_values = _flatten_latent_values(
            latent_paths_batch[idx]
            if latent_paths_batch is not None and idx < len(latent_paths_batch)
            else None
        )

        sample_tensors = [x for x in batch_values if isinstance(x, torch.Tensor)]
        sample_paths = [x for x in batch_values if isinstance(x, str)]
        sample_paths.extend(x for x in path_values if isinstance(x, str))

        if sample_tensors:
            materialized.append(sample_tensors)
            continue
        if sample_paths:
            loaded = _load_latent_tensors(sample_paths)
            if len(loaded) != len(sample_paths):
                msg = (
                    f"[Qwen3VL Latent] Materialize mismatch on sample {idx}: "
                    f"paths={len(sample_paths)} loaded={len(loaded)}. "
                    f"path_preview={sample_paths[:3]}"
                )
                if strict:
                    raise ValueError(msg)
                logger.warning(msg)
            materialized.append(loaded)
            continue

        materialized.append([])

    return materialized


# Initialize cache with LRU eviction
# Using module-level variables to avoid issues after function wrapping
# Note: Cache is already initialized above at module level


def _evict_old_cache_entries():
    """Evict oldest cache entries when cache is full (LRU policy).

    Using OrderedDict.popitem(last=False) for O(1) eviction.
    """
    cache = _LATENT_TENSOR_CACHE
    max_size = _LATENT_TENSOR_MAX_CACHE_SIZE

    # Only evict when cache exceeds capacity by 10% (avoid frequent small evictions)
    if len(cache) <= int(max_size * 1.1):
        return

    # Evict excess entries
    excess = len(cache) - max_size
    for _ in range(excess):
        if not cache:
            break
        cache.popitem(last=False)


# Patch _load_latent_tensors to use LRU eviction
_original_load_latent_tensors = _load_latent_tensors


def _load_latent_tensors_with_lru(paths: List[str]) -> List[torch.Tensor]:
    """Wrapper that adds LRU eviction to tensor loading.

    Use OrderedDict.move_to_end() for O(1) LRU updates.
    """
    result = _original_load_latent_tensors(paths)

    # Update cache order for accessed items - O(1) with OrderedDict
    cache = _LATENT_TENSOR_CACHE
    for path in paths:
        if isinstance(path, str) and path in cache:
            # move_to_end marks item as recently used - O(1)
            cache.move_to_end(path)

    # Evict old entries if cache is too large
    _evict_old_cache_entries()

    return result


# Replace with LRU version
_load_latent_tensors = _load_latent_tensors_with_lru
_load_latent_tensors._io_workers = int(os.environ.get("DATALOADER_NUM_WORKERS", "4"))


def _find_latent_positions(
    input_ids: torch.Tensor,
    latent_token_id: int,
    thinking_start_id: int,
    thinking_end_id: int,
) -> torch.BoolTensor:
    """Find positions of latent tokens in input_ids."""
    batch_size, seq_len = input_ids.shape
    latent_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for b in range(batch_size):
        in_thinking = False
        for i in range(seq_len):
            tok = input_ids[b, i].item()
            if tok == thinking_start_id:
                in_thinking = True
                continue
            if tok == thinking_end_id:
                in_thinking = False
                continue
            if in_thinking and tok == latent_token_id:
                latent_mask[b, i] = True

    # Debug: Log if any latent positions found (first call only, rank 0 only)
    if not hasattr(_find_latent_positions, '_logged_found') and latent_mask.any():
        should_log = _is_rank0()

        if should_log:
            num_found = latent_mask.sum().item()
            logger.debug(f"[Qwen3VL Latent] Found {num_found} latent positions in batch")
        _find_latent_positions._logged_found = True

    return latent_mask


# Apply patches on import
_patch_once()
