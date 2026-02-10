"""
Minimal LlamaFactory integration for DPSK OCR + Qwen3-VL.

Effects:
  1) Registers OCRQwen3VLConfig to load our model when using Qwen3-VL checkpoints
  2) Replaces Qwen3VLProcessor.image_processor with DPSKOCRImageProcessor
  3) Registers OCRVL template to avoid eager image loading
  4) Patches FSDP+PEFT incompatibility for custom models with additional_target
  5) Converts all model parameters to bfloat16 AFTER FSDP wrapping (checkpoint loading resets dtypes)
  6) Adds connector parameters to optimizer (FSDP-compatible replacement for additional_target)
  7) Patches Accelerate FSDP + PEFT adapter-only saving on non-rank0 processes

Note: Checkpoint loading with FSDP resets parameter dtypes to float32, so we must
convert AFTER FSDP wrapping, not before.
"""

from __future__ import annotations

import functools
import logging
import os

import torch
import torch.multiprocessing as mp
import torch.nn as nn

# FD limit: Use file_system sharing strategy to avoid FD-per-tensor
mp.set_sharing_strategy("file_system")
# Only log from rank 0 to avoid spam (check environment variable set by torchrun)
if os.environ.get("LOCAL_RANK", "0") == "0":
    logging.warning("[OCRVL] Set torch multiprocessing sharing strategy to 'file_system' to avoid FD limits")
from accelerate.utils import fsdp_utils as accelerate_fsdp_utils
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
)

# OCRVL imports
import OCRVL.llamafactory as ocrvl_llamafactory
from OCRVL.llamafactory.dpsk_ocr_image_processor import DPSKOCRImageProcessor
from OCRVL.model.language_model.ocr_qwen3_vl import (
    OCRQwen3VLConfig,
    OCRQwen3VLForConditionalGeneration,
)

# Qwen imports
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

# These may fail if modules aren't available
try:
    from Qwen.scripts import llamafactory_integration
except ImportError:
    llamafactory_integration = None

try:
    from peft.utils.other import fsdp_auto_wrap_policy as original_fsdp_auto_wrap_policy
except ImportError:
    original_fsdp_auto_wrap_policy = None

try:
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
except ImportError:
    CustomSeq2SeqTrainer = None


def _patch_once() -> None:
    # PID-based guard for multiprocessing
    pid = str(os.getpid())
    if os.environ.get("OCRVL_LLAMAFACTORY_PATCHED_PID", "") == pid:
        return
    os.environ["OCRVL_LLAMAFACTORY_PATCHED_PID"] = pid

    # Import Qwen3VL latent supervision integration for thinking training
    if llamafactory_integration is not None:
        import logging
        logger = logging.getLogger(__name__)
        latent_supervision = os.environ.get("QWEN3VL_LATENT_SUPERVISION", "0")
        logger.warning(f"[sitecustomize] QWEN3VL_LATENT_SUPERVISION={latent_supervision}, llamafactory_integration={llamafactory_integration}")
        llamafactory_integration._patch_once()
    else:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("[sitecustomize] llamafactory_integration is None - import failed")

    # 1) Register model class for our config
    AutoModelForCausalLM.register(OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration, exist_ok=True)
    AutoModelForVision2Seq.register(OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration, exist_ok=True)

    # 2) Convert Qwen3VLConfig to OCRQwen3VLConfig when loading
    # Only apply OCRVL wrapper for models that explicitly need it (controlled by env var or model path)
    orig_autoconfig_from_pretrained = AutoConfig.from_pretrained

    @classmethod  # type: ignore[misc]
    def wrapped_autoconfig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = orig_autoconfig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        # Only convert to OCRQwen3VLConfig if:
        # 1) OCRVL_ENABLE_WRAPPER=1 is set, OR
        # 2) Model path contains OCRVL/DPSK keywords (legacy behavior for OCRVL models)
        if isinstance(config, Qwen3VLConfig) and not isinstance(config, OCRQwen3VLConfig):
            should_wrap = os.environ.get("OCRVL_ENABLE_WRAPPER", "0") == "1"

            # Also auto-wrap if model path indicates it's an OCRVL model
            if not should_wrap:
                model_path_lower = str(pretrained_model_name_or_path).lower()
                should_wrap = any(keyword in model_path_lower for keyword in ["ocrvl", "dpsk", "ocr_"])

            if should_wrap:
                ocr_config_dict = config.to_dict()
                ocr_config_dict.pop("model_type", None)
                logging.warning(f"[OCRVL] Converting {pretrained_model_name_or_path} to OCRQwen3VLConfig")
                return OCRQwen3VLConfig(**ocr_config_dict)

            logging.info(f"[OCRVL] Using native Qwen3VLConfig for {pretrained_model_name_or_path} (OCRVL wrapper disabled)")

        return config

    AutoConfig.from_pretrained = wrapped_autoconfig_from_pretrained

    # 3) Patch AutoProcessor to use DPSKOCRImageProcessor
    orig_processor_from_pretrained = AutoProcessor.from_pretrained

    @classmethod  # type: ignore[misc]
    @functools.wraps(orig_processor_from_pretrained)
    def wrapped_processor_from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        processor = orig_processor_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if processor is None:
            return processor

        # Only patch Qwen3-VL processors when OCRVL wrapper is enabled or model path indicates OCRVL/DPSK
        if processor.__class__.__name__ == "Qwen3VLProcessor":
            should_wrap = os.environ.get("OCRVL_ENABLE_WRAPPER", "0") == "1"

            if not should_wrap:
                model_path_lower = str(pretrained_model_name_or_path).lower()
                should_wrap = any(keyword in model_path_lower for keyword in ["ocrvl", "dpsk", "ocr_"])

            if should_wrap:
                merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)
                processor.image_processor = DPSKOCRImageProcessor(merge_size=merge_size)
                logging.warning(f"[OCRVL] Using DPSKOCRImageProcessor for {pretrained_model_name_or_path}")

        return processor

    AutoProcessor.from_pretrained = wrapped_processor_from_pretrained

    # 3b) Add special thinking tokens for latent training
    # These tokens need to be single tokens, not multiple tokens
    orig_tokenizer_from_pretrained = AutoTokenizer.from_pretrained

    @classmethod  # type: ignore[misc]
    @functools.wraps(orig_tokenizer_from_pretrained)
    def wrapped_tokenizer_from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        """Wrapped tokenizer that adds thinking special tokens."""
        tokenizer = orig_tokenizer_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if tokenizer is None:
            return tokenizer

        # Only add tokens for Qwen3-VL models.
        # Note: Qwen3-VL uses Qwen2TokenizerFast, so type checks alone are insufficient.
        model_hint = str(pretrained_model_name_or_path)
        name_hint = getattr(tokenizer, "name_or_path", "")
        if "Qwen3-VL" not in model_hint and "Qwen3-VL" not in name_hint and "Qwen3-VL" not in str(type(tokenizer)):
            return tokenizer

        # Special tokens to add (must be single tokens)
        # <|latent_step|>: latent injection positions
        # <|thinking_sep|>: separator between thinking steps (kept as single token)
        special_tokens = [
            "<|latent_step|>",
            "<|thinking_sep|>",
        ]

        # Check if tokens are already single tokens
        added_count = 0
        for token in special_tokens:
            encoded = tokenizer.encode(token, add_special_tokens=False)
            if len(encoded) > 1:
                # Token is not single, add it
                num_added = tokenizer.add_special_tokens(
                    {"additional_special_tokens": [token]},
                    replace_additional_special_tokens=False
                )
                added_count += num_added

        # Set up token ID environment variables for llamafactory_integration
        # Thinking tokens are already in Qwen3VL vocab (fixed IDs)
        # Latent step token will be added dynamically
        os.environ["QWEN3VL_THINKING_START_ID"] = "151667"
        os.environ["QWEN3VL_THINKING_END_ID"] = "151668"
        logging.warning("[OCRVL] Using Qwen3VL native thinking tokens: start=151667, end=151668")

        if added_count > 0:
            logging.warning(f"[OCRVL] Added {added_count} special latent tokens to tokenizer")

        # Store token IDs for llamafactory_integration (always override to be safe)
        for token in special_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id >= 0:  # Found in vocab
                if token == "<|latent_step|>":
                    os.environ["QWEN3VL_LATENT_TOKEN_ID"] = str(token_id)
                    logging.warning(f"[OCRVL] {token} = ID {token_id} (env: QWEN3VL_LATENT_TOKEN_ID)")
                elif token == "<|thinking_sep|>":
                    os.environ["QWEN3VL_THINKING_SEP_ID"] = str(token_id)
                    logging.warning(f"[OCRVL] {token} = ID {token_id} (env: QWEN3VL_THINKING_SEP_ID)")

        return tokenizer

    AutoTokenizer.from_pretrained = wrapped_tokenizer_from_pretrained

    # NOTE: Vision tower compilation is now handled by TransparentEvalCallback.on_train_begin
    # to avoid FSDP wrapping conflicts. The callback runs AFTER accelerator.prepare().

    # 3) Register OCRVL template to avoid eager image loading
    try:
        ocrvl_llamafactory.register_ocrvl_templates()
        import logging
        logging.getLogger(__name__).warning("[sitecustomize] ✓ Registered OCRVL templates")
    except Exception as e:
        # "already registered" or "already exists" is expected on subsequent ranks, don't fail
        error_str = str(e).lower()
        if "already registered" not in error_str and "already exists" not in error_str:
            import logging
            import traceback
            logging.getLogger(__name__).error(f"[sitecustomize] Failed to register OCRVL templates: {e}")
            logging.getLogger(__name__).error(traceback.format_exc())
            raise
        else:
            import logging
            logging.getLogger(__name__).debug(f"[sitecustomize] Templates already registered (OK on non-rank-0)")

    # 5) Patch FSDP+PEFT incompatibility for custom models
    # The issue: transformers._fsdp_qlora_plugin_updates() calls fsdp_auto_wrap_policy(model)
    # on the PEFT-wrapped model, which can't find transformer layer classes
    if original_fsdp_auto_wrap_policy is not None:
        try:
            @functools.wraps(original_fsdp_auto_wrap_policy)
            def patched_fsdp_auto_wrap_policy(model):
                """Patched version that returns None if transformer layers not found.

                Returning None allows FSDP to use the fsdp_auto_wrap_policy from config
                (SIZE_BASED_WRAP) instead of PEFT's transformer-based wrapping.
                """
                try:
                    return original_fsdp_auto_wrap_policy(model)
                except Exception as e:
                    if "Could not find the transformer layer class to wrap in the model" in str(e):
                        logging.warning(
                            "[OCRVL] FSDP auto-wrap policy could not find transformer layer class. "
                            "Returning None to use config-based wrapping policy. This is expected for custom OCRVL models."
                        )
                        # Return None to let FSDP use the config's fsdp_auto_wrap_policy (SIZE_BASED_WRAP)
                        return None
                    raise

            # Patch the peft.utils.other module
            import peft.utils.other
            peft.utils.other.fsdp_auto_wrap_policy = patched_fsdp_auto_wrap_policy
            import peft.utils.other as peft_other_module
            if hasattr(peft_other_module, '__dict__'):
                peft_other_module.__dict__['fsdp_auto_wrap_policy'] = patched_fsdp_auto_wrap_policy
        except Exception as e:
            logging.warning(f"[OCRVL] Failed to patch FSDP auto-wrap policy: {e}")

    # 6) Convert all model parameters to bfloat16 on each iteration (before train/eval)
    # This is required because FSDP may reset dtypes when switching between train/eval modes
    # Must happen in _inner_training_loop which runs AFTER accelerator.prepare()
    try:
        # Store original _inner_training_loop method
        original_inner_training_loop = Trainer._inner_training_loop

        @functools.wraps(original_inner_training_loop)
        def patched_inner_training_loop(self, *args, **kwargs):
            """Convert all params to bfloat16 BEFORE each train/eval iteration."""
            # Convert on each iteration (before both training and eval)
            # This runs AFTER FSDP wrapping is complete
            # NOTE: SAM encoder dtype fix wrapper handles SAM params, so we skip LayerNorm
            if hasattr(self.model, 'modules'):
                converted = 0

                for module in self.model.modules():
                    # Convert Conv2d/Linear layers (skip SAM encoder, handled by wrapper)
                    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
                        # Skip if this is part of SAM encoder (handled by dtype fix wrapper)
                        if hasattr(module, 'name') and 'sam' in str(module.name).lower():
                            continue
                        if hasattr(module, 'bias') and module.bias is not None:
                            if module.bias.dtype != torch.bfloat16:
                                module.bias.data = module.bias.data.to(torch.bfloat16)
                                converted += 1
                        if hasattr(module, 'weight') and module.weight is not None:
                            if module.weight.dtype != torch.bfloat16:
                                module.weight.data = module.weight.data.to(torch.bfloat16)
                                converted += 1

                if converted > 0:
                    logging.warning(f"[OCRVL] Converted {converted} params to bfloat16 (pre-iteration)")

            return original_inner_training_loop(self, *args, **kwargs)

        # Apply patch
        Trainer._inner_training_loop = patched_inner_training_loop
    except Exception as e:
        logging.warning(f"[OCRVL] Failed to patch trainer _inner_training_loop: {e}")

    # 6b) Patch model.eval() to ensure dtype conversion when switching to eval mode
    # This is needed because FSDP can rematerialize parameters with wrong dtypes during eval mode switch
    # DISABLED: Eval mode conversion was causing dtype mismatches during checkpoint saving
    # The SAM encoder dtype fix wrapper in dpsk_ocr_encoder.py is sufficient for training
    logging.debug("[OCRVL] Skipping nn.Module.eval patch (SAM dtype fix wrapper is sufficient)")

    # 7) Freeze LLM when OCRVL_FREEZE_LLM=1 (connector-only training)
    # This enables training ONLY connectors without LoRA on LLM layers
    if os.environ.get("OCRVL_FREEZE_LLM", "0") == "1":
        try:
            original_create_optimizer = Trainer.create_optimizer

            @functools.wraps(original_create_optimizer)
            def patched_create_optimizer_freeze_llm(self):
                """Freeze LLM parameters before optimizer creation, keeping only connectors trainable."""
                if hasattr(self.model, "model"):
                    frozen_count = 0
                    trainable_count = 0

                    for name, param in self.model.named_parameters():
                        # Freeze language model (but NOT connectors)
                        if "language_model" in name and "ocr" not in name:
                            if param.requires_grad:
                                param.requires_grad = False
                                frozen_count += 1
                        # Count trainable connector params
                        elif "ocr_connector" in name or "ocr_deepstack_connector" in name:
                            if param.requires_grad:
                                trainable_count += 1

                    if frozen_count > 0:
                        logging.warning(
                            f"[OCRVL] Froze {frozen_count} LLM parameters (connector-only training mode). "
                            f"{trainable_count} connector parameters remain trainable."
                        )

                # Call original create_optimizer
                return original_create_optimizer(self)

            Trainer.create_optimizer = patched_create_optimizer_freeze_llm
            logging.debug("[OCRVL] LLM freeze patch applied (OCRVL_FREEZE_LLM=1)")
        except Exception as e:
            logging.warning(f"[OCRVL] Failed to patch trainer for LLM freezing: {e}")

    # 8) Ensure OCR connectors are always trainable and in optimizer
    # Connectors must be trainable in all stages (alignment, VQA, etc.)
    # This patch ensures they're properly registered even if something goes wrong with additional_target
    try:
        original_create_optimizer = Trainer.create_optimizer

        @functools.wraps(original_create_optimizer)
        def patched_create_optimizer_ensure_connectors(self):
            """Ensure OCR connectors are trainable and in optimizer."""
            # Step 1: Ensure connector gradients are enabled
            connector_params = []
            for name, param in self.model.named_parameters():
                if 'ocr_connector' in name or 'deepstack_connector' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                        logging.warning(f"[OCRVL] Re-enabled gradient for connector: {name}")
                    if param.requires_grad:
                        connector_params.append((name, param))

            # Step 2: Create optimizer
            optimizer = original_create_optimizer(self)

            # Step 3: Verify connectors are in optimizer, add if missing
            if connector_params:
                existing_param_ids = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
                missing_params = [(n, p) for n, p in connector_params if id(p) not in existing_param_ids]

                if missing_params:
                    param_names = [n for n, p in missing_params]
                    logging.warning(f"[OCRVL] Adding {len(missing_params)} connector params to optimizer: {param_names}")
                    optimizer.add_param_group({"params": [p for n, p in missing_params]})
                else:
                    logging.debug(f"[OCRVL] All {len(connector_params)} connector params already in optimizer")

            return optimizer

        Trainer.create_optimizer = patched_create_optimizer_ensure_connectors
        logging.debug("[OCRVL] Applied connector training patch (ensures connectors trainable in all stages)")
    except Exception as e:
        logging.warning(f"[OCRVL] Failed to patch trainer for connector training: {e}")

    # 9) Patch Accelerate FSDP + PEFT adapter-only saving on non-rank0 processes.
    # Accelerate's save_fsdp_model() calls _get_model_state_dict() on every rank even when
    # FULL_STATE_DICT + rank0_only is enabled; in that case non-rank0 ranks receive an empty
    # state_dict, and PEFT's get_peft_model_state_dict can crash with KeyError on modules_to_save.
    try:
        original_get_model_state_dict = accelerate_fsdp_utils._get_model_state_dict

        @functools.wraps(original_get_model_state_dict)
        def patched_get_model_state_dict(model, adapter_only=False, sd_options=None):
            if adapter_only:
                try:
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        if torch.distributed.get_rank() != 0:
                            # Participate in any collective state_dict gathering, but don't try to
                            # materialize adapter/modules_to_save state dict on non-rank0.
                            try:
                                _ = model.state_dict()
                            except Exception:
                                pass
                            return {}
                except Exception:
                    pass

            return original_get_model_state_dict(model, adapter_only=adapter_only, sd_options=sd_options)

        accelerate_fsdp_utils._get_model_state_dict = patched_get_model_state_dict
    except Exception as e:
        logging.warning(f"[OCRVL] Failed to patch accelerate FSDP save for PEFT: {e}")

    # 10) Inject TransparentEvalCallback into LlamaFactory trainer
    # This enables transparent evaluation during training when eval_dataset='ocrvl_transparent_eval'
    if CustomSeq2SeqTrainer is not None:
        try:
            original_trainer_init = CustomSeq2SeqTrainer.__init__

            @functools.wraps(original_trainer_init)
            def patched_trainer_init(self, model=None, args=None, callbacks=None, **kwargs):
                """Patched __init__ that injects TransparentEvalCallback."""
                # Always inject TransparentEvalCallback
                # It will auto-enable when eval_dataset='ocrvl_transparent_eval'
                try:
                    from OCRVL.llamafactory.transparent_eval_callback import TransparentEvalCallback

                    # Extract tokenizer and processor from kwargs
                    # LlamaFactory passes these via **tokenizer_module
                    tokenizer = kwargs.get('tokenizer')
                    processor = kwargs.get('processor')

                    if model is not None:
                        # Create TransparentEvalCallback
                        transparent_eval_callback = TransparentEvalCallback(
                            model=model,
                            tokenizer=tokenizer,
                            processor=processor
                        )

                        # Add to callbacks list
                        if callbacks is None:
                            callbacks = []
                        elif not isinstance(callbacks, list):
                            callbacks = list(callbacks)

                        callbacks.append(transparent_eval_callback)
                        logging.warning("[OCRVL] ✓ Injected TransparentEvalCallback into trainer")

                except Exception as e:
                    logging.warning(f"[OCRVL] Failed to create TransparentEvalCallback: {e}")
                    import traceback
                    logging.warning(traceback.format_exc())

                # Call original __init__ with updated callbacks
                return original_trainer_init(self, model=model, args=args, callbacks=callbacks, **kwargs)

            # Apply patch
            CustomSeq2SeqTrainer.__init__ = patched_trainer_init
            logging.warning("[OCRVL] ✓ Patched CustomSeq2SeqTrainer.__init__ for TransparentEvalCallback injection")

        except Exception as e:
            logging.warning(f"[OCRVL] Failed to patch CustomSeq2SeqTrainer for TransparentEvalCallback: {e}")


# Apply patches on import
_patch_once()
