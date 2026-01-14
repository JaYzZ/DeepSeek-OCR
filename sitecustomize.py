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
import os


def _patch_once() -> None:
    # PID-based guard for multiprocessing
    pid = str(os.getpid())
    if os.environ.get("OCRVL_LLAMAFACTORY_PATCHED_PID", "") == pid:
        return
    os.environ["OCRVL_LLAMAFACTORY_PATCHED_PID"] = pid

    # Import OCRVL modules
    import OCRVL.llamafactory as ocrvl_llamafactory
    from OCRVL.model.language_model.ocr_qwen3_vl import (
        OCRQwen3VLConfig,
        OCRQwen3VLForConditionalGeneration,
    )
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from OCRVL.llamafactory.dpsk_ocr_image_processor import DPSKOCRImageProcessor

    # 1) Register model class for our config
    AutoModelForCausalLM.register(OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration, exist_ok=True)
    AutoModelForVision2Seq.register(OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration, exist_ok=True)

    # 2) Convert Qwen3VLConfig to OCRQwen3VLConfig when loading
    orig_autoconfig_from_pretrained = AutoConfig.from_pretrained

    @classmethod  # type: ignore[misc]
    def wrapped_autoconfig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = orig_autoconfig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if isinstance(config, Qwen3VLConfig) and not isinstance(config, OCRQwen3VLConfig):
            ocr_config_dict = config.to_dict()
            ocr_config_dict.pop("model_type", None)
            return OCRQwen3VLConfig(**ocr_config_dict)
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

        # Only patch Qwen3-VL processors
        if processor.__class__.__name__ == "Qwen3VLProcessor":
            merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)
            processor.image_processor = DPSKOCRImageProcessor(merge_size=merge_size)

        return processor

    AutoProcessor.from_pretrained = wrapped_processor_from_pretrained

    # 4) Register OCRVL template to avoid eager image loading
    ocrvl_llamafactory.register_llamafactory_extensions()

    # 5) Patch FSDP+PEFT incompatibility for custom models
    # The issue: transformers._fsdp_qlora_plugin_updates() calls fsdp_auto_wrap_policy(model)
    # on the PEFT-wrapped model, which can't find transformer layer classes
    try:
        from peft.utils.other import fsdp_auto_wrap_policy as original_fsdp_auto_wrap_policy

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
                    import logging
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
        import logging
        logging.warning(f"[OCRVL] Failed to patch FSDP auto-wrap policy: {e}")

    # 6) Convert all model parameters to bfloat16 on each iteration (before train/eval)
    # This is required because FSDP may reset dtypes when switching between train/eval modes
    # Must happen in _inner_training_loop which runs AFTER accelerator.prepare()
    try:
        from transformers import Trainer as OriginalTrainer
        import torch
        import logging
        import torch.nn as nn

        # Store original _inner_training_loop method
        original_inner_training_loop = OriginalTrainer._inner_training_loop

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
        OriginalTrainer._inner_training_loop = patched_inner_training_loop
    except Exception as e:
        import logging
        logging.warning(f"[OCRVL] Failed to patch trainer _inner_training_loop: {e}")

    # 6b) Patch model.eval() to ensure dtype conversion when switching to eval mode
    # This is needed because FSDP can rematerialize parameters with wrong dtypes during eval mode switch
    # DISABLED: Eval mode conversion was causing dtype mismatches during checkpoint saving
    # The SAM encoder dtype fix wrapper in dpsk_ocr_encoder.py is sufficient for training
    try:
        import logging
        logging.info("[OCRVL] Skipping nn.Module.eval patch (SAM dtype fix wrapper is sufficient)")
    except Exception as e:
        import logging
        logging.warning(f"[OCRVL] Failed to skip nn.Module.eval patch: {e}")

    # 7) Add connector parameters to optimizer (replacement for additional_target)
    # This is needed because additional_target (PEFT's modules_to_save) is incompatible with FSDP
    try:
        from transformers import Trainer as OriginalTrainer
        import torch

        # Store original create_optimizer method
        original_create_optimizer = OriginalTrainer.create_optimizer

        @functools.wraps(original_create_optimizer)
        def patched_create_optimizer(self):
            """Wrapper that adds OCR connector parameters to the optimizer."""
            # Call original create_optimizer
            optimizer = original_create_optimizer(self)

            # Add connector parameters to optimizer if they exist and are trainable.
            # Important: avoid duplicating params already present in optimizer groups.
            if hasattr(self.model, "model"):
                import logging

                connector_params = []
                connector_param_names = []

                for name, param in self.model.named_parameters():
                    if "ocr_connector" in name and param.requires_grad:
                        connector_params.append(param)
                        connector_param_names.append(name)

                if connector_params:
                    existing_param_ids = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
                    missing_params = [p for p in connector_params if id(p) not in existing_param_ids]

                    if missing_params:
                        logging.warning(
                            f"[OCRVL] Adding {len(missing_params)} OCR connector params to optimizer (missing from default groups)."
                        )
                        optimizer.add_param_group({"params": missing_params})
                    else:
                        logging.warning(
                            "[OCRVL] OCR connector params already present in optimizer groups (no-op)."
                        )

            return optimizer

        # Apply patch
        OriginalTrainer.create_optimizer = patched_create_optimizer
    except Exception as e:
        import logging
        logging.warning(f"[OCRVL] Failed to patch trainer for connector params: {e}")

    # 7) Patch Accelerate FSDP + PEFT adapter-only saving on non-rank0 processes.
    # Accelerate's save_fsdp_model() calls _get_model_state_dict() on every rank even when
    # FULL_STATE_DICT + rank0_only is enabled; in that case non-rank0 ranks receive an empty
    # state_dict, and PEFT's get_peft_model_state_dict can crash with KeyError on modules_to_save.
    try:
        from accelerate.utils import fsdp_utils as accelerate_fsdp_utils

        original_get_model_state_dict = accelerate_fsdp_utils._get_model_state_dict

        @functools.wraps(original_get_model_state_dict)
        def patched_get_model_state_dict(model, adapter_only=False, sd_options=None):
            if adapter_only:
                try:
                    import torch

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
        import logging
        logging.warning(f"[OCRVL] Failed to patch accelerate FSDP save for PEFT: {e}")

    # 8) Inject TransparentEvalCallback into LlamaFactory trainer
    # This enables transparent evaluation during training when OCRVL_ENABLE_TRANSPARENT_EVAL=1
    try:
        from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

        original_trainer_init = CustomSeq2SeqTrainer.__init__

        @functools.wraps(original_trainer_init)
        def patched_trainer_init(self, model=None, args=None, callbacks=None, **kwargs):
            """Patched __init__ that injects TransparentEvalCallback."""
            import logging

            # Debug: Always log that we're in the patched init
            logging.warning(f"[OCRVL DEBUG] Patched trainer __init__ called, OCRVL_ENABLE_TRANSPARENT_EVAL={os.environ.get('OCRVL_ENABLE_TRANSPARENT_EVAL', 'NOT_SET')}")

            # Check if transparent eval is enabled
            if os.environ.get("OCRVL_ENABLE_TRANSPARENT_EVAL", "0") == "1":
                try:
                    from OCRVL.llamafactory.transparent_eval_callback import TransparentEvalCallback

                    # Extract tokenizer and processor from kwargs
                    # LlamaFactory passes these via **tokenizer_module
                    tokenizer = kwargs.get('tokenizer')
                    processor = kwargs.get('processor')

                    logging.warning(f"[OCRVL DEBUG] model={model is not None}, tokenizer={tokenizer is not None}, processor={processor is not None}")

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
        import logging
        logging.warning("[OCRVL] ✓ Patched CustomSeq2SeqTrainer.__init__ for TransparentEvalCallback injection")

    except Exception as e:
        import logging
        logging.warning(f"[OCRVL] Failed to patch CustomSeq2SeqTrainer for TransparentEvalCallback: {e}")


# Apply patches on import
_patch_once()
