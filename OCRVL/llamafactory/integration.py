"""
OCRVL-specific LlamaFactory patches.

This module contains all OCRVL-specific patches that were previously in sitecustomize.py.
These patches are applied when using OCRVL models.

Patches:
1. OCRQwen3VLConfig registration
2. DPSKOCRImageProcessor patching
3. OCRVL template registration
4. FSDP+PEFT incompatibility patch
5. Model parameter dtype conversion
6. LLM freeze patch (OCRVL_FREEZE_LLM)
7. Connector training patch
8. Accelerate FSDP + PEFT saving patch
"""

from __future__ import annotations

import functools
import logging
import os

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq, AutoProcessor

logger = logging.getLogger(__name__)


def apply_ocrvl_patches(logger) -> None:
    """Apply all OCRVL-specific LlamaFactory patches.

    This function should be called from sitecustomize.py after importing.
    """
    # PID-based guard for multiprocessing
    pid = str(os.getpid())
    if os.environ.get("OCRVL_LLAMAFACTORY_PATCHED_PID", "") == pid:
        return
    os.environ["OCRVL_LLAMAFACTORY_PATCHED_PID"] = pid

    # 1) Register OCRQwen3VLConfig
    _register_ocrvl_config(logger)

    # 2) Patch AutoProcessor for DPSKOCRImageProcessor
    _patch_processor_for_dpsk(logger)

    # 3) Register OCRVL templates
    _register_ocrvl_templates(logger)

    # 4) Patch FSDP+PEFT incompatibility
    _patch_fsdp_peft(logger)

    # 5) Patch Trainer for dtype conversion
    _patch_trainer_for_dtype(logger)

    # 6) Patch Trainer for LLM freezing
    _patch_trainer_for_llm_freeze(logger)

    # 7) Patch Trainer for connector training
    _patch_trainer_for_connectors(logger)

    # 8) Patch Accelerate FSDP save
    _patch_accelerate_fsdp_save(logger)

    logger.info("[OCRVL] ✓ All OCRVL patches applied")


def _register_ocrvl_config(logger) -> None:
    """Register OCRQwen3VLConfig to replace Qwen3VLConfig."""
    try:
        from OCRVL.model.language_model.ocr_qwen3_vl import OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration

        AutoModelForCausalLM.register(OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration, exist_ok=True)
        AutoModelForVision2Seq.register(OCRQwen3VLConfig, OCRQwen3VLForConditionalGeneration, exist_ok=True)
        logger.debug("[OCRVL] Registered OCRQwen3VLConfig")
    except ImportError as e:
        logger.warning(f"[OCRVL] Could not register OCRQwen3VLConfig: {e}")
        return

    # Patch AutoConfig.from_pretrained to convert Qwen3VLConfig to OCRQwen3VLConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    orig_autoconfig_from_pretrained = AutoConfig.from_pretrained

    @classmethod  # type: ignore[misc]
    def wrapped_autoconfig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = orig_autoconfig_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        if isinstance(config, Qwen3VLConfig):
            should_wrap = os.environ.get("OCRVL_ENABLE_WRAPPER", "0") == "1"
            if not should_wrap:
                model_path_lower = str(pretrained_model_name_or_path).lower()
                should_wrap = any(keyword in model_path_lower for keyword in ["ocrvl", "dpsk", "ocr_"])

            if should_wrap:
                ocr_config_dict = config.to_dict()
                ocr_config_dict.pop("model_type", None)
                logger.warning(f"[OCRVL] Converting {pretrained_model_name_or_path} to OCRQwen3VLConfig")
                return OCRQwen3VLConfig(**ocr_config_dict)

            logger.info(f"[OCRVL] Using native Qwen3VLConfig for {pretrained_model_name_or_path}")

        return config

    AutoConfig.from_pretrained = wrapped_autoconfig_from_pretrained
    logger.debug("[OCRVL] Patched AutoConfig.from_pretrained for OCRVL")


def _patch_processor_for_dpsk(logger) -> None:
    """Patch AutoProcessor to use DPSKOCRImageProcessor."""
    from OCRVL.llamafactory.dpsk_ocr_image_processor import DPSKOCRImageProcessor

    orig_processor_from_pretrained = AutoProcessor.from_pretrained

    @classmethod  # type: ignore[misc]
    @functools.wraps(orig_processor_from_pretrained)
    def wrapped_processor_from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs):
        processor = orig_processor_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        if processor is None:
            return processor

        if processor.__class__.__name__ == "Qwen3VLProcessor":
            should_wrap = os.environ.get("OCRVL_ENABLE_WRAPPER", "0") == "1"
            if not should_wrap:
                model_path_lower = str(pretrained_model_name_or_path).lower()
                should_wrap = any(keyword in model_path_lower for keyword in ["ocrvl", "dpsk", "ocr_"])

            if should_wrap:
                merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)
                processor.image_processor = DPSKOCRImageProcessor(merge_size=merge_size)
                logger.warning(f"[OCRVL] Using DPSKOCRImageProcessor for {pretrained_model_name_or_path}")

        return processor

    AutoProcessor.from_pretrained = wrapped_processor_from_pretrained
    logger.debug("[OCRVL] Patched AutoProcessor for DPSKOCRImageProcessor")


def _register_ocrvl_templates(logger) -> None:
    """Register OCRVL templates."""
    import traceback
    try:
        import OCRVL.llamafactory as ocrvl_llamafactory
        ocrvl_llamafactory.register_ocrvl_templates()
        logger.warning("[OCRVL] ✓ Registered OCRVL templates")
    except Exception as e:
        error_str = str(e).lower()
        if "already registered" not in error_str and "already exists" not in error_str:
            logger.error(f"[OCRVL] Failed to register OCRVL templates: {e}")
            logger.error(traceback.format_exc())
            raise
        else:
            logger.debug(f"[OCRVL] Templates already registered (OK on non-rank-0)")


def _patch_fsdp_peft(logger) -> None:
    """Patch FSDP+PEFT incompatibility for custom models."""
    try:
        from peft.utils.other import fsdp_auto_wrap_policy as original_fsdp_auto_wrap_policy
    except ImportError:
        return

    try:
        @functools.wraps(original_fsdp_auto_wrap_policy)
        def patched_fsdp_auto_wrap_policy(model):
            try:
                return original_fsdp_auto_wrap_policy(model)
            except Exception as e:
                if "Could not find the transformer layer class to wrap in the model" in str(e):
                    logger.warning(
                        "[OCRVL] FSDP auto-wrap policy could not find transformer layer class. "
                        "Returning None to use config-based wrapping policy."
                    )
                    return None
                raise

        import peft.utils.other
        peft.utils.other.fsdp_auto_wrap_policy = patched_fsdp_auto_wrap_policy
        import peft.utils.other as peft_other_module
        if hasattr(peft_other_module, '__dict__'):
            peft_other_module.__dict__['fsdp_auto_wrap_policy'] = patched_fsdp_auto_wrap_policy
        logger.debug("[OCRVL] Patched FSDP+PEFT auto-wrap policy")
    except Exception as e:
        logger.warning(f"[OCRVL] Failed to patch FSDP auto-wrap policy: {e}")


def _patch_trainer_for_dtype(logger) -> None:
    """Patch Trainer._inner_training_loop to convert params to bfloat16."""
    from transformers import Trainer

    try:
        original_inner_training_loop = Trainer._inner_training_loop

        @functools.wraps(original_inner_training_loop)
        def patched_inner_training_loop(self, *args, **kwargs):
            if hasattr(self.model, 'modules'):
                converted = 0
                for module in self.model.modules():
                    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Conv1d)):
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
                    logger.warning(f"[OCRVL] Converted {converted} params to bfloat16 (pre-iteration)")

            return original_inner_training_loop(self, *args, **kwargs)

        Trainer._inner_training_loop = patched_inner_training_loop
        logger.debug("[OCRVL] Patched Trainer._inner_training_loop for dtype conversion")
    except Exception as e:
        logger.warning(f"[OCRVL] Failed to patch trainer _inner_training_loop: {e}")


def _patch_trainer_for_llm_freeze(logger) -> None:
    """Patch Trainer.create_optimizer to freeze LLM when OCRVL_FREEZE_LLM=1."""
    from transformers import Trainer

    if os.environ.get("OCRVL_FREEZE_LLM", "0") != "1":
        return

    try:
        original_create_optimizer = Trainer.create_optimizer

        @functools.wraps(original_create_optimizer)
        def patched_create_optimizer_freeze_llm(self):
            if hasattr(self.model, "model"):
                frozen_count = 0
                for name, param in self.model.named_parameters():
                    if "language_model" in name and "ocr" not in name:
                        if param.requires_grad:
                            param.requires_grad = False
                            frozen_count += 1

                if frozen_count > 0:
                    logger.warning(f"[OCRVL] Froze {frozen_count} LLM parameters (connector-only training)")

            return original_create_optimizer(self)

        Trainer.create_optimizer = patched_create_optimizer_freeze_llm
        logger.debug("[OCRVL] Patched Trainer.create_optimizer for LLM freezing")
    except Exception as e:
        logger.warning(f"[OCRVL] Failed to patch trainer for LLM freezing: {e}")


def _patch_trainer_for_connectors(logger) -> None:
    """Patch Trainer.create_optimizer to ensure OCR connectors are always trainable."""
    from transformers import Trainer

    try:
        original_create_optimizer = Trainer.create_optimizer

        @functools.wraps(original_create_optimizer)
        def patched_create_optimizer_ensure_connectors(self):
            # Step 1: Ensure connector gradients are enabled
            connector_params = []
            for name, param in self.model.named_parameters():
                if 'ocr_connector' in name or 'deepstack_connector' in name:
                    if not param.requires_grad:
                        param.requires_grad = True
                        logger.warning(f"[OCRVL] Re-enabled gradient for connector: {name}")
                    if param.requires_grad:
                        connector_params.append((name, param))

            # Step 2: Create optimizer
            optimizer = original_create_optimizer(self)

            # Step 3: Verify connectors are in optimizer
            if connector_params:
                existing_param_ids = {id(p) for group in optimizer.param_groups for p in group.get("params", [])}
                missing_params = [(n, p) for n, p in connector_params if id(p) not in existing_param_ids]

                if missing_params:
                    param_names = [n for n, p in missing_params]
                    logger.warning(f"[OCRVL] Adding {len(missing_params)} connector params to optimizer: {param_names}")
                    optimizer.add_param_group({"params": [p for n, p in missing_params]})
                else:
                    logger.debug(f"[OCRVL] All {len(connector_params)} connector params already in optimizer")

            return optimizer

        Trainer.create_optimizer = patched_create_optimizer_ensure_connectors
        logger.debug("[OCRVL] Patched Trainer.create_optimizer for connector training")
    except Exception as e:
        logger.warning(f"[OCRVL] Failed to patch trainer for connector training: {e}")


def _patch_accelerate_fsdp_save(logger) -> None:
    """Patch Accelerate FSDP + PEFT adapter-only saving on non-rank0."""
    try:
        from accelerate.utils import fsdp_utils as accelerate_fsdp_utils
    except ImportError:
        return

    try:
        original_get_model_state_dict = accelerate_fsdp_utils._get_model_state_dict

        @functools.wraps(original_get_model_state_dict)
        def patched_get_model_state_dict(model, adapter_only=False, sd_options=None):
            if adapter_only:
                try:
                    if torch.distributed.is_available() and torch.distributed.is_initialized():
                        if torch.distributed.get_rank() != 0:
                            try:
                                _ = model.state_dict()
                            except Exception:
                                pass
                            return {}
                except Exception:
                    pass

            return original_get_model_state_dict(model, adapter_only=adapter_only, sd_options=sd_options)

        accelerate_fsdp_utils._get_model_state_dict = patched_get_model_state_dict
        logger.debug("[OCRVL] Patched accelerate FSDP save for PEFT")
    except Exception as e:
        logger.warning(f"[OCRVL] Failed to patch accelerate FSDP save: {e}")
