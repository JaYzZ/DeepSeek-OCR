#!/usr/bin/env python3
"""
Curriculum Learning Callback for LlamaFactory Training (Qwen3VL-specific)

Implements per-epoch curriculum for latent supervision training.
All settings are controlled via environment variables.

Usage:
    export QWEN3VL_CURRICULUM_ENABLE=1

    # Default curriculum (3 stages):
    # export QWEN3VL_CURRICULUM_EPOCHS="0,1,2"           # epoch boundaries
    # export QWEN3VL_CURRICULUM_LOSS_TYPES="pre_think_mse,pre_think_mse,ot+pre_think_mse"
    # export QWEN3VL_CURRICULUM_LATENT_CE="0,1,1"   # 0=off, 1=on

Custom example:
    export QWEN3VL_CURRICULUM_EPOCHS="0,2,4"
    export QWEN3VL_CURRICULUM_LOSS_TYPES="pre_think_mse,mse,ot+pre_think_mse"
    export QWEN3VL_CURRICULUM_LATENT_CE="0,1,1"
"""

import logging
import os
from typing import Dict, Any

import torch.distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from Qwen.llamafactory.runtime_env import get_env

logger = logging.getLogger(__name__)


def _apply_stage_trainability(model, lora_trainable: bool, vae_trainable: bool) -> None:
    """Apply LoRA/VAE trainability outside forward() to keep checkpoint replay stable."""
    if model is None:
        return

    lora_updated = False
    vae_updated = False
    for name, param in model.named_parameters():
        lower_name = name.lower()
        if "lora_" in lower_name and param.requires_grad != lora_trainable:
            param.requires_grad = lora_trainable
            lora_updated = True
        if "latent_vae" in name and param.requires_grad != vae_trainable:
            param.requires_grad = vae_trainable
            vae_updated = True

    if lora_updated:
        state = "trainable" if lora_trainable else "frozen"
        logger.info(f"[Qwen3VL Latent] Set LoRA parameters to {state}")
    if vae_updated:
        state = "trainable" if vae_trainable else "frozen"
        logger.info(f"[Qwen3VL Latent] Set VAE parameters to {state}")


class QwenCurriculumCallback(TrainerCallback):
    """
    Callback that adjusts latent supervision loss configuration per epoch.

    Curriculum settings are read from environment variables:
    - QWEN3VL_CURRICULUM_EPOCHS: epoch boundaries (default: "0,1,2")
    - QWEN3VL_CURRICULUM_LOSS_TYPES: loss types per stage (default: "pre_think_mse,pre_think_mse,ot+pre_think_mse")
    - QWEN3VL_CURRICULUM_LATENT_CE: latent CE per stage (default: "0,1,1")
    """

    def __init__(self):
        env_val = os.environ.get("QWEN3VL_CURRICULUM_ENABLE", "0")
        self.enabled = env_val == "1"

        try:
            dist_initialized = dist.is_initialized()
            if dist_initialized:
                current_rank = dist.get_rank()
                self._is_main = current_rank == 0
            else:
                self._is_main = True
        except Exception:
            self._is_main = True

        if not self.enabled:
            logger.debug("[QwenCurriculum] Disabled")
            return

        epochs_str = get_env("QWEN3VL_CURRICULUM_EPOCHS", "3")
        loss_types_str = get_env(
            "QWEN3VL_CURRICULUM_LOSS_TYPES",
            "ce+mse:0.4+ot:0.4,ce+vae:0.4+mse:0.4+ot:0.4,ce+vae_ce+vae:0.4+mse:0.4+ot:0.4"
        )
        latent_ce_config_str = get_env("QWEN3VL_CURRICULUM_LATENT_CE", "1")

        if "," in epochs_str:
            epochs = [float(x.strip()) for x in epochs_str.split(",") if x.strip()]
        else:
            num_stages = int(epochs_str)
            epochs = list(range(num_stages))

        loss_types = [x.strip() for x in loss_types_str.split(",") if x.strip()]
        if "," in latent_ce_config_str:
            latent_ce = [x.strip() == "1" for x in latent_ce_config_str.split(",") if x.strip()]
        else:
            latent_ce = [latent_ce_config_str.strip() == "1"] * len(epochs)

        lora_trainable_str = get_env("QWEN3VL_CURRICULUM_LORA_TRAINABLE", "1")
        if "," in lora_trainable_str:
            lora_trainable = [x.strip() == "1" for x in lora_trainable_str.split(",") if x.strip()]
        else:
            lora_trainable = [lora_trainable_str.strip() == "1"] * len(epochs)

        vae_trainable_str = get_env("QWEN3VL_CURRICULUM_VAE_TRAINABLE", "1")
        if "," in vae_trainable_str:
            vae_trainable = [x.strip() == "1" for x in vae_trainable_str.split(",") if x.strip()]
        else:
            vae_trainable = [vae_trainable_str.strip() == "1"] * len(epochs)

        aux_source_str = get_env("QWEN3VL_CURRICULUM_AUX_SOURCE", "hidden")
        if "," in aux_source_str:
            aux_source = [x.strip() for x in aux_source_str.split(",") if x.strip()]
        else:
            aux_source = [aux_source_str.strip()] * len(epochs)

        # Validate lengths match
        num_stages = len(epochs)
        if not (len(loss_types) == len(latent_ce) == len(lora_trainable) == len(vae_trainable) == len(aux_source) == num_stages):
            raise ValueError(
                f"[QwenCurriculum] Mismatch in curriculum config: "
                f"epochs={num_stages}, loss_types={len(loss_types)}, "
                f"latent_ce={len(latent_ce)}, lora_trainable={len(lora_trainable)}, "
                f"vae_trainable={len(vae_trainable)}, aux_source={len(aux_source)}"
            )

        # Build stages from env vars
        self.stages = []
        for i in range(num_stages):
            self.stages.append({
                "epoch": epochs[i],
                "loss_type": loss_types[i],
                "latent_ce": latent_ce[i],
                "lora_trainable": lora_trainable[i],
                "vae_trainable": vae_trainable[i],
                "aux_source": aux_source[i],
                "description": f"Stage {i+1}: {loss_types[i]}"
            })

        self.current_stage = 0
        self.last_logged_epoch = None

        if self._is_main:
            logger.info(f"[QwenCurriculum] Enabled with {len(self.stages)} stages:")
            for i, stage in enumerate(self.stages):
                logger.info(
                    f"  Stage {i+1}: epoch>={stage['epoch']}, loss={stage['loss_type']}, "
                    f"lora_trainable={stage['lora_trainable']}, vae_trainable={stage['vae_trainable']}, "
                    f"aux_source={stage['aux_source']}, latent_ce={stage['latent_ce']}"
                )

    def _get_stage_for_epoch(self, epoch: float) -> Dict[str, Any]:
        """Get the appropriate stage for a given epoch."""
        # Find the highest stage whose epoch threshold is met
        current_stage = self.stages[0]
        for stage in self.stages:
            if epoch >= stage["epoch"]:
                current_stage = stage
        return current_stage

    def _apply_stage(self, stage: Dict[str, Any]) -> None:
        """Apply a curriculum stage by setting environment variables."""
        loss_type = stage["loss_type"]
        latent_ce = stage.get("latent_ce", False)
        lora_trainable = stage.get("lora_trainable", True)
        vae_trainable = stage.get("vae_trainable", False)
        aux_source = stage.get("aux_source", "hidden")

        # Always enable latent supervision
        os.environ["QWEN3VL_LATENT_SUPERVISION"] = "1"

        # Set loss type; weighting should stay inside loss_type itself.
        os.environ["QWEN3VL_LOSS_TYPE"] = loss_type

        os.environ["QWEN3VL_LATENT_CE_ACTIVE"] = "1" if latent_ce else "0"

        has_vae = "vae" in loss_type
        has_vae_ce = "vae_ce" in loss_type

        os.environ["QWEN3VL_VAE_TRAINABLE"] = "1" if (has_vae and vae_trainable) else "0"
        os.environ["QWEN3VL_VAE_CE_ENABLE"] = "1" if has_vae_ce else "0"
        os.environ["QWEN3VL_LORA_TRAINABLE"] = "1" if lora_trainable else "0"
        os.environ["QWEN3VL_LATENT_AUX_LOSS_SOURCE"] = aux_source

        if self._is_main:
            logger.info(
                f"[QwenCurriculum] Applied: loss_type={loss_type}, "
                f"has_vae={has_vae}, vae_trainable={vae_trainable}, vae_ce={has_vae_ce}, "
                f"lora_trainable={lora_trainable}, "
                f"aux_source={aux_source}, latent_ce={latent_ce}"
            )

    def _apply_stage_to_model(self, model, stage: Dict[str, Any]) -> None:
        _apply_stage_trainability(
            model=model,
            lora_trainable=stage.get("lora_trainable", True),
            vae_trainable=stage.get("vae_trainable", False),
        )

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Apply initial curriculum stage at training start."""
        if not self.enabled:
            return

        initial_stage = self._get_stage_for_epoch(0)
        self._apply_stage(initial_stage)
        self._apply_stage_to_model(kwargs.get("model"), initial_stage)
        self.current_stage = 0

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ):
        """Check if we need to transition to next stage at epoch start."""
        if not self.enabled:
            return

        epoch = state.epoch
        stage = self._get_stage_for_epoch(epoch)
        stage_idx = self.stages.index(stage)
        if stage_idx > self.current_stage:
            logger.info(
                f"[QwenCurriculum] Transitioning from stage {self.current_stage + 1} "
                f"to {stage_idx + 1} at epoch {epoch:.1f}"
            )
            self._apply_stage(stage)
            self._apply_stage_to_model(kwargs.get("model"), stage)
            self.current_stage = stage_idx

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, Any] = None,
        **kwargs
    ):
        """Log curriculum info periodically."""
        if not self.enabled:
            return
        if hasattr(state, "is_world_process_zero") and not state.is_world_process_zero:
            return

        epoch = state.epoch
        if self.last_logged_epoch is None or (epoch - self.last_logged_epoch) >= 0.5:
            stage = self._get_stage_for_epoch(epoch)
            logger.info(f"[QwenCurriculum] Epoch {epoch:.1f}: stage={stage['description']}, "
                       f"loss_type={stage['loss_type']}")
            self.last_logged_epoch = epoch


def register_curriculum_callback():
    """Register the curriculum callback with llamafactory."""
    # This function can be called from sitecustomize.py or training script
    # to register the callback
    return QwenCurriculumCallback
