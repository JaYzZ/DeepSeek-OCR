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
    # export QWEN3VL_CURRICULUM_LATENT_STEP_CE="0,1,1"   # 0=off, 1=on

Custom example:
    export QWEN3VL_CURRICULUM_EPOCHS="0,2,4"
    export QWEN3VL_CURRICULUM_LOSS_TYPES="pre_think_mse,mse,ot+pre_think_mse"
    export QWEN3VL_CURRICULUM_LATENT_STEP_CE="0,1,1"
"""

import logging
import os
from typing import Dict, Any

import torch
import torch.distributed as dist
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class QwenCurriculumCallback(TrainerCallback):
    """
    Callback that adjusts latent supervision loss configuration per epoch.

    Curriculum settings are read from environment variables:
    - QWEN3VL_CURRICULUM_EPOCHS: epoch boundaries (default: "0,1,2")
    - QWEN3VL_CURRICULUM_LOSS_TYPES: loss types per stage (default: "pre_think_mse,pre_think_mse,ot+pre_think_mse")
    - QWEN3VL_CURRICULUM_LATENT_STEP_CE: latent_step CE per stage (default: "0,1,1")
    """

    def __init__(self):
        self.enabled = os.environ.get("QWEN3VL_CURRICULUM_ENABLE", "0") == "1"
        try:
            self._is_main = (not dist.is_initialized()) or dist.get_rank() == 0
        except Exception:
            self._is_main = True

        if not self.enabled:
            logger.debug("[QwenCurriculum] Disabled - set QWEN3VL_CURRICULUM_ENABLE=1 to enable")
            return

        # Parse curriculum configuration from environment variables
        epochs_str = os.environ.get("QWEN3VL_CURRICULUM_EPOCHS", "0,1,2")
        loss_types_str = os.environ.get(
            "QWEN3VL_CURRICULUM_LOSS_TYPES",
            "pre_think_mse,pre_think_mse,ot+pre_think_mse"
        )
        latent_step_ce_str = os.environ.get("QWEN3VL_CURRICULUM_LATENT_STEP_CE", "0,1,1")

        epochs = [float(x.strip()) for x in epochs_str.split(",")]
        loss_types = [x.strip() for x in loss_types_str.split(",")]
        latent_step_ce = [x.strip() == "1" for x in latent_step_ce_str.split(",")]

        # Validate lengths match
        num_stages = len(epochs)
        if not (len(loss_types) == len(latent_step_ce) == num_stages):
            raise ValueError(
                f"[QwenCurriculum] Mismatch in curriculum config: "
                f"epochs={num_stages}, loss_types={len(loss_types)}, "
                f"latent_step_ce={len(latent_step_ce)}"
            )

        # Build stages from env vars
        self.stages = []
        for i in range(num_stages):
            self.stages.append({
                "epoch": epochs[i],
                "loss_type": loss_types[i],
                "latent_step_ce": latent_step_ce[i],
                "description": f"Stage {i+1}: {loss_types[i]}"
            })

        self.current_stage = 0
        self.last_logged_epoch = None

        if self._is_main:
            logger.info(f"[QwenCurriculum] Enabled with {len(self.stages)} stages:")
            for i, stage in enumerate(self.stages):
                logger.info(
                    f"  Stage {i+1}: epoch>={stage['epoch']}, loss={stage['loss_type']}, "
                    f"latent_step_ce={stage['latent_step_ce']}"
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
        latent_step_ce = stage.get("latent_step_ce", False)

        # Always enable latent supervision
        os.environ["QWEN3VL_LATENT_SUPERVISION"] = "1"

        # Set loss type; weighting should stay inside loss_type itself.
        os.environ["QWEN3VL_LOSS_TYPE"] = loss_type

        # Control latent-step CE masking in the main process.
        os.environ["QWEN3VL_LATENT_STEP_CE_ACTIVE"] = "1" if latent_step_ce else "0"

        if self._is_main:
            logger.info(
                f"[QwenCurriculum] Applied: loss_type={loss_type}, "
                f"latent_step_ce={latent_step_ce}"
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

        # Apply on every rank: curriculum settings are process-local env vars.
        # Apply initial stage (epoch 0)
        initial_stage = self._get_stage_for_epoch(0)
        self._apply_stage(initial_stage)
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

        # Check if we need to transition
        stage_idx = self.stages.index(stage)
        if stage_idx > self.current_stage:
            logger.info(f"[QwenCurriculum] Transitioning from stage {self.current_stage+1} to {stage_idx+1} "
                       f"at epoch {epoch:.1f}")
            self._apply_stage(stage)
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
