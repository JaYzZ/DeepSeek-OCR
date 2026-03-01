"""VAE Save Callback for Qwen3VL training."""

import os
import logging
from collections import deque
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainingArguments

logger = logging.getLogger(__name__)


def _is_rank_zero() -> bool:
    """Check if current process is rank 0 or distributed is not initialized."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def _resolve_latent_vae(model):
    """Find latent_vae through common wrapper layers (FSDP/PEFT/Trainer wrappers)."""
    if model is None:
        return None

    queue = deque([model])
    seen = set()
    while queue:
        node = queue.popleft()
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)

        vae = getattr(node, "latent_vae", None)
        if vae is not None:
            return vae

        for attr in ("module", "model", "base_model", "wrapped_module", "_fsdp_wrapped_module"):
            child = getattr(node, attr, None)
            if child is not None:
                queue.append(child)

    return None


def save_vae_checkpoint(model, output_dir: str) -> None:
    """Save VAE weights separately from the model checkpoint.

    Only rank 0 should save to avoid duplicate files.

    Args:
        model: The model with latent_vae module
        output_dir: Directory to save VAE checkpoint
    """
    vae_module = _resolve_latent_vae(model)
    if vae_module is None:
        return

    import safetensors.torch

    # IMPORTANT: call state_dict() on all ranks.
    # Under FSDP this may run collectives; rank-zero-only can deadlock.
    vae_state_dict = vae_module.state_dict()

    if not _is_rank_zero():
        return

    # Canonical layout: <checkpoint_dir>/vae.safetensors
    os.makedirs(output_dir, exist_ok=True)

    cpu_state_dict = {k: v.detach().cpu().contiguous() for k, v in vae_state_dict.items()}

    # Save VAE weights
    vae_path = os.path.join(output_dir, "vae.safetensors")
    safetensors.torch.save_file(cpu_state_dict, vae_path)
    logger.info(f"[Qwen3VL Latent] Saved VAE checkpoint to {vae_path}")


class VAESaveCallback(TrainerCallback):
    """Callback to save VAE weights at model save intervals.

    Automatically saves VAE weights when the model is checkpointed.
    """

    def __init__(self, output_dir: str, save_steps: int = 500):
        self.output_dir = output_dir
        self.save_steps = save_steps

    def on_save(self, args: "TrainingArguments", state, control, model=None, **kwargs):
        """Called when model is saved."""
        if model is None:
            return

        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        save_vae_checkpoint(model, checkpoint_dir)
