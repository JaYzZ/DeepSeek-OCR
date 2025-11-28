"""
Utility Functions for OCRFlow

Helper functions for training, inference, and general use.
"""

import torch
import random
import numpy as np
import os
from typing import Union


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: Union[str, torch.device, None] = None) -> torch.device:
    """
    Get torch device.

    Args:
        device: Device specification (None for auto-detect, "cuda", "cpu", or torch.device)

    Returns:
        torch.device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(device, str):
        device = torch.device(device)

    return device


def count_parameters(model: torch.nn.Module, only_trainable: bool = False) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model
        only_trainable: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_grad_norm(model: torch.nn.Module) -> float:
    """
    Get gradient norm of model parameters.

    Args:
        model: PyTorch model

    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    loss: float,
    save_path: str,
    **kwargs
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        global_step: Global training step
        loss: Current loss value
        save_path: Path to save checkpoint
        **kwargs: Additional data to save
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'loss': loss,
        **kwargs
    }

    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer = None,
    device: str = "cuda",
):
    """
    Load training checkpoint.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint
        optimizer: Optimizer to load state into (optional)
        device: Device to load to

    Returns:
        Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Return metadata
    metadata = {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'loss': checkpoint.get('loss', float('inf')),
    }

    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"  Epoch: {metadata['epoch']}")
    print(f"  Global step: {metadata['global_step']}")
    print(f"  Loss: {metadata['loss']:.4f}")

    return metadata


class AverageMeter:
    """
    Computes and stores the average and current value.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0

    def __str__(self):
        return f"{self.name}: {self.avg:.4f} (current: {self.val:.4f})"


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test set_seed
    set_seed(42)
    r1 = torch.rand(3)
    set_seed(42)
    r2 = torch.rand(3)
    assert torch.allclose(r1, r2), "set_seed not working"
    print("✓ set_seed test passed")

    # Test get_device
    device = get_device()
    print(f"✓ get_device: {device}")

    # Test count_parameters
    model = torch.nn.Linear(10, 5)
    n_params = count_parameters(model)
    expected = 10 * 5 + 5  # weights + bias
    assert n_params == expected, f"count_parameters failed: {n_params} != {expected}"
    print(f"✓ count_parameters: {n_params} params")

    # Test format_time
    assert format_time(45) == "45s"
    assert format_time(90) == "1m 30s"
    assert format_time(3725) == "1h 2m 5s"
    print("✓ format_time test passed")

    # Test AverageMeter
    meter = AverageMeter("loss")
    meter.update(1.0)
    meter.update(2.0)
    meter.update(3.0)
    assert meter.avg == 2.0, "AverageMeter failed"
    print(f"✓ AverageMeter: {meter}")

    print("\nAll utility tests passed!")
