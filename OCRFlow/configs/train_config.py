"""
OCRFlow Training Configuration

Defines training hyperparameters and settings for rectified flow training.
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class TrainConfig:
    """Training configuration for OCRFlow"""

    # ============================================================================
    # Data Configuration
    # ============================================================================
    train_data_path: str = ""  # Path to training data (images + captions)
    val_data_path: Optional[str] = None  # Path to validation data

    # Data preprocessing
    image_size: int = 640  # Image size for DeepSeek OCR
    base_size: int = 1024  # Base size for global view
    crop_mode: bool = True  # Use dynamic cropping

    # Data loading
    batch_size: int = 16  # Per-GPU batch size
    num_workers: int = 8  # DataLoader workers
    pin_memory: bool = True
    prefetch_factor: int = 2

    # ============================================================================
    # Model Configuration
    # ============================================================================
    model_size: str = "small"  # Model size: tiny, small, base, large, xl
    num_image_tokens: int = 1600  # Depends on image_size and patch_size
    token_dim: int = 1280  # DeepSeek OCR projection dimension

    # Text encoders
    use_dual_text_encoders: bool = True
    t5_model: str = "google/t5-v1_1-base"
    clip_model: str = "openai/clip-vit-large-patch14"
    freeze_text_encoders: bool = True  # Keep text encoders frozen

    # DeepSeek OCR vision encoder
    deepseek_model_path: str = "deepseek-ai/DeepSeek-OCR"
    freeze_vision_encoder: bool = True  # Keep vision encoder frozen

    # ============================================================================
    # Training Hyperparameters
    # ============================================================================
    num_epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, linear, constant
    warmup_steps: int = 1000
    warmup_ratio: float = 0.05  # If warmup_steps not set, use this ratio

    # Optimizer
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # ============================================================================
    # Rectified Flow Training
    # ============================================================================
    loss_type: str = "huber"  # huber, mse, l1
    huber_delta: float = 1.0

    # Classifier-free guidance training
    cfg_dropout_prob: float = 0.1  # Probability of dropping text conditioning

    # Timestep sampling
    timestep_sampling: str = "uniform"  # uniform, logit_normal

    # ============================================================================
    # Precision and Performance
    # ============================================================================
    mixed_precision: str = "bf16"  # bf16, fp16, no
    gradient_accumulation_steps: int = 1

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    local_rank: int = 0
    dist_backend: str = "nccl"

    # ============================================================================
    # Checkpointing and Logging
    # ============================================================================
    output_dir: str = "./outputs/ocrflow"
    save_every: int = 1000  # Save checkpoint every N steps
    save_epochs: bool = True  # Save at end of each epoch
    keep_last_n_checkpoints: int = 3

    # Validation
    val_every: int = 1000  # Validate every N steps
    val_samples: int = 4  # Number of samples to generate during validation
    val_prompts: Optional[List[str]] = None  # Custom validation prompts

    # Logging
    log_every: int = 50
    use_wandb: bool = False
    wandb_project: str = "ocrflow"
    wandb_name: Optional[str] = None

    # ============================================================================
    # Inference Configuration (for validation)
    # ============================================================================
    num_sampling_steps: int = 20  # Euler sampling steps
    cfg_scale: float = 3.0  # Classifier-free guidance scale

    # ============================================================================
    # Miscellaneous
    # ============================================================================
    seed: int = 42
    device: str = "cuda"
    compile_model: bool = False  # Use torch.compile for speedup

    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Set warmup steps if not provided
        if self.warmup_steps == 0 and self.warmup_ratio > 0:
            # Will be calculated based on total steps during training
            pass

        # Set default validation prompts if not provided
        if self.val_prompts is None:
            self.val_prompts = [
                "<image>\n<|grounding|>Convert the document to markdown.",
                "<image>\n<|grounding|>OCR this image.",
                "<image>\nFree OCR.",
                "<image>\nDescribe this image in detail.",
            ]

        # Validate model size
        valid_sizes = ["tiny", "small", "base", "large", "xl"]
        if self.model_size not in valid_sizes:
            raise ValueError(
                f"model_size must be one of {valid_sizes}, got {self.model_size}"
            )

        # Validate loss type
        valid_losses = ["huber", "mse", "l1"]
        if self.loss_type not in valid_losses:
            raise ValueError(
                f"loss_type must be one of {valid_losses}, got {self.loss_type}"
            )


def get_default_train_config(**kwargs) -> TrainConfig:
    """
    Get default training configuration with optional overrides.

    Args:
        **kwargs: Configuration overrides

    Returns:
        TrainConfig instance

    Example:
        >>> config = get_default_train_config(
        ...     model_size="base",
        ...     batch_size=32,
        ...     learning_rate=2e-4
        ... )
    """
    return TrainConfig(**kwargs)


# Predefined training configurations for different scenarios
TRAIN_CONFIGS = {
    "quick_test": {
        "model_size": "tiny",
        "batch_size": 4,
        "num_epochs": 1,
        "save_every": 100,
        "val_every": 100,
        "log_every": 10,
    },
    "small_dataset": {
        "model_size": "small",
        "batch_size": 16,
        "num_epochs": 20,
        "learning_rate": 1e-4,
        "save_every": 500,
        "val_every": 500,
    },
    "production": {
        "model_size": "base",
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 5e-5,
        "distributed": True,
        "use_wandb": True,
    },
}


def get_preset_config(preset: str, **kwargs) -> TrainConfig:
    """
    Get a preset training configuration.

    Args:
        preset: One of ["quick_test", "small_dataset", "production"]
        **kwargs: Additional overrides

    Returns:
        TrainConfig instance
    """
    if preset not in TRAIN_CONFIGS:
        raise ValueError(
            f"preset must be one of {list(TRAIN_CONFIGS.keys())}, got {preset}"
        )

    config_dict = TRAIN_CONFIGS[preset].copy()
    config_dict.update(kwargs)

    return TrainConfig(**config_dict)


if __name__ == "__main__":
    # Test configurations
    print("Default config:")
    config = get_default_train_config()
    print(f"  Model size: {config.model_size}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")

    print("\nQuick test config:")
    test_config = get_preset_config("quick_test")
    print(f"  Model size: {test_config.model_size}")
    print(f"  Epochs: {test_config.num_epochs}")
    print(f"  Val every: {test_config.val_every}")
