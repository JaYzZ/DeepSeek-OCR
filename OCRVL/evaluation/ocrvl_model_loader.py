#!/usr/bin/env python3
"""
OCRVL Model Loader for Benchmarks

This module provides utilities to load OCRVL-trained models for evaluation.
It wraps the standard Qwen3-VL model loading with OCRVL connector loading.

Usage in inference scripts:
    from ocrvl_model_loader import load_ocrvl_model

    # Instead of:
    # model = AutoModelForCausalLM.from_pretrained(model_path)

    # Use:
    model = load_ocrvl_model(
        base_model_path=model_path,
        checkpoint_path=os.environ.get('OCRVL_CHECKPOINT_PATH')
    )
"""

import os
import logging
import torch
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_ocrvl_model(
    base_model_path: str,
    checkpoint_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
    trust_remote_code: bool = True,
    **kwargs
):
    """
    Load OCRVL model with trained connectors.

    Args:
        base_model_path: Path to base Qwen3-VL model
        checkpoint_path: Path to OCRVL checkpoint directory (with connectors.pt)
                        If None, checks OCRVL_CHECKPOINT_PATH environment variable
        device_map: Device map for model loading
        torch_dtype: Model dtype
        trust_remote_code: Whether to trust remote code
        **kwargs: Additional arguments passed to from_pretrained

    Returns:
        Loaded OCRQwen3VL model with connectors
    """
    # Check if OCRVL mode is enabled
    ocrvl_mode = os.environ.get('OCRVL_MODE', '0') == '1'

    if not ocrvl_mode:
        # Standard Qwen3-VL loading
        logger.info(f"Loading standard Qwen3-VL model from {base_model_path}")
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            base_model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            **kwargs
        )

    # OCRVL mode: load with connectors
    if checkpoint_path is None:
        checkpoint_path = os.environ.get('OCRVL_CHECKPOINT_PATH')

    if not checkpoint_path:
        raise ValueError(
            "OCRVL_MODE=1 but no checkpoint path provided. "
            "Set OCRVL_CHECKPOINT_PATH environment variable or pass checkpoint_path argument."
        )

    checkpoint_path = Path(checkpoint_path)
    connectors_path = checkpoint_path / "connectors.pt"

    if not connectors_path.exists():
        raise FileNotFoundError(f"Connectors not found: {connectors_path}")

    logger.info(f"Loading OCRVL model:")
    logger.info(f"  Base model: {base_model_path}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Connectors: {connectors_path}")

    # Import OCRVL model class
    try:
        import sys
        ocrvl_root = Path(__file__).parent.parent
        if str(ocrvl_root) not in sys.path:
            sys.path.insert(0, str(ocrvl_root))

        from OCRVL.model.language_model.ocr_qwen3_vl import OCRQwen3VLForConditionalGeneration
    except ImportError as e:
        raise ImportError(
            f"Failed to import OCRVL model class. Make sure OCRVL is in PYTHONPATH. Error: {e}"
        )

    # Load base model with OCRVL architecture
    logger.info("Loading OCRQwen3VL model...")
    model = OCRQwen3VLForConditionalGeneration.from_pretrained(
        base_model_path,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        **kwargs
    )

    # Load connector weights
    logger.info(f"Loading connectors from {connectors_path}...")
    state = torch.load(connectors_path, map_location='cpu')

    if 'ocr_connector' in state and hasattr(model.model, 'ocr_connector'):
        model.model.ocr_connector.load_state_dict(state['ocr_connector'])
        logger.info("  ✓ Loaded ocr_connector")

    if 'deepstack_connectors' in state and hasattr(model.model, '_ocr_deepstack_connectors'):
        for k, v in state['deepstack_connectors'].items():
            if k in model.model._ocr_deepstack_connectors:
                model.model._ocr_deepstack_connectors[k].load_state_dict(v)
        logger.info(f"  ✓ Loaded {len(state['deepstack_connectors'])} deepstack connectors")

    logger.info("✓ OCRVL model loaded successfully")

    return model


def is_ocrvl_mode() -> bool:
    """Check if OCRVL mode is enabled via environment variable."""
    return os.environ.get('OCRVL_MODE', '0') == '1'


def get_ocrvl_checkpoint_info():
    """Get information about the current OCRVL checkpoint."""
    checkpoint_path = os.environ.get('OCRVL_CHECKPOINT_PATH')
    if not checkpoint_path:
        return None

    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.json"

    info = {
        'checkpoint_path': str(checkpoint_path),
        'connectors_path': str(checkpoint_path / "connectors.pt"),
        'config_path': str(config_path) if config_path.exists() else None,
    }

    # Try to load config
    if info['config_path']:
        try:
            import json
            with open(info['config_path']) as f:
                config = json.load(f)
                info['config'] = config
        except Exception:
            pass

    return info
