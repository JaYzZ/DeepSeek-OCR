"""Vision encoders for Qwen models.

Provides standalone vision encoders extracted from Qwen VL models.
Supports:
- Qwen3-VL: 406M parameter ViT encoder
- Qwen2.5-VL: 675M parameter ViT encoder
"""

try:
    from .qwen3vl_encoder import Qwen3VLEncoder, Qwen3VLEncoderOutput
except ImportError:
    Qwen3VLEncoder = None
    Qwen3VLEncoderOutput = None

try:
    from .qwen25vl_encoder import Qwen25VLEncoder, Qwen25VLEncoderOutput
except ImportError:
    Qwen25VLEncoder = None
    Qwen25VLEncoderOutput = None

__all__ = [
    "Qwen3VLEncoder",
    "Qwen3VLEncoderOutput",
    "Qwen25VLEncoder",
    "Qwen25VLEncoderOutput",
]
