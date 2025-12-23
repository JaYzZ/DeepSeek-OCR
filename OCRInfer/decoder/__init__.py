"""vLLM decoder module with embedding support"""

# Import embedding_patch first to apply monkey-patches
from . import embedding_patch  # noqa: F401
from .dpsk_ocr_decoder import VLLMEmbeddingDecoder

__all__ = ["VLLMEmbeddingDecoder", "embedding_patch"]
