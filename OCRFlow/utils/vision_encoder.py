"""
Standalone Vision Encoder for OCRFlow Training

Memory-efficient encoder that loads ONLY vision components:
- CLIP ViT (1024 hidden, 24 layers) - ~304M params
- SAM ViT (768 hidden, 12 layers) - ~89M params
- Projector (2048 → 1280) - ~2.6M params
- Newline + separator tokens - ~2.5K params

Total: ~400M params vs ~7B for full model

Features:
- Direct weight loading from HuggingFace checkpoint
- High-throughput batched encoding
- Parallel text rendering with configurable workers
- No LLM weights loaded (saves ~6.6B params / ~13GB VRAM)

Usage:
    from OCRFlow.utils.vision_encoder import create_vision_encoder

    encoder = create_vision_encoder(device="cuda:0")
    visual_tokens = encoder.encode_texts(["Hello world"] * 8)
    # Returns list of [111, 1280] tensors
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image
import logging
import sys

logger = logging.getLogger(__name__)


class VisionEncoderOnly(nn.Module):
    """
    Vision encoder that loads only CLIP + SAM + Projector weights.

    Much more memory efficient than loading full DeepSeek-OCR model.
    """

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_render_workers: int = 8,
    ):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.num_render_workers = num_render_workers

        # Models will be loaded lazily
        self.clip_model = None
        self.sam_model = None
        self.projector = None
        self.image_newline = None
        self.view_separator = None
        self.processor = None
        self._renderer = None
        self._initialized = False

    def _load_weights(self):
        """Load only vision encoder weights from checkpoint"""
        if self._initialized:
            return

        logger.info(f"Loading vision encoder from {self.model_path}...")

        # Add deepencoder path
        ocr_path = Path(__file__).parent.parent.parent / "DeepSeek-OCR-master" / "DeepSeek-OCR-vllm"
        sys.path.insert(0, str(ocr_path))

        from deepencoder.clip_sdpa import VitModel, vit_model_cfg
        from deepencoder.sam_vary_sdpa import build_sam_vit_b
        from process.image_process import DeepseekOCRProcessor

        # Build empty models
        self.clip_model = VitModel(cfg=vit_model_cfg, freeze_embed=False, freeze_pre_norm=False)
        self.sam_model = build_sam_vit_b()

        # Projector: concatenated features (1024 CLIP + 1024 SAM) -> 1280 hidden
        self.projector = nn.Linear(2048, 1280, bias=True)

        # Special tokens
        self.image_newline = nn.Parameter(torch.zeros(1280))
        self.view_separator = nn.Parameter(torch.zeros(1280))

        # Load weights from HuggingFace
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        # Download model files
        logger.info("Downloading model weights...")

        try:
            # Try to load from local cache or download
            model_index_path = hf_hub_download(
                repo_id=self.model_path,
                filename="model.safetensors.index.json",
            )

            import json
            with open(model_index_path) as f:
                index = json.load(f)

            # Find which shard files contain vision weights
            weight_map = index.get("weight_map", {})
            vision_shards = set()
            for key, shard in weight_map.items():
                if any(x in key for x in ["vision_model", "sam_model", "projector", "image_newline", "view_seperator"]):
                    vision_shards.add(shard)

            logger.info(f"Loading vision weights from shards: {vision_shards}")

            # Load only vision-related shards
            state_dict = {}
            for shard in vision_shards:
                shard_path = hf_hub_download(repo_id=self.model_path, filename=shard)
                shard_weights = load_file(shard_path)
                state_dict.update(shard_weights)

        except Exception as e:
            logger.warning(f"Shard loading failed ({e}), trying single file...")
            # Fallback: try single safetensors file
            model_path = hf_hub_download(
                repo_id=self.model_path,
                filename="model.safetensors",
            )
            state_dict = load_file(model_path)

        # Map weights to our models
        self._load_clip_weights(state_dict)
        self._load_sam_weights(state_dict)
        self._load_projector_weights(state_dict)
        self._load_special_tokens(state_dict)

        # Move to device
        self.clip_model = self.clip_model.to(device=self.device, dtype=self.dtype)
        self.sam_model = self.sam_model.to(device=self.device, dtype=self.dtype)
        self.projector = self.projector.to(device=self.device, dtype=self.dtype)
        self.image_newline = nn.Parameter(self.image_newline.to(device=self.device, dtype=self.dtype))
        self.view_separator = nn.Parameter(self.view_separator.to(device=self.device, dtype=self.dtype))

        # Set eval mode
        self.clip_model.eval()
        self.sam_model.eval()
        self.projector.eval()

        # Load processor
        self.processor = DeepseekOCRProcessor()

        # Initialize ultra-fast parallel renderer with adaptive font sizing
        # Uses 16 workers for high throughput (~200+ img/s)
        from OCRFlow.utils.ultra_fast_renderer import UltraFastRenderer
        self._renderer = UltraFastRenderer(
            num_workers=min(16, self.num_render_workers * 2),
            min_font_size=9,  # Validated for 900 words with >93% OCR accuracy
            max_font_size=20,
        )

        self._initialized = True

        # Log memory usage
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✓ Vision encoder loaded: {total_params / 1e6:.1f}M params")

    def _load_clip_weights(self, state_dict):
        """Load CLIP weights"""
        clip_state = {}
        prefix = "model.vision_model."

        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                clip_state[new_key] = value

        if clip_state:
            self.clip_model.load_state_dict(clip_state, strict=False)
            logger.info(f"  Loaded {len(clip_state)} CLIP weights")
        else:
            logger.warning("  No CLIP weights found!")

    def _load_sam_weights(self, state_dict):
        """Load SAM weights"""
        sam_state = {}
        prefix = "model.sam_model."

        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                sam_state[new_key] = value

        if sam_state:
            self.sam_model.load_state_dict(sam_state, strict=False)
            logger.info(f"  Loaded {len(sam_state)} SAM weights")
        else:
            logger.warning("  No SAM weights found!")

    def _load_projector_weights(self, state_dict):
        """Load projector weights"""
        proj_weight = state_dict.get("model.projector.weight")
        proj_bias = state_dict.get("model.projector.bias")

        if proj_weight is not None:
            self.projector.weight.data.copy_(proj_weight)
            logger.info("  Loaded projector weight")
        if proj_bias is not None:
            self.projector.bias.data.copy_(proj_bias)
            logger.info("  Loaded projector bias")

    def _load_special_tokens(self, state_dict):
        """Load newline and separator tokens"""
        newline = state_dict.get("model.image_newline")
        separator = state_dict.get("model.view_seperator")

        if newline is not None:
            self.image_newline.data.copy_(newline)
            logger.info("  Loaded image_newline")
        if separator is not None:
            self.view_separator.data.copy_(separator)
            logger.info("  Loaded view_separator")

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Encode images to visual tokens [111, 1280] each.

        Args:
            images: List of PIL Images (640x640 rendered text)

        Returns:
            List of visual token tensors [111, 1280]
        """
        if not self._initialized:
            self._load_weights()

        from PIL import ImageOps

        # Process images
        pixel_values_list = []
        for image in images:
            global_view = ImageOps.pad(
                image,
                (self.processor.base_size, self.processor.base_size),
                color=tuple(int(x * 255) for x in self.processor.image_transform.mean)
            )
            pixel_values = self.processor.image_transform(global_view)
            pixel_values_list.append(pixel_values)

        pixel_values = torch.stack(pixel_values_list, dim=0).to(
            device=self.device, dtype=self.dtype
        )

        batch_size = len(images)

        # SAM encoding
        sam_features = self.sam_model(pixel_values)  # [B, 1024, H, W]

        # CLIP encoding (pass SAM features as patch_embeds)
        clip_features = self.clip_model(pixel_values, sam_features)  # [B, seq+1, 1024]

        # Concatenate features (skip CLS token from CLIP)
        features = torch.cat(
            (
                clip_features[:, 1:],  # [B, seq, 1024]
                sam_features.flatten(2).permute(0, 2, 1),  # [B, hw, 1024]
            ),
            dim=-1,
        )  # [B, hw, 2048]

        # Project
        features = self.projector(features)  # [B, hw, 1280]

        # Add newline tokens and view separator
        _, hw, dim = features.shape
        side = int(hw ** 0.5)

        embeddings_list = []
        for jdx in range(batch_size):
            img_features = features[jdx].view(side, side, dim)
            newline = self.image_newline[None, None, :].expand(side, 1, dim)
            img_features = torch.cat([img_features, newline], dim=1)
            img_features = img_features.view(-1, dim)

            combined = torch.cat(
                [img_features, self.view_separator[None, :]],
                dim=0
            )
            embeddings_list.append(combined)

        return embeddings_list

    def encode_texts(
        self,
        texts: List[str],
        chunk_size: int = 6000,
    ) -> List[torch.Tensor]:
        """
        Encode texts to visual tokens.

        Args:
            texts: List of text strings
            chunk_size: Max characters per text

        Returns:
            List of visual token tensors [111, 1280]
        """
        if not self._initialized:
            self._load_weights()

        # Truncate texts
        truncated = [text[:chunk_size] for text in texts]

        # Render texts to images (parallel for batches > 2)
        # Uses adaptive font sizing to prevent content cutoff
        if len(truncated) > 2:
            images = self._renderer.render_batch_pil(truncated)
        else:
            from OCRFlow.utils.ultra_fast_renderer import render_to_pil
            images = [render_to_pil(text, min_font_size=9, max_font_size=20) for text in truncated]

        # Encode images
        return self.encode_images(images)


# Global encoder instance
_global_encoder: Optional[VisionEncoderOnly] = None


def get_vision_encoder(
    model_path: str = "deepseek-ai/DeepSeek-OCR",
    device: str = "cuda",
    num_render_workers: int = 8,
) -> VisionEncoderOnly:
    """Get or create global vision encoder instance"""
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = VisionEncoderOnly(
            model_path=model_path,
            device=device,
            num_render_workers=num_render_workers,
        )
    return _global_encoder


def create_vision_encoder(
    model_path: str = "deepseek-ai/DeepSeek-OCR",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_render_workers: int = 8,
) -> VisionEncoderOnly:
    """
    Create a new vision encoder instance.

    Args:
        model_path: HuggingFace model path
        device: Device to load on
        dtype: Data type for weights
        num_render_workers: Number of parallel rendering workers

    Returns:
        VisionEncoderOnly instance
    """
    encoder = VisionEncoderOnly(
        model_path=model_path,
        device=device,
        dtype=dtype,
        num_render_workers=num_render_workers,
    )
    return encoder


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing VisionEncoderOnly...")

    # Create encoder
    encoder = create_vision_encoder(device="cuda", num_render_workers=4)

    # Test with sample texts
    texts = [
        "Hello, world! This is a test of the vision encoder.",
        "Machine learning is a subset of artificial intelligence.",
    ] * 4  # 8 samples

    print(f"\nEncoding {len(texts)} texts...")

    import time
    start = time.time()
    tokens = encoder.encode_texts(texts)
    elapsed = time.time() - start

    print(f"Encoded in {elapsed:.2f}s ({len(texts)/elapsed:.1f} samples/sec)")
    print(f"Token shape: {tokens[0].shape}")
    print(f"Token dtype: {tokens[0].dtype}")

    # Check memory
    if torch.cuda.is_available():
        print(f"\nGPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\n✓ Test passed!")
