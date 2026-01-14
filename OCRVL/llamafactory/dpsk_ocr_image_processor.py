from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import ImageOps

from OCRInfer.process.image_process import DeepseekOCRProcessor


def _infer_image_grid_thw(num_tokens: int, merge_size: int) -> torch.LongTensor:
    """
    LlamaFactory token expansion expects image_grid_thw to represent the *pre-merge* grid.
    It then divides by merge_size^2 to get the number of image placeholder tokens.

    Our DPSK OCR encoder typically outputs a 10x10 grid (=100 tokens) when remove_separators=True.
    With merge_size=2 (Qwen-VL default), the corresponding pre-merge grid is 20x20:
      (20*20) / (2*2) = 100.
    """
    side = int(round(num_tokens**0.5))
    if side * side != num_tokens:
        # Fallback to a degenerate strip.
        h, w = num_tokens * (merge_size**2), 1
    else:
        h = side * merge_size
        w = side * merge_size
    return torch.tensor([1, h, w], dtype=torch.long)


@dataclass
class DPSKOCRImageProcessor:
    """
    Drop-in replacement for Qwen3VLProcessor.image_processor that returns:
      - pixel_values: [B,3,H,W] (CPU-preprocessed DeepSeek-OCR inputs)
      - image_grid_thw: [B,3]

    It intentionally performs NO CUDA work so it is safe to run in DataLoader/
    datasets multiprocessing workers. The actual OCR encoding happens inside
    the model forward pass (training cycle) via the DPSKVisionTowerAdapter.

    NOTE: Returns "pixel_values" (not "ocr_pixel_values") for compatibility
    with LlamaFactory's data processing pipeline.
    """

    merge_size: int = 2

    def __call__(self, images: List[Any], return_tensors: str = "pt", **_: Any) -> Dict[str, Any]:
        if return_tensors != "pt":
            raise ValueError("DPSKOCRImageProcessor only supports return_tensors='pt'.")

        if len(images) == 0:
            return {"pixel_values": torch.empty((0, 3, 0, 0)), "image_grid_thw": torch.empty((0, 3), dtype=torch.long)}

        # CRITICAL: Skip processing during tokenization phase
        # During tokenization, images come as dicts with bytes=None (set by _regularize_images patch)
        # We return dummy tensors that will be ignored. Actual processing happens during training.
        if len(images) > 0 and isinstance(images[0], dict) and images[0].get("bytes") is None:
            # Tokenization phase: return dummy tensors
            # These are cached in the Arrow dataset but not used during training
            remove_separators = os.environ.get("OCRVL_DPSK_REMOVE_SEPARATORS", "1").strip() != "0"
            expected_tokens = 100 if remove_separators else 111
            # Return tiny dummy tensor (will be discarded during training)
            dummy_pixel_values = torch.zeros((len(images), 3, 640, 640), dtype=torch.float32)
            grid = _infer_image_grid_thw(expected_tokens, self.merge_size)
            dummy_image_grid_thw = grid.unsqueeze(0).expand(len(images), -1).contiguous()
            return {"pixel_values": dummy_pixel_values, "image_grid_thw": dummy_image_grid_thw}

        # Training phase: Process images normally
        # This happens when the DataLoader actually loads images for training
        proc = _get_dpsk_image_preprocessor()
        pixel_values_list = []
        for image in images:
            # Handle lazy loading: if image is a dict with path, load it now
            if isinstance(image, dict):
                if "path" in image:
                    from PIL import Image
                    image = Image.open(image["path"]).convert("RGB")
                elif "bytes" in image and image["bytes"] is not None:
                    from PIL import Image
                    import io
                    image = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                else:
                    raise ValueError(f"Image dict must contain 'path' or 'bytes': {image}")

            global_view = ImageOps.pad(
                image,
                (proc.base_size, proc.base_size),
                color=tuple(int(x * 255) for x in proc.image_transform.mean),
            )
            pixel_values_list.append(proc.image_transform(global_view))
        pixel_values = torch.stack(pixel_values_list, dim=0).to(device="cpu")

        remove_separators = os.environ.get("OCRVL_DPSK_REMOVE_SEPARATORS", "1").strip() != "0"
        expected_tokens = 100 if remove_separators else 111
        grid = _infer_image_grid_thw(expected_tokens, self.merge_size)
        image_grid_thw = grid.unsqueeze(0).expand(pixel_values.shape[0], -1).contiguous()

        # Return as "pixel_values" for LlamaFactory compatibility
        # The model will route these through the DPSKVisionTowerAdapter
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def save_pretrained(self, save_directory: str, **kwargs: Any) -> None:
        """
        Save processor configuration to a JSON file.
        Follows the standard transformers pattern for compatibility with LlamaFactory.
        """
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "preprocessor_config.json")

        # Save the configuration as JSON
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump({"merge_size": self.merge_size}, f, indent=2)


_PREPROCESSOR_SINGLETON: Optional[DeepseekOCRProcessor] = None


def _get_dpsk_image_preprocessor() -> DeepseekOCRProcessor:
    global _PREPROCESSOR_SINGLETON
    if _PREPROCESSOR_SINGLETON is None:
        _PREPROCESSOR_SINGLETON = DeepseekOCRProcessor()
    return _PREPROCESSOR_SINGLETON
