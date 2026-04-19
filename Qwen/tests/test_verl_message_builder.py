from __future__ import annotations

import copy
from io import BytesIO

from PIL import Image
from verl.utils.dataset.rl_dataset import RLHFDataset

from verl_compat.bootstrap import apply_runtime_compat_patches


class _DummyDataset:
    prompt_key = "prompt"
    image_key = "images"
    video_key = "videos"
    processor = object()


def _png_bytes() -> bytes:
    image = Image.new("RGB", (2, 2), color=(255, 255, 255))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_verl_message_builder_preserves_image_payloads() -> None:
    apply_runtime_compat_patches()

    row = {
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "<image>\nSolve it."},
        ],
        "images": [{"bytes": _png_bytes(), "path": None}],
    }

    messages = RLHFDataset._build_messages(_DummyDataset(), copy.deepcopy(row))

    assert isinstance(messages[1]["content"], list)
    assert messages[1]["content"][0]["type"] == "image"
    assert messages[1]["content"][0]["bytes"] == row["images"][0]["bytes"]
    assert messages[1]["content"][0]["path"] is None
    assert "image" in messages[1]["content"][0]
    assert messages[1]["content"][1] == {"type": "text", "text": "\nSolve it."}


def test_verl_message_builder_keeps_structured_content() -> None:
    apply_runtime_compat_patches()

    structured = [{"type": "image", "image": Image.new("RGB", (1, 1))}, {"type": "text", "text": "Solve it."}]
    row = {
        "prompt": [
            {"role": "user", "content": structured},
        ],
        "images": [{"bytes": _png_bytes(), "path": None}],
    }

    messages = RLHFDataset._build_messages(_DummyDataset(), copy.deepcopy(row))

    assert messages[0]["content"] == structured
