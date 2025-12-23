import types

import torch
from PIL import Image

from OCRVL import Qwen25VLOCRTextAdapter


class DummyEncoder:
    def __init__(self):
        self.calls = 0

    def encode_images(self, images, return_global=False, return_local=True):
        self.calls += len(images)
        return [torch.ones(2, 3) for _ in images]


def dummy_render_texts(chunks):
    return [Image.new("RGB", (10, 10), color="white") for _ in chunks]


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        toks = text.split()
        return types.SimpleNamespace(
            input_ids=torch.tensor([list(range(len(toks)))], dtype=torch.long)
        )


def test_prompt_builder_matches_chunk_count():
    prompt = Qwen25VLOCRTextAdapter.build_prompt_with_placeholders(
        "summarize",
        2,
        placeholder_token="<image>",
        joiner="\n",
    )
    assert prompt.count("<image>") == 2
    assert prompt.startswith("summarize")


def test_prepare_inputs_uses_encoder_chunks():
    adapter = Qwen25VLOCRTextAdapter(
        encoder=DummyEncoder(),
        chunk_tokens=4,
        render_width=160,
        render_height=160,
        render_font_size=12,
        render_texts=dummy_render_texts,
    )
    tokenizer = DummyTokenizer()
    long_text = "one two three four five six seven eight"

    input_ids, feats = adapter.prepare_qwen_inputs(
        instruction="read the document and answer",
        dense_text=long_text,
        tokenizer=tokenizer,
        placeholder_token="<image>",
    )

    assert len(feats) >= 2
    assert input_ids.ndim == 2
    assert all(isinstance(f, torch.Tensor) for f in feats)


def test_prepare_inputs_from_images_encodes_each_image():
    adapter = Qwen25VLOCRTextAdapter(
        encoder=DummyEncoder(),
        render_texts=dummy_render_texts,
    )
    tokenizer = DummyTokenizer()
    images = [Image.new("RGB", (8, 8), color="white") for _ in range(3)]

    input_ids, feats = adapter.prepare_qwen_inputs_from_images(
        instruction="summarize",
        images=images,
        tokenizer=tokenizer,
        placeholder_token="<image>",
    )

    assert len(feats) == 3
    assert input_ids.shape[1] >= 3
