import types

import torch
from PIL import Image

from OCRVL import Qwen3VLOCRTextAdapter


class DummyEncoder:
    def __init__(self):
        self.calls = 0

    def encode_images(self, images, return_global=False, return_local=True):
        # Return one tiny tensor per image chunk
        self.calls += len(images)
        return [torch.ones(2, 3) for _ in images]

    def encode_images_with_deepstack(self, images):
        # Mirror DPSKOCREncoder API: (final_feats, deepstack_feats)
        self.calls += len(images)
        final = [torch.ones(2, 3) for _ in images]
        # Qwen3 expects 3 deepstack levels per image.
        deepstack = [[torch.ones(2, 3) for _ in range(3)] for _ in images]
        return final, deepstack


def dummy_render_texts(chunks):
    # Produce a blank image per chunk
    return [Image.new("RGB", (10, 10), color="white") for _ in chunks]


class DummyTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        # Naive whitespace tokenization to mimic HF output shape
        toks = text.split()
        return types.SimpleNamespace(
            input_ids=torch.tensor([list(range(len(toks)))], dtype=torch.long)
        )


def test_prompt_builder_matches_chunk_count():
    prompt = Qwen3VLOCRTextAdapter.build_prompt_with_placeholders(
        "summarize",
        3,
        placeholder_token="<image>",
        joiner="\n",
    )
    assert prompt.count("<image>") == 3
    assert prompt.startswith("summarize")


def test_prepare_inputs_uses_encoder_chunks():
    adapter = Qwen3VLOCRTextAdapter(
        encoder=DummyEncoder(),
        chunk_tokens=5,
        render_width=160,
        render_height=160,
        render_font_size=12,
        render_texts=dummy_render_texts,
    )
    tokenizer = DummyTokenizer()
    long_text = "one two three four five six seven eight nine ten"

    input_ids, ocr_features = adapter.prepare_qwen_inputs(
        instruction="read the document and answer",
        dense_text=long_text,
        tokenizer=tokenizer,
        placeholder_token="<image>",
        return_deepstack=True,
    )

    feats, deepstack = ocr_features

    # Expect at least two chunks for the short chunk_tokens threshold
    assert len(feats) >= 2
    assert len(deepstack) == len(feats)
    assert all(len(levels) == 3 for levels in deepstack)
    assert input_ids.ndim == 2
    # Ensure features are torch tensors
    assert all(isinstance(f, torch.Tensor) for f in feats)


def test_prepare_inputs_from_images_encodes_each_image():
    adapter = Qwen3VLOCRTextAdapter(
        encoder=DummyEncoder(),
        render_texts=dummy_render_texts,
    )
    tokenizer = DummyTokenizer()
    images = [Image.new("RGB", (8, 8), color="white") for _ in range(2)]

    input_ids, ocr_features = adapter.prepare_qwen_inputs_from_images(
        instruction="summarize",
        images=images,
        tokenizer=tokenizer,
        placeholder_token="<image>",
        return_deepstack=True,
    )

    feats, deepstack = ocr_features

    assert len(feats) == 2
    assert len(deepstack) == 2
    assert all(len(levels) == 3 for levels in deepstack)
    assert input_ids.shape[1] >= 2  # two placeholders + BOS/EOS
