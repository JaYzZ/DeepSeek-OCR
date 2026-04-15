"""
Quick Donut inference example using the installed `donut-python`.

This loads a public CORD-finetuned checkpoint, runs it on a sample image from
the upstream donut repo, and prints the decoded JSON result.
"""

from __future__ import annotations

import pathlib

import torch
from donut import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
from project_paths import hf_path


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parent.parent / "donut"
    sample_image = repo_root / "misc" / "sample_image_cord_test_receipt_00004.png"
    local_model_dir = hf_path("naver-clova-ix", "donut-base-finetuned-cord-v2")
    model_id = str(local_model_dir) if local_model_dir.is_dir() else "naver-clova-ix/donut-base-finetuned-cord-v2"

    if not sample_image.is_file():
        raise SystemExit(f"Sample image not found: {sample_image}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load processor + model (weights download from HF if not cached).
    processor = DonutProcessor.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id).to(device)

    image = Image.open(sample_image).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    task_prompt = "<s_cord-v2>"  # dataset/task prefix for this checkpoint
    decoder_input_ids = processor.tokenizer(
        task_prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)

    outputs = model.generate(
        pixel_values=pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.config.decoder.max_length,
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
    )

    # Decode model output to JSON
    sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    result = processor.token2json(sequence)

    print(f"Input image: {sample_image}")
    print("Decoded result:")
    print(result)


if __name__ == "__main__":
    main()
