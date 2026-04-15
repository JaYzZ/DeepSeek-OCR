# OCRVL

OCR-aware vision-language wrappers and tooling for Qwen-VL and LLaVA-style models.

## Current Layout

```text
OCRVL/
├── builder.py
├── decoder/
├── evaluation/
├── llamafactory/
├── model/
│   └── language_model/
├── scripts/
├── tests/
├── train.py
└── utils/
```

Current package exports from [OCRVL/__init__.py](__init__.py):

- `OCRQwen3VLForConditionalGeneration`
- `Qwen3VLOCRTextAdapter`
- `OCRQwen25VLForConditionalGeneration`
- `Qwen25VLOCRTextAdapter`

## Structure Notes

- Qwen-specific encoder and generic decoder helpers now live in [Qwen/encoder](../Qwen/encoder) and [Qwen/decoder](../Qwen/decoder).
- `OCRVL/` owns model wrappers, adapters, OCRVL-specific vLLM processors, LlamaFactory integration, and OCR-aware evaluation helpers.
- Dataset builders for the Qwen training flow now live in `Qwen/data/`, while OCRVL keeps its own helpers under [OCRVL/scripts](scripts).

## Quick Usage

```python
from PIL import Image
from transformers import AutoTokenizer

from OCRVL import OCRQwen3VLForConditionalGeneration, Qwen3VLOCRTextAdapter

model_path = "$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

adapter = Qwen3VLOCRTextAdapter(device="cuda")
images = [Image.open("doc.png").convert("RGB")]
input_ids, ocr_features = adapter.prepare_qwen_inputs_from_images(
    instruction="Describe the document.",
    images=images,
    tokenizer=tokenizer,
    return_deepstack=True,
)

model = OCRQwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype="bfloat16",
    device_map="cuda",
)
outputs = model.generate(
    input_ids=input_ids.to(model.device),
    ocr_image_features=ocr_features,
    max_new_tokens=256,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Main Entry Points

- [builder.py](builder.py): OCR-aware LLaVA model loader wrapper.
- [train.py](train.py): multi-stage OCRVL training entrypoint.
- [model/language_model/ocr_qwen3_vl.py](model/language_model/ocr_qwen3_vl.py): Qwen3-VL wrapper and adapter implementation.
- [model/language_model/ocr_qwen25_vl.py](model/language_model/ocr_qwen25_vl.py): Qwen2.5-VL wrapper and adapter implementation.
- [model/language_model/ocr_llava_llama.py](model/language_model/ocr_llava_llama.py): OCR-aware LLaVA wrapper, imported directly from the module rather than via `OCRVL/__init__.py`.
- [decoder/](decoder): OCRVL-specific vLLM processors.
- [Qwen/decoder](../Qwen/decoder): generic Qwen vLLM decoders and decoder utilities.
- [llamafactory/](llamafactory): OCRVL integration hooks for LlamaFactory workflows.
- [evaluation/](evaluation): benchmark runners and evaluation helpers.

## Tests

The lightest CPU-safe adapter tests are:

- [tests/test_ocr_qwen3vl.py](tests/test_ocr_qwen3vl.py)
- [tests/test_ocr_qwen25vl.py](tests/test_ocr_qwen25vl.py)

GPU-dependent generic Qwen decoder and encoder tests now live under [Qwen/tests](../Qwen/tests) and should not be run on hosts without CUDA.
