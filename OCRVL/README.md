## OCRVL Helpers for Qwen2.5-VL & Qwen3-VL

`Qwen25VLOCRTextAdapter` mirrors the Qwen3 adapter but targets the Qwen2.5-VL
pretrained checkpoints. It renders long text to DeepSeek-OCR visual tokens and
returns `(input_ids, ocr_image_features)` so you can keep prompts short while
feeding dense content through the OCR encoder. Combine it with
`OCRQwen25VLForConditionalGeneration` to get the same OCR-aware forward/generate
path as Qwen3-VL.

Use `Qwen3VLOCRTextAdapter` to turn long text into DeepSeek-OCR visual tokens and feed them to the OCR-Qwen3-VL wrapper. The prompt stays short (instruction + `<image>` placeholders) while dense content lives in `ocr_image_features`. Text is rendered with the Vello GPU renderer when available (fallback to PIL), and the same DeepSeek-OCR encoder is used for both rendered text and raw images.

```python
from OCRVL import (
    OCRQwen3VLForConditionalGeneration,
    OCRQwen25VLForConditionalGeneration,
    Qwen25VLOCRTextAdapter,
    Qwen3VLOCRTextAdapter,
)
from transformers import AutoTokenizer
from OCRInfer.utils.model_paths import resolve_model_path

qwen_path = resolve_model_path("Qwen/Qwen3-VL-7B")
ocr_path = resolve_model_path("deepseek-ai/DeepSeek-OCR")

tokenizer = AutoTokenizer.from_pretrained(qwen_path)
adapter = Qwen3VLOCRTextAdapter(
    encoder_model_path=ocr_path,  # local mirror preferred
    device="cuda",
    chunk_tokens=900,  # chunk long text before rendering
)

instruction = "Read the document images and answer the question."
dense_text = open("long.txt").read()
input_ids, ocr_feats = adapter.prepare_qwen_inputs(
    instruction=instruction,
    dense_text=dense_text,
    tokenizer=tokenizer,
    placeholder_token="<image>",  # match your tokenizer's image token
)

model = OCRQwen3VLForConditionalGeneration.from_pretrained(qwen_path, torch_dtype="bfloat16")
out = model.generate(
    input_ids=input_ids.to(model.device),
    ocr_image_features=[f.to(model.device) for f in ocr_feats],
    max_new_tokens=64,
)
print(tokenizer.decode(out[0], skip_special_tokens=True))

# Encode raw images with the same DeepSeek-OCR encoder
from PIL import Image
images = [Image.open("doc1.png"), Image.open("doc2.png")]
input_ids, ocr_feats = adapter.prepare_qwen_inputs_from_images(
    instruction="Read the document images and answer the question.",
    images=images,
    tokenizer=tokenizer,
    placeholder_token="<image>",
)
out = model.generate(
    input_ids=input_ids.to(model.device),
    ocr_image_features=[f.to(model.device) for f in ocr_feats],
    max_new_tokens=64,
)

# Qwen2.5-VL mirrors the same pattern
qwen25_path = resolve_model_path("Qwen/Qwen2.5-VL-7B-Instruct")
tokenizer25 = AutoTokenizer.from_pretrained(qwen25_path)
adapter25 = Qwen25VLOCRTextAdapter(encoder_model_path=ocr_path, device="cuda")
input_ids, ocr_feats = adapter25.prepare_qwen_inputs(
    instruction=instruction,
    dense_text=dense_text,
    tokenizer=tokenizer25,
    placeholder_token="<image>",
)
model25 = OCRQwen25VLForConditionalGeneration.from_pretrained(qwen25_path, torch_dtype="bfloat16")
out = model25.generate(
    input_ids=input_ids.to(model25.device),
    ocr_image_features=[f.to(model25.device) for f in ocr_feats],
    max_new_tokens=64,
)
```

Tips:
- Keep `instruction` short; the adapter renders only `dense_text` into vision tokens.
- `placeholder_token` should be whatever your tokenizer uses for images (e.g., `<image>` or a dedicated vision token block).
- Finetune with LoRA on the language stack; you only need `input_ids` + `ocr_image_features` + labels. Vision weights can stay frozen.
- All HuggingFace models are mirrored under `/share/project/xiyan/huggingface/{model_id}`; `resolve_model_path` will pick the local copy automatically and fall back to the raw model id if missing.
