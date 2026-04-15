# OCRInfer

Standalone DeepSeek-OCR inference utilities with an explicit encoder/decoder split.

## Current Layout

```text
OCRInfer/
├── config.py
├── encoder/
│   ├── dpsk_ocr_encoder.py
│   ├── dpsk_ocr_cross_attention.py
│   └── build_linear.py
├── decoder/
│   ├── dpsk_ocr_decoder.py
│   ├── embedding_patch.py
│   └── transformers_patch.py
├── model/
├── process/
├── tests/
│   ├── roundtrip.py
│   ├── benchmark_encoder.py
│   └── verify_optimizations.py
└── utils/
    └── model_paths.py
```

The public package exports are:

- `OCRInfer.DPSKOCREncoder`
- `OCRInfer.EncoderOutput`
- `OCRInfer.VLLMEmbeddingDecoder`

## Quick Start

```python
from PIL import Image

from OCRInfer import DPSKOCREncoder, VLLMEmbeddingDecoder
from OCRInfer.utils.model_paths import resolve_model_path

model_path = resolve_model_path("deepseek-ai/DeepSeek-OCR")

encoder = DPSKOCREncoder(model_path=model_path, device="cuda", dtype="bfloat16")
image = Image.open("document.png").convert("RGB")
visual_tokens = encoder.encode_images([image], return_local=True)[0]

decoder = VLLMEmbeddingDecoder(model_path=model_path, dtype="bfloat16")
text = decoder.decode(
    visual_embeddings=visual_tokens,
    prompt="Transcribe all text:",
    max_tokens=2048,
)
print(text)
```

`resolve_model_path()` first checks the local mirror under `$ROOT_DIR/huggingface/{model_id}` and falls back to the raw Hugging Face model id if the mirror does not exist.

## Main Components

- [encoder/dpsk_ocr_encoder.py](encoder/dpsk_ocr_encoder.py): standalone DeepSeek-OCR vision encoder.
- [encoder/dpsk_ocr_cross_attention.py](encoder/dpsk_ocr_cross_attention.py): cross-attention variant used by some experiments.
- [decoder/dpsk_ocr_decoder.py](decoder/dpsk_ocr_decoder.py): vLLM embedding decoder entrypoint.
- [decoder/embedding_patch.py](decoder/embedding_patch.py): monkey-patches that let vLLM consume precomputed visual embeddings.
- [decoder/transformers_patch.py](decoder/transformers_patch.py): compatibility patching for the transformers path.
- [utils/model_paths.py](utils/model_paths.py): model path resolution.

## Tests and Benchmarks

These scripts are GPU-oriented; do not run them on CPU-only hosts:

- [tests/roundtrip.py](tests/roundtrip.py): end-to-end text -> render -> encode -> decode validation.
- [tests/benchmark_encoder.py](tests/benchmark_encoder.py): encoder throughput benchmark.
- [tests/verify_optimizations.py](tests/verify_optimizations.py): attention/backend inspection and large-batch checks.

Typical round-trip invocation:

```bash
CUDA_VISIBLE_DEVICES=0 python OCRInfer/tests/roundtrip.py
```

Round-trip artifacts are written under `OCRInfer/outputs/test_images/`.
