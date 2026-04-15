# DeepSeek-OCR Workspace

This repository is a working research workspace around DeepSeek-OCR rather than a single-package release. The active code is split into a few focused modules, while generated outputs and vendored upstream code live alongside them.

## Repository Map

- `Qwen/`: active Qwen3-VL training, dataset building, evaluation, visualization, and Qwen-owned encoder/decoder helpers.
- `OCRVL/`: OCR-aware vision-language wrappers, adapters, OCRVL-specific processors, and integration tests.
- `OCRInfer/`: standalone DeepSeek-OCR encoder/decoder inference toolkit.
- `OCRFlow/`: separate OCR training experiments and data pipelines.
- `Renderer/`: renderer backends used by OCRFlow/OCRInfer.
- `docs/`: repo-local engineering notes and workflow docs.
- `DeepSeek-OCR-master/`: vendored upstream DeepSeek-OCR snapshot; treat this as reference code.

## Source vs Runtime Artifacts

These top-level paths are runtime or generated content, not primary source modules:

- `logs/`
- `outputs/`
- `swanlog/`
- `Qwen/checkpoints/`

When adding new code, keep it under the owning module instead of expanding the root with more ad hoc scripts.

## Entry Points

- Qwen training: [Qwen/README.md](Qwen/README.md)
- OCR-aware VL wrappers: [OCRVL/README.md](OCRVL/README.md)
- DeepSeek-OCR inference split: [OCRInfer/README.md](OCRInfer/README.md)
- OCRFlow experiments: [OCRFlow/README.md](OCRFlow/README.md)
- Repo docs index: [docs/README.md](docs/README.md)

## Current Structural Conventions

- Qwen-specific encoder and generic decoder helpers now live in [Qwen/encoder](Qwen/encoder) and [Qwen/decoder](Qwen/decoder), not under `OCRVL/`.
- Dataset builders for the Qwen flow live under `Qwen/data/`.
- Vendored upstream code under `DeepSeek-OCR-master/` should be updated only when intentionally syncing that snapshot.
