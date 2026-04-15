# Docs Index

This directory contains repo-local documentation for the engineering workspace in this repository. The root project contains multiple active modules, so this folder is for cross-cutting workflow notes rather than release-facing product docs.

## Start Here

- Repo overview: [README.md](../README.md)
- Qwen training flow: [Qwen/README.md](../Qwen/README.md)
- OCR-aware vision-language wrappers: [OCRVL/README.md](../OCRVL/README.md)
- DeepSeek-OCR inference split: [OCRInfer/README.md](../OCRInfer/README.md)
- OCRFlow experiments: [OCRFlow/README.md](../OCRFlow/README.md)

## Documents In This Folder

- [CLAUDE.md](CLAUDE.md): repo-specific working notes and operational conventions.
- [LLAMAFACTORY.md](LLAMAFACTORY.md): how the repo overlays LlamaFactory for latent-token training.

## Upstream vs Local Code

This repo also vendors an upstream DeepSeek-OCR snapshot in [DeepSeek-OCR-master](../DeepSeek-OCR-master). Keep these roles separate:

- `docs/`: local workflow and engineering docs for this workspace.
- `DeepSeek-OCR-master/`: vendored upstream reference code and server examples.

If you are looking for the local vLLM server wrapper, start from [DeepSeek-OCR-master/DeepSeek-OCR-vllm/server/README.md](../DeepSeek-OCR-master/DeepSeek-OCR-vllm/server/README.md) and the surrounding server files, but treat that subtree as a reference snapshot rather than the main place for new work.
