# OCRFlow

OCRFlow contains encoder-centric OCR training experiments, dataset pipelines, and example launchers.

## Current Layout

```text
OCRFlow/
├── configs/
├── examples/
│   ├── generate.py
│   ├── train.py
│   ├── train_mmdit.py
│   └── train_rolling_cache.py
├── models/
├── scripts/
├── training/
└── utils/
```

## Quick Start

```bash
pip install -r requirements.txt

# Optional but recommended: build the renderer backend used by OCRFlow
cd $ROOT_DIR/sources/DeepSeek-OCR/Renderer
pip install maturin
maturin develop --release

cd $ROOT_DIR/sources/DeepSeek-OCR/OCRFlow
python examples/train.py --max_steps 50000
```

For a tmux-managed launch, use [scripts/start_training.sh](scripts/start_training.sh).

## Main Entry Points

- [examples/train.py](examples/train.py): main OCRFlow training entrypoint.
- [examples/train_rolling_cache.py](examples/train_rolling_cache.py): rolling-cache training variant.
- [examples/train_mmdit.py](examples/train_mmdit.py): MMDiT-focused experiment entrypoint.
- [examples/generate.py](examples/generate.py): generation/inference utility for OCRFlow models.
- [scripts/precompute_vistok.py](scripts/precompute_vistok.py): optional offline cache preparation.
- [scripts/prepare_text_data.py](scripts/prepare_text_data.py): text-data preparation helper.
- [configs/train_config.py](configs/train_config.py): shared training defaults.

## Renderer Dependency

OCRFlow relies on renderer backends provided by [Renderer/](../Renderer) at the repo root. The old `utils/vello_renderer` path is no longer part of this repo layout.

To benchmark renderers directly:

```bash
python $ROOT_DIR/sources/DeepSeek-OCR/Renderer/benchmark_renderers.py
```

## Diagnostics

```bash
python -c "from Renderer import VelloRenderer; print('Renderer: OK')"
python -c "from OCRInfer.encoder import DPSKOCREncoder; print('Encoder: OK')"
```

If CJK glyphs render incorrectly, install `fonts-noto-cjk` and rebuild the renderer under `Renderer/`.
