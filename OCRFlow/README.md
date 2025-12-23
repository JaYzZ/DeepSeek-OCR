# OCRFlow

High-performance training system for OCR models with GPU-accelerated rendering and optimized encoder-decoder architecture.

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Build Vello renderer (12x faster than PIL) - see SETUP.md
sudo apt-get install -y libvulkan-dev fonts-noto-cjk
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
cd utils/vello_renderer && pip install maturin && maturin develop --release

# 3. Run training (7 encoder + 1 training GPU)
python examples/train.py --max_steps 50000

# 4. Monitor training in tmux
tmux attach -t ocrflow_training
```

**Note:** First run takes 7-21 min for torch.compile warmup. Subsequent runs start instantly (0s warmup).

## Features

- **High Performance**: 12x faster rendering with Vello, ~2,079 pairs/s training throughput
- **Memory Efficient**: Vision encoder only (401M params vs 7B full model), ~800MB per encoder GPU
- **Zero Warmup**: torch.compile cache artifacts enable 0s startup on subsequent runs
- **Multilingual**: Full CJK support (Chinese, Japanese, Korean) via Noto Sans fonts
- **MAR Training**: Masked autoregressive self-supervised learning with diffusion-based prediction
- **Dedicated Pool**: Separate encoding workers (GPUs 1-7) and training worker (GPU 0)

## Performance

### Rendering Speed
| Renderer | Speed (img/s) | Speedup vs PIL | CJK Support |
|----------|---------------|----------------|-------------|
| PIL (default) | 130 | 1x | ✅ |
| Skia | 778 | 6x | ✅ |
| **Vello** | **1,565** | **12x** | **✅** |

### Training Throughput
| Configuration | Encoding Speed | Training Speed | GPU Util |
|---------------|----------------|----------------|----------|
| 8 GPUs (7+1) | 2,079 pairs/s | 2,278 pairs/s | 91% |
| 4 GPUs (3+1) | 891 pairs/s | 2,278 pairs/s | 39% |
| 2 GPUs (1+1) | 297 pairs/s | 2,278 pairs/s | 13% |

### Memory Usage
- **Encoder GPUs (1-7):** ~800MB each (vision encoder only, no LLM)
- **Training GPU (0):** ~20GB (full model + optimizer)
- **System RAM:** ~8GB (dataset + cache)

## Architecture

```
┌────────────────────────────────────────────────────┐
│                 OCRFlow Training                   │
├────────────────────────────────────────────────────┤
│                                                    │
│  ┌──────────────┐         ┌──────────────┐       │
│  │ Vello        │  1,565  │ Vision       │  297  │
│  │ Renderer     │─ img/s ─│ Encoder      │pairs/s│
│  │ (CPU-based)  │         │ (GPU 1-7)    │×7 GPUs│
│  └──────────────┘         └──────┬───────┘       │
│                                   │               │
│                                   ▼               │
│                          ┌─────────────────┐     │
│                          │ Rolling Cache   │     │
│                          │ (50K pairs)     │     │
│                          └────────┬────────┘     │
│                                   │               │
│                                   ▼               │
│                          ┌─────────────────┐     │
│                          │ Training Worker │     │
│                          │ (GPU 0)         │     │
│                          │ - Next-chunk    │     │
│                          │ - MAR diffusion │     │
│                          └─────────────────┘     │
│                                                    │
│  Throughput: ~2,079 pairs/s                       │
│  Training GPU Util: ~91%                          │
└────────────────────────────────────────────────────┘
```

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / Linux
- **Python**: 3.10+
- **CUDA**: 12.1+ (for GPU acceleration)
- **RAM**: 32GB+
- **GPU**: 1x NVIDIA GPU with 24GB+ VRAM (for basic training)

### Recommended for Full Training
- **GPUs**: 8x NVIDIA H100/A100 (7 for encoding + 1 for training)
- **RAM**: 128GB+
- **Storage**: 500GB+ SSD (for datasets)

## Training Configuration

### Basic Training

```bash
# Basic training (8 GPUs)
python examples/train.py --max_steps 50000

# Custom configuration
python examples/train.py \
    --max_steps 100000 \
    --encode_batch_size 24 \
    --train_batch_size 16 \
    --learning_rate 1e-4
```

### With MAR (Masked Autoregressive) Training

Add self-supervised vision learning with diffusion-based prediction:

```bash
python examples/train.py \
    --max_steps 50000 \
    --enable_mar \
    --mar_loss_weight 0.1
```

**Benefits:**
- Self-supervised vision learning on 100 pure visual tokens
- Diffusion-based continuous token prediction (not simple L2)
- High mask ratios (70-100%) via truncated Gaussian
- Better vision representations
- ~10% training overhead

### Configuration Options

Key parameters in `examples/train.py`:

```python
# Encoder configuration
ENCODER_GPUS = [1, 2, 3, 4, 5, 6, 7]  # GPUs for vision encoding
TRAINING_GPU = 0                       # GPU for training

# Dataset configuration
CACHE_SIZE = 50000                     # Rolling cache size
ENCODE_BATCH_SIZE = 24                 # Batch size per encoder (optimal: 329 img/s)
TRAIN_BATCH_SIZE = 16                  # Training batch size

# Optimization
LEARNING_RATE = 1e-4
MAX_STEPS = 50000
GRADIENT_ACCUMULATION = 4
```

## Troubleshooting

### Quick Diagnostics

```bash
# 1. Check Python environment
python --version  # Should be 3.10+

# 2. Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 3. Check Vello renderer
python -c "from Renderer import VelloRenderer; print('Vello: OK')"

# 4. Check vision encoder
python -c "from OCRInfer.encoder.dpsk_ocr_encoder import DPSKOCREncoder; print('Encoder: OK')"
```

### Common Issues

**Vello renderer not found:**
```bash
cd ../Renderer
maturin develop --release
```

**CJK characters show as boxes:**
```bash
sudo apt-get install -y fonts-noto-cjk
cd utils/vello_renderer && maturin develop --release
```

**Training OOM (Out of Memory):**
```python
# Reduce batch sizes in examples/train.py
ENCODE_BATCH_SIZE = 12  # Default: 24 (optimal)
TRAIN_BATCH_SIZE = 8   # Default: 16
CACHE_SIZE = 25000     # Default: 50000
```

**torch.compile warmup taking too long (first run):**
- Expected: 1-3 minutes per GPU (one-time)
- If hanging > 5 minutes, check CUDA compatibility and GPU memory
- Subsequent runs will be instant (0s warmup)

### Monitor Training

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check training logs
tail -f logs/training_*.log

# Check encoder throughput (should see ~297 pairs/s per encoder)
grep "pairs/s" logs/training_*.log

# Verify Vello is being used
grep "Vello renderer" logs/training_*.log
```

## Benchmarking

```bash
# Benchmark renderers (see Renderer/ module at repo root)
python ../Renderer/benchmark_renderers.py

# Expected output:
# Vello Renderer: 1,565 img/s (GPU)
# Skia Renderer: 778 img/s (CPU)
# PIL Renderer: 130 img/s (CPU)
```

## File Structure

```
OCRFlow/
├── README.md                      # This file
├── SETUP.md                       # Vello renderer setup guide
├── examples/
│   └── train.py                   # Main training script
├── training/
│   ├── rolling_cache_dataset.py   # Dataset with rolling cache
│   ├── mar_diffusion.py           # MAR training with diffusion loss
│   └── diffloss.py                # Diffusion loss module
├── utils/
│   └── image_augmentation.py      # Document-style augmentation
├── scripts/
│   ├── precompute_vistok.py       # Offline vistok cache builder (optional)
│   ├── prepare_text_data.py       # HF text download helpers (optional)
│   └── start_training.sh          # tmux training launcher
└── requirements.txt               # Python dependencies

Note: All renderers (Vello/Skia/PIL) are in ../Renderer/ module at repo root
```

## Performance Optimization

### 1. Use Vello Renderer (12x faster)

See [SETUP.md](SETUP.md) for installation instructions.

### 2. Encoder Throughput

OCRFlow uses the OCRInfer `DPSKOCREncoder` directly for text→vistok. If you
need higher throughput, precompute vistok caches with
`scripts/precompute_vistok.py` and train from disk.

### 3. Optimize GPU Allocation

Use 7 encoder GPUs + 1 training GPU for ~91% training GPU utilization:
```python
ENCODER_GPUS = [1, 2, 3, 4, 5, 6, 7]
TRAINING_GPU = 0
```

### 4. Tune Batch Sizes

Balance memory and throughput:
```python
ENCODE_BATCH_SIZE = 24  # Per encoder GPU (optimal: 329 img/s on H100)
TRAIN_BATCH_SIZE = 16   # Training GPU
```

## Support

For help:
1. Check [SETUP.md](SETUP.md) for Vello renderer installation
2. Review [Troubleshooting](#troubleshooting) section
3. Run diagnostic commands above
4. Check logs in `logs/`

---

**Version:** 1.0.0
**Last Updated:** 2025-12-05
