# OCRFlow: Visual Token Learning

Train transformer models on visual tokens from DeepSeek-OCR for large-scale text understanding.

## Overview

**OCRFlow** learns visual token representations using text datasets and DeepSeek-OCR as the encoder. Two training paradigms:

### 1. BERT (Bidirectional) - Reconstruction
- Masked token prediction (like BERT MLM)
- Bidirectional attention
- Good for understanding/encoding

### 2. Markovian Decoder (Autoregressive) - Generation
- Next-token prediction (like GPT)
- Causal attention (Markovian thinking)
- Good for generation/reasoning

**Both support:**
- ✅ **Large-scale pretraining** on text corpora (FineWeb-Edu, etc.)
- ✅ **GPT-style training practices** (large batches, cosine LR, etc.)
- ✅ **~305M params** (fast training)
- ✅ **No need for document images** (text-only datasets)

## Quick Start

### 1. Start DeepSeek-OCR Server

```bash
cd DeepSeek-OCR-master/DeepSeek-OCR-vllm/server
python deepseek_ocr_server.py --port 8010 --gpu-devices 0
```

### 2a. Train BERT (Bidirectional Reconstruction)

```bash
python OCRFlow/examples/train.py \
    --dataset_type fineweb \
    --train_data /share/project/xiyan/huggingface/HuggingFaceFW/fineweb-edu \
    --server_url http://localhost:8010 \
    --cache_dir ./vistok_cache \
    --output_dir ./checkpoints/bert_fineweb \
    --model_size large \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --use_masking \
    --use_amp \
    --max_steps 50000
```

### 2b. Train Markovian Decoder (Autoregressive Generation)

```bash
python OCRFlow/examples/train_markovian.py \
    --dataset_type fineweb \
    --train_data /share/project/xiyan/huggingface/HuggingFaceFW/fineweb-edu \
    --server_url http://localhost:8010 \
    --cache_dir ./vistok_cache \
    --output_dir ./checkpoints/markovian_fineweb \
    --model_size large \
    --batch_size 32 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --use_amp \
    --max_steps 50000
```

## Architectures

### 1. BERT (Bidirectional) - for Understanding

```
Input: Text → DeepSeek OCR → Visual Tokens [111, 1280]
       ↓
Mask 15% randomly → [111, 1280]
       ↓
BERT Transformer (bidirectional attention)
       ↓
Output: Reconstructed Visual Tokens [111, 1280]
       ↓
Loss: MSE on masked positions
```

### 2. Markovian Decoder (Autoregressive) - for Generation

```
Input: Text → DeepSeek OCR → Visual Tokens [111, 1280]
       ↓
Input: [v1, v2, ..., vN-1]
Target: [v2, v3, ..., vN]
       ↓
GPT Transformer (causal attention)
       ↓
Output: Next Token Predictions [110, 1280]
       ↓
Loss: MSE on next-token prediction
```

**Model Sizes (both architectures):**
- **base**: 768 hidden, 12 layers, ~150M params
- **large**: 1024 hidden, 24 layers, ~305M params
- **xl**: 1280 hidden, 32 layers, ~550M params

## Training Features

### GPT-Style Best Practices

1. **Large Effective Batch Sizes**
   - Use gradient accumulation: `--gradient_accumulation_steps 4`
   - Effective batch: 32 × 4 = 128 samples

2. **Cosine LR Schedule**
   - Warmup: 5% of total steps (adjustable)
   - Decay to 10% of peak LR
   - `--warmup_ratio 0.05 --min_lr_ratio 0.1`

3. **GPT-Style Optimizer**
   - AdamW with β2=0.95 (not 0.999)
   - Weight decay: 0.1
   - `--beta2 0.95 --weight_decay 0.1`

4. **Mixed Precision Training**
   - BF16/FP16 automatic mixed precision
   - `--use_amp`

5. **Masked Token Prediction**
   - BERT-style 15% masking
   - `--use_masking --mask_ratio 0.15`

## Dataset Support

### 1. FineWeb-Edu (Recommended for Large-Scale Training)

```bash
python OCRFlow/examples/train.py \
    --dataset_type fineweb \
    --train_data /path/to/fineweb-edu \
    --cache_dir ./vistok_cache \
    --min_tokens 100 \
    --max_tokens 1200 \
    --max_steps 100000
```

**Features:**
- Streaming from parquet files (memory efficient)
- Automatic caching of visual tokens
- Filters by text length (curriculum learning)
- High-quality educational web content

### 2. Custom Text Dataset

```bash
python OCRFlow/examples/train.py \
    --dataset_type vistok \
    --train_data ./data/train \
    --val_data ./data/val
```

**Format:** Place `.jsonl` file or `.txt` files in data directory:
```json
{"text": "Your text content here..."}
{"text": "Another document..."}
```

## Training Arguments

### Dataset
- `--dataset_type`: `fineweb` or `vistok`
- `--train_data`: Path to training data
- `--server_url`: DeepSeek OCR server URL (default: `http://localhost:8010`)
- `--cache_dir`: Cache directory for visual tokens

### Model
- `--model_size`: `base`, `large`, or `xl`
- `--use_masking`: Enable BERT-style masked prediction
- `--mask_ratio`: Masking ratio (default: 0.15)

### Training
- `--batch_size`: Batch size per GPU (default: 32)
- `--gradient_accumulation_steps`: Accumulation steps (default: 4)
- `--learning_rate`: Peak LR (default: 2e-4)
- `--weight_decay`: Weight decay (default: 0.1)
- `--beta2`: Adam beta2 (default: 0.95)
- `--warmup_ratio`: Warmup fraction (default: 0.05)
- `--num_epochs`: Training epochs (default: 10)
- `--max_steps`: Max steps (overrides epochs)
- `--use_amp`: Enable mixed precision

### System
- `--output_dir`: Output directory
- `--resume_from`: Resume from checkpoint
- `--save_every`: Save checkpoint every N steps (default: 1000)
- `--log_every`: Log metrics every N steps (default: 100)

## Inference

```bash
python OCRFlow/examples/infer_bert_baseline.py \
    --checkpoint ./checkpoints/final_model.pt \
    --text "# Test Document\n\nSample text for reconstruction." \
    --server_url http://localhost:8010 \
    --model_size large
```

## Performance Tips

1. **Enable Caching**: Always use `--cache_dir` to avoid repeated server calls
2. **Increase Batch Size**: Use gradient accumulation for larger effective batches
3. **Mixed Precision**: Use `--use_amp` for faster training
4. **FineWeb Streaming**: For large-scale training, FineWeb streams data efficiently

## File Structure

```
OCRFlow/
├── models/
│   └── bert_baseline.py          # BERT model implementation
├── training/
│   ├── vistok_dataset.py         # Pre-converted vistok dataset
│   └── fineweb_dataset.py        # FineWeb-Edu streaming dataset
├── examples/
│   ├── train.py                  # Unified training script
│   └── infer_bert_baseline.py   # Inference script
└── README.md                     # This file
```

## How It Works

### Training Pipeline

```
FineWeb-Edu Text
    ↓
Filter by length (100-1200 tokens)
    ↓
DeepSeek OCR Server:8010
    ↓ /text-to-vistok
Visual Tokens [111, 1280]
    ↓ Cache to disk
BERT Model
    ↓
Reconstructed Tokens
    ↓
MSE Loss (masked or full)
```

### Visual Token Format

- **Input shape**: `[111, 1280]` per chunk
- **111 tokens**: 100 visual + 10 newline + 1 separator
- **1280 dims**: DeepSeek OCR projection dimension
- **~1000 text tokens** → 1 visual chunk (111 tokens)

## Text Rendering Settings (FastBatchRenderer)

For Markovian training with 50-900 words per image on 640x640:

### Validated OCR Roundtrip Settings

| Words | Max Font Size | OCR Accuracy |
|-------|---------------|--------------|
| 900 | 9 | 97.3% |
| 500 | 10 | 98%+ |
| 200 | 14 | 98%+ |

**OCR Prompt**: Always use `<image>\nTranscribe the text in the image.`

### Key Findings

1. **Maximum font size for 900 words on 640x640**: Font size 9
   - 57 lines needed, 60 max lines available
   - Text remains readable for OCR

2. **OCR Prompt**: `<image>\nTranscribe the text in the image.`
   - Achieves 97.3% word accuracy on 900-word images
   - Produces correct word count (no hallucination)
   - "Free OCR" prompt hallucinates on dense text (>500 words) - DO NOT USE

3. **Adaptive Font Sizing**: `FastBatchRenderer` with `adaptive=True`:
   - Automatically calculates font size to fit all content
   - Range: min_font_size=9 to max_font_size=24
   - For 900 words: automatically selects font 9

### Usage Example

```python
from OCRFlow.utils.ultra_fast_renderer import UltraFastRenderer, render_to_pil

# High-throughput batch rendering (200+ img/s with 16 workers)
renderer = UltraFastRenderer(num_workers=16, min_font_size=9, max_font_size=20)
images = renderer.render_batch_pil(texts)  # List[PIL.Image]

# Single image rendering
img = render_to_pil(text, min_font_size=9, max_font_size=20)

# With prefetching for pipelined training
from OCRFlow.utils.ultra_fast_renderer import PrefetchingRenderer
prefetch_renderer = PrefetchingRenderer(num_workers=16)
prefetch_renderer.start(first_batch)
for next_batch in batches:
    images = prefetch_renderer.get_and_prefetch(next_batch)
    # GPU encodes while next batch renders
```

### Renderer Performance

| Renderer | Workers | Rate | Use Case |
|----------|---------|------|----------|
| FastBatchRenderer | 8 | ~124 img/s | Default |
| UltraFastRenderer | 8 | ~131 img/s | Optimized |
| UltraFastRenderer | 16 | **~213 img/s** | High throughput |

VisionEncoderOnly uses UltraFastRenderer by default for maximum throughput.

## Why This Approach?

1. **Leverage Large Text Corpora**: Train on billions of text tokens (FineWeb-Edu, etc.)
2. **Simple Baseline**: BERT is simpler than diffusion/flow models
3. **Fast Training**: 305M params, trains in hours not days
4. **Proven Practices**: GPT-style training works well
5. **Scalable**: Text-only datasets are abundant

## Next Steps

After training the BERT baseline:

1. **Evaluate** reconstruction quality
2. **Compare** with MMDiT + Sana/Qwen approaches
3. **Scale up** model size or training data
4. **Fine-tune** on downstream tasks

## Citation

If you use OCRFlow, please cite DeepSeek-OCR:

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```
