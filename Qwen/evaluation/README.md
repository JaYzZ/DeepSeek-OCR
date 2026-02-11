# Qwen3-VL Evaluation Suite

Unified evaluation benchmark suite for Qwen3-VL models with support for base models and LoRA adapters.

## Benchmarks

| Benchmark | Description | Metric | Samples |
|-----------|-------------|--------|---------|
| **MMMU** | Massive Multi-discipline Multimodal Understanding | Accuracy | 1,377 |
| **MathVision** | Mathematical visual reasoning | Accuracy | 16,230 |
| **RealWorldQA** | Real-world question answering | Accuracy | 766 |
| **ODinW-13** | Object detection (13 datasets) | mAP | ~7,700 |

## Quick Start

### 1. Setup Data

```bash
cd Qwen/evaluation

# Download all datasets
bash setup_data.sh

# Verify downloads
bash verify_data.sh
```

### 2. Install Dependencies

```bash
# Install all dependencies for all benchmarks
pip install -r requirements.txt
```

### 3. Run Full Evaluation (Base Model)

```bash
# Run all 4 benchmarks with 100 samples each
python run_all_benchmarks.py \
    --num-samples 100 \
    --model-path /path/to/Qwen3-VL-2B-Thinking \
    --gpus 0,1,2,3
```

### 4. Run Evaluation with LoRA Checkpoint

```bash
# Evaluate your trained LoRA adapter
python run_all_benchmarks.py \
    --num-samples 100 \
    --model-path Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking \
    --enable-lora \
    --lora-path Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_20260209_235504/checkpoint-1000 \
    --lora-name r1_onevision_lora \
    --gpus 0,1,2,3
```

## Command Reference

### `run_all_benchmarks.py` (Main Entry Point)

Runs inference and evaluation for all benchmarks in one command.

```bash
python run_all_benchmarks.py [OPTIONS]

Options:
  --num-samples N          Samples per benchmark (default: 100)
  --model-path PATH        Path to base model
  --gpus ID,ID,ID          GPU IDs for tensor parallel (default: auto-select 4)
  --benchmarks LIST        Comma-separated benchmarks (default: all 4)

  # LoRA Options
  --enable-lora            Enable LoRA adapter mode
  --lora-path PATH         Path to LoRA checkpoint
  --lora-name NAME         LoRA adapter name (default: default)
  --max-lora-rank N        Maximum LoRA rank (default: 64)

  # Phase Control
  --skip-infer             Skip inference, run evaluation only
  --skip-eval              Skip evaluation, run inference only
  --run-dir PATH           Existing run directory (for --skip-infer)
```

### `run_all_inference.py` (Inference Only)

Runs inference on all benchmarks with a single model load (faster).

```bash
python run_all_inference.py \
    --model-path /path/to/model \
    --output-dir results/run_$(date +%Y%m%d_%H%M%S) \
    --num-samples 100 \
    --tensor-parallel-size 4

# With LoRA
python run_all_inference.py \
    --model-path /path/to/base \
    --enable-lora --lora-path /path/to/lora \
    --output-dir results/...
```

## Examples

```bash
# Run specific benchmarks only
python run_all_benchmarks.py --benchmarks MMMU,MathVision --num-samples 100

# Run inference only (skip evaluation)
python run_all_benchmarks.py --num-samples 100 --skip-eval

# Run evaluation on existing inference results
python run_all_benchmarks.py --skip-infer --run-dir results/run_20260210_120000

# Use custom judge server (default: http://47.111.147.142:8600)
export JUDGE_SERVER_URL="http://localhost:8000"
python run_all_benchmarks.py --num-samples 100
```

## Directory Structure

```
Qwen/evaluation/
├── config.py                    # Configuration (paths, defaults)
├── run_all_benchmarks.py        # Main entry point (inference + eval)
├── run_all_inference.py         # Unified inference (single model load)
├── setup_data.sh                # Dataset download script
├── verify_data.sh               # Dataset verification script
├── test_judge_server.sh         # Test judge server connectivity
├── test_lora.py                 # Test LoRA checkpoint loading
│
├── data/                        # Dataset files
│   ├── MMMU/MMMU_DEV_VAL.tsv
│   ├── MathVision/MathVision.tsv
│   ├── RealWorldQA/RealWorldQA.tsv
│   └── ODinW-13/odinw/          # Images downloaded during inference
│
├── results/                     # Evaluation outputs (timestamped)
│   └── run_YYYYMMDD_HHMMSS/
│       ├── mmmu_inference.jsonl
│       ├── mathvision_inference.jsonl
│       ├── realworldqa_inference.jsonl
│       ├── odinw_inference.jsonl
│       ├── *_eval_result.csv
│       └── SUMMARY.md
│
└── benchmarks/                  # Per-benchmark modules
    ├── mmmu/
    │   ├── run_mmmu.py          # Individual benchmark runner
    │   ├── dataset_utils.py     # Dataset loading
    │   ├── eval_utils.py        # Evaluation metrics
    │   └── common_utils.py
    ├── MathVision/              # (same structure)
    ├── RealWorldQA/             # (same structure)
    └── ODinW-13/                # (same structure)
```

## LoRA Training Integration

For evaluating LoRA checkpoints trained with the training pipeline:

```bash
# Training produces checkpoint at:
# Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_XXXXXX/checkpoint-N

# Evaluate the checkpoint:
python run_all_benchmarks.py \
    --model-path Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking \
    --enable-lora \
    --lora-path Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_XXXXXX/checkpoint-1000
```

**Important**: The `--model-path` must match the base model used during training. See your training config yaml for the `model_name_or_path`.

## Output Format

Results are saved to `results/run_YYYYMMDD_HHMMSS/`:

| File | Description |
|------|-------------|
| `*_inference.jsonl` | Raw model predictions (one JSON per line) |
| `*_eval_result.csv` | Evaluation metrics |
| `*_eval_result_acc.json` | Accuracy breakdown |
| `SUMMARY.md` | Combined summary of all benchmarks |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JUDGE_SERVER_URL` | Judge model API endpoint | `http://47.111.147.142:8600` |
| `EVAL_NUM_SAMPLES` | Sample limit for dataset loaders | None |
| `DATA_DIR` | Override data directory | `./data` |

## Troubleshooting

### Judge Server Connection

```bash
# Test judge server
bash test_judge_server.sh

# Or manually:
curl http://47.111.147.142:8600/health
```

### GPU Memory Issues

```bash
# Reduce memory utilization
python run_all_benchmarks.py \
    --gpus 0,1,2,3 \
    # Edit run_all_inference.py to set:
    # --gpu-memory-utilization 0.70  (default: 0.75)
```

### LoRA Loading Issues

```bash
# Test LoRA checkpoint independently
python test_lora.py \
    --base-model-path Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking \
    --lora-path Qwen/checkpoints/.../checkpoint-1000
```

## Reference

Based on [Qwen3-VL evaluation suite](https://github.com/QwenLM/Qwen3-VL) with unified scripting and LoRA support.
