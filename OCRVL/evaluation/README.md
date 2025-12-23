# OCRVL Benchmark Evaluation

This directory contains scripts for evaluating OCRVL-trained models on standard VLM benchmarks.

## Overview

The OCRVL benchmark system wraps the Qwen3-VL evaluation suite to:
- Load OCRVL-trained connectors on top of the base Qwen3-VL model
- Run the same benchmarks (MMMU, MathVision, RealWorldQA, ODinW-13)
- Output results to `OCRVL/results/*` with the same naming convention as Qwen3-VL

## Quick Start

### 1. Wait for Checkpoint

Checkpoints are saved every 1000 steps during training:
```bash
# Check training progress
tail -f OCRVL/checkpoints/alignment_60k_*/training.log

# List saved checkpoints
find OCRVL/checkpoints -name "step_*"
```

### 2. Run Benchmarks

Once a checkpoint is available (e.g., `step_1000`):

```bash
# Set checkpoint path
export CHECKPOINT_PATH=OCRVL/checkpoints/alignment_60k_20251221_023626/step_1000

# Run all benchmarks
./OCRVL/evaluation/run_all_benchmarks.sh
```

### 3. View Results

Results are saved to `OCRVL/results/<checkpoint>_<timestamp>/`:

```
OCRVL/results/alignment_60k_20251221_023626_step_1000_20251221_040000/
├── mmmu/
│   ├── mmmu_dev_val_predictions.jsonl
│   └── mmmu_dev_val_metrics.json
├── mathvision/
│   ├── mathvision_test_predictions.jsonl
│   └── mathvision_test_metrics.json
├── realworldqa/
│   ├── realworldqa_test_predictions.jsonl
│   └── realworldqa_test_metrics.json
├── odinw13/
│   ├── odinw13_predictions.jsonl
│   └── odinw13_metrics.json
├── summary.json      # Aggregated results (JSON)
├── summary.csv       # Aggregated results (CSV)
└── run_all_benchmarks.log  # Full execution log
```

## Configuration

### Required Environment Variables

- `CHECKPOINT_PATH`: Path to OCRVL checkpoint directory (must contain `connectors.pt`)

### Optional Environment Variables

Same as Qwen3-VL benchmarks:

- `MODEL_PATH`: Base Qwen3-VL model path (auto-detected from checkpoint config if not set)
- `DATA_DIR`: Benchmark data directory (default: `../Qwen3-VL/data/VLMEval`)
- `ODINW_DIR`: ODinW-13 data directory (default: `../Qwen3-VL/data/odinw`)
- `RUN_TAG`: Custom run identifier (default: auto-generated from checkpoint name + timestamp)
- `NUM_GPUS`: Number of GPUs to use (default: auto-detect free GPUs)
- `GPUS`: Comma-separated GPU IDs (e.g., `"0,1,2"`)

### Text-as-Images Mode

To render text inputs as images (OCR mode):

```bash
export RENDER_TEXT=1
export RENDER_TEXT_BACKEND=vello  # vello|skia|pil
export RENDER_TEXT_SIZE=640
export RENDER_TEXT_CHUNK_TOKENS=900
./OCRVL/evaluation/run_all_benchmarks.sh
```

## Architecture

### Components

1. **`run_all_benchmarks.sh`**: Main entry point
   - Validates OCRVL checkpoint
   - Sets up environment variables
   - Delegates to Qwen3-VL benchmark suite
   - Redirects output to `OCRVL/results/*`

2. **`ocrvl_model_loader.py`**: Model loading utility
   - Loads base Qwen3-VL model
   - Applies OCRVL-trained connectors
   - Used by inference scripts via environment variable detection

3. **Qwen3-VL evaluation scripts**: (reused as-is)
   - `../Qwen3-VL/evaluation/run_all_benchmarks.sh`
   - `../Qwen3-VL/evaluation/*/run_*.py`

### Integration Points

The system uses environment variables to enable OCRVL mode:

- `OCRVL_MODE=1`: Enables OCRVL connector loading
- `OCRVL_CHECKPOINT_PATH`: Path to checkpoint directory
- `OCRVL_CONNECTORS_PATH`: Path to `connectors.pt` file

Inference scripts can detect OCRVL mode and load connectors:

```python
from OCRVL.evaluation.ocrvl_model_loader import load_ocrvl_model, is_ocrvl_mode

if is_ocrvl_mode():
    model = load_ocrvl_model(model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path)
```

## Current Status

⚠️ **Inference Script Integration Pending**

The benchmark runner is ready, but the Qwen3-VL inference scripts need to be modified to use `ocrvl_model_loader.load_ocrvl_model()` instead of standard `AutoModelForCausalLM.from_pretrained()`.

### Next Steps

1. **Modify Qwen3-VL inference scripts** to use OCRVL model loader:
   - `../Qwen3-VL/evaluation/mmmu/run_mmmu.py`
   - `../Qwen3-VL/evaluation/MathVision/run_mathv.py`
   - `../Qwen3-VL/evaluation/RealWorldQA/run_realworldqa.py`
   - `../Qwen3-VL/evaluation/ODinW-13/run_odinw.py`

2. **Test with trained checkpoint** (once step_1000 is reached):
   ```bash
   export CHECKPOINT_PATH=OCRVL/checkpoints/.../step_1000
   ./OCRVL/evaluation/run_all_benchmarks.sh
   ```

## Example: Modify Inference Script

In `../Qwen3-VL/evaluation/mmmu/run_mmmu.py`:

```python
# Before:
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(args.model_path, ...)

# After:
import sys
sys.path.insert(0, '/path/to/DeepSeek-OCR')
from OCRVL.evaluation.ocrvl_model_loader import load_ocrvl_model
model = load_ocrvl_model(args.model_path, ...)
```

Or use automatic detection:

```python
from OCRVL.evaluation.ocrvl_model_loader import load_ocrvl_model, is_ocrvl_mode

if is_ocrvl_mode():
    model = load_ocrvl_model(args.model_path)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_path, ...)
```

## Troubleshooting

### Checkpoint not found
```bash
# List available checkpoints
find OCRVL/checkpoints -type d -name "step_*"

# Check current training step
tail OCRVL/checkpoints/*/training.log | grep "Step"
```

### Missing connectors.pt
Ensure training has reached a save interval (default: every 1000 steps).

### Model loading errors
Verify that:
1. `MODEL_PATH` points to a valid Qwen3-VL model directory
2. `CHECKPOINT_PATH` contains `connectors.pt`
3. OCRVL Python modules are in PYTHONPATH

## See Also

- Qwen3-VL benchmarks: `../Qwen3-VL/evaluation/README.md`
- OCRVL training: `../OCRVL/README.md`
- Model architecture: `../OCRVL/model/README.md`
