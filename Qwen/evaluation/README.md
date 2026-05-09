# Qwen3-VL Evaluation

Unified evaluation runners for the Qwen training flow.

Current defaults match the code in this repo:

- base model default: `$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking`
- default benchmark set: `MathVision,MMMU,RealWorldQA,M3CoT`
- LoRA evaluation is optional through `--enable-lora --lora-path ...`
- `ODinW-13`, `M3CoT`, and `ScienceQA` are available when their data is prepared

## Quick Start

```bash
cd $ROOT_DIR/sources/DeepSeek-OCR/Qwen/evaluation
bash setup_data.sh
bash verify_data.sh
pip install -r requirements.txt
```

Run the default benchmark set:

```bash
python run_all_benchmarks.py \
  --start-server \
  --num-samples 100 \
  --model-path $ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --gpus 0,1,2,3
```

Run against a LoRA checkpoint:

```bash
python run_all_benchmarks.py \
  --start-server \
  --num-samples 100 \
  --lora-path Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_XXXXXX/checkpoint-1000 \
  --run-dir Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_XXXXXX/checkpoint-1000/bench \
  --gpus 0,1,2,3
```

Run a designated benchmark subset on a designated checkpoint:

```bash
python run_all_benchmarks.py \
  --start-server \
  --lora-path /abs/path/to/checkpoint-XXXX \
  --run-dir /abs/path/to/checkpoint-XXXX/bench \
  --gpus 1,2 \
  --benchmarks MathVision,MMMU,RealWorldQA,M3CoT \
  --num-samples 100
```

## Standard Eval Process

Use one canonical process for post-training eval.

### 1. Transparent Backfill

Use this when you want transparent reasoning samples and debug traces for a specific checkpoint.

```bash
CUDA_VISIBLE_DEVICES=0 python Qwen/inference/backfill_transparent_eval.py \
  --checkpoint_dir /abs/path/to/run_or_ckpt_parent \
  --checkpoint checkpoint-1650 \
  --gpu_memory_utilization 0.8
```

Rules:

- `--checkpoint_dir` points to the parent directory that contains the checkpoint.
- `--checkpoint` selects the concrete checkpoint directory name, such as `checkpoint-1650` or `checkpoint_latest`.
- If `--checkpoint` is omitted, the script resolves the latest checkpoint automatically.
- `CUDA_VISIBLE_DEVICES` is the device assignment for backfill.
- Tensor parallel defaults to auto-infer from `CUDA_VISIBLE_DEVICES`, so no extra flag is needed in the normal case.
- Outputs are written under `<checkpoint>/eval_results/`.

### 2. Benchmark

Use this when you want benchmark scores for a specific checkpoint.

```bash
CUDA_VISIBLE_DEVICES=1,2 python Qwen/evaluation/run_all_benchmarks.py \
  --start-server \
  --lora-path /abs/path/to/checkpoint-1650 \
  --run-dir /abs/path/to/checkpoint-1650/bench \
  --gpus 1,2 \
  --benchmarks MathVision,MMMU,RealWorldQA,M3CoT \
  --num-samples 100
```

Rules:

- `--lora-path` points directly to the checkpoint directory to evaluate.
- `--run-dir` is the benchmark output directory. Standard location is `<checkpoint>/bench/`.
- `--gpus` must match the GPU subset you want the benchmark server to use.
- `--benchmarks` is the exact dataset list to run.
- `--num-samples` controls benchmark sample count.
- Do not use `--output-dir` here. The benchmark entrypoint uses `--run-dir`.

### 3. Recommended Runtime Convention

For the current OPSD / RLSD discrete evaluation path:

```bash
export VLLM_THINKING=0
```

That means:

- discrete AR evaluation
- `<think>` is appended through the prompt path when enabled by the repo logic
- no continuous-AR inference mode is requested from vLLM

Inference-only mode still exists and requires an explicit output directory:

```bash
python run_all_inference.py \
  --model-path $ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --output-dir results/run_$(date +%Y%m%d_%H%M%S) \
  --num-samples 100 \
  --tensor-parallel-size 4
```

## Main Entry Points

- [run_all_benchmarks.py](run_all_benchmarks.py): inference + eval orchestration, optional local server startup, LoRA loading, and summary output
- [run_all_inference.py](run_all_inference.py): inference-only runner with a shared model load
- [config.py](config.py): default model and data paths

## Benchmarks

Current benchmark modules in this directory:

- [mathvision](mathvision)
- [MMMU](mmmu)
- [realworldqa](realworldqa)
- [m3cot](m3cot)
- [scienceqa](scienceqa)
- [odinw](odinw)

## Outputs

Runs write into `Qwen/evaluation/results/run_YYYYMMDD_HHMMSS/` unless `--run-dir` is supplied.

Common files:

- `*_inference.jsonl`
- `*_eval_result.csv`
- `*_eval_result_acc.json`
- `SUMMARY.md`
- `benchmark.log`

## Notes

- `run_all_benchmarks.py` defaults to `MathVision,MMMU,RealWorldQA,M3CoT`.
- The model path should match the base model used during training when evaluating LoRA adapters.
- For LoRA checkpoints, prefer passing `--lora-path` and let the repo resolve the matching base model automatically.
- Some evaluation paths rely on a judge server. Use [test_judge_server.sh](test_judge_server.sh) to validate connectivity first.
