# Qwen3-VL Evaluation

Unified evaluation runners for the Qwen training flow.

Current defaults match the code in this repo:

- base model default: `$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking`
- default benchmark set: `MathVision,MMMU,RealWorldQA`
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
  --num-samples 100 \
  --model-path $ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --gpus 0,1,2,3
```

Run against a LoRA checkpoint:

```bash
python run_all_benchmarks.py \
  --num-samples 100 \
  --model-path $ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --enable-lora \
  --lora-path Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_XXXXXX/checkpoint-1000 \
  --gpus 0,1,2,3
```

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

- `--benchmarks` defaults to `MathVision,MMMU,RealWorldQA`, not the full benchmark list.
- The model path should match the base model used during training when evaluating LoRA adapters.
- Some evaluation paths rely on a judge server. Use [test_judge_server.sh](test_judge_server.sh) to validate connectivity first.
