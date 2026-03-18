# Qwen3VL R1-OneVision Latent Thinking Training

This repo currently maintains **one** training pipeline: Qwen3-VL R1-OneVision SFT with latent injection + latent losses (+ curriculum).

## Run Training (First Priority Command)

```bash
tmux new-session -d -s r1_sft 'CUDA_VISIBLE_DEVICES=0,1,2,3 bash Qwen/scripts/train_qwen3vl_r1onevision.sh Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml'
```

What the script does (high level):
- Verifies the OCRFlow python at `../../envs/ocrflow/bin/python`.
- Requires the dataset file `Qwen/data/r1_onevision_thinking.jsonl` to exist.
- Exports `QWEN3VL_*` env vars from the YAML (loss spec, token ids, curriculum, etc.).
- Runs `python -m llamafactory.cli train ...` (FSDP in the provided YAML).
- After training succeeds, runs **post-training backfill** (`Qwen/scripts/backfill_transparent_eval.py`) for checkpoints.

## Build The Dataset

The trainer expects `Qwen/data/r1_onevision_thinking.jsonl`.

Recommended two-phase build (CPU render -> GPU encode):

```bash
# Phase 1 (CPU): render images (question image + thinking-chunk images)
python Qwen/scripts/build_r1_onevision_thinking.py --render-only --all

# Phase 2 (GPU): encode features into .pt files and write JSONL
CUDA_VISIBLE_DEVICES=0 python Qwen/scripts/build_r1_onevision_thinking.py --encode-only --all

# Multi-GPU encode (example: 4 GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 python Qwen/scripts/build_r1_onevision_thinking.py --encode-only --all --num-gpus 4
```

## JSONL Sample Format

Dataset is ShareGPT-style (for LlamaFactory) with extra latent fields:

```json
{
  "id": "sample_id",
  "messages": [
    {"role": "user", "content": "<image>\nQuestion text..."},
    {"role": "assistant", "content": "<think><latent><think_sep><latent></think>Answer text..."}
  ],
  "images": ["/abs/or/relative/path/to/question_image.png"],
  "latent_ground_truth": ["/path/to/thinking_0.pt", "/path/to/thinking_1.pt"],
  "latent_supervision": ["/path/to/question_image.pt"],
  "num_latent_steps": 2,
  "cot": "optional raw thinking text (for debugging/CE labels)"
}
```

Notes:
- If `num_latent_steps == 1`, assistant content is typically `<think><latent></think>{answer}` (no `<think_sep>`).
- `latent_ground_truth` is **one entry per thinking chunk** (question_text image is excluded).
- `latent_supervision` is usually a **single** entry: the main question image feature (used as target for OT/NCE losses).

## Training Structure (Sequence, Expansion, Injection)

### 1) Sequence

The model is trained on the normal chat sequence (user message + assistant message). The thinking region lives inside the assistant content:

`<think> ... </think>Answer`

### 2) Placeholder Expansion

In the collator (`Qwen/llamafactory/integration.py`), each `<latent>` token inside `<think>...</think>` is expanded into a run of repeated `<latent>` tokens:

- For the i-th thinking chunk, let `L_i = latent_seq_len(latent_ground_truth[i])` (e.g. ~100 tokens, depends on encoder output).
- The single `<latent>` becomes `<latent>` repeated `L_i` times.
- This makes the number of `<latent>` tokens match the number of latent vectors that will be injected.

### 3) Latent Injection

Still in `Qwen/llamafactory/integration.py`, the input embedding at every expanded `<latent>` position is replaced by the pre-extracted `latent_ground_truth` embedding (concatenated across steps).

Important: `<latent>` is a **token id** used as a placeholder/mask location; the injected embedding is what carries the information.

## Losses (What Is Actually Optimized)

Total loss is a combination of:

1) **CE loss (standard SFT)**
- Always on normal answer tokens.
- On expanded `<latent>` positions:
  - Controlled by curriculum-driven latent-step CE masking in the main process, and/or
  - If the dataset provides `cot`, code can sample CoT tokens and use them as labels for latent positions.

2) **Thinking losses on hidden states at latent positions** (configurable by `QWEN3VL_LOSS_TYPE`)

Supported terms:
- `mse`: MSE between **shifted** hidden states (position p-1) and the injected `latent_ground_truth` targets.
- `repa`: cosine alignment between **shifted** hidden states (position p-1) and `latent_ground_truth`.
- `ot`, `nce`: compare **shifted** hidden states (position p-1) to `latent_supervision` targets (usually the main image feature), using `QWEN3VL_MATCH_STRATEGY` where applicable.

3) **VAE loss (optional, when `vae` is included in `QWEN3VL_LOSS_TYPE`)**
- Adds a `latent_vae` module to map shifted hidden states -> latent distribution.
- Computes NLL against the `latent_ground_truth` targets.
- Also runs a **second forward** (`pred_embed_forward`) with VAE-sampled latents inserted, and reuses the **same CE supervision mask/targets as the main forward**.

## Curriculum Learning

Curriculum is enabled by default in `Qwen/scripts/train_qwen3vl_r1onevision.sh` via:
- `QWEN3VL_CURRICULUM_ENABLE=1`
- `QWEN3VL_CURRICULUM_EPOCHS`, `QWEN3VL_CURRICULUM_LOSS_TYPES`
- `QWEN3VL_CURRICULUM_LATENT_STEP_CE` (toggles CE on latent positions per stage)

Implementation: `Qwen/llamafactory/curriculum_callback.py` (it updates env vars at epoch boundaries).

## Transparent Eval / Backfill

During-training `QwenTransparentEvalCallback` is **not required** for the training pipeline.

The default training script runs **post-training backfill** after training completes (see `Qwen/scripts/train_qwen3vl_r1onevision.sh`).

## vLLM Thinking Plugin (Inference)

Inference is intentionally different from training.

`vllm_thinking_plugin` provides a continuous mode in vLLM that is **prompt-driven** (it enters continuous mode when the prompt contains an unclosed `<think>`). Do not assume vLLM outputs the same placeholder formatting as training; evaluation code should generally strip the thinking region and score only the final answer.

### How The Plugin Switches Modes (Discrete vs Continuous)

Implementation: `vllm_thinking_plugin/vllm_thinking/runner_patch.py` (patches vLLM V1 `GPUModelRunner`).

Per request, the runner tracks a small state machine:
- `discrete` mode: normal vLLM token generation (logits -> token -> embedding(token) -> next step).
- `continuous` mode: feed the previous step **hidden state** back as `inputs_embeds` (no tokenization for the thinking steps).

Entry condition (prompt-driven):
- If a new request prompt contains `<think>` **without** a matching `</think>` after it, the plugin enters `continuous` mode for that request.

Continuous loop behavior:
- In `_preprocess`: if `continuous`, replace the next-step `inputs_embeds` with `prev_hidden` stored from the last model forward.
- In `_model_forward`: store the last hidden state as `prev_hidden` for the next step.
  - If `VLLM_LORA_CHECKPOINT_PATH/vae.safetensors` exists, the plugin loads `latent_vae` and samples from it before storing `prev_hidden` (to better match the VAE sampling path used in training).
  - If the file does not exist, the plugin **skips** VAE sampling and directly uses the previous hidden state.
- In `_sample`: still projects to logits to decide whether to exit.
  - Exit when the model selects `</think>` (ID `151668`), or when `max_steps` is reached (then the plugin forces `</think>` by boosting its logit).

Required env var:
```bash
export VLLM_THINKING_MODE_ENABLED=1
```

Useful env vars:
```bash
# Helps the plugin find `vae.safetensors` for latent_vae loading (optional).
export VLLM_LORA_CHECKPOINT_PATH=/path/to/your/checkpoint-dir
```

## Qwen3-VL Linear Base Checkpoint (Patch-Embed Conv3d -> Linear)

Qwen3-VL uses a Conv3d patch embedding layer in the vision tower, which can be slower/inefficient in some environments.
This repo supports a **linearized** base checkpoint that replaces the Conv3d patch embed with an equivalent Linear layer
(same outputs; faster runtime in practice).

- Converter script: `Qwen/scripts/convert_qwen3vl_patch_embed_to_linear.py`
- Default destination: `Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking`

Why it is safe for this training:
- The replacement is output-equivalent for the patch embedding.
- Our LoRA adapters and the `latent_vae` module are applied on the language model side; they do not depend on whether the
  vision patch embed is implemented as Conv3d or Linear.
- You can still load/run with the original base checkpoint (`Qwen3-VL-2B-Thinking`) if you prefer; the LoRA+VAE logic does not
  require the linearized base.

To generate the linear checkpoint:
```bash
python Qwen/scripts/convert_qwen3vl_patch_embed_to_linear.py \
  --src /share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --dst Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking
```

## Key Files

- Training entry:
  - `Qwen/scripts/train_qwen3vl_r1onevision.sh`
  - `Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml`
- Dataset build:
  - `Qwen/scripts/build_r1_onevision_thinking.py`
- LlamaFactory patches (latent injection + losses + curriculum):
  - `sitecustomize.py`
  - `Qwen/llamafactory/integration.py`
  - `Qwen/llamafactory/curriculum_callback.py`
