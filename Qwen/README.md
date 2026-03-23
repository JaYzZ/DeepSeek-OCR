# Qwen3VL Training

This repo currently maintains two active Qwen3-VL training paths:

- **R1-OneVision SFT** with latent injection + latent losses (+ curriculum).
- **DeepVision-103K GSPO** on VERL with vLLM rollout and a repo-local compatibility patch path.

## Run Training (First Priority Command)

```bash
tmux new-session -d -s r1_sft 'CUDA_VISIBLE_DEVICES=0,1,2,3 bash Qwen/scripts/train_qwen3vl_r1onevision.sh Qwen/configs/qwen3vl_native_r1onevision_thinking.yaml'
```

## DeepVision GSPO (VERL + vLLM)

### 1. Build RL parquet from local DeepVision-103K

The local dataset path used here is:

`/share/project/xiyan/huggingface/skylenage/DeepVision-103K`

Build VERL-ready train/val parquet:

```bash
/share/project/xiyan/envs/ocrflow/bin/python Qwen/scripts/build_deepvision_verl_dataset.py
```

Default output:

`Qwen/data/deepvision_103k_verl/train.parquet`

`Qwen/data/deepvision_103k_verl/val.parquet`

Notes:
- The converter preserves the original multi-message prompt and raw image bytes from DeepVision parquet.
- The local corpus is the full DeepVision-103K release: `math-77k.parquet` (77,135 rows) + `visual_logic-26k.parquet` (26,368 rows).
- Reward supervision is taken from `reward_model.ground_truth` plus `equivalent_answers`.
- Validation is a deterministic per-source split because DeepVision-103K ships as train-only parquet.

### 2. Launch single-node GSPO with vLLM rollout

Default YAML:

- `Qwen/configs/rl/deepvision_gspo.yaml`

Example:

```bash
tmux new-session -d -s deepvision_gspo 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash Qwen/scripts/train_qwen3vl_deepvision_gspo.sh'
```

What the GSPO launcher does:
- Runs `Qwen/scripts/run_verl_ppo.py`, which merges VERL's base PPO config with our project YAML.
- Writes a single authoritative log at `OUTPUT_DIR/training.log`; launcher, Ray, and worker output all stream there.
- Keeps `algorithm.adv_estimator=grpo`, matching the official DeepVision setup of GSPO policy loss with GRPO advantage estimation.
- Uses `Qwen/scripts/deepvision_gspo_reward.py` as the reward function.
- Uses a pure rule-based correctness reward: `+1` for a correct final answer, `0` otherwise.
- Does not add any format reward or judge reward.
- Uses a repo-local batched reward manager (`dapo_batch`) so reward scoring stays compatible with DAPO-style bookkeeping while avoiding the slow per-sample decode/score loop.
- Uses **vLLM** as the rollout backend.
- Forces rollout through the repo-local plugin + compat path by exporting:
  - `PYTHONPATH=$REPO_ROOT/vllm_thinking_plugin:$REPO_ROOT`
  - `VLLM_PLUGINS=vllm_thinking`
- Injects the local VERL worker patch through Ray runtime env plus `worker_process_setup_hook=verl_compat.worker_setup.apply_worker_compat_patches`.
- Keeps continuous thinking mode **disabled** by default for the baseline:
  - `VLLM_THINKING=0`
- Keeps the base model configurable through `MODEL_PATH`; the rule logic remains DeepVision-style regardless of which supported base you plug in.
- Starts with LoRA on the language stack and excludes `visual` modules from LoRA.
- Freezes the vision tower for the initial baseline.
- Defaults to `FILTER_OVERLONG_PROMPTS_WORKERS=1`; for this stack that has been the fastest setting in practice.
- Leaves multimodal preprocessor cache enabled by default in rollout; set `ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE=true` only if host-memory pressure forces it.

### 3. Multi-node compatibility

The script is written to stay compatible with multinode Ray/VERL runs:

```bash
NNODES=2 \
RAY_ADDRESS=http://your-ray-head:8265 \
bash Qwen/scripts/train_qwen3vl_deepvision_gspo.sh
```

Practical note:
- The launcher itself does not create a Ray cluster for you.
- For `NNODES>1`, bring up Ray separately, then pass `RAY_ADDRESS` and the desired trainer overrides.

Initialize RL from an existing SFT LoRA checkpoint:

```bash
tmux new-session -d -s deepvision_gspo_sft_init 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 INIT_LORA_PATH=Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_current_sota/checkpoint_latest bash Qwen/scripts/train_qwen3vl_deepvision_gspo.sh'
```

Resume an interrupted VERL run from its own checkpoint:

```bash
tmux new-session -d -s deepvision_gspo_resume 'CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 RESUME_MODE=resume_path RESUME_FROM_PATH=/abs/path/to/verl_run/global_step_40 bash Qwen/scripts/train_qwen3vl_deepvision_gspo.sh'
```

Important:
- `INIT_LORA_PATH` is for bootstrapping RL from a HuggingFace/PEFT LoRA adapter such as `checkpoint_latest`.
- `RESUME_MODE=resume_path` is only for resuming a previous VERL run, whose checkpoint folder must contain `global_step_*` actor/critic state.
- If the init LoRA contains visual-module adapters, vLLM will warn that those visual LoRA weights are ignored during rollout. That is expected for the current language-only rollout path.

What the script does (high level):
- Verifies the OCRFlow python at `../../envs/ocrflow/bin/python`.
- Requires `Qwen/data/deepvision_103k_verl/{train,val}.parquet` to exist.
- Keeps the base model configurable through `MODEL_PATH`.
- Enables the repo-local VERL compatibility bootstrap through `sitecustomize.py` and `./verl_compat/`.
- Loads training-critical RL settings from `Qwen/configs/rl/*.yaml`.
- Runs VERL PPO with GSPO loss, GRPO advantage estimation, vLLM rollout, LoRA on the language stack, and DeepVision-style rule reward.

### 4. Practical speed knobs for DeepVision-103K

The current training logs show generation time dominates each step, with reward evaluation as the next-largest cost. The safest speed knobs are:

- Keep `FILTER_OVERLONG_PROMPTS_WORKERS=1` unless you have measured a real improvement on your machine.
- `ROLLOUT_DISABLE_MM_PREPROCESSOR_CACHE=false` to keep image preprocessing cached in rollout.
- `ROLLOUT_MAX_BATCHED_TOKENS` to tune batching separately from `ROLLOUT_MAX_MODEL_LEN`.
- `MAX_RESPONSE_LENGTH` to cap runaway long generations if you want faster smoke tests.
- `ROLLOUT_N` to reduce samples per prompt for quick checks.

Code-path optimizations already wired into this repo:

- DeepVision reward scoring uses a batched reward manager plus cached symbolic parsing instead of VERL's default per-sample DAPO decode loop.

Use more aggressive changes only if you have already confirmed stability on this stack:

- `ROLLOUT_ENFORCE_EAGER=false`
- `free_cache_engine=false` via an overlay config
- higher `GPU_MEMORY_UTILIZATION`

## Build The Dataset

The trainer expects `Qwen/data/r1ov_thinking.jsonl`.

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
