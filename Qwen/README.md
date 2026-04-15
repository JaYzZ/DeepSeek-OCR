# Qwen3VL Training

This directory currently has two primary training flows:

- `R1-OneVision` supervised fine-tuning with latent injection and latent losses.
- `Chimera` GSPO reinforcement learning on VERL with vLLM rollout.

The commands below match the active wrappers in this repo as of April 4, 2026.

## Current State

The launcher command forms did not change during the recent refactor. These are still the active entrypoints:

- SFT: `tmux new-session -d -s r1_sft 'bash Qwen/scripts/train_qwen3vl_r1onevision.sh Qwen/configs/qwen3vl_r1onevision_thinking.yaml'`
- RL: `tmux new-session -d -s chimera_gspo 'INIT_LORA_PATH=Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_vae_mse_ot_2ce_subcot/checkpoint-729 bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh'`

Current implementation state:

- Qwen3-VL now defaults to the original Hugging Face checkpoint, not the old linearized checkpoint.
- Patch-embed is fixed by monkey-patching the original Qwen3-VL path; the linearized option is retained only for compatibility.
- Qwen-owned Qwen-VL encoder helpers now live under [Qwen/encoder](encoder).
- Qwen-owned generic vLLM decoders now live under [Qwen/decoder](decoder).
- R1-OneVision SFT now runs a 3-stage curriculum split by the wrapper.
- Stage 2 is true main-path `vae` training without normal token CE.
- Stage 3 keeps the frozen VAE active in forward/loss so LoRA realigns to the trained latent policy.
- The SFT wrapper now explicitly hands `vae.safetensors` from stage 2 into later stages.
- The Qwen shell launchers share a common helper library at [qwen3vl_common.sh](scripts/qwen3vl_common.sh), so command syntax stayed stable while duplicated bootstrap logic was removed.

## Active Launch Commands

R1-OneVision SFT:

```bash
tmux new-session -d -s r1_sft 'bash Qwen/scripts/train_qwen3vl_r1onevision.sh Qwen/configs/qwen3vl_r1onevision_thinking.yaml'
```

Chimera GSPO RL initialized from an SFT LoRA:

```bash
tmux new-session -d -s chimera_gspo 'INIT_LORA_PATH=Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_vae_mse_ot_2ce_subcot/checkpoint-729 bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh'
```

## Pipeline Map

### SFT

Entry files:

- [train_qwen3vl_r1onevision.sh](scripts/train_qwen3vl_r1onevision.sh)
- [qwen3vl_r1onevision_thinking.yaml](configs/qwen3vl_r1onevision_thinking.yaml)
- [qwen3vl_runtime_env.yaml](configs/qwen3vl_runtime_env.yaml)
- [train_qwen3vl_dataset_mix.sh](scripts/train_qwen3vl_dataset_mix.sh)
- [integration.py](llamafactory/integration.py)

The wrapper:

- requires `$ROOT_DIR/envs/ocrflow/bin/python`
- snapshots the main config and runtime env config into the run directory
- materializes dataset mixtures and optional per-dataset ratios
- sets latent-runtime env vars used by training, backfill, and benchmark subprocesses
- launches LlamaFactory training
- writes a consolidated log to `run_*/training.log`
- runs post-training transparent-eval backfill
- optionally runs benchmarks on the latest checkpoint

### RL

Entry files:

- [train_qwen3vl_chimera_gspo.sh](scripts/train_qwen3vl_chimera_gspo.sh)
- [chimera_gspo.yaml](configs/rl/chimera_gspo.yaml)
- [run_verl_ppo.py](verl/run_verl_ppo.py)
- [chimera_gspo_reward.py](verl/chimera_gspo_reward.py)
- [build_chimera_verl_dataset.py](data/build_chimera_verl_dataset.py)

The RL wrapper:

- requires the same OCRFlow Python env
- expects prebuilt `Qwen/data/chimera_verl/train.parquet` and `val.parquet`
- exports repo-local VERL and vLLM compatibility paths
- merges VERL PPO base config with the project YAML
- resolves the final runtime config into the output directory
- logs launcher and worker output to a single `training.log`

## R1-OneVision SFT

### Model and Training Config

The default training config is [qwen3vl_r1onevision_thinking.yaml](configs/qwen3vl_r1onevision_thinking.yaml).

Current defaults:

- base model: `$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking`
- stage: `sft`
- finetuning: LoRA
- LoRA rank/alpha/dropout: `8 / 16 / 0.05`
- LoRA targets: `q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj`
- vision tower: frozen
- template: `qwen3vl_latent`
- cutoff length: `8192`
- per-device batch size: `32`
- gradient accumulation: `2`
- precision: `bf16`, `pure_bf16`
- distributed mode from config: FSDP, not DeepSpeed

Important wrapper behavior:

- if `CUDA_VISIBLE_DEVICES` is unset, the wrapper defaults to `0,1,2,3,4,5,6,7`
- for single-node runs, it auto-selects a free `MASTER_PORT`
- it sets `TMPDIR=/tmp` unless already exported
- it loads repo-local `.env` if present so keys like `SWANLAB_API_KEY` are picked up
- it prepends `../LlamaFactory/src` and repo root to `PYTHONPATH`

### Dataset Contract

The config references dataset `r1_onevision_thinking`, which is defined in [dataset_info.json](data/dataset_info.json) and currently points to:

- `Qwen/data/sft/r1ov_thinking.jsonl`

Expected fields:

- `messages`
- `images`
- `latent_ground_truth`
- `latent_supervision`
- `latent_seq_lens`
- `cot_chunk_token_ids`
- `num_latent_steps`

### Building the SFT Dataset

Builder:

- [build_r1onevision_thinking.py](data/build_r1onevision_thinking.py)

Recommended two-phase build:

```bash
python Qwen/data/build_r1onevision_thinking.py --render-only --all
CUDA_VISIBLE_DEVICES=0 python Qwen/data/build_r1onevision_thinking.py --encode-only --all
```

Multi-GPU encode example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python Qwen/data/build_r1onevision_thinking.py --encode-only --all --num-gpus 4
```

Builder behavior:

- renders question images and thinking-chunk images
- encodes both through the native Qwen3VL vision stack into LLM hidden-size features
- stores latent tensors in cache files and writes the final JSONL manifest
- computes `cot_chunk_token_ids` from text chunks for latent-step CE supervision

### Sequence and Latent Semantics

The assistant target is structured as:

- `<think><latent>...<think_sep>... </think>{answer}`

Common single-step case:

- `<think><latent></think>{answer}`

At train time:

1. The collator expands each `<latent>` token so its token span matches the latent sequence length for that thinking step.
2. Those expanded positions receive injected embeddings from `latent_ground_truth`.
3. Standard CE still supervises structural tokens like `<think>`, `</think>`, and `<think_sep>`.
4. Latent-region CE can also supervise the expanded positions.
5. Hidden-state latent losses are computed on shifted hidden states around those positions.

Implementation lives in [integration.py](llamafactory/integration.py).

### Loss Terms

The runtime env config in [qwen3vl_runtime_env.yaml](configs/qwen3vl_runtime_env.yaml) is the shared source of truth for latent-runtime behavior used by:

- training wrappers
- transparent backfill
- the vLLM thinking plugin
- benchmark evaluation

Current default runtime latent settings:

- `loss_type: ce+vae`
- `latent_aux_loss_source: hidden`
- `latent_ce_token: 0`
- `match_strategy: truncate`
- `latent_supervision: 1`

What each loss term means in the active implementation:

- `ce`: standard token CE
- `mse`: shifted hidden states vs `latent_ground_truth`
- `repa`: cosine-style alignment vs `latent_ground_truth`
- `ot`: optimal transport vs `latent_supervision`
- `nce`: contrastive loss vs `latent_supervision`
- `vae`: latent VAE negative log-likelihood against `latent_ground_truth`
- `vae_ce`: second forward pass with sampled latent embeddings inserted, reusing the same CE targets

The second forward implementation is in `_compute_pred_embed_forward_loss` inside [integration.py](llamafactory/integration.py).

### Curriculum and Stage Splitting

The SFT wrapper does not just pass the config through unchanged. It can split training into separate stage runs based on runtime curriculum boundaries.

Current runtime defaults from [qwen3vl_runtime_env.yaml](configs/qwen3vl_runtime_env.yaml):

- `curriculum_enable: 1`
- `curriculum_epochs: 0,1,2`
- `curriculum_loss_types: ce+mse:0.4+ot:0.4,vae,ce+vae`
- `curriculum_vae_trainable: 0,1,0`
- `curriculum_lora_trainable: 1,0,1`
- `curriculum_aux_source: hidden,hidden,hidden`
- `curriculum_latent_ce: 1,0,0`

With the wrapper's stage splitting enabled, these become:

- Stage 1: epoch window `[0,1)`, `num_train_epochs=1.0`, LoRA trainable, VAE disabled, gold-latent-conditioned sequence modeling
- Stage 2: epoch window `[1,2)`, `num_train_epochs=1.0`, LoRA frozen, VAE trainable, main-path `vae` only
- Stage 3: epoch window `[2,3)`, `num_train_epochs=1.0`, LoRA trainable, VAE frozen but active, `ce+vae`

Important details:

- even though the YAML itself has `num_train_epochs: 1.0`, the wrapper derives a total three-stage run because the final curriculum span is inferred from the boundary list
- later stages load the previous stage handoff from `run_*/checkpoint_latest`
- if `vae.safetensors` exists in the handoff directory, the wrapper exports it so the trained VAE is restored explicitly

Stage outputs are written to:

- `run_*/stage_1`
- `run_*/stage_2`

After a successful stage, the wrapper copies that stage's `checkpoint_latest` to the top-level handoff directory:

- `run_*/checkpoint_latest`

### Backfill and Benchmarks

If training succeeds, the wrapper can launch:

- transparent-eval backfill across all discovered checkpoints
- benchmarks on the latest checkpoint

Relevant scripts:

- [backfill_transparent_eval.py](inference/backfill_transparent_eval.py)
- [run_all_benchmarks.py](evaluation/run_all_benchmarks.py)

Current runtime defaults:

- `backfill_enable: 1`
- `benchmark_enable: 1`
- `benchmark_list: MathVision,MMMU,RealWorldQA`
- `benchmark_num_samples: 100`

The wrapper also emits:

- `training.log`
- `dataset_mix_summary.json` when dataset mixing or ratios are used
- `rerun_evals.sh` for rerunning backfill and benchmark jobs later

## Chimera GSPO RL

### Dataset Build

Builder:

- [build_chimera_verl_dataset.py](data/build_chimera_verl_dataset.py)

Default source data:

- `$ROOT_DIR/huggingface/TianHongZXY/CHIMERA/Qwen3.5-397B`

Default output:

- `Qwen/data/chimera_verl/train.parquet`
- `Qwen/data/chimera_verl/val.parquet`

Build command:

```bash
$ROOT_DIR/envs/ocrflow/bin/python Qwen/data/build_chimera_verl_dataset.py --data-dir "$ROOT_DIR/huggingface/TianHongZXY/CHIMERA/Qwen3.5-397B"
```

Builder behavior:

- reads Chimera parquet shards
- uses pre-rendered question images from `Qwen/data/chimera_images` when available
- falls back to text-only prompts if an image is missing
- writes VERL-format records with prompt, image bytes, reward metadata, and extra per-sample info

Each RL record includes:

- `prompt`
- `images`
- `reward_model.ground_truth`
- `reward_model.equivalent_answers`
- `extra_info.question`
- `extra_info.solution`
- `extra_info.original_solution`

### RL Launcher Defaults

The active wrapper is [train_qwen3vl_chimera_gspo.sh](scripts/train_qwen3vl_chimera_gspo.sh).

Current defaults:

- base config: `Qwen/configs/rl/chimera_gspo.yaml`
- base model: `$ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking`
- dataset dir: `Qwen/data/chimera_verl`
- images dir: `Qwen/data/chimera_images`
- output dir: `Qwen/checkpoints/qwen3vl-2b/verl/chimera_gspo/run_*`
- project name: `qwen3vl-chimera-gspo`
- resume mode: `disable`

The wrapper:

- auto-counts GPUs from `CUDA_VISIBLE_DEVICES` or `torch.cuda.device_count()`
- if `CUDA_VISIBLE_DEVICES` is unset, exports all visible GPUs as a comma list
- validates dataset files before launch
- validates `INIT_LORA_PATH` if provided
- validates `RESUME_FROM_PATH` when `RESUME_MODE=resume_path`
- copies config snapshots into the run directory
- writes `resolved_runtime_config.yaml`
- writes `launcher_effective_env.snapshot.txt`

### RL Config Semantics

The project config is [chimera_gspo.yaml](configs/rl/chimera_gspo.yaml).

Key settings:

- reward function: `Qwen/verl/chimera_gspo_reward.py::compute_score_batch`
- reward manager: `dapo_batch`
- policy loss mode: `gspo`
- advantage estimator: `grpo`
- rollout backend: `vllm`
- rollout mode: `sync`
- rollout samples per prompt: `n=8`
- rollout tensor parallel size: defaults to `1` unless overridden
- actor LoRA target modules: `all-linear`
- actor LoRA exclude modules: `.*visual.*`
- actor freeze vision tower: `true`
- actor and ref FSDP param offload: enabled
- actor KL loss: disabled by default
- `trainer.total_epochs`: `2`

Selected default batch and length values:

- `data.gen_batch_size`: GPU-count-adaptive
- `data.train_batch_size`: GPU-count-adaptive
- `data.val_batch_size`: `64`
- `data.max_prompt_length`: `2048`
- `data.max_response_length`: `16384`
- `actor.ppo_max_token_len_per_gpu`: `36864`
- `ref.log_prob_max_token_len_per_gpu`: `55296`
- `rollout.max_model_len`: GPU-count-adaptive, defaulting to `18432`

### LoRA Init and Resume Modes

Bootstrapping RL from a prior SFT LoRA:

```bash
INIT_LORA_PATH=Qwen/checkpoints/qwen3vl-2b/lora/r1_onevision_thinking/run_vae_mse_ot_2ce_subcot/checkpoint-729 \
bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh
```

Resuming a VERL run:

```bash
RESUME_MODE=resume_path \
RESUME_FROM_PATH=/abs/path/to/verl_run/global_step_40 \
bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh
```

Meaning:

- `INIT_LORA_PATH` points to a PEFT LoRA adapter to initialize actor training and rollout
- `RESUME_MODE=resume_path` points to a VERL checkpoint tree containing trainer state

The wrapper also exports:

- `VLLM_LORA_CHECKPOINT_PATH=$INIT_LORA_PATH` when init LoRA is provided

This allows the vLLM thinking plugin to find the same adapter directory, including optional `vae.safetensors`.

### Reward Function

Reward implementation:

- [chimera_gspo_reward.py](verl/chimera_gspo_reward.py)

Behavior:

- strips `<think>` regions and other XML-like tags
- extracts candidate answers from boxed expressions, explicit answer markers, last line, or full text
- builds acceptable gold candidates from `ground_truth` and `equivalent_answers`
- normalizes multiple-choice answers
- tries symbolic equivalence with `sympy` and `latex2sympy2`
- returns `1.0` for a correct answer and `0.0` otherwise
- scores batches in parallel using a thread pool

Parallelism knob:

- `REWARD_COMPUTATION_WORKERS`, default `16`

### Runtime and Compatibility Path

The RL wrapper exports:

- `PYTHONPATH=$REPO_ROOT:$REPO_ROOT/vllm_thinking_plugin:${PYTHONPATH:-}`
- `VLLM_PLUGINS=vllm_thinking`
- `VLLM_THINKING=1`
- `VLLM_WORKER_MULTIPROC_METHOD=spawn`
- `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`

[run_verl_ppo.py](verl/run_verl_ppo.py) then:

- loads VERL's generated PPO base config
- merges one or more project config overlays
- normalizes accidental `+ray_kwargs` overrides back into `ray_kwargs`
- injects repo-local Ray worker runtime env patches
- launches the real VERL PPO trainer

### Overlay Config Caveat

The shell wrapper comment claims positional overlay config support:

```bash
bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh [overlay_config.yaml]
```

But the current implementation does not consume `$1` into `PROJECT_CONFIG`. Right now overlay configs only work when passed via environment variable:

```bash
PROJECT_CONFIG=/abs/path/to/overlay.yaml bash Qwen/scripts/train_qwen3vl_chimera_gspo.sh
```

Treat the environment-variable form as authoritative until the wrapper is fixed.

## vLLM Thinking Plugin

The repo-local plugin is used by benchmark and RL-serving flows.

Relevant env vars:

- `VLLM_THINKING`
- `VLLM_FORCE_THINK`
- `MIN_CONTINUOUS_STEPS`
- `VLLM_LORA_CHECKPOINT_PATH`

The plugin can load `vae.safetensors` from the LoRA checkpoint directory when present. If that file is missing, it falls back to using the previous hidden state directly instead of VAE sampling.

Inference is intentionally not identical to SFT training:

- training uses explicit `<latent>` placeholders with injected latent embeddings
- vLLM continuous thinking re-feeds hidden states during generation based on request-time mode switching

Do not assume the inference path emits the same placeholder formatting as the training target.

## Linearized Patch-Embed Compatibility

This repo supports a linearized version of the Qwen3-VL patch embedding:

- converter: [convert_qwen3vl_patch_embed_to_linear.py](compat/convert_qwen3vl_patch_embed_to_linear.py)
- default converted checkpoint: `Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking`

Converter example:

```bash
python Qwen/compat/convert_qwen3vl_patch_embed_to_linear.py \
  --src $ROOT_DIR/huggingface/Qwen/Qwen3-VL-2B-Thinking \
  --dst Qwen/checkpoints/Qwen3-VL-Linear-2B-Thinking
```

This converted checkpoint is optional. The active SFT, RL, evaluation, and visualization paths now default to the original Hugging Face Qwen3-VL checkpoint and apply compatibility patches at runtime.

## Key Files

SFT:

- [train_qwen3vl_r1onevision.sh](scripts/train_qwen3vl_r1onevision.sh)
- [qwen3vl_r1onevision_thinking.yaml](configs/qwen3vl_r1onevision_thinking.yaml)
- [qwen3vl_runtime_env.yaml](configs/qwen3vl_runtime_env.yaml)
- [build_r1onevision_thinking.py](data/build_r1onevision_thinking.py)
- [integration.py](llamafactory/integration.py)

RL:

- [train_qwen3vl_chimera_gspo.sh](scripts/train_qwen3vl_chimera_gspo.sh)
- [chimera_gspo.yaml](configs/rl/chimera_gspo.yaml)
- [run_verl_ppo.py](verl/run_verl_ppo.py)
- [build_chimera_verl_dataset.py](data/build_chimera_verl_dataset.py)
- [chimera_gspo_reward.py](verl/chimera_gspo_reward.py)
