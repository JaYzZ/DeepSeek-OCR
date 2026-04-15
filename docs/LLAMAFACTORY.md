# LlamaFactory Infra in This Repo

This document is repo-focused. It explains how this repo uses upstream LlamaFactory, where our overlay lives, and where we should make changes.

Upstream repo:

- `$ROOT_DIR/sources/LlamaFactory`

Repo-local overlay:

- `$ROOT_DIR/sources/DeepSeek-OCR/Qwen/llamafactory`
- `$ROOT_DIR/sources/DeepSeek-OCR/sitecustomize.py`
- `$ROOT_DIR/sources/DeepSeek-OCR/Qwen/scripts/train_*.sh`

The operating rule is:

- upstream LlamaFactory provides the generic training framework
- this repo provides Qwen-specific behavior through wrappers and monkey patches
- we should prefer changing this repo, not `../LlamaFactory`

## Main Control Flow

Training usually starts from one of:

- [`Qwen/scripts/train_qwen3vl_r1onevision.sh`](../Qwen/scripts/train_qwen3vl_r1onevision.sh)
- [`Qwen/scripts/train_qwen3vl_chimera.sh`](../Qwen/scripts/train_qwen3vl_chimera.sh)

These wrappers:

- load the main training yaml
- load the shared runtime env yaml
- export `QWEN3VL_*` env vars
- set `PYTHONPATH` to include both `../LlamaFactory/src` and this repo root
- launch upstream LlamaFactory

Because this repo root is on `PYTHONPATH`, Python imports:

- [`sitecustomize.py`](../sitecustomize.py)

That file is the patch entrypoint.

## `sitecustomize.py`

File:

- [`sitecustomize.py`](../sitecustomize.py)

Purpose:

- acts as the early patch hub
- enables Qwen patches when `QWEN3VL_LATENT_SUPERVISION=1`
- imports `Qwen.llamafactory.integration`
- applies patches before most LlamaFactory code runs

It also sets torch multiprocessing sharing strategy to `file_system`.

This is the main reason we can keep upstream LlamaFactory untouched.

## Repo-Local LlamaFactory Overlay

Main files:

- [`Qwen/llamafactory/integration.py`](../Qwen/llamafactory/integration.py)
- [`Qwen/llamafactory/vae_callback.py`](../Qwen/llamafactory/vae_callback.py)
- [`Qwen/llamafactory/curriculum_callback.py`](../Qwen/llamafactory/curriculum_callback.py)
- [`Qwen/llamafactory/transparent_eval_callback.py`](../Qwen/llamafactory/transparent_eval_callback.py)

This is our actual integration layer.

What it owns:

- Qwen3VL latent supervision behavior
- extra dataset fields not understood by upstream LlamaFactory
- latent token expansion in the collator
- latent embedding injection
- custom latent losses
- latent VAE creation and save behavior
- auto-registration of Qwen-specific callbacks
- final save layout policy such as `checkpoint_latest/`

What it does not own:

- generic trainer lifecycle
- generic dataset loading framework
- generic optimizer/scheduler/training loop logic

Those remain upstream.

## Upstream LlamaFactory Areas We Depend On

Important upstream areas:

- `../LlamaFactory/src/llamafactory/data`
- `../LlamaFactory/src/llamafactory/model`
- `../LlamaFactory/src/llamafactory/train`
- `../LlamaFactory/src/llamafactory/hparams`

For SFT, the most relevant upstream flow is:

- `llamafactory.cli train`
- `llamafactory/train/sft/workflow.py`
- `llamafactory/train/sft/trainer.py`
- `llamafactory/train/callbacks.py`

The conceptual flow is:

1. parse configs and CLI overrides
2. load tokenizer, template, dataset, and model
3. build trainer
4. run training
5. save periodic checkpoints to `output_dir/checkpoint-N`
6. save final artifacts to `output_dir`

Our overlay intercepts only selected parts of that flow.

## Data Contract Owned by This Repo

Dataset builders in this repo create fields upstream LlamaFactory does not natively know about.

Main builders:

- [`Qwen/data/build_r1onevision_thinking.py`](../Qwen/data/build_r1onevision_thinking.py)
- [`Qwen/data/build_chimera_thinking.py`](../Qwen/data/build_chimera_thinking.py)
- [`Qwen/data/build_deepvision_thinking.py`](../Qwen/data/build_deepvision_thinking.py)

Important extra fields:

- `latent_ground_truth`
- `latent_supervision`
- `latent_seq_lens`
- `cot_chunk_token_ids`
- `num_latent_steps`

The meaning of those fields is implemented in:

- [`Qwen/llamafactory/integration.py`](../Qwen/llamafactory/integration.py)

So if a dataset field changes, the first place to inspect is our repo-local integration layer, not upstream LlamaFactory.

## Runtime Env Contract

Shared runtime env yaml:

- [`Qwen/configs/qwen3vl_runtime_env.yaml`](../Qwen/configs/qwen3vl_runtime_env.yaml)

The training wrappers export those values as environment variables, such as:

- token ids
- loss spec
- curriculum settings
- latent-step CE settings
- transparent eval settings
- vLLM-related thinking settings

This yaml is the shared contract across training, backfill, and vLLM-side scripts.

If training behavior changes and inference must stay aligned, this file is usually part of the change.

## How We Patch Trainer Behavior

The trainer patching entrypoint is inside:

- [`Qwen/llamafactory/integration.py`](../Qwen/llamafactory/integration.py)

Important design:

- we import upstream classes
- we keep original methods
- we wrap only the parts we need

Examples of trainer-facing behavior owned here:

- callback auto-registration
- VAE optimizer integration
- final `save_model()` redirection to `checkpoint_latest/`
- processor save redirection at train end

This is the preferred place for repo-specific save policy or callback policy changes.

## Where To Change Things

### If you want to change training launch behavior

Edit:

- `Qwen/scripts/train_qwen3vl_r1onevision.sh`
- `Qwen/scripts/train_qwen3vl_chimera.sh`

Examples:

- env var exports
- output directory conventions
- backfill launch policy
- benchmark launch policy

### If you want to change loss behavior

Edit:

- `Qwen/configs/qwen3vl_runtime_env.yaml`
- `Qwen/llamafactory/integration.py`

Examples:

- add/remove a loss term
- change latent-step CE behavior
- change VAE weighting
- change matching strategy

### If you want to change latent data semantics

Edit:

- dataset builder in `Qwen/data/build_*.py`
- `Qwen/llamafactory/integration.py`

Examples:

- new metadata fields
- different latent target structure
- different expansion lengths

### If you want to change final save layout

Edit:

- `Qwen/llamafactory/integration.py`

Do not start by editing upstream `workflow.py` in `../LlamaFactory`.

### If you want to change transparent eval or curriculum behavior

Edit:

- `Qwen/llamafactory/transparent_eval_callback.py`
- `Qwen/llamafactory/curriculum_callback.py`
- `Qwen/llamafactory/vae_callback.py`

## What We Should Avoid

Avoid direct edits to:

- `../LlamaFactory/src/...`

unless the issue is truly generic and intended to be maintained upstream.

Reasons:

- upstream changes are harder to track
- our Qwen-specific logic does not belong there
- upgrades become harder if we fork the upstream training framework

## Fragile Areas

Most fragile repo-local file:

- [`Qwen/llamafactory/integration.py`](../Qwen/llamafactory/integration.py)

Why:

- it patches upstream internals
- it depends on upstream trainer/data/model signatures
- it contains both data-path and model-path logic

When changing it:

- keep patches narrow
- preserve original methods
- guard against double patching
- verify with at least `python -m py_compile`

## Recommended Debug Order

If training behavior is wrong, inspect in this order:

1. [`Qwen/scripts/train_qwen3vl_r1onevision.sh`](../Qwen/scripts/train_qwen3vl_r1onevision.sh)
2. [`Qwen/configs/qwen3vl_runtime_env.yaml`](../Qwen/configs/qwen3vl_runtime_env.yaml)
3. [`sitecustomize.py`](../sitecustomize.py)
4. [`Qwen/llamafactory/integration.py`](../Qwen/llamafactory/integration.py)
5. the relevant upstream path under `../LlamaFactory/src/llamafactory`

## Short Policy

For LlamaFactory-related changes in this workspace:

- prefer wrappers, runtime env, and `Qwen/llamafactory`
- treat `../LlamaFactory` as upstream infra
- only edit upstream when the change is genuinely generic
