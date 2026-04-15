# vLLM Infra in This Repo

This document is repo-focused. It explains how this repo uses upstream vLLM, where our overlay lives, and where we should make changes.

Upstream repo:

- `$ROOT_DIR/sources/vllm`

Repo-local overlay:

- `$ROOT_DIR/sources/DeepSeek-OCR/vllm_thinking_plugin`
- `$ROOT_DIR/sources/DeepSeek-OCR/Qwen/scripts/vllm_server.py`
- `$ROOT_DIR/sources/DeepSeek-OCR/Qwen/scripts/vllm_utils.py`
- `$ROOT_DIR/sources/DeepSeek-OCR/Qwen/scripts/backfill_transparent_eval.py`

The operating rule is:

- upstream vLLM provides the runtime and serving engine
- this repo provides Qwen3VL-specific behavior through a plugin and wrappers
- we should prefer changing this repo, not `../vllm`

## Main Control Flow

The common serving entrypoint is:

- [`Qwen/scripts/vllm_server.py`](../Qwen/scripts/vllm_server.py)

That script:

- loads the shared runtime env yaml through helper code
- ensures `VLLM_PLUGINS` includes `vllm_thinking`
- imports the thinking plugin before importing `vllm`
- constructs `LLM(...)`
- optionally prepares LoRA usage
- serves a FastAPI HTTP interface

Backfill and benchmark paths use the same general contract:

- shared runtime env
- plugin-enabled vLLM process
- optional LoRA checkpoint path

## Repo-Local vLLM Overlay

Main files:

- [`vllm_thinking_plugin/vllm_thinking/__init__.py`](../vllm_thinking_plugin/vllm_thinking/__init__.py)
- [`vllm_thinking_plugin/vllm_thinking/runner_patch.py`](../vllm_thinking_plugin/vllm_thinking/runner_patch.py)
- [`vllm_thinking_plugin/vllm_thinking/qwen3vl_patch_embed_patch.py`](../vllm_thinking_plugin/vllm_thinking/qwen3vl_patch_embed_patch.py)
- [`vllm_thinking_plugin/setup.py`](../vllm_thinking_plugin/setup.py)

This is our actual vLLM patch layer.

What it owns:

- Qwen3-VL patch-embed compatibility for our checkpoints
- thinking-mode runtime behavior
- continuous latent AR behavior
- VAE loading from LoRA/model paths
- request state tracking for thinking mode

What it does not own:

- generic scheduler/runtime engine
- generic OpenAI API server behavior
- generic LoRA engine behavior

Those remain upstream.

## Plugin Entry Design

The plugin entrypoint is exposed by:

- [`vllm_thinking_plugin/setup.py`](../vllm_thinking_plugin/setup.py)

The runtime bootstrap is:

- [`vllm_thinking_plugin/vllm_thinking/__init__.py`](../vllm_thinking_plugin/vllm_thinking/__init__.py)

Its logic is:

1. apply the Qwen3-VL patch-embed compatibility patch
2. check `VLLM_THINKING`
3. if enabled, apply the thinking-mode runtime patch

This means the plugin has two layers:

- model-loading compatibility
- decode/runtime behavior

## Model Loading Compatibility Patch

File:

- [`vllm_thinking_plugin/vllm_thinking/qwen3vl_patch_embed_patch.py`](../vllm_thinking_plugin/vllm_thinking/qwen3vl_patch_embed_patch.py)

Purpose:

- patches vLLM Qwen3-VL patch embed weight loading
- allows linearized patch-embed weights to load correctly

This is the correct place for model-load compatibility fixes that are specific to our checkpoints.

It is not the right place for generation-mode logic.

## Thinking Runtime Patch

File:

- [`vllm_thinking_plugin/vllm_thinking/runner_patch.py`](../vllm_thinking_plugin/vllm_thinking/runner_patch.py)

Purpose:

- patches `vllm.v1.worker.gpu_model_runner.GPUModelRunner`
- manages continuous hidden-state AR for Qwen3VL thinking mode
- resolves token IDs from shared env vars
- loads the latent VAE from checkpoint/model path
- decides whether a request starts in discrete mode or continuous mode
- uses prompt-embed paths for next-step latent embedding injection

This is the most version-sensitive part of the vLLM overlay.

## Runtime Env Contract

Shared env loader:

- [`Qwen/scripts/vllm_utils.py`](../Qwen/scripts/vllm_utils.py)

Shared runtime yaml:

- [`Qwen/configs/qwen3vl_runtime_env.yaml`](../Qwen/configs/qwen3vl_runtime_env.yaml)

Important env values for vLLM:

- `VLLM_THINKING`
- `VLLM_FORCE_THINK`
- `VLLM_ENFORCE_EAGER`
- `MIN_CONTINUOUS_STEPS`
- `QWEN3VL_THINKING_START_ID`
- `QWEN3VL_THINKING_END_ID`
- `QWEN3VL_LATENT_TOKEN_ID`
- `QWEN3VL_THINKING_SEP_ID`

This is the alignment layer between training semantics and inference semantics.

If token ids or thinking behavior change in training, these values must stay aligned.

## Server Wrapper Design

Server wrapper:

- [`Qwen/scripts/vllm_server.py`](../Qwen/scripts/vllm_server.py)

This file owns:

- env setup before vLLM import
- plugin activation
- request preprocessing
- chat template handling
- server API shape used by our local tools

This is the right place for:

- server startup policy
- request conversion policy
- LoRA selection policy
- API-facing behavior

This is not the right place for low-level decode loop patching.

## Backfill and Evaluation Paths

Main file:

- [`Qwen/scripts/backfill_transparent_eval.py`](../Qwen/scripts/backfill_transparent_eval.py)

This script uses the same vLLM-side conventions:

- shared runtime env
- consistent token IDs
- checkpoint-aware loading
- plugin-enabled inference behavior

If evaluation behavior diverges from server behavior, this script is one of the first places to inspect.

## Upstream vLLM Areas We Depend On

Important upstream areas:

- `../vllm/vllm/plugins`
- `../vllm/vllm/v1/worker`
- `../vllm/vllm/model_executor/models`
- `../vllm/vllm/lora`
- `../vllm/vllm/entrypoints`

The most important direct dependency for our runtime patch is:

- `vllm.v1.worker.gpu_model_runner.GPUModelRunner`

This means our plugin is tightly coupled to upstream vLLM internals.

## Where To Change Things

### If you want to change server startup or API behavior

Edit:

- `Qwen/scripts/vllm_server.py`
- `Qwen/scripts/vllm_utils.py`

Examples:

- startup env setup
- plugin enable policy
- request preprocessing
- tensor parallel inference policy

### If you want to change thinking-mode runtime behavior

Edit:

- `vllm_thinking_plugin/vllm_thinking/runner_patch.py`

Examples:

- mode switching rules
- latent injection behavior
- VAE sampling behavior
- request state structure

### If you want to change model-loading compatibility

Edit:

- `vllm_thinking_plugin/vllm_thinking/qwen3vl_patch_embed_patch.py`

Examples:

- checkpoint weight reshaping
- Qwen3-VL-specific load quirks

### If you want to change shared vLLM runtime flags

Edit:

- `Qwen/configs/qwen3vl_runtime_env.yaml`
- `Qwen/scripts/vllm_utils.py`

Examples:

- force-think behavior
- minimum continuous steps
- eager mode policy

## What We Should Avoid

Avoid direct edits to:

- `../vllm/vllm/...`

unless the issue is truly generic and meant to be maintained upstream.

Reasons:

- our thinking-mode behavior is project-specific
- plugin-based overlay is easier to maintain
- upstream updates are simpler when we keep our changes local

## Fragile Areas

Most fragile repo-local file:

- [`vllm_thinking_plugin/vllm_thinking/runner_patch.py`](../vllm_thinking_plugin/vllm_thinking/runner_patch.py)

Why:

- it patches internal vLLM classes
- it relies on method signatures and runtime data structures
- vLLM upgrades can break it even if public APIs look unchanged

When changing it:

- inspect the matching upstream class in `../vllm`
- keep patches narrow
- preserve original methods
- guard against double patching
- verify importability and basic execution path

## Recommended Debug Order

If vLLM behavior is wrong, inspect in this order:

1. [`Qwen/scripts/vllm_server.py`](../Qwen/scripts/vllm_server.py)
2. [`Qwen/scripts/vllm_utils.py`](../Qwen/scripts/vllm_utils.py)
3. [`Qwen/configs/qwen3vl_runtime_env.yaml`](../Qwen/configs/qwen3vl_runtime_env.yaml)
4. [`vllm_thinking_plugin/vllm_thinking/__init__.py`](../vllm_thinking_plugin/vllm_thinking/__init__.py)
5. [`vllm_thinking_plugin/vllm_thinking/runner_patch.py`](../vllm_thinking_plugin/vllm_thinking/runner_patch.py)
6. relevant upstream code under `../vllm/vllm`

## Short Policy

For vLLM-related changes in this workspace:

- prefer `Qwen/scripts/vllm_*` and `vllm_thinking_plugin`
- treat `../vllm` as upstream infra
- only edit upstream when the change is genuinely generic
