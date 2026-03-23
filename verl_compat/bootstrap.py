"""Bootstrap entrypoint for repo-local VERL compatibility patches."""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
_PATCHED = False
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SETUP_HOOK = "verl_compat.worker_setup.apply_worker_compat_patches"


def build_repo_pythonpath(existing_pythonpath: str) -> str:
    pythonpath_parts = [str(REPO_ROOT), str(REPO_ROOT / "vllm_thinking_plugin")]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return os.pathsep.join(part for part in pythonpath_parts if part)


def patch_worker_env_vars(existing_env_vars: dict | None) -> dict:
    env_vars = dict(existing_env_vars or {})
    env_vars["PYTHONPATH"] = build_repo_pythonpath(env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", "")))
    env_vars["QWEN3VL_APPLY_VERL_PATCHES"] = "1"
    env_vars["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = os.environ.get(
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "1",
    )

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        env_vars["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    runtime_env_stamp = os.environ.get("QWEN3VL_RUNTIME_ENV_STAMP")
    if runtime_env_stamp:
        env_vars["QWEN3VL_RUNTIME_ENV_STAMP"] = runtime_env_stamp

    return env_vars


def patch_runtime_env(runtime_env: dict | None) -> dict:
    patched_runtime_env = dict(runtime_env or {})
    patched_runtime_env["env_vars"] = patch_worker_env_vars(patched_runtime_env.get("env_vars"))
    patched_runtime_env.setdefault("worker_process_setup_hook", WORKER_SETUP_HOOK)
    return patched_runtime_env


def _patch_verl_vllm_module() -> None:
    """Patch the active external ``verl.utils.vllm`` module in place."""

    from .vllm_shim import TensorLoRARequest, VLLMHijack, is_version_ge

    external_utils = importlib.import_module("verl.utils.vllm.utils")
    external_pkg = importlib.import_module("verl.utils.vllm")

    # Patch the already-imported external classes/functions in place so later
    # imports or external re-hijack calls cannot restore the stale behavior.
    if hasattr(external_utils, "TensorLoRARequest"):
        external_utils.TensorLoRARequest.peft_config = TensorLoRARequest.peft_config
        external_utils.TensorLoRARequest.lora_tensors = TensorLoRARequest.lora_tensors

    if hasattr(external_utils, "VLLMHijack"):
        external_utils.VLLMHijack.hijack = staticmethod(VLLMHijack.hijack)

    external_utils.TensorLoRARequest = TensorLoRARequest
    external_utils.VLLMHijack = VLLMHijack
    external_utils.is_version_ge = is_version_ge
    external_utils.__all__ = ["TensorLoRARequest", "VLLMHijack", "is_version_ge"]

    external_pkg.TensorLoRARequest = TensorLoRARequest
    external_pkg.VLLMHijack = VLLMHijack
    external_pkg.is_version_ge = is_version_ge
    external_pkg.__all__ = ["TensorLoRARequest", "VLLMHijack", "is_version_ge"]

    # Reinstall the actual runtime hooks on vLLM classes.
    VLLMHijack.hijack()
    logger.warning("Patched external verl.utils.vllm in place with repo-local compatibility shim.")


def _patch_imported_runtime_aliases() -> None:
    """Repair stale `from verl.utils.vllm import ...` aliases in already-imported modules."""

    from .vllm_shim import TensorLoRARequest, VLLMHijack, is_version_ge

    patched_modules = []
    for module_name in (
        "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
        "verl.workers.sharding_manager.fsdp_vllm",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue

        module.TensorLoRARequest = TensorLoRARequest
        module.VLLMHijack = VLLMHijack
        module.is_version_ge = is_version_ge
        patched_modules.append(module_name)

    if patched_modules:
        VLLMHijack.hijack()
        logger.warning(
            "Patched already-imported VERL runtime aliases in modules: %s",
            ", ".join(sorted(patched_modules)),
        )


def _patch_ray_actor_runtime_env() -> None:
    """Ensure VERL rollout actors inherit the local patch env and setup hook."""

    external_ray_base = importlib.import_module("verl.single_controller.ray.base")
    ray_class_with_init = external_ray_base.RayClassWithInitArgs
    original_update_options = ray_class_with_init.update_options

    if getattr(original_update_options, "_qwen3vl_compat_patch", False):
        return

    def compat_update_options(self, options: dict) -> None:
        merged_options = dict(options)
        merged_options["runtime_env"] = patch_runtime_env(merged_options.get("runtime_env"))

        original_update_options(self, merged_options)

    compat_update_options._qwen3vl_compat_patch = True
    ray_class_with_init.update_options = compat_update_options
    logger.warning("Patched VERL Ray actor runtime env to carry repo-local compat hooks.")


def apply_runtime_compat_patches() -> None:
    """Apply repo-local compatibility patches for the active VERL stack."""

    global _PATCHED
    if _PATCHED:
        return

    _patch_verl_vllm_module()
    _patch_imported_runtime_aliases()
    _patch_ray_actor_runtime_env()
    _PATCHED = True
    logger.warning("Applied repo-local VERL compatibility patches.")
