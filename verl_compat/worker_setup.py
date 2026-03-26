"""Worker setup hook for repo-local VERL compatibility patches."""

from __future__ import annotations

import os
import sys

from .bootstrap import apply_runtime_compat_patches
from vllm.lora.models import LoRAModel
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager
from verl.utils.vllm.utils import VLLMHijack


def collect_patch_diagnostics() -> dict:
    rollout_module = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_rollout_spmd")
    sharding_module = sys.modules.get("verl.workers.sharding_manager.fsdp_vllm")

    return {
        "compat_env_flag": os.environ.get("QWEN3VL_APPLY_VERL_PATCHES"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "external_hijack_module": VLLMHijack.__module__,
        "local_rank": os.environ.get("LOCAL_RANK"),
        "lora_from_tensors_module": LoRAModel.from_lora_tensors.__module__,
        "lora_from_tensors_patched": bool(
            getattr(LoRAModel.from_lora_tensors, "_qwen3vl_compat_patch", False)
            or getattr(getattr(LoRAModel.from_lora_tensors, "__func__", None), "_qwen3vl_compat_patch", False)
        ),
        "python_executable": sys.executable,
        "python_path_head": sys.path[:5],
        "ray_noset_cuda_visible_devices": os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"),
        "worker_load_adapter_module": WorkerLoRAManager._load_adapter.__module__,
        "worker_load_adapter_patched": bool(
            getattr(WorkerLoRAManager._load_adapter, "_qwen3vl_compat_patch", False)
        ),
        "lru_load_adapter_module": LRUCacheWorkerLoRAManager._load_adapter.__module__,
        "lru_load_adapter_patched": bool(
            getattr(LRUCacheWorkerLoRAManager._load_adapter, "_qwen3vl_compat_patch", False)
        ),
        "rollout_module_loaded": rollout_module is not None,
        "rollout_hijack_module": getattr(getattr(rollout_module, "VLLMHijack", None), "__module__", None),
        "rollout_tensor_request_module": getattr(
            getattr(rollout_module, "TensorLoRARequest", None),
            "__module__",
            None,
        ),
        "sharding_module_loaded": sharding_module is not None,
        "sharding_hijack_module": getattr(getattr(sharding_module, "VLLMHijack", None), "__module__", None),
        "sharding_tensor_request_module": getattr(
            getattr(sharding_module, "TensorLoRARequest", None),
            "__module__",
            None,
        ),
    }


def apply_worker_compat_patches() -> None:
    apply_runtime_compat_patches()
