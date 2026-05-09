"""Worker setup hook for repo-local VERL compatibility patches."""

from __future__ import annotations

import os
import sys
import importlib.abc


def _apply_deferred_runtime_patches() -> None:
    if os.environ.get("QWEN3VL_DEFER_VERL_PATCHES") == "1":
        os.environ.pop("QWEN3VL_DEFER_VERL_PATCHES", None)
        import sitecustomize

        sitecustomize._configure_torch_sharing_strategy()
        sitecustomize._patch_once()

    from verl_compat import apply_runtime_compat_patches

    apply_runtime_compat_patches()


class _DeferredRuntimePatchFinder(importlib.abc.MetaPathFinder):
    _TARGETS = {
        "verl.single_controller.base.worker": "_patch_worker_module",
        "verl.workers.rollout.vllm_rollout.vllm_async_server": "_patch_runtime_module",
    }

    def find_spec(self, fullname, path=None, target=None):
        patch_name = self._TARGETS.get(fullname)
        if patch_name is None:
            return None

        for finder in sys.meta_path:
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            spec = find_spec(fullname, path, target)
            if spec is not None:
                if spec.loader is not None:
                    spec.loader = _DeferredRuntimePatchLoader(spec.loader, patch_name)
                return spec
        return None


class _DeferredRuntimePatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader, patch_name: str):
        self._wrapped_loader = wrapped_loader
        self._patch_name = patch_name

    def create_module(self, spec):
        create_module = getattr(self._wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        if self._patch_name == "_patch_worker_module":
            _patch_worker_class(module.Worker)
        else:
            _apply_deferred_runtime_patches()


def _patch_worker_class(worker_cls) -> None:
    original_init = worker_cls.__init__
    if getattr(original_init, "_qwen3vl_compat_patch", False):
        return

    def compat_init(self, *args, **kwargs):
        result = original_init(self, *args, **kwargs)
        _apply_deferred_runtime_patches()
        return result

    compat_init._qwen3vl_compat_patch = True
    worker_cls.__init__ = compat_init


def _install_deferred_runtime_patch() -> None:
    worker_target = "verl.single_controller.base.worker"
    loaded_module = sys.modules.get(worker_target)
    if loaded_module is not None and hasattr(loaded_module, "Worker"):
        _patch_worker_class(loaded_module.Worker)

    server_target = "verl.workers.rollout.vllm_rollout.vllm_async_server"
    if server_target in sys.modules:
        _apply_deferred_runtime_patches()

    if not any(isinstance(finder, _DeferredRuntimePatchFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _DeferredRuntimePatchFinder())


def collect_patch_diagnostics() -> dict:
    # These imports intentionally stay lazy: Ray imports this setup-hook module
    # before assigning actor-local CUDA_VISIBLE_DEVICES.
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager
    from verl.utils.vllm.utils import VLLMHijack

    rollout_module = sys.modules.get("verl.workers.rollout.vllm_rollout.vllm_rollout")
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
    os.environ["QWEN3VL_APPLY_VERL_PATCHES"] = "1"
    _install_deferred_runtime_patch()
