"""Bootstrap entrypoint for repo-local VERL compatibility patches."""

from __future__ import annotations

import importlib
import functools
import logging
import os
import shutil
import sys
from pathlib import Path

import torch.distributed as dist
from safetensors.torch import save_file

from .continuous_replay import _restore_policy_latent_vae, apply_continuous_replay_patches
from .vllm_shim import TensorLoRARequest, VLLMHijack, is_version_ge

logger = logging.getLogger(__name__)
_PATCHED = False
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_SETUP_HOOK = "verl_compat.worker_setup.apply_worker_compat_patches"
_VAE_ARTIFACT_LOGGED = False


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

    for passthrough_name in (
        "VLLM_THINKING",
        "VLLM_PLUGINS",
        "VLLM_LORA_CHECKPOINT_PATH",
        "VLLM_MODEL_PATH",
        "QWEN3VL_LATENT_SUPERVISION",
        "QWEN3VL_LOSS_TYPE",
        "QWEN3VL_VAE_INTERMEDIATE_SIZE",
    ):
        passthrough_value = os.environ.get(passthrough_name)
        if passthrough_value:
            env_vars[passthrough_name] = passthrough_value

    return env_vars


def patch_runtime_env(runtime_env: dict | None) -> dict:
    patched_runtime_env = dict(runtime_env or {})
    patched_runtime_env["env_vars"] = patch_worker_env_vars(patched_runtime_env.get("env_vars"))
    patched_runtime_env.setdefault("worker_process_setup_hook", WORKER_SETUP_HOOK)
    return patched_runtime_env


def _patch_verl_vllm_module() -> None:
    """Patch the active external ``verl.utils.vllm`` module in place."""

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

def _patch_imported_runtime_aliases() -> None:
    """Repair stale `from verl.utils.vllm import ...` aliases in already-imported modules."""

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

def _patch_fsdp_checkpoint_manager() -> None:
    """Guard VERL FSDP checkpoint export from PEFT source-layout issues."""

    external_ckpt = importlib.import_module("verl.utils.checkpoint.fsdp_checkpoint_manager")
    original_custom_object_save = external_ckpt.custom_object_save

    if getattr(original_custom_object_save, "_qwen3vl_compat_patch", False):
        return

    def compat_custom_object_save(obj, folder, config=None):
        try:
            return original_custom_object_save(obj, folder, config=config)
        except FileNotFoundError as exc:
            missing_path = str(getattr(exc, "filename", "") or exc)
            normalized_path = missing_path.replace("\\", "/")
            if "/site-packages/peft/" in normalized_path and normalized_path.endswith(".py"):
                logger.warning(
                    "Skipping custom_object_save because transformers referenced missing PEFT source file: %s",
                    missing_path,
                )
                return None
            raise

    compat_custom_object_save._qwen3vl_compat_patch = True
    external_ckpt.custom_object_save = compat_custom_object_save

def _resolve_latent_vae_state_dict(model) -> dict | None:
    """Find a latent_vae module through common model wrappers and return a CPU state dict."""

    visited = set()
    queue = [model]
    while queue:
        node = queue.pop(0)
        if node is None:
            continue
        node_id = id(node)
        if node_id in visited:
            continue
        visited.add(node_id)

        vae = getattr(node, "latent_vae", None)
        if vae is not None:
            return {name: tensor.detach().cpu() for name, tensor in vae.state_dict().items()}

        for attr_name in ("module", "_fsdp_wrapped_module", "model", "base_model", "pretrained_model"):
            child = getattr(node, attr_name, None)
            if child is not None:
                queue.append(child)

    return None


def _resolve_policy_latent_vae_state_dict(policy) -> dict | None:
    vae = getattr(policy, "latent_vae", None)
    if vae is None:
        return None
    return {name: tensor.detach().cpu() for name, tensor in vae.state_dict().items()}


def _copy_vae_artifact_for_actor(self, local_path: str) -> None:
    global _VAE_ARTIFACT_LOGGED
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    actor_model = getattr(self, "actor_module", None) or getattr(self, "actor_module_fsdp", None)
    vae_state_dict = _resolve_policy_latent_vae_state_dict(getattr(self, "actor", None))
    if vae_state_dict is None:
        vae_state_dict = _resolve_latent_vae_state_dict(actor_model)

    source_vae_path = None
    for candidate_dir in (
        self.config.model.get("lora_adapter_path"),
        os.environ.get("VLLM_LORA_CHECKPOINT_PATH"),
    ):
        if not candidate_dir:
            continue
        candidate_path = os.path.join(candidate_dir, "vae.safetensors")
        if os.path.isfile(candidate_path):
            source_vae_path = candidate_path
            break

    if vae_state_dict is None and source_vae_path is None:
        return

    output_paths = [os.path.join(local_path, "vae.safetensors")]
    if getattr(self, "_is_lora", False):
        output_paths.append(os.path.join(local_path, "lora_adapter", "vae.safetensors"))

    if rank == 0:
        for output_path in output_paths:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if vae_state_dict is not None:
                save_file(vae_state_dict, output_path)
            else:
                shutil.copy2(source_vae_path, output_path)
        if not _VAE_ARTIFACT_LOGGED:
            logger.warning("Saved latent VAE artifact(s) to %s", ", ".join(output_paths))
            _VAE_ARTIFACT_LOGGED = True

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _patch_actor_checkpoint_save() -> None:
    """Ensure RL actor checkpoints carry VAE artifacts alongside LoRA export."""

    external_workers = importlib.import_module("verl.workers.fsdp_workers")
    worker_cls = external_workers.ActorRolloutRefWorker
    original_save_checkpoint = worker_cls.save_checkpoint
    original_load_checkpoint = worker_cls.load_checkpoint

    if getattr(original_save_checkpoint, "_qwen3vl_compat_patch", False):
        return

    def compat_save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        result = original_save_checkpoint(
            self,
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        try:
            _copy_vae_artifact_for_actor(self, local_path)
        except Exception:
            logger.exception("Failed to export latent VAE artifact for actor checkpoint %s", local_path)
        return result

    compat_save_checkpoint = functools.wraps(original_save_checkpoint)(compat_save_checkpoint)
    compat_save_checkpoint.__dict__.update(original_save_checkpoint.__dict__)
    compat_save_checkpoint._qwen3vl_compat_patch = True
    worker_cls.save_checkpoint = compat_save_checkpoint

    def compat_load_checkpoint(self, local_path, hdfs_path=None, del_local_after_load=False):
        result = original_load_checkpoint(
            self,
            local_path=local_path,
            hdfs_path=hdfs_path,
            del_local_after_load=del_local_after_load,
        )
        try:
            if getattr(self, "actor", None) is not None:
                _restore_policy_latent_vae(self.actor, local_path)
            if getattr(self, "ref_policy", None) is not None:
                _restore_policy_latent_vae(self.ref_policy, local_path)
        except Exception:
            logger.exception("Failed to restore latent VAE after checkpoint load from %s", local_path)
        return result

    compat_load_checkpoint = functools.wraps(original_load_checkpoint)(compat_load_checkpoint)
    compat_load_checkpoint.__dict__.update(original_load_checkpoint.__dict__)
    compat_load_checkpoint._qwen3vl_compat_patch = True
    worker_cls.load_checkpoint = compat_load_checkpoint

def apply_runtime_compat_patches() -> None:
    """Apply repo-local compatibility patches for the active VERL stack."""

    global _PATCHED
    if _PATCHED:
        return

    _patch_verl_vllm_module()
    _patch_imported_runtime_aliases()
    _patch_ray_actor_runtime_env()
    _patch_fsdp_checkpoint_manager()

    apply_continuous_replay_patches()
    _patch_actor_checkpoint_save()
    _PATCHED = True
