#!/usr/bin/env python3
"""Load VERL PPO base config, merge project YAML overlays, then run PPO."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from verl_compat import patch_runtime_env, patch_worker_env_vars


WORKER_SETUP_HOOK = "verl_compat.worker_setup.apply_worker_compat_patches"


def _resolve_verl_ppo_config() -> Path:
    import verl

    return Path(verl.__file__).resolve().parent / "trainer" / "config" / "_generated_ppo_trainer.yaml"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-config",
        action="append",
        default=[],
        help="Project YAML overlay to merge on top of VERL PPO base config. Can be passed multiple times.",
    )
    parser.add_argument(
        "--check-runtime-only",
        action="store_true",
        help="Validate the Ray worker runtime uses the repo-local VERL/vLLM compatibility patch, then exit.",
    )
    return parser.parse_known_args()


def _normalize_ray_kwargs(config) -> None:
    """Merge accidental `+ray_kwargs` overrides back into `ray_kwargs`.

    The VERL trainer reads `config.ray_kwargs`, so patch/runtime env overrides
    must land there rather than under an additive `+ray_kwargs` key.
    """

    additive_ray_kwargs = config.get("+ray_kwargs")
    if additive_ray_kwargs is None:
        return

    base_ray_kwargs = config.get("ray_kwargs")
    if base_ray_kwargs is None:
        config["ray_kwargs"] = OmegaConf.create({})
        base_ray_kwargs = config["ray_kwargs"]

    config["ray_kwargs"] = OmegaConf.merge(base_ray_kwargs, additive_ray_kwargs)
    del config["+ray_kwargs"]


def _inject_runtime_env(config) -> None:
    """Ensure Ray workers can import repo-local compatibility modules."""

    ray_init = OmegaConf.select(config, "ray_kwargs.ray_init")
    if ray_init is None:
        if OmegaConf.select(config, "ray_kwargs") is None:
            config["ray_kwargs"] = OmegaConf.create({})
        config["ray_kwargs"]["ray_init"] = OmegaConf.create({})
        ray_init = config["ray_kwargs"]["ray_init"]

    if OmegaConf.select(config, "ray_kwargs.ray_init.include_dashboard") is None:
        ray_init["include_dashboard"] = False

    runtime_env = OmegaConf.select(config, "ray_kwargs.ray_init.runtime_env")
    runtime_env_dict = OmegaConf.to_container(runtime_env, resolve=False) if runtime_env is not None else {}
    patched_runtime_env = patch_runtime_env(runtime_env_dict)
    ray_init["runtime_env"] = OmegaConf.create(patched_runtime_env)


def _check_runtime_env(config) -> int:
    import ray

    ray_init_kwargs = OmegaConf.to_container(
        OmegaConf.select(config, "ray_kwargs.ray_init"),
        resolve=True,
    ) or {}
    ray_init_kwargs.setdefault("include_dashboard", False)
    print(f"Runtime preflight ray init kwargs: {json.dumps(ray_init_kwargs, sort_keys=True)}", flush=True)

    if ray.is_initialized():
        ray.shutdown()
    ray.init(**ray_init_kwargs)
    print("Runtime preflight: ray initialized", flush=True)

    @ray.remote(num_cpus=1, num_gpus=1)
    class RuntimeProbe:
        def inspect(self, actor_model_cfg: dict, actor_cfg: dict):
            import json
            import sys

            import torch
            from omegaconf import OmegaConf
            from peft import PeftModel, TaskType, get_peft_model, LoraConfig
            from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

            from verl_compat.worker_setup import collect_patch_diagnostics
            from verl_compat.continuous_replay import LatentVAE, inspect_replay_binding
            from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager
            from verl.utils.vllm.utils import VLLMHijack
            from verl.utils.fs import copy_to_local
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            VLLMHijack.hijack()

            actor_model_cfg = OmegaConf.create(actor_model_cfg)
            actor_cfg = OmegaConf.create(actor_cfg)
            local_path = copy_to_local(actor_model_cfg.path, use_shm=actor_model_cfg.get("use_shm", False))
            trust_remote_code = actor_model_cfg.get("trust_remote_code", False)
            override_model_config = OmegaConf.to_container(OmegaConf.create(actor_model_cfg.get("override_config", {})))
            attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
            if not torch.cuda.is_available() and attn_implementation == "flash_attention_2":
                attn_implementation = "eager"
            actor_model_config = AutoConfig.from_pretrained(
                local_path,
                attn_implementation=attn_implementation,
                trust_remote_code=trust_remote_code,
            )

            has_remote_code = hasattr(actor_model_config, "auto_map") and any(
                actor_model_config.architectures[0] in val for val in actor_model_config.auto_map.values()
            )
            if has_remote_code:
                auto_class = next(
                    k for k, v in actor_model_config.auto_map.items() if actor_model_config.architectures[0] in v
                )
                match auto_class:
                    case "AutoModelForVision2Seq":
                        actor_module_class = AutoModelForVision2Seq
                    case "AutoModelForCausalLM":
                        actor_module_class = AutoModelForCausalLM
                    case "AutoModelForImageTextToText":
                        actor_module_class = AutoModelForImageTextToText
                    case _:
                        actor_module_class = AutoModel
            else:
                if type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
                    actor_module_class = AutoModelForVision2Seq
                elif type(actor_model_config) in AutoModelForCausalLM._model_mapping.keys():
                    actor_module_class = AutoModelForCausalLM
                elif type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
                    actor_module_class = AutoModelForImageTextToText
                else:
                    actor_module_class = AutoModel

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch.bfloat16,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=(False if not torch.cuda.is_available() else actor_model_cfg.get("use_remove_padding", False)),
                ulysses_sp_size=actor_cfg.get("ulysses_sequence_parallel_size", 1),
                use_fused_kernels=actor_model_cfg.get("use_fused_kernels", False),
                fused_kernels_backend=(actor_model_cfg.get("fused_kernel_options", {}) or {}).get("impl_backend"),
            )

            lora_adapter_path = actor_model_cfg.get("lora_adapter_path")
            lora_rank = actor_model_cfg.get("lora_rank", 0)
            if lora_adapter_path is not None:
                actor_module.enable_input_require_grads()
                local_adapter_path = copy_to_local(lora_adapter_path, use_shm=actor_model_cfg.get("use_shm", False))
                actor_module = PeftModel.from_pretrained(actor_module, local_adapter_path, is_trainable=True)
                peft_config = actor_module.peft_config["default"]
                if isinstance(peft_config.task_type, str):
                    peft_config.task_type = TaskType.CAUSAL_LM
            elif lora_rank and int(lora_rank) > 0:
                actor_module.enable_input_require_grads()
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": actor_model_cfg.lora_rank,
                    "lora_alpha": actor_model_cfg.lora_alpha,
                    "target_modules": OmegaConf.to_container(OmegaConf.create(actor_model_cfg.target_modules)),
                    "exclude_modules": OmegaConf.to_container(OmegaConf.create(actor_model_cfg.exclude_modules)),
                    "bias": "none",
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))

            runtime_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            actor_module = actor_module.to(runtime_device)

            payload = {
                "python_executable": sys.executable,
                "python_path_head": sys.path[:5],
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_current_device": (int(torch.cuda.current_device()) if torch.cuda.is_available() else None),
                "runtime_device": str(runtime_device),
                "worker_load_adapter": {
                    "module": WorkerLoRAManager._load_adapter.__module__,
                    "name": WorkerLoRAManager._load_adapter.__name__,
                    "patched": bool(getattr(WorkerLoRAManager._load_adapter, "_qwen3vl_compat_patch", False)),
                },
                "lru_load_adapter": {
                    "module": LRUCacheWorkerLoRAManager._load_adapter.__module__,
                    "name": LRUCacheWorkerLoRAManager._load_adapter.__name__,
                    "patched": bool(getattr(LRUCacheWorkerLoRAManager._load_adapter, "_qwen3vl_compat_patch", False)),
                },
                "replay_binding": inspect_replay_binding(actor_module),
            }
            hidden_size = getattr(actor_model_config, "hidden_size", None)
            if hidden_size is None:
                hidden_size = getattr(getattr(actor_model_config, "text_config", None), "hidden_size", None)
            bos_token_id = getattr(actor_model_config, "bos_token_id", None)
            if bos_token_id is None:
                bos_token_id = getattr(getattr(actor_model_config, "text_config", None), "bos_token_id", 0)
            replay_smoke = {"ok": False}
            try:
                seq_len = 8
                input_ids = torch.full((1, seq_len), int(bos_token_id or 0), dtype=torch.long, device=runtime_device)
                attention_mask = torch.ones((1, seq_len), dtype=torch.long, device=runtime_device)
                position_ids = torch.arange(seq_len, dtype=torch.long, device=runtime_device).unsqueeze(0)
                replay_row_ids = torch.full((1, seq_len), -1, dtype=torch.long, device=runtime_device)
                replay_row_ids[0, 4] = 0
                replay_row_ids[0, 5] = 1
                latent_vae = LatentVAE(hidden_size=int(hidden_size), deterministic=False).to(
                    device=runtime_device,
                    dtype=torch.bfloat16,
                )
                replay_hidden_states = torch.randn((2, int(hidden_size)), device=runtime_device, dtype=torch.bfloat16)
                with torch.no_grad():
                    replay_output = actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        continuous_replay_row_ids=replay_row_ids,
                        continuous_replay_hidden_states=replay_hidden_states,
                        continuous_replay_latent_vae=latent_vae,
                        use_cache=False,
                    )
                replay_smoke = {
                    "ok": True,
                    "logits_shape": list(replay_output.logits.shape),
                }
            except Exception as exc:
                replay_smoke = {
                    "ok": False,
                    "error": repr(exc),
                }
            payload["replay_forward_smoke"] = replay_smoke
            payload.update(collect_patch_diagnostics())
            print(f"Runtime preflight worker payload: {json.dumps(payload, sort_keys=True)}", flush=True)
            return payload

    probe = RuntimeProbe.options(runtime_env=patch_runtime_env({})).remote()
    ref = probe.inspect.remote(
        OmegaConf.to_container(OmegaConf.select(config, "actor_rollout_ref.model"), resolve=True),
        OmegaConf.to_container(OmegaConf.select(config, "actor_rollout_ref.actor"), resolve=True),
    )
    print("Runtime preflight: submitted remote inspection actor", flush=True)
    try:
        payload = ray.get(ref, timeout=120)
    except Exception as exc:
        print(f"Runtime preflight failed before result: {exc!r}", file=sys.stderr, flush=True)
        ray.shutdown()
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    ray.shutdown()

    checks = [payload["worker_load_adapter"], payload["lru_load_adapter"]]
    ok = (
        payload.get("compat_env_flag") == "1"
        and payload.get("lora_from_tensors_patched") is True
        and payload.get("replay_binding", {}).get("ok") is True
        and payload.get("replay_forward_smoke", {}).get("ok") is True
        and all(item["patched"] and item["module"] == "verl_compat.vllm_shim" for item in checks)
    )
    return 0 if ok else 1


def main() -> int:
    args, overrides = parse_args()

    from verl_compat import apply_runtime_compat_patches
    import verl_compat.reward_manager  # noqa: F401

    apply_runtime_compat_patches()

    from verl.trainer.main_ppo import run_ppo

    verl_ppo_config = _resolve_verl_ppo_config()
    config = OmegaConf.load(verl_ppo_config)
    for config_path in args.project_config:
        config = OmegaConf.merge(config, OmegaConf.load(config_path))
    if overrides:
        config = OmegaConf.merge(config, OmegaConf.from_dotlist(overrides))

    _normalize_ray_kwargs(config)
    _inject_runtime_env(config)
    OmegaConf.resolve(config)

    if args.check_runtime_only:
        return _check_runtime_env(config)

    run_ppo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
