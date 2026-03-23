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

    @ray.remote(num_cpus=1)
    class RuntimeProbe:
        def inspect(self):
            import json
            import sys

            from verl_compat.worker_setup import collect_patch_diagnostics
            from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager
            from verl.utils.vllm.utils import VLLMHijack

            VLLMHijack.hijack()

            payload = {
                "python_executable": sys.executable,
                "python_path_head": sys.path[:5],
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
            }
            payload.update(collect_patch_diagnostics())
            print(f"Runtime preflight worker payload: {json.dumps(payload, sort_keys=True)}", flush=True)
            return payload

    probe = RuntimeProbe.options(runtime_env=patch_runtime_env({})).remote()
    ref = probe.inspect.remote()
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
