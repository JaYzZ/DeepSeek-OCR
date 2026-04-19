from __future__ import annotations

import os

from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_custom_reward_fn, load_reward_manager


_VERL_BASE_CONFIG = "/home/jianzhan/sources/verl/verl/trainer/config/_generated_ppo_trainer.yaml"


def _load_project_config(path: str):
    os.environ["ROOT_DIR"] = "/home/jianzhan"
    return OmegaConf.merge(
        OmegaConf.load(_VERL_BASE_CONFIG),
        OmegaConf.load(path),
    )


def test_chimera_reward_config_uses_v017_reward_contract() -> None:
    config = _load_project_config("Qwen/configs/rl/chimera_gspo.yaml")

    assert config.reward.reward_manager.name == "dapo"
    assert config.reward.custom_reward_function.name == "compute_score"

    reward_fn = get_custom_reward_fn(config)
    assert reward_fn is not None

    reward_manager = load_reward_manager(config, tokenizer=None)
    assert reward_manager.__class__.__name__ == "DAPORewardManager"


def test_deepvision_reward_config_uses_v017_reward_contract() -> None:
    config = _load_project_config("Qwen/configs/rl/deepvision_gspo.yaml")

    assert config.reward.reward_manager.name == "dapo"
    assert config.reward.custom_reward_function.name == "compute_score"

    reward_fn = get_custom_reward_fn(config)
    assert reward_fn is not None

    reward_manager = load_reward_manager(config, tokenizer=None)
    assert reward_manager.__class__.__name__ == "DAPORewardManager"
