"""Repo-local reward managers for VERL training."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("dapo_batch")
class DAPOBatchRewardManager(AbstractRewardManager):
    """Batch-oriented DeepVision reward manager with DAPO-style semantics."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, "max_resp_len must be provided when overlong buffer is enabled"
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def _build_batch_inputs(
        self, data: DataProto, valid_response_lengths: torch.Tensor
    ) -> tuple[list[str], list[str], list[str], list[dict[str, Any]]]:
        response_ids = data.batch["responses"]
        data_sources = list(data.non_tensor_batch[self.reward_fn_key])
        ground_truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in data]
        rollout_reward_scores = data.non_tensor_batch.get("reward_scores", [{} for _ in range(len(data))])
        extra_infos = list(data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))]))

        response_token_lists = [
            response_ids[i][: valid_response_lengths[i].item()].tolist()
            for i in range(len(data))
        ]
        response_strs = self.tokenizer.batch_decode(response_token_lists, skip_special_tokens=True)

        eos_token = self.tokenizer.eos_token
        if eos_token:
            response_strs = [
                response[:-len(eos_token)] if response.endswith(eos_token) else response
                for response in response_strs
            ]

        normalized_extra_infos: list[dict[str, Any]] = []
        for extra_info, rollout_score in zip(extra_infos, rollout_reward_scores, strict=False):
            copied = dict(extra_info or {})
            copied["rollout_reward_scores"] = rollout_score
            normalized_extra_infos.append(copied)

        return data_sources, response_strs, ground_truths, normalized_extra_infos

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        prompt_len = data.batch["prompts"].shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_prompt_lengths = attention_mask[:, :prompt_len].sum(dim=-1)
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)

        data_sources, response_strs, ground_truths, extra_infos = self._build_batch_inputs(data, valid_response_lengths)
        scores = self.compute_score(
            data_sources=data_sources,
            solution_strs=response_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )

        already_printed: dict[str, int] = {}

        for i, (score_result, response_str, ground_truth, data_source) in enumerate(
            zip(scores, response_strs, ground_truths, data_sources, strict=False)
        ):
            valid_response_length = valid_response_lengths[i].item()

            if isinstance(score_result, dict):
                score = score_result["score"]
                for key, value in score_result.items():
                    reward_extra_info[key].append(value)
            else:
                score = score_result
                reward_extra_info["acc"].append(score)

            reward = score
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if already_printed.get(data_source, 0) < self.num_examine:
                valid_prompt_length = valid_prompt_lengths[i].item()
                prompt_ids = data.batch["prompts"][i][-valid_prompt_length:]
                prompt_str = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[score]", score_result)
                already_printed[data_source] = already_printed.get(data_source, 0) + 1

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
