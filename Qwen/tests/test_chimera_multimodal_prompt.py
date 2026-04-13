#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Qwen.data import build_chimera_verl_dataset as chimera_build
from Qwen.verl import chimera_gspo_reward
from Qwen.verl import chimera_rl_dataset
from verl_compat.bootstrap import _patch_rlhf_dataset_message_builder
from verl_compat.continuous_replay import (
    _inject_latent_log_probs_into_rollout,
    _trim_request_trace_to_actual_response_length,
    _validate_multimodal_token_match,
)


def test_build_chimera_verl_dataset_emits_structured_image_content(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    sample_id = "chimera_0000000"
    image_path = image_dir / f"{sample_id}_question.png"
    Image.new("RGB", (8, 8), color="white").save(image_path)

    prompt, images = chimera_build._normalize_prompt(
        {"question": "Q", "subject": "Mathematics"},
        sample_id=sample_id,
        chimera_images_dir=image_dir,
    )

    assert len(images) == 1
    assert prompt == [
        {
            "role": "system",
            "content": chimera_build.CHIMERA_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": "The image contains the full problem statement. Solve it carefully. This is a Mathematics problem.",
                },
            ],
        }
    ]


def test_chimera_rl_dataset_helper_emits_structured_image_content():
    assert chimera_rl_dataset._build_user_content("Solve it.", has_image=True) == [
        {"type": "image"},
        {"type": "text", "text": "Solve it."},
    ]
    assert chimera_rl_dataset._build_user_content("Solve it.", has_image=False) == "Solve it."


def test_build_chimera_record_keeps_incorrect_trace_rows(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    record = chimera_build._build_record(
        {
            "question": "Q",
            "answer": "A",
            "solution": "S",
            "original_solution": "OS",
            "subject": "Math",
            "topic": "Geometry",
            "index": 0,
            "correctness": False,
        },
        source_name="train",
        row_idx=0,
        chimera_images_dir=image_dir,
    )

    assert record is not None
    assert record["reward_model"]["ground_truth"] == "A"
    assert record["extra_info"]["correctness"] is False


def test_rlhf_dataset_patch_preserves_structured_multimodal_content():
    _patch_rlhf_dataset_message_builder()

    from verl.utils.dataset.rl_dataset import RLHFDataset

    dummy_dataset = object.__new__(RLHFDataset)
    dummy_dataset.prompt_key = "prompt"
    dummy_dataset.image_key = "images"
    dummy_dataset.video_key = "videos"

    structured_example = {
        "prompt": [
            {
                "role": "user",
                "content": np.array(
                    [
                        {"type": "image"},
                        {"type": "text", "text": "Solve this question shown in the image."},
                    ],
                    dtype=object,
                ),
            }
        ],
        "images": ["placeholder.png"],
    }
    structured_messages = dummy_dataset._build_messages(structured_example)
    assert structured_messages[0]["content"].tolist() == [
        {"type": "image"},
        {"type": "text", "text": "Solve this question shown in the image."},
    ]

    legacy_example = {
        "prompt": [{"role": "user", "content": "<image>\nSolve this question shown in the image."}],
        "images": ["placeholder.png"],
    }
    legacy_messages = dummy_dataset._build_messages(legacy_example)
    assert legacy_messages[0]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "\nSolve this question shown in the image."},
    ]


def test_multimodal_validation_ignores_generated_image_token_when_prompt_mask_is_provided():
    input_ids = torch.tensor([[151655, 151655, 101, 202, 151655]], dtype=torch.long)
    prompt_mask = torch.tensor([[True, True, True, False, False]])

    mask = _validate_multimodal_token_match(
        modality="image",
        input_ids=input_ids,
        token_id=151655,
        feature_count=2,
        grid_thw=None,
        prompt_positions_mask=prompt_mask,
    )

    assert mask.tolist() == [[True, True, False, False, False]]


def test_continuous_replay_logprob_injection_detects_position_overflow():
    curr_log_prob = [0.0, 0.0]
    mask_row = np.array([False, True, False, True], dtype=np.bool_)
    latent_log_probs = np.array([-0.5, -0.7], dtype=np.float32)

    try:
        _inject_latent_log_probs_into_rollout(
            curr_log_prob=curr_log_prob,
            mask_row=mask_row,
            latent_log_probs=latent_log_probs,
            request_id="req-overflow",
        )
    except RuntimeError as exc:
        assert "position overflow" in str(exc)
        assert "req-overflow" in str(exc)
    else:
        raise AssertionError("Expected overflow check to raise RuntimeError")


def test_continuous_replay_trace_is_trimmed_to_actual_response_length():
    hidden_row = np.arange(6, dtype=np.float16).reshape(3, 2)
    latent_row = (np.arange(6, dtype=np.float16) + 10).reshape(3, 2)
    mask_row = np.array([True, False, True, True], dtype=np.bool_)
    latent_log_probs = np.array([-0.1, -0.2, -0.3], dtype=np.float32)

    trimmed_hidden, trimmed_latent, trimmed_mask, trimmed_log_probs = (
        _trim_request_trace_to_actual_response_length(
            hidden_row=hidden_row,
            latent_row=latent_row,
            mask_row=mask_row,
            latent_log_probs=latent_log_probs,
            actual_response_length=3,
            request_id="req-trim",
        )
    )

    assert trimmed_mask.tolist() == [True, False, True]
    assert trimmed_hidden.tolist() == hidden_row[:2].tolist()
    assert trimmed_latent.tolist() == latent_row[:2].tolist()
    assert trimmed_log_probs.tolist() == latent_log_probs[:2].tolist()


def test_chimera_reward_accepts_structured_multipart_answers():
    prediction = "(a) (1/3, 1/3, 1/3)\n(b) 1\n(c) 1/3\n(d) (1/3)(1 - e^{-3t})"
    ground_truth = (
        "(a) π = (1/3, 1/3, 1/3).\n"
        "(b) E[T | X_0 = A] = 1.\n"
        "(c) Probability = 1/3.\n"
        "(d) P(X_t = B | X_t ∈ {A,B,C}, X_0 = A) = (1/3)(1 - e^{-3t})."
    )

    score = chimera_gspo_reward.compute_score(
        data_source="chimera::math",
        solution_str=prediction,
        ground_truth=ground_truth,
        extra_info={},
    )

    assert score == 1.0
