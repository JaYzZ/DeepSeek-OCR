#!/usr/bin/env python3
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Qwen.llamafactory import integration as lfi
from Qwen.llamafactory.curriculum_callback import QwenCurriculumCallback


THINK_START_ID = 151667
THINK_END_ID = 151668
LATENT_TOKEN_ID = 151669


class DummyLogger:
    def warning(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None


class DummyCollator:
    def __init__(self):
        self.tokenizer = SimpleNamespace(pad_token_id=0, model_max_length=64)
        self.label_pad_token_id = -100
        self.block_diag_attn = False
        self.max_length = 64


class DummySecondForwardModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        image_grid_thw=None,
        logits_to_keep=None,
    ):
        assert input_ids is None
        assert inputs_embeds is not None
        keep = logits_to_keep or inputs_embeds.shape[1]
        logits = inputs_embeds[:, -keep:, : self.vocab_size].contiguous()
        return SimpleNamespace(logits=logits)


@pytest.fixture(autouse=True)
def latent_env(monkeypatch):
    monkeypatch.setenv("QWEN3VL_LATENT_TOKEN_ID", str(LATENT_TOKEN_ID))
    monkeypatch.setenv("QWEN3VL_THINKING_START_ID", str(THINK_START_ID))
    monkeypatch.setenv("QWEN3VL_THINKING_END_ID", str(THINK_END_ID))
    monkeypatch.setenv("QWEN3VL_CUTOFF_LEN", "64")
    monkeypatch.setenv("QWEN3VL_LATENT_CE_ACTIVE", "1")
    monkeypatch.setenv("QWEN3VL_LATENT_CE_TOKEN", "0")


def test_full_latent_pipeline_from_paths(tmp_path):
    hidden_dim = 4
    latent_gt = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    latent_sup = torch.tensor([[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0]])

    gt_path = tmp_path / "gt.pt"
    sup_path = tmp_path / "sup.pt"
    torch.save({"latent": latent_gt}, gt_path)
    torch.save({"l_features": latent_sup}, sup_path)

    batch = [{
        "input_ids": [10, THINK_START_ID, LATENT_TOKEN_ID, THINK_END_ID, 11],
        "labels": [THINK_START_ID, LATENT_TOKEN_ID, THINK_END_ID, 11, -100],
    }]
    latent_fields = [{
        "latent_ground_truth": [str(gt_path)],
        "latent_supervision": [str(sup_path)],
        "latent_seq_lens": [latent_gt.shape[0]],
        "num_latent_steps": 1,
    }]

    packed_features, packed_latent_fields = lfi._pack_features_after_injection(
        batch=batch,
        latent_fields_list=latent_fields,
        collator=DummyCollator(),
        logger=DummyLogger(),
    )

    assert len(packed_features) == 1
    assert packed_features[0]["input_ids"][:6] == [10, THINK_START_ID, LATENT_TOKEN_ID, LATENT_TOKEN_ID, THINK_END_ID, 11]
    assert packed_features[0]["input_ids"][6:] == [0] * (len(packed_features[0]["input_ids"]) - 6)
    assert len(packed_features[0]["input_ids"]) == 65
    assert len(packed_features[0]["labels"]) == len(packed_features[0]["input_ids"])

    collated = {
        "input_ids": torch.tensor([packed_features[0]["input_ids"]], dtype=torch.long),
    }
    collated = lfi._add_latent_supervision_to_batch(
        collated=collated,
        batch=[packed_latent_fields[0]],
        logger=DummyLogger(),
    )

    assert collated["latent_positions"].sum().item() == latent_gt.shape[0]
    assert collated["latent_ground_truth"] == [[]]
    assert collated["latent_ground_truth_paths"] == [[str(gt_path)]]
    assert collated["latent_supervision_paths"] == [[str(sup_path)]]

    materialized_gt = lfi._materialize_latent_batch(
        collated["latent_ground_truth"],
        collated["latent_ground_truth_paths"],
        strict=True,
    )
    materialized_sup = lfi._materialize_latent_batch(
        collated["latent_supervision"],
        collated["latent_supervision_paths"],
        strict=True,
    )

    assert len(materialized_gt) == 1
    assert torch.equal(materialized_gt[0][0], latent_gt)
    assert torch.equal(materialized_sup[0][0], latent_sup)

    hidden_states = torch.zeros((1, len(packed_features[0]["input_ids"]), hidden_dim), dtype=torch.float32)
    hidden_states[0, :6, :] = torch.tensor(
        [
            [9.0, 9.0, 9.0, 9.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [4.0, 4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0, 5.0],
            [6.0, 6.0, 6.0, 6.0],
        ],
        dtype=torch.float32,
    )

    pre_loss = lfi._compute_mse_loss(
        hidden_states=hidden_states,
        latent_ground_truth=materialized_gt,
        latent_positions=collated["latent_positions"],
    )
    assert pre_loss is not None
    assert torch.isclose(pre_loss, torch.tensor(0.0))

    vae = lfi.LatentVAE(hidden_size=hidden_dim, intermediate_size=8, deterministic=False)
    vae_loss, entropy, _, sampled_latents, batch_indices, seq_indices = lfi._compute_vae_loss(
        vae=vae,
        hidden_states=hidden_states,
        latent_positions=collated["latent_positions"],
        latent_ground_truth_packed=torch.cat(materialized_gt[0], dim=0),
    )
    assert vae_loss is not None
    assert entropy is not None
    assert sampled_latents.shape == (latent_sup.shape[0], hidden_dim)
    assert batch_indices.tolist() == [0, 0]
    assert seq_indices.tolist() == [2, 3]

    assert lfi._compute_ot_loss(hidden_states, materialized_sup, collated["latent_positions"])[0] is not None
    assert lfi._compute_mse_loss(hidden_states, materialized_sup, collated["latent_positions"]) is not None


def test_pred_embed_forward_loss_matches_shifted_labels():
    model = DummySecondForwardModel()
    inputs_embeds = torch.tensor(
        [[[0.0] * 32, [0.0] * 32, [0.0] * 32, [0.0] * 32]],
        dtype=torch.float32,
    )
    sampled_latents = torch.tensor([[0.0] * 32, [0.0] * 32], dtype=torch.float32)
    sampled_latents[0, 5] = 10.0
    sampled_latents[1, 7] = 10.0
    batch_indices = torch.tensor([0, 0], dtype=torch.long)
    seq_indices = torch.tensor([1, 2], dtype=torch.long)
    labels = torch.tensor([[1, 5, 7, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(labels)

    loss = lfi._compute_pred_embed_forward_loss(
        model=model,
        original_inputs_embeds=inputs_embeds,
        sampled_latents=sampled_latents,
        batch_indices=batch_indices,
        seq_indices=seq_indices,
        labels=labels,
        vae_ce_mask=None,
        attention_mask=attention_mask,
        pixel_values=None,
        image_grid_thw=None,
    )

    assert loss is not None
    expected = torch.nn.functional.cross_entropy(
        torch.tensor(
            [
                [0.0] * 32,
                [0.0] * 5 + [10.0] + [0.0] * 26,
                [0.0] * 7 + [10.0] + [0.0] * 24,
                [0.0] * 32,
            ]
        ),
        torch.tensor([5, 7, 9, -100]),
        ignore_index=-100,
    )
    assert torch.isclose(loss, expected, atol=1e-5)


def test_parse_loss_spec_preserves_explicit_weights():
    parsed = lfi._parse_loss_spec("mse:0.8+repa:0.2+ot:1.0")
    assert parsed == [
        ("mse", pytest.approx(0.8)),
        ("repa", pytest.approx(0.2)),
        ("ot", pytest.approx(1.0)),
    ]


def test_parse_loss_spec_defaults_unweighted_terms_to_one():
    parsed = lfi._parse_loss_spec("vae+mse:0.8+repa:0.2+ot")
    assert parsed == [
        ("vae", pytest.approx(1.0)),
        ("mse", pytest.approx(0.8)),
        ("repa", pytest.approx(0.2)),
        ("ot", pytest.approx(1.0)),
    ]


def test_curriculum_applies_on_non_world_zero(monkeypatch):
    monkeypatch.setenv("QWEN3VL_CURRICULUM_ENABLE", "1")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_EPOCHS", "1")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_LOSS_TYPES", "repa")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_LATENT_CE", "1")
    monkeypatch.setenv("QWEN3VL_LOSS_TYPE", "mse")
    monkeypatch.setenv("QWEN3VL_LATENT_CE_ACTIVE", "0")

    callback = QwenCurriculumCallback()
    state = SimpleNamespace(is_world_process_zero=False, epoch=0.0)
    callback.on_train_begin(args=None, state=state, control=None)

    assert os.environ["QWEN3VL_LOSS_TYPE"] == "repa"
    assert os.environ["QWEN3VL_LATENT_CE_ACTIVE"] == "1"


def test_curriculum_single_number_uses_zero_based_boundaries(monkeypatch):
    monkeypatch.setenv("QWEN3VL_CURRICULUM_ENABLE", "1")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_EPOCHS", "3")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_LOSS_TYPES", "stage0,stage1,stage2")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_LATENT_CE", "0,0,0")

    callback = QwenCurriculumCallback()

    assert [stage["epoch"] for stage in callback.stages] == [0, 1, 2]
    assert callback._get_stage_for_epoch(0.0)["loss_type"] == "stage0"
    assert callback._get_stage_for_epoch(1.0)["loss_type"] == "stage1"
    assert callback._get_stage_for_epoch(2.0)["loss_type"] == "stage2"


def test_curriculum_applies_trainability_to_model(monkeypatch):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_a = torch.nn.Parameter(torch.ones(1))
            self.latent_vae_weight = torch.nn.Parameter(torch.ones(1))

    monkeypatch.setenv("QWEN3VL_CURRICULUM_ENABLE", "1")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_EPOCHS", "1")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_LOSS_TYPES", "ce+mse")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_LATENT_CE", "0")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_LORA_TRAINABLE", "0")
    monkeypatch.setenv("QWEN3VL_CURRICULUM_VAE_TRAINABLE", "1")

    callback = QwenCurriculumCallback()
    model = DummyModel()
    state = SimpleNamespace(is_world_process_zero=True, epoch=0.0)
    callback.on_train_begin(args=None, state=state, control=None, model=model)

    assert model.lora_a.requires_grad is False
    assert model.latent_vae_weight.requires_grad is True


def test_should_save_vae_checkpoint_respects_active_loss_spec(monkeypatch):
    monkeypatch.setenv("QWEN3VL_LOSS_TYPE", "ce+mse:0.4+ot:0.4")
    assert lfi._should_save_vae_checkpoint() is False

    monkeypatch.setenv("QWEN3VL_LOSS_TYPE", "ce+vae_ce+vae:0.4+mse:0.4+ot:0.4")
    assert lfi._should_save_vae_checkpoint() is True


def test_unbalanced_sinkhorn_prefers_selective_matches_over_full_average():
    cost_matrix = torch.tensor(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss, transport = lfi._compute_unbalanced_sinkhorn_cost(
        cost_matrix=cost_matrix,
        epsilon=0.1,
        tau=1.0,
        num_iters=20,
    )

    assert transport.shape == cost_matrix.shape
    assert torch.isfinite(loss)
    assert float(loss.item()) < float(cost_matrix.mean().item())


def test_ot_sample_loss_is_more_selective_than_pairwise_average(monkeypatch):
    monkeypatch.setenv("QWEN3VL_OT_SAMPLE_K", "none")
    monkeypatch.setenv("QWEN3VL_OT_EPSILON", "0.1")
    monkeypatch.setenv("QWEN3VL_OT_TAU", "1.0")
    monkeypatch.setenv("QWEN3VL_OT_SINKHORN_ITERS", "20")

    pred_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target_tokens = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    loss, cost_matrix = lfi._compute_ot_sample_loss(pred_tokens, target_tokens)

    assert cost_matrix.shape == (2, 3)
    assert torch.isfinite(loss)
    assert float(loss.item()) < float(cost_matrix.mean().item())
