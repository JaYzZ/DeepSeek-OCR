#!/usr/bin/env python3

from types import SimpleNamespace

import torch

from Qwen.swift.opsd_span import (
    _build_interleave_segments,
    _build_replay_states,
    _compute_segment_replay_logps_and_entropies,
    _run_batched_replay_prefill,
)


LATENT_TOKEN_ID = 99


class DummyReplayModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 32):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size)
        self.forward_token_count = 0

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        output_hidden_states=False,
        logits_to_keep=None,
        **kwargs,
    ):
        if inputs_embeds is None:
            hidden = self.embedding(input_ids)
        else:
            hidden = inputs_embeds
        self.forward_token_count += int(hidden.shape[0] * hidden.shape[1])
        hidden = self.proj(hidden)
        logits = self.lm_head(hidden)

        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]
        device = hidden.device
        if past_key_values is None:
            past_key_values = ((torch.zeros(batch_size, 1, seq_len, 1, device=device), torch.zeros(batch_size, 1, seq_len, 1, device=device)),)
        else:
            prev_len = int(past_key_values[0][0].shape[2])
            total_len = prev_len + seq_len
            past_key_values = ((torch.zeros(batch_size, 1, total_len, 1, device=device), torch.zeros(batch_size, 1, total_len, 1, device=device)),)

        outputs = SimpleNamespace(logits=logits, past_key_values=past_key_values)
        if output_hidden_states:
            outputs.hidden_states = [hidden]
        return outputs


class DummyTrainer:
    def __init__(self):
        self.temperature = 1.0
        self.args = SimpleNamespace(logits_to_keep=None)
        self.model_kwarg_keys = {"input_ids", "attention_mask", "position_ids", "inputs_embeds", "past_key_values", "use_cache", "output_hidden_states"}
        self.tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda token: LATENT_TOKEN_ID if token == "<latent>" else 0, unk_token_id=-1)

    def _prepare_model_inputs(self, inputs):
        return dict(inputs)


class DummyMultimodalReplayModel(DummyReplayModel):
    def __init__(self, vocab_size: int = 128, hidden_dim: int = 32):
        super().__init__(vocab_size=vocab_size, hidden_dim=hidden_dim)
        self.forward_input_lengths = []
        self.forward_token_ids = []

    def forward(self, input_ids=None, inputs_embeds=None, pixel_values=None, **kwargs):
        current_len = int(input_ids.shape[1]) if input_ids is not None else int(kwargs["inputs_embeds"].shape[1])
        self.forward_input_lengths.append(current_len)
        if input_ids is not None:
            self.forward_token_ids.append(input_ids.detach().clone())
        if pixel_values is not None and current_len < 3:
            raise ValueError("Image features and image tokens do not match")
        return super().forward(input_ids=input_ids, inputs_embeds=inputs_embeds, pixel_values=pixel_values, **kwargs)


def test_build_interleave_segments_groups_consecutive_latents():
    segments = _build_interleave_segments(latent_positions=[2, 3, 6], seq_len=8)
    assert segments == [(0, 2, "normal"), (2, 4, "latent"), (4, 6, "normal"), (6, 7, "latent"), (7, 8, "normal")]


def test_segment_replay_batch_interleaved_shape_and_efficiency(monkeypatch):
    monkeypatch.setenv("OPSD_SPAN_STAGE", "gspo")
    monkeypatch.setenv("OPSD_SPAN_REPLAY_MODE", "segment_mixed")
    monkeypatch.setenv("OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX", "12")
    monkeypatch.setenv("OPSD_DELTA_MEMORY_ENABLED", "0")

    trainer = DummyTrainer()
    model = DummyReplayModel()

    input_ids = torch.tensor(
        [
            [11, 12, 13, 14, LATENT_TOKEN_ID, 21, LATENT_TOKEN_ID, 31],
            [41, 42, 43, LATENT_TOKEN_ID, LATENT_TOKEN_ID, 51, 52, 53],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "completion_mask": completion_mask,
        "logits_to_keep": 7,
    }

    logps, entropies, metrics = _compute_segment_replay_logps_and_entropies(
        trainer=trainer,
        model=model,
        inputs=inputs,
        memory_states=[None, None],
        compute_entropy=True,
    )

    assert logps.shape == (2, 7)
    assert entropies is not None and entropies.shape == (2, 7)
    assert metrics["replay_segment_mixed"] == 1.0
    assert metrics["replay_segment_samples"] == 2.0
    assert metrics["segment_replay_latent_steps_max"] == 2.0
    assert model.forward_token_count == int(input_ids.numel())


def test_segment_replay_prefill_keeps_multimodal_prefix(monkeypatch):
    monkeypatch.setenv("OPSD_SPAN_STAGE", "gspo")
    monkeypatch.setenv("OPSD_SPAN_REPLAY_MODE", "segment_mixed")
    monkeypatch.setenv("OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX", "12")
    monkeypatch.setenv("OPSD_DELTA_MEMORY_ENABLED", "0")

    trainer = DummyTrainer()
    model = DummyMultimodalReplayModel()

    input_ids = torch.tensor([[7, 8, 9, 10, LATENT_TOKEN_ID, 11, 12]], dtype=torch.long)
    completion_mask = torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.bool)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "completion_mask": completion_mask,
        "logits_to_keep": 4,
        "pixel_values": torch.randn(1, 3, 4),
    }

    states, max_prefix_len = _build_replay_states(
        input_ids=input_ids,
        completion_mask=completion_mask,
        latent_token_id=LATENT_TOKEN_ID,
        logits_to_keep=4,
    )

    assert len(states) == 1
    assert states[0].window_start == 2
    assert states[0].prefix_len == 1

    _run_batched_replay_prefill(
        trainer=trainer,
        model=model,
        inputs=inputs,
        states=states,
        max_prefix_len=max_prefix_len,
    )

    assert model.forward_input_lengths[0] == 3


def test_segment_replay_multimodal_chunk_uses_absolute_positions(monkeypatch):
    monkeypatch.setenv("OPSD_SPAN_STAGE", "gspo")
    monkeypatch.setenv("OPSD_SPAN_REPLAY_MODE", "segment_mixed")
    monkeypatch.setenv("OPSD_SPAN_REPLAY_LEGIT_LATENT_COUNT_MAX", "12")
    monkeypatch.setenv("OPSD_DELTA_MEMORY_ENABLED", "0")

    trainer = DummyTrainer()
    model = DummyMultimodalReplayModel()

    input_ids = torch.tensor([[7, 8, 9, 10, LATENT_TOKEN_ID, 11, 12]], dtype=torch.long)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "completion_mask": torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.bool),
        "logits_to_keep": 4,
        "pixel_values": torch.randn(1, 3, 4),
    }

    logps, entropies, metrics = _compute_segment_replay_logps_and_entropies(
        trainer=trainer,
        model=model,
        inputs=inputs,
        memory_states=[None],
        compute_entropy=True,
    )

    assert logps.shape == (1, 4)
    assert entropies is not None and entropies.shape == (1, 4)
    assert metrics["replay_segment_mixed"] == 1.0
    assert model.forward_input_lengths[0] == 3
    assert any(torch.equal(tokens, torch.tensor([[10]])) for tokens in model.forward_token_ids)
