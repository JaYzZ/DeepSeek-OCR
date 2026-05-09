#!/usr/bin/env python3

import torch

from Qwen.swift.opsd_span import _build_interleave_segments, _delta_memory_enabled


def test_segment_builder_complex_multispan():
    segments = _build_interleave_segments(
        latent_positions=[1, 2, 5, 6, 7, 10],
        seq_len=12,
    )
    assert segments == [
        (0, 1, "normal"),
        (1, 3, "latent"),
        (3, 5, "normal"),
        (5, 8, "latent"),
        (8, 10, "normal"),
        (10, 11, "latent"),
        (11, 12, "normal"),
    ]


def test_delta_memory_env_switch(monkeypatch):
    monkeypatch.setenv("OPSD_DELTA_MEMORY_ENABLED", "1")
    assert _delta_memory_enabled() is True
    monkeypatch.setenv("OPSD_DELTA_MEMORY_ENABLED", "0")
    assert _delta_memory_enabled() is False


def test_segment_builder_empty_latent():
    segments = _build_interleave_segments(latent_positions=[], seq_len=5)
    assert segments == [(0, 5, "normal")]
