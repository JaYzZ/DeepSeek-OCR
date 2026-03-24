"""Per-request continuous thinking traces captured inside vLLM workers."""

from __future__ import annotations

from threading import Lock
from typing import Any

import numpy as np
import torch

_TRACE_LOCK = Lock()
_REQUEST_TRACES: dict[str, dict[str, list[Any]]] = {}


def reset_request_trace(request_id: str) -> None:
    request_id = str(request_id)
    with _TRACE_LOCK:
        _REQUEST_TRACES[request_id] = {
            "continuous_hidden_states": [],
            "continuous_latent_embeddings": [],
            "continuous_latent_log_probs": [],
            "continuous_token_mask": [],
        }


def record_request_step(
    request_id: str,
    hidden_state: torch.Tensor | None,
    latent_embedding: torch.Tensor | None,
    latent_logprob: torch.Tensor | None,
    use_continuous_embedding: bool,
) -> None:
    request_id = str(request_id)
    with _TRACE_LOCK:
        trace = _REQUEST_TRACES.setdefault(
            request_id,
            {
                "continuous_hidden_states": [],
                "continuous_latent_embeddings": [],
                "continuous_latent_log_probs": [],
                "continuous_token_mask": [],
            },
        )
        trace["continuous_token_mask"].append(bool(use_continuous_embedding))
        if hidden_state is not None and use_continuous_embedding:
            hidden = hidden_state.detach()
            if hidden.ndim > 1:
                hidden = hidden[0]
            hidden = hidden.to(device="cpu", dtype=torch.float16, non_blocking=False).contiguous()
            trace["continuous_hidden_states"].append(hidden.numpy())
        if latent_embedding is not None and use_continuous_embedding:
            latent = latent_embedding.detach()
            if latent.ndim > 1:
                latent = latent[0]
            latent = latent.to(device="cpu", dtype=torch.float16, non_blocking=False).contiguous()
            trace["continuous_latent_embeddings"].append(latent.numpy())
        if latent_logprob is not None and use_continuous_embedding:
            logprob = latent_logprob.detach()
            if logprob.ndim > 0:
                logprob = logprob.reshape(-1)[0]
            trace["continuous_latent_log_probs"].append(float(logprob.item()))


def pop_request_trace(request_id: str) -> dict[str, np.ndarray] | None:
    request_id = str(request_id)
    with _TRACE_LOCK:
        trace = _REQUEST_TRACES.pop(request_id, None)

    if trace is None:
        return None

    packed_hidden_states = trace["continuous_hidden_states"]
    if packed_hidden_states:
        hidden_array = np.stack(packed_hidden_states, axis=0)
    else:
        hidden_array = np.empty((0, 0), dtype=np.float16)
    packed_latent_embeddings = trace["continuous_latent_embeddings"]
    if packed_latent_embeddings:
        latent_array = np.stack(packed_latent_embeddings, axis=0)
    else:
        latent_array = np.empty((0, 0), dtype=np.float16)
    packed_latent_log_probs = trace["continuous_latent_log_probs"]
    if packed_latent_log_probs:
        latent_logprob_array = np.asarray(packed_latent_log_probs, dtype=np.float32)
    else:
        latent_logprob_array = np.empty((0,), dtype=np.float32)

    return {
        "continuous_hidden_states": hidden_array,
        "continuous_latent_embeddings": latent_array,
        "continuous_latent_log_probs": latent_logprob_array,
        "continuous_token_mask": np.asarray(trace["continuous_token_mask"], dtype=np.bool_),
    }


__all__ = ["pop_request_trace", "record_request_step", "reset_request_trace"]
