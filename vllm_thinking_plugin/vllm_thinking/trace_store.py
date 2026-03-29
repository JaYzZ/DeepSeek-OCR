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
            "all_hidden_states": [],
            "all_token_ids": [],
            "attention_weights": [],
        }


def record_request_step(
    request_id: str,
    hidden_state: torch.Tensor | None,
    latent_embedding: torch.Tensor | None,
    latent_logprob: torch.Tensor | None,
    use_continuous_embedding: bool,
    all_hidden_states: torch.Tensor | None = None,
    all_token_ids: list[int] | None = None,
    attention_weights: torch.Tensor | None = None,
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
                "all_hidden_states": [],
                "all_token_ids": [],
                "attention_weights": [],
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

        # Store all hidden states (for visualization)
        if all_hidden_states is not None:
            all_hidden = all_hidden_states.detach()
            if all_hidden.ndim > 2:
                all_hidden = all_hidden[0]  # Remove batch dim
            all_hidden = all_hidden.to(device="cpu", dtype=torch.float32, non_blocking=False).contiguous()
            trace["all_hidden_states"].append(all_hidden.numpy())

        # Store all token IDs
        if all_token_ids is not None:
            trace["all_token_ids"].extend(all_token_ids)

        # Store attention weights (for visualization)
        if attention_weights is not None:
            attn = attention_weights.detach()
            if attn.ndim > 3:
                attn = attn[0]  # Remove batch dim if present
            attn = attn.to(device="cpu", dtype=torch.float32, non_blocking=False).contiguous()
            trace["attention_weights"].append(attn.numpy())


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

    # Pack all hidden states
    all_hidden_states = trace.get("all_hidden_states", [])
    if all_hidden_states:
        # Concatenate along sequence dimension
        all_hidden_array = np.concatenate(all_hidden_states, axis=0)
    else:
        all_hidden_array = np.empty((0, 0), dtype=np.float32)

    # Pack all token IDs
    all_token_ids = trace.get("all_token_ids", [])

    # Pack attention weights
    attention_weights = trace.get("attention_weights", [])
    if attention_weights:
        # Concatenate or stack attention weights depending on shape
        # For now, just stack them
        try:
            attention_array = np.stack(attention_weights, axis=0)
        except ValueError:
            # If shapes don't match, just return the first one or concatenate
            attention_array = attention_weights[0] if attention_weights else np.empty((0, 0), dtype=np.float32)
    else:
        attention_array = np.empty((0, 0), dtype=np.float32)

    return {
        "continuous_hidden_states": hidden_array,
        "continuous_latent_embeddings": latent_array,
        "continuous_latent_log_probs": latent_logprob_array,
        "continuous_token_mask": np.asarray(trace["continuous_token_mask"], dtype=np.bool_),
        "all_hidden_states": all_hidden_array,
        "all_token_ids": all_token_ids,
        "attention_weights": attention_array,
    }


def get_request_trace(request_id: str) -> dict[str, np.ndarray] | None:
    """Get request trace without popping it."""
    request_id = str(request_id)
    with _TRACE_LOCK:
        trace = _REQUEST_TRACES.get(request_id)
        if trace is None:
            return None

        # Return a copy to avoid mutation issues
        return {
            "continuous_hidden_states": trace.get("continuous_hidden_states", []),
            "continuous_latent_embeddings": trace.get("continuous_latent_embeddings", []),
            "continuous_latent_log_probs": trace.get("continuous_latent_log_probs", []),
            "continuous_token_mask": trace.get("continuous_token_mask", []),
            "all_hidden_states": trace.get("all_hidden_states", []),
            "all_token_ids": trace.get("all_token_ids", []),
            "attention_weights": trace.get("attention_weights", []),
        }


def get_latest_trace() -> dict[str, np.ndarray] | None:
    """Get the most recent trace (useful for single-request scenarios)."""
    with _TRACE_LOCK:
        if not _REQUEST_TRACES:
            return None
        # Get the most recently added trace
        request_id = list(_REQUEST_TRACES.keys())[-1]
        return get_request_trace(request_id)


def list_request_ids() -> list[str]:
    """List all active request IDs."""
    with _TRACE_LOCK:
        return list(_REQUEST_TRACES.keys())


__all__ = [
    "pop_request_trace",
    "get_request_trace",
    "get_latest_trace",
    "list_request_ids",
    "record_request_step",
    "reset_request_trace",
]
