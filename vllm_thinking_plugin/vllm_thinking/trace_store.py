"""Per-request continuous thinking traces captured inside vLLM workers."""

from __future__ import annotations

import os
from threading import Lock
from typing import Any
import queue

import numpy as np
import torch

_STORE_VISUALIZATION_DATA = os.environ.get("VLLM_STORE_VISUALIZATION_DATA", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Global lock for trace dictionary operations only (not for per-request appends)
_TRACE_DICT_LOCK = Lock()
_REQUEST_TRACES: dict[str, dict[str, list[Any]]] = {}
# Per-request queues for lock-free trace recording
_REQUEST_QUEUES: dict[str, queue.Queue] = {}


def reset_request_trace(request_id: str) -> None:
    request_id = str(request_id)
    with _TRACE_DICT_LOCK:
        trace_dict = {
            "continuous_hidden_states": [],
            "continuous_latent_embeddings": [],
            "continuous_latent_log_probs": [],
            "continuous_token_mask": [],
            "all_token_ids": [],
            "all_token_logprobs": [],
        }
        if _STORE_VISUALIZATION_DATA:
            trace_dict["all_hidden_states"] = []
            trace_dict["all_token_embeddings"] = []
            trace_dict["attention_weights"] = []
        _REQUEST_TRACES[request_id] = trace_dict
        # Create per-request queue for lock-free recording
        _REQUEST_QUEUES[request_id] = queue.Queue()


def record_request_step(
    request_id: str,
    hidden_state: torch.Tensor | None,
    latent_embedding: torch.Tensor | None,
    latent_logprob: torch.Tensor | None,
    use_continuous_embedding: bool,
    all_hidden_states: torch.Tensor | None = None,
    all_token_ids: list[int] | None = None,
    all_token_logprobs: list[float | None] | None = None,
    attention_weights: torch.Tensor | None = None,
    token_embeddings: torch.Tensor | None = None,
) -> None:
    request_id = str(request_id)

    # Use per-request queue for lock-free recording (optimization 2)
    q = _REQUEST_QUEUES.get(request_id)
    if q is None:
        # Fallback to dict if queue not initialized
        with _TRACE_DICT_LOCK:
            trace = _REQUEST_TRACES.setdefault(
                request_id,
                {
                    "continuous_hidden_states": [],
                    "continuous_latent_embeddings": [],
                    "continuous_latent_log_probs": [],
                    "continuous_token_mask": [],
                    "all_token_ids": [],
                    "all_token_logprobs": [],
                    "all_hidden_states": [],
                    "all_token_embeddings": [],
                    "attention_weights": [],
                },
            )
        _record_to_trace(
            trace,
            hidden_state,
            latent_embedding,
            latent_logprob,
            use_continuous_embedding,
            all_hidden_states,
            all_token_ids,
            all_token_logprobs,
            attention_weights,
            token_embeddings,
        )
    else:
        # Queue the data for batch processing (lock-free)
        q.put((
            hidden_state,
            latent_embedding,
            latent_logprob,
            use_continuous_embedding,
            all_hidden_states,
            all_token_ids,
            all_token_logprobs,
            attention_weights,
            token_embeddings,
        ))


def _record_to_trace(
    trace: dict[str, list],
    hidden_state: torch.Tensor | None,
    latent_embedding: torch.Tensor | None,
    latent_logprob: torch.Tensor | None,
    use_continuous_embedding: bool,
    all_hidden_states: torch.Tensor | None = None,
    all_token_ids: list[int] | None = None,
    all_token_logprobs: list[float | None] | None = None,
    attention_weights: torch.Tensor | None = None,
    token_embeddings: torch.Tensor | None = None,
) -> None:
    """Internal helper to record data directly to trace dict."""
    trace["continuous_token_mask"].append(bool(use_continuous_embedding))
    if hidden_state is not None and use_continuous_embedding:
        hidden = hidden_state.detach()
        if hidden.ndim > 1:
            hidden = hidden[0]
        # OPTIMIZATION 3: non_blocking=True with pinned memory
        hidden = hidden.to(device="cpu", dtype=torch.float16, non_blocking=True).contiguous()
        trace["continuous_hidden_states"].append(hidden.numpy())
    if latent_embedding is not None and use_continuous_embedding:
        latent = latent_embedding.detach()
        if latent.ndim > 1:
            latent = latent[0]
        # OPTIMIZATION 3: non_blocking=True with pinned memory
        latent = latent.to(device="cpu", dtype=torch.float16, non_blocking=True).contiguous()
        trace["continuous_latent_embeddings"].append(latent.numpy())
    if latent_logprob is not None and use_continuous_embedding:
        logprob = latent_logprob.detach()
        if logprob.ndim > 0:
            logprob = logprob.reshape(-1)[0]
        trace["continuous_latent_log_probs"].append(float(logprob.item()))

    if all_token_ids is not None:
        trace["all_token_ids"].extend(int(token_id) for token_id in all_token_ids)
    if all_token_logprobs is not None:
        trace["all_token_logprobs"].extend(
            float(logprob) if logprob is not None else float("nan")
            for logprob in all_token_logprobs
        )

    if _STORE_VISUALIZATION_DATA:
        if all_hidden_states is not None:
            hidden_states = all_hidden_states.detach().to(
                device="cpu",
                dtype=torch.float16,
                non_blocking=True,
            ).contiguous().numpy()
            trace.setdefault("all_hidden_states", []).append(hidden_states)
        if token_embeddings is not None:
            embeddings = token_embeddings.detach().to(
                device="cpu",
                dtype=torch.float16,
                non_blocking=True,
            ).contiguous().numpy()
            trace.setdefault("all_token_embeddings", []).append(embeddings)
        if attention_weights is not None:
            attention = attention_weights.detach().to(
                device="cpu",
                dtype=torch.float32,
                non_blocking=True,
            ).contiguous().numpy()
            trace.setdefault("attention_weights", []).append(attention)


def _drain_request_queue(request_id: str) -> None:
    q = _REQUEST_QUEUES.get(request_id)
    if q is None:
        return

    with _TRACE_DICT_LOCK:
        trace = _REQUEST_TRACES.get(request_id)
        if trace is None:
            return
        while not q.empty():
            (
                hidden_state,
                latent_embedding,
                latent_logprob,
                use_continuous_embedding,
                all_hidden_states,
                all_token_ids,
                all_token_logprobs,
                attention_weights,
                token_embeddings,
            ) = q.get_nowait()
            _record_to_trace(
                trace,
                hidden_state,
                latent_embedding,
                latent_logprob,
                use_continuous_embedding,
                all_hidden_states,
                all_token_ids,
                all_token_logprobs,
                attention_weights,
                token_embeddings,
            )


def _pack_trace(trace: dict[str, list[Any]]) -> dict[str, np.ndarray | list[int]]:
    packed_hidden_states = trace.get("continuous_hidden_states", [])
    packed_latent_embeddings = trace.get("continuous_latent_embeddings", [])
    packed_latent_log_probs = trace.get("continuous_latent_log_probs", [])

    result: dict[str, np.ndarray | list[int]] = {
        "continuous_hidden_states": np.stack(packed_hidden_states, axis=0) if packed_hidden_states else np.empty((0, 0), dtype=np.float16),
        "continuous_latent_embeddings": np.stack(packed_latent_embeddings, axis=0) if packed_latent_embeddings else np.empty((0, 0), dtype=np.float16),
        "continuous_latent_log_probs": np.asarray(packed_latent_log_probs, dtype=np.float32) if packed_latent_log_probs else np.empty((0,), dtype=np.float32),
        "continuous_token_mask": np.asarray(list(trace.get("continuous_token_mask", [])), dtype=np.bool_),
        "all_token_ids": list(trace.get("all_token_ids", [])),
        "all_token_logprobs": np.asarray(list(trace.get("all_token_logprobs", [])), dtype=np.float32),
    }
    if trace.get("all_hidden_states"):
        result["all_hidden_states"] = np.array(trace["all_hidden_states"], dtype=np.float16)
    if trace.get("all_token_embeddings"):
        result["all_token_embeddings"] = np.array(trace["all_token_embeddings"], dtype=np.float16)
    if trace.get("attention_weights"):
        result["attention_weights"] = np.array(trace["attention_weights"], dtype=np.float32)

    return result

def pop_request_trace(request_id: str) -> dict[str, np.ndarray] | None:
    request_id = str(request_id)

    # Flush per-request queue first (optimization 2)
    _drain_request_queue(request_id)
    _REQUEST_QUEUES.pop(request_id, None)

    with _TRACE_DICT_LOCK:
        trace = _REQUEST_TRACES.pop(request_id, None)

    if trace is None:
        return None

    return _pack_trace(trace)


def get_request_trace(request_id: str) -> dict[str, np.ndarray] | None:
    """Get request trace without popping it."""
    request_id = str(request_id)

    # Flush queue first to get latest data
    _drain_request_queue(request_id)

    with _TRACE_DICT_LOCK:
        trace = _REQUEST_TRACES.get(request_id)
        if trace is None:
            return None
        return _pack_trace(trace)


def get_latest_trace() -> dict[str, np.ndarray] | None:
    """Get the most recent trace (useful for single-request scenarios)."""
    with _TRACE_DICT_LOCK:
        if not _REQUEST_TRACES:
            return None
        # Get the most recently added trace
        request_id = list(_REQUEST_TRACES.keys())[-1]

    # Call get_request_trace WITHOUT holding the lock
    return get_request_trace(request_id)


def list_request_ids() -> list[str]:
    """List all active request IDs."""
    with _TRACE_DICT_LOCK:
        return list(_REQUEST_TRACES.keys())


__all__ = [
    "pop_request_trace",
    "get_request_trace",
    "get_latest_trace",
    "list_request_ids",
    "record_request_step",
    "reset_request_trace",
]
