"""
Thinking Mode Patch for vLLM - Continuous Latent AR
"""

import time
import functools
import json
import logging
import os
import sys

import torch
import torch.nn as nn
from safetensors.torch import load_file
from project_paths import hf_path

from vllm_thinking.trace_store import record_request_step, reset_request_trace
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


logger = logging.getLogger(__name__)
_THINKING_DEBUG = os.environ.get("VLLM_THINKING_DEBUG", "0") == "1"
_TARGET_PROB_IO_WARNED = False
_VLLM_VAE_LOGGED = False
_CONTINUOUS_AR_CERT_LOGGED = False
_FORCED_THINK_END_LOGPROB_SYNC_LOGGED = False
_DISCRETE_LATENT_CARRY_LOGGED = False


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _thinking_mode_enabled() -> bool:
    return os.environ.get("VLLM_THINKING", "0").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(msg: str):
    """Write debug message to stderr - visible in worker output."""
    if not _THINKING_DEBUG:
        return
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    log_msg = f"[{timestamp}] [VLLM_THINK] {msg}"
    print(log_msg, flush=True, file=sys.stderr)


def _append_target_prob(record: dict):
    global _TARGET_PROB_IO_WARNED
    target_prob_path = os.environ.get("VLLM_THINKING_TARGET_PROB_PATH", "")
    if not target_prob_path:
        return
    try:
        with open(target_prob_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Never fail generation because of debug I/O.
        if not _TARGET_PROB_IO_WARNED:
            _TARGET_PROB_IO_WARNED = True
            logger.warning("[Thinking] Failed to write target prob trace to %s", target_prob_path)


def _get_logprob_row(logits_for_trace: torch.Tensor | None, req_idx: int) -> torch.Tensor | None:
    """Return the per-request logits row used to build sampled-token logprobs."""
    if logits_for_trace is None:
        return None
    if logits_for_trace.ndim == 2:
        return logits_for_trace[req_idx].float()
    if logits_for_trace.ndim == 3:
        return logits_for_trace[req_idx, -1, :].float()
    return None


def _sampled_token_logprob(logits_for_trace: torch.Tensor | None, req_idx: int, token_id: int) -> float | None:
    logits_row = _get_logprob_row(logits_for_trace, req_idx)
    if logits_row is None:
        return None
    vocab_size = int(logits_row.shape[-1])
    if not 0 <= int(token_id) < vocab_size:
        return None
    return float((logits_row[int(token_id)] - torch.logsumexp(logits_row, dim=-1)).item())


def _store_visualization_data_enabled() -> bool:
    return os.environ.get("VLLM_STORE_VISUALIZATION_DATA", "0").strip().lower() in {"1", "true", "yes", "on"}


def _delta_memory_enabled() -> bool:
    return _env_flag("OPSD_DELTA_MEMORY_ENABLED", default=False)


def _delta_memory_gamma() -> float:
    raw = os.environ.get("OPSD_DELTA_MEMORY_GAMMA", "0.5").strip()
    try:
        return float(raw)
    except ValueError:
        return 0.5


def _sync_forced_token_logprobs(
    result,
    *,
    req_idx: int,
    forced_token_id: int,
    logits_row: torch.Tensor | None,
    req_id: str,
) -> None:
    """Keep ModelRunnerOutput token ids and logprobs aligned after token rewriting."""
    global _FORCED_THINK_END_LOGPROB_SYNC_LOGGED

    if getattr(result, "logprobs", None) is None:
        return

    logprobs = result.logprobs
    cu_num_generated_tokens = logprobs.cu_num_generated_tokens
    if cu_num_generated_tokens:
        pos_idx = int(cu_num_generated_tokens[req_idx])
    else:
        pos_idx = req_idx

    if pos_idx >= logprobs.logprob_token_ids.shape[0]:
        logger.warning(
            "[Thinking] Cannot sync forced token logprobs: req_id=%s req_idx=%s pos_idx=%s rows=%s",
            req_id,
            req_idx,
            pos_idx,
            int(logprobs.logprob_token_ids.shape[0]),
        )
        return

    logprobs.logprob_token_ids[pos_idx, 0] = int(forced_token_id)

    if logits_row is not None:
        vocab_size = int(logits_row.shape[-1])
        if 0 <= forced_token_id < vocab_size:
            forced_logprob = float((logits_row[forced_token_id] - torch.logsumexp(logits_row, dim=-1)).item())
            forced_rank = int((logits_row > logits_row[forced_token_id]).sum().item() + 1)
            logprobs.logprobs[pos_idx, 0] = forced_logprob
            logprobs.sampled_token_ranks[pos_idx] = forced_rank
            # vLLM/Swift later materializes a per-token dict keyed by token_id.
            # Keep that mapping aligned as well, otherwise forced token ids can
            # trigger KeyError during logprob formatting.
            if hasattr(logprobs, "topk_token_ids") and hasattr(logprobs, "topk_logprobs"):
                topk_token_ids = logprobs.topk_token_ids
                topk_logprobs = logprobs.topk_logprobs
                if pos_idx < topk_token_ids.shape[0] and topk_token_ids.shape[1] > 0:
                    topk_token_ids[pos_idx, 0] = int(forced_token_id)
                    topk_logprobs[pos_idx, 0] = forced_logprob
            if not _FORCED_THINK_END_LOGPROB_SYNC_LOGGED:
                logger.warning(
                    "[Thinking] Synced forced </think> token with rollout logprobs: req_id=%s token_id=%s rank=%s",
                    req_id,
                    forced_token_id,
                    forced_rank,
                )
                _FORCED_THINK_END_LOGPROB_SYNC_LOGGED = True
            return

    if not _FORCED_THINK_END_LOGPROB_SYNC_LOGGED:
        logger.warning(
            "[Thinking] Forced </think> token without logits row; reused existing sampled-token logprob slot. req_id=%s token_id=%s",
            req_id,
            forced_token_id,
        )
        _FORCED_THINK_END_LOGPROB_SYNC_LOGGED = True


def _sync_forced_token_ids(
    result,
    *,
    req_idx: int,
    forced_token_id: int,
) -> None:
    """Keep sampled token ids aligned across async and sync result containers."""
    try:
        sampled_token_ids = getattr(result, "sampled_token_ids", None)
        if sampled_token_ids is not None:
            sampled_token_ids[req_idx][0] = int(forced_token_id)
    except Exception:
        pass

    try:
        sampled_token_ids_cpu = getattr(result, "sampled_token_ids_cpu", None)
        async_copy_ready_event = getattr(result, "async_copy_ready_event", None)
        if sampled_token_ids_cpu is not None:
            if async_copy_ready_event is not None:
                async_copy_ready_event.synchronize()
            sampled_token_ids_cpu[req_idx, 0] = int(forced_token_id)
    except Exception:
        pass

    try:
        model_runner_output = getattr(result, "_model_runner_output", None)
        if model_runner_output is not None and req_idx < len(model_runner_output.sampled_token_ids):
            row = model_runner_output.sampled_token_ids[req_idx]
            if row:
                row[0] = int(forced_token_id)
    except Exception:
        pass


def _get_thinking_token_ids():
    return {
        "think_start": int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667")),
        "think_end": int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668")),
        "latent": int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669")),
        "think_sep": int(os.environ.get("QWEN3VL_THINKING_SEP_ID", "151670")),
    }


_TOKEN_IDS = None


def _get_token_id(name):
    global _TOKEN_IDS
    if _TOKEN_IDS is None:
        _TOKEN_IDS = _get_thinking_token_ids()
        logger.info(
            "[Thinking] Token IDs resolved: start=%s, end=%s, latent=%s, sep=%s",
            _TOKEN_IDS["think_start"],
            _TOKEN_IDS["think_end"],
            _TOKEN_IDS["latent"],
            _TOKEN_IDS["think_sep"],
        )
    return _TOKEN_IDS[name]


class LatentVAE(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int = 512, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(intermediate_size, intermediate_size, dtype=dtype),
            nn.LayerNorm(intermediate_size, dtype=dtype),
        )
        self.mean = nn.Linear(intermediate_size, hidden_size, dtype=dtype)
        self.log_std = nn.Linear(intermediate_size, hidden_size, dtype=dtype)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.distributions.Normal:
        """Map hidden states to latent distribution.

        Args:
            x: Hidden states [batch, seq_len, hidden_size] or [N, hidden_size]
            temperature: Temperature for sampling (higher = more random)

        Returns:
            Normal distribution over latent space
        """
        h = self.fc(x)
        mean = self.mean(h)
        log_std = self.log_std(h)
        std = log_std.exp() * temperature
        return torch.distributions.Normal(mean, std)

    def sample(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """Sample latent from the distribution.

        Args:
            x: Hidden states [batch, seq_len, hidden_size] or [N, hidden_size]
            temperature: Temperature for sampling

        Returns:
            Sampled latent embeddings [batch, seq_len, hidden_size]
        """
        dist = self.forward(x, temperature)
        return dist.rsample()


def _align_vae_to_hidden_state(vae: nn.Module, hidden_state: torch.Tensor) -> nn.Module:
    """Keep rollout VAE on the same device/dtype as the hidden state stream."""
    param = next(vae.parameters(), None)
    if param is None or param.device != hidden_state.device or param.dtype != hidden_state.dtype:
        vae = vae.to(device=hidden_state.device, dtype=hidden_state.dtype)
    vae.eval()
    return vae


# Token IDs now resolved lazily via _get_token_id() - see section above
DEFAULT_THINKING_LENGTH = 1024  # Fallback only when per-request limits are unavailable.
DEFAULT_MIN_CONTINUOUS_STEPS = 0


def _get_min_continuous_steps() -> int:
    """Minimum continuous AR steps before allowing </think> to exit mode."""
    try:
        return max(0, int(os.environ.get("MIN_CONTINUOUS_STEPS", str(DEFAULT_MIN_CONTINUOUS_STEPS))))
    except Exception:
        return DEFAULT_MIN_CONTINUOUS_STEPS


def _get_max_continuous_steps(max_tokens: int | None = None) -> int:
    """Maximum continuous AR steps before forcing </think>."""
    derived = max(1, int(max_tokens or 2048) // 2)
    raw = os.environ.get("MAX_CONTINUOUS_STEPS", "")
    if not str(raw).strip():
        return derived
    try:
        return max(1, int(raw))
    except Exception:
        return derived


def _get_req_num_sched_for_forward(model_runner, req_idx: int) -> int:
    """Scheduled-token count for the current v0.17+ GPU input batch."""
    input_batch = model_runner.input_batch
    computed = int(input_batch.num_computed_tokens_cpu[req_idx])
    return max(0, int(input_batch.num_tokens_no_spec[req_idx]) - computed)


def _get_req_token_pos_after_sample(model_runner, req_idx: int, req_id: str) -> int:
    """Last generated token position after sample_tokens() on v0.17+."""
    req_state = model_runner.requests.get(req_id)
    if req_state is None:
        return -1
    return int(req_state.num_tokens) - 1


def _get_result_req_ids(result) -> list[str]:
    if result is None:
        return []
    req_ids = getattr(result, "req_ids", None)
    if req_ids is not None:
        return list(req_ids)
    output = getattr(result, "_model_runner_output", None)
    if output is None:
        return []
    return list(output.req_ids)


def _get_result_sampled_token_ids(result) -> list[list[int]]:
    if result is None:
        return []
    sampled_token_ids = getattr(result, "sampled_token_ids", None)
    if sampled_token_ids:
        return sampled_token_ids

    ready_event = getattr(result, "async_copy_ready_event", None)
    sampled_token_ids_cpu = getattr(result, "sampled_token_ids_cpu", None)
    if ready_event is None or sampled_token_ids_cpu is None:
        return []

    ready_event.synchronize()
    sampled_list = result.sampled_token_ids_cpu.tolist()
    invalid_req_indices = set(result._invalid_req_indices)
    for req_idx in invalid_req_indices:
        if 0 <= req_idx < len(sampled_list):
            sampled_list[req_idx] = []
    return sampled_list


def _resolve_token_embedding_weight(model_runner) -> torch.Tensor | None:
    model = getattr(model_runner, "model", None)
    if model is None:
        return None
    if (
        hasattr(model, "language_model")
        and hasattr(model.language_model, "model")
        and hasattr(model.language_model.model, "embed_tokens")
    ):
        return model.language_model.model.embed_tokens.weight
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens.weight
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        language_model = model.model.language_model
        if hasattr(language_model, "model") and hasattr(language_model.model, "embed_tokens"):
            return language_model.model.embed_tokens.weight
        if hasattr(language_model, "embed_tokens"):
            return language_model.embed_tokens.weight
    if hasattr(model, "model") and hasattr(model.model, "llm_model") and hasattr(model.model.llm_model, "embed_tokens"):
        return model.model.llm_model.embed_tokens.weight
    return None


def _build_discrete_latent_carry_embedding(
    *,
    model_runner,
    latent_token_id: int,
    hidden_state: torch.Tensor | None,
    previous_memory: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if hidden_state is None:
        return None
    embed_weight = _resolve_token_embedding_weight(model_runner)
    if embed_weight is None:
        return None
    vocab_size = int(embed_weight.shape[0])
    if not 0 <= int(latent_token_id) < vocab_size:
        return None

    token_embedding = embed_weight[latent_token_id : latent_token_id + 1].detach()
    carry_hidden = hidden_state.detach()
    if carry_hidden.ndim == 1:
        carry_hidden = carry_hidden.unsqueeze(0)
    if carry_hidden.ndim == 2:
        carry_hidden = carry_hidden[:, :].clone()
    if _delta_memory_enabled():
        if previous_memory is not None:
            memory = previous_memory.detach()
            if memory.ndim == 1:
                memory = memory.unsqueeze(0)
            gamma = _delta_memory_gamma()
            carry_hidden = memory + gamma * (carry_hidden - memory)
        else:
            carry_hidden = carry_hidden.clone()
    if carry_hidden.dtype != token_embedding.dtype:
        carry_hidden = carry_hidden.to(dtype=token_embedding.dtype)
    if carry_hidden.device != token_embedding.device:
        carry_hidden = carry_hidden.to(device=token_embedding.device)
    return token_embedding + carry_hidden

_patch_applied = False


def apply_thinking_mode_patch():
    _debug_log("[PATCH] Starting apply_thinking_mode_patch...")
    _debug_log(f"[PATCH] GPUModelRunner: {GPUModelRunner}")
    _debug_log(f"[PATCH] hasattr execute_model: {hasattr(GPUModelRunner, 'execute_model')}")
    _debug_log(f"[PATCH] hasattr sample_tokens: {hasattr(GPUModelRunner, 'sample_tokens')}")

    global _patch_applied
    if _patch_applied:
        return
    if hasattr(GPUModelRunner, '_thinking_patch_applied'):
        return
    target_prob_path = os.environ.get("VLLM_THINKING_TARGET_PROB_PATH", "")
    target_prob_max_steps = int(os.environ.get("VLLM_THINKING_TARGET_PROB_MAX_STEPS", "8"))
    if target_prob_path:
        logger.info(
            "[Thinking] Target prob trace enabled: path=%s max_steps=%s",
            target_prob_path,
            target_prob_max_steps,
        )

    # ============== VAE Loading ==============
    def _load_latent_vae_from_checkpoint(self):
        global _VLLM_VAE_LOGGED
        if hasattr(self, 'latent_vae') and self.latent_vae is not None:
            return

        hidden_size = self.model.config.text_config.hidden_size
        self.latent_vae = None

        # Try LoRA path first
        lora_path = os.environ.get("VLLM_LORA_CHECKPOINT_PATH")
        if lora_path:
            vae_path = os.path.join(lora_path, "vae.safetensors")
            if os.path.exists(vae_path):
                try:
                    vae_state = load_file(vae_path)
                    loaded_vae = LatentVAE(hidden_size=hidden_size)
                    loaded_vae.load_state_dict(vae_state, strict=False)
                    loaded_vae = loaded_vae.cuda()
                    loaded_vae.eval()
                    self.latent_vae = loaded_vae
                    if not _VLLM_VAE_LOGGED:
                        logger.warning("[Thinking] vLLM rollout VAE loaded: path=%s hidden_size=%s", vae_path, hidden_size)
                        _VLLM_VAE_LOGGED = True
                    return
                except Exception as e:
                    self.latent_vae = None
                    logger.warning(f"[Thinking] Load VAE failed: {e}")

        # Try model path
        model_path = os.environ.get("VLLM_MODEL_PATH", str(hf_path("Qwen", "Qwen3-VL-2B-Thinking")))
        vae_path = os.path.join(model_path, "vae.safetensors")
        if os.path.exists(vae_path):
            try:
                vae_state = load_file(vae_path)
                loaded_vae = LatentVAE(hidden_size=hidden_size)
                loaded_vae.load_state_dict(vae_state, strict=False)
                loaded_vae = loaded_vae.cuda()
                loaded_vae.eval()
                self.latent_vae = loaded_vae
                if not _VLLM_VAE_LOGGED:
                    logger.warning("[Thinking] vLLM rollout VAE loaded: path=%s hidden_size=%s", vae_path, hidden_size)
                    _VLLM_VAE_LOGGED = True
                return
            except Exception as e:
                self.latent_vae = None
                logger.warning(f"[Thinking] Load VAE failed: {e}")

        logger.debug("[Thinking] No VAE found")

    def _detect_initial_mode_from_prompt(prompt_ids) -> str:
        if not _thinking_mode_enabled():
            return "discrete"
        prompt_ids = list(prompt_ids or [])
        if not prompt_ids:
            return "discrete"

        think_start_id = _get_token_id("think_start")
        think_end_id = _get_token_id("think_end")
        last_start = -1
        last_end = -1
        for idx, tok in enumerate(prompt_ids):
            if tok == think_start_id:
                last_start = idx
            elif tok == think_end_id:
                last_end = idx
        return "continuous" if last_start > last_end else "discrete"

    # ============== Init ==============
    _orig_init = GPUModelRunner.__init__

    @functools.wraps(_orig_init)
    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._thinking_state = {}
        self._vae_loaded = False
        self.latent_vae = None
        self.hidden_size = self.model_config.get_hidden_size()
        # Use vLLM's native mixed token/embed input path so the model call
        # signature stays static across decode iterations.
        self.enable_prompt_embeds = True

    def _get_or_create_req_embed_storage(self, req_idx: int) -> torch.Tensor:
        storage = self.input_batch.req_prompt_embeds.get(req_idx)
        if storage is None or storage.shape[0] != self.max_model_len:
            storage = torch.zeros(
                (self.max_model_len, self.hidden_size),
                dtype=self.dtype,
                device="cpu",
                pin_memory=self.pin_memory,
            )
            self.input_batch.req_prompt_embeds[req_idx] = storage
        return storage

    def _set_next_decode_embedding(
        self,
        req_idx: int,
        token_pos: int,
        embedding: torch.Tensor | None,
    ) -> None:
        if token_pos < 0 or token_pos >= self.max_model_len:
            return
        if embedding is None:
            self.input_batch.is_token_ids[req_idx, token_pos] = True
            return

        storage = _get_or_create_req_embed_storage(self, req_idx)
        emb = embedding.detach()
        if emb.ndim == 3:
            emb = emb[0, 0]
        elif emb.ndim == 2:
            emb = emb[0]
        if emb.device.type != "cpu":
            emb = emb.to(device="cpu", non_blocking=False)
        if emb.dtype != self.dtype:
            emb = emb.to(dtype=self.dtype)
        storage[token_pos].copy_(emb)
        self.input_batch.is_token_ids[req_idx, token_pos] = False

    # ============== Execute Model ==============
    _orig_execute_model = GPUModelRunner.execute_model
    _orig_prepare_input_ids = GPUModelRunner._prepare_input_ids
    _orig_preprocess = GPUModelRunner._preprocess
    _orig_model_forward = GPUModelRunner._model_forward

    @functools.wraps(_orig_execute_model)
    def patched_execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
        if not self._vae_loaded:
            _load_latent_vae_from_checkpoint(self)
            self._vae_loaded = True
            _debug_log(f"[DEBUG] VAE loaded: {self.latent_vae is not None}")

        # Track new requests
        if hasattr(scheduler_output, 'scheduled_new_reqs') and scheduler_output.scheduled_new_reqs:
            for req in scheduler_output.scheduled_new_reqs:
                if req.req_id not in self._thinking_state:
                    reset_request_trace(req.req_id)
                    # Initial mode detection from prompt
                    prompt_ids = req.prompt_token_ids or []
                    mode = _detect_initial_mode_from_prompt(prompt_ids)

                    max_tokens = getattr(req, 'max_tokens', 2048) or 2048
                    thinking_length = _get_max_continuous_steps(max_tokens)

                    self._thinking_state[req.req_id] = {
                        'mode': mode,
                        'step': 0,
                        'embedding': None,
                        'delta_memory': None,
                        'thinking_length': thinking_length,
                        'max_thinking_length': thinking_length,
                        'min_continuous_steps': _get_min_continuous_steps(),
                    }
                    _debug_log(
                        f"[DEBUG] New req {req.req_id}: mode={mode}, "
                        f"thinking_length={thinking_length}, "
                        f"prompt_len={len(prompt_ids)}, "
                        f"start_count={sum(1 for t in prompt_ids if t == _get_token_id('think_start'))}, "
                        f"end_count={sum(1 for t in prompt_ids if t == _get_token_id('think_end'))}, "
                        f"prompt_tail={prompt_ids[-12:]}"
                    )

        # Track cached requests
        if hasattr(scheduler_output, 'scheduled_cached_reqs') and scheduler_output.scheduled_cached_reqs:
            for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
                if req_id not in self._thinking_state:
                    reset_request_trace(req_id)
                    self._thinking_state[req_id] = {
                        'mode': 'discrete',
                        'step': 0,
                        'embedding': None,
                        'delta_memory': None,
                    }
                    _debug_log(f"[DEBUG] Cached req {req_id}: init as discrete")

        # Clean finished
        if hasattr(scheduler_output, 'finished_req_ids'):
            for req_id in scheduler_output.finished_req_ids:
                self._thinking_state.pop(req_id, None)
                _debug_log(f"[DEBUG] Finished req {req_id}: removed from state")

        return _orig_execute_model(self, scheduler_output, intermediate_tensors)

    @functools.wraps(_orig_prepare_input_ids)
    def patched_prepare_input_ids(self, scheduler_output, total_num_scheduled_tokens, cu_num_tokens):
        embed_rows: list[int] = []
        if self.enable_prompt_embeds:
            req_ids = getattr(self.input_batch, "req_ids", ())
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu
            is_token_ids = self.input_batch.is_token_ids
            num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})

            flat_idx = 0
            for req_idx, req_id in enumerate(req_ids):
                num_sched = int(num_scheduled_tokens.get(req_id, 0))
                if num_sched <= 0:
                    continue

                start = int(num_computed_tokens[req_idx])
                end = start + num_sched
                if end > start:
                    row_mask = is_token_ids[req_idx, start:end]
                    for local_idx, is_token in enumerate(row_mask):
                        if not bool(is_token):
                            embed_rows.append(flat_idx + local_idx)
                flat_idx += num_sched

        _orig_prepare_input_ids(
            self,
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )

        if not embed_rows or not self.enable_prompt_embeds:
            return

        # Keep vLLM's async token backfill for normal decode rows, then patch
        # only the embed-driven rows so multimodal embed_input_ids never sees
        # async placeholder token IDs such as -1.
        for flat_idx in embed_rows:
            self.input_ids.gpu[flat_idx] = 0
            if flat_idx < total_num_scheduled_tokens:
                self.inputs_embeds.gpu[flat_idx].copy_(
                    self.inputs_embeds.cpu[flat_idx].to(
                        device=self.inputs_embeds.gpu.device,
                        dtype=self.inputs_embeds.gpu.dtype,
                        non_blocking=False,
                    )
                )
                self.is_token_ids.gpu[flat_idx] = False
        return

    @functools.wraps(_orig_preprocess)
    def patched_preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors=None):
        (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        ) = _orig_preprocess(self, scheduler_output, num_input_tokens, intermediate_tensors)

        # For multimodal models, upstream `_preprocess` rebuilds the full scheduled
        # embedding tensor and overwrites `self.inputs_embeds.gpu[...]`. Restore only
        # the rows explicitly marked as embed-driven by our continuous mode patch.
        if (
            self.enable_prompt_embeds
            and self.supports_mm_inputs
            and inputs_embeds is not None
        ):
            req_ids = getattr(self.input_batch, "req_ids", ())
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu
            num_prompt_tokens = self.input_batch.num_prompt_tokens
            is_token_ids = self.input_batch.is_token_ids
            num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})

            flat_idx = 0
            for req_idx, req_id in enumerate(req_ids):
                num_sched = int(num_scheduled_tokens.get(req_id, 0))
                if num_sched <= 0:
                    continue

                start_pos = int(num_computed_tokens[req_idx])
                storage = self.input_batch.req_prompt_embeds.get(req_idx)
                if storage is None:
                    flat_idx += num_sched
                    continue

                for offset in range(num_sched):
                    token_pos = start_pos + offset
                    if token_pos >= storage.shape[0]:
                        break
                    # Never overwrite prompt-time multimodal embeddings. Our
                    # continuous embeddings are only valid for decode positions.
                    if token_pos < int(num_prompt_tokens[req_idx]):
                        continue
                    if bool(is_token_ids[req_idx, token_pos]):
                        continue
                    inputs_embeds[flat_idx + offset].copy_(
                        storage[token_pos].to(
                            device=inputs_embeds.device,
                            dtype=inputs_embeds.dtype,
                            non_blocking=False,
                        )
                    )
                flat_idx += num_sched

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    @functools.wraps(_orig_model_forward)
    def patched_model_forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        if _THINKING_DEBUG and hasattr(self, "input_batch"):
            req_ids = getattr(self.input_batch, "req_ids", ())
            num_reqs = len(req_ids)
            if num_reqs > 0:
                num_computed = self.input_batch.num_computed_tokens_cpu
                flat_idx = 0
                for req_idx, req_id in enumerate(req_ids):
                    state = self._thinking_state.get(req_id)
                    num_sched = _get_req_num_sched_for_forward(self, req_idx)
                    if num_sched <= 0:
                        continue

                    if state and state.get("mode") == "continuous":
                        decode_pos = int(num_computed[req_idx])
                        is_token = bool(self.input_batch.is_token_ids[req_idx, decode_pos])
                        embed_norm = None
                        embed_diff = None
                        if inputs_embeds is not None and flat_idx < inputs_embeds.shape[0]:
                            try:
                                embed_norm = float(inputs_embeds[flat_idx].float().norm().item())
                                storage = self.input_batch.req_prompt_embeds.get(req_idx)
                                if storage is not None and decode_pos < storage.shape[0]:
                                    expected = storage[decode_pos].to(
                                        device=inputs_embeds.device,
                                        dtype=inputs_embeds.dtype,
                                        non_blocking=False,
                                    )
                                    embed_diff = float(
                                        (inputs_embeds[flat_idx] - expected).abs().max().item()
                                    )
                            except Exception:
                                embed_norm = None
                        _debug_log(
                            f"[FORWARD] req={req_id} mode=continuous decode_pos={decode_pos} "
                            f"flat_idx={flat_idx} num_sched={num_sched} input_ids_is_none={input_ids is None} "
                            f"inputs_embeds_is_none={inputs_embeds is None} is_token_ids={is_token} "
                            f"embed_norm={embed_norm if embed_norm is not None else 'N/A'} "
                            f"embed_max_abs_diff={embed_diff if embed_diff is not None else 'N/A'}"
                        )
                    flat_idx += num_sched

        return _orig_model_forward(
            self,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # ============== Sample Tokens ==============
    # This is where hidden_states is available from execute_model_state
    _orig_sample_tokens = GPUModelRunner.sample_tokens

    @functools.wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output):
        # Cache ephemeral outputs before calling original sample_tokens().
        # vLLM clears execute_model_state inside sample_tokens.
        hidden_states = None
        sample_hidden_states = None
        logits_for_trace = None
        attention_weights = None
        store_visualization_data = _store_visualization_data_enabled()
        if hasattr(self, 'execute_model_state') and self.execute_model_state is not None:
            hidden_states = self.execute_model_state.hidden_states
            sample_hidden_states = self.execute_model_state.sample_hidden_states
            logits_for_trace = self.execute_model_state.logits
            if store_visualization_data:
                attention_weights = getattr(self.execute_model_state, 'attention_weights', None)

        # Call original sample_tokens
        result = _orig_sample_tokens(self, grammar_output)
        if result is None:
            return None

        # Get batch_req_ids to match with thinking state
        batch_req_ids = _get_result_req_ids(result)
        if not batch_req_ids and hasattr(self, 'input_batch') and hasattr(self.input_batch, 'req_ids'):
            batch_req_ids = list(self.input_batch.req_ids)

        sampled_tokens_list = _get_result_sampled_token_ids(result) if batch_req_ids else []

        if batch_req_ids and sampled_tokens_list:
            for i, req_id in enumerate(batch_req_ids):
                if req_id not in self._thinking_state:
                    continue
                state = self._thinking_state[req_id]
                # sampled_token_ids is list of list, get first token of first (and only) generation
                sampled = sampled_tokens_list[i][0] if i < len(sampled_tokens_list) and sampled_tokens_list[i] else 0
                mode = state.get('mode', 'discrete')
                step = state.get('step', 0)

                # Get hidden states output for preparing next input embedding
                last_hidden = None
                if sample_hidden_states is not None and i < sample_hidden_states.shape[0]:
                    try:
                        last_hidden = sample_hidden_states[i:i+1, :].detach().clone()
                        if last_hidden.dtype != torch.bfloat16:
                            last_hidden = last_hidden.to(dtype=torch.bfloat16)
                        state['step'] = step + 1
                    except Exception as e:
                        _debug_log(f"[ERROR] sample_tokens: failed to extract sample_hidden: {e}")
                        continue
                elif hidden_states is not None:
                    try: 
                        last_hidden = hidden_states[i:i+1, :].detach().clone()
                        # Ensure correct dtype and device
                        if last_hidden.dtype != torch.bfloat16:
                            last_hidden = last_hidden.to(dtype=torch.bfloat16)
                        state['step'] = step + 1
                    except Exception as e:
                        _debug_log(f"[ERROR] sample_tokens: failed to extract hidden: {e}")
                        continue

                # Optional targeted probability tracing:
                target_prob_path = os.environ.get("VLLM_THINKING_TARGET_PROB_PATH", "")
                target_prob_max_steps = int(os.environ.get("VLLM_THINKING_TARGET_PROB_MAX_STEPS", "8"))
                if (
                    target_prob_path
                    and step < target_prob_max_steps
                    and logits_for_trace is not None
                ):
                    try:
                        logits = logits_for_trace
                        if logits.ndim == 2:
                            row = logits[i].float()
                        elif logits.ndim == 3:
                            row = logits[i, -1, :].float()
                        else:
                            row = None
                        if row is not None:
                            denom = torch.logsumexp(row, dim=-1)
                            targets = {
                                "think_start": _get_token_id("think_start"),
                                "think_end": _get_token_id("think_end"),
                                "latent": _get_token_id("latent"),
                                "think_sep": _get_token_id("think_sep"),
                            }
                            target_logprobs = {}
                            vocab = int(row.shape[-1])
                            for name, tid in targets.items():
                                if 0 <= tid < vocab:
                                    tval = row[tid]
                                    lp = float((tval - denom).item())
                                    rank = int((row > tval).sum().item() + 1)
                                    target_logprobs[name] = {"token_id": int(tid), "logprob": lp, "rank": rank}
                                else:
                                    target_logprobs[name] = {"token_id": int(tid), "logprob": None, "rank": None}
                            _append_target_prob(
                                {
                                    "req_id": str(req_id),
                                    "step": int(step),
                                    "mode": mode,
                                    "sampled_token_id": int(sampled),
                                    "target_logprobs": target_logprobs,
                                }
                            )
                    except Exception:
                        pass

                # Monitor sampled token for mode switching
                if _thinking_mode_enabled() and sampled == _get_token_id('think_start'):
                    state['mode'] = 'continuous'
                    state['step'] = 0
                    _debug_log(f"[DEBUG] sample_tokens: switched to continuous (think_start)")
                elif _thinking_mode_enabled() and sampled == _get_token_id('think_end'):
                    min_steps = int(state.get('min_continuous_steps', _get_min_continuous_steps()))
                    if mode == 'continuous' and step < min_steps:
                        _debug_log(
                            f"[DEBUG] sample_tokens: ignore early think_end req={req_id} step={step} min_steps={min_steps}"
                        )
                    else:
                        state['mode'] = 'discrete'
                        state['embedding'] = None
                        state['delta_memory'] = None
                        _debug_log(f"[DEBUG] sample_tokens: switched to discrete (think_end)")
                elif _thinking_mode_enabled() and sampled == _get_token_id('think_sep'):
                    state['thinking_length'] = state.get('max_thinking_length', DEFAULT_THINKING_LENGTH)
                    state['step'] = 0
                    _debug_log(f"[DEBUG] sample_tokens: reset thinking length to {state['thinking_length']} (think_sep)")

                thinking_length = state.get('thinking_length', DEFAULT_THINKING_LENGTH)
                latent_embedding = None
                latent_logprob = None
                thinking_enabled = _thinking_mode_enabled()
                if thinking_enabled and state.get('mode') == 'continuous' and step >= thinking_length:
                    if i < len(sampled_tokens_list) and sampled_tokens_list[i]:
                        sampled_tokens_list[i][0] = _get_token_id('think_end')
                        sampled = sampled_tokens_list[i][0]
                        _sync_forced_token_ids(
                            result,
                            req_idx=i,
                            forced_token_id=int(sampled),
                        )
                        _sync_forced_token_logprobs(
                            result,
                            req_idx=i,
                            forced_token_id=int(sampled),
                            logits_row=_get_logprob_row(logits_for_trace, i),
                            req_id=str(req_id),
                        )
                        req_pos = _get_req_token_pos_after_sample(self, i, req_id)
                        if req_pos >= 0:
                            self.input_batch.token_ids_cpu[i, req_pos] = sampled
                            req_state = self.requests.get(req_id)
                            if req_state is not None and req_state.output_token_ids:
                                req_state.output_token_ids[-1] = sampled
                        _debug_log(f"[DEBUG] sample_tokens: forced think_end for req={req_id}")
                    state['mode'] = 'discrete'
                    state['embedding'] = None
                    state['delta_memory'] = None
                    state['thinking_length'] = state.get('max_thinking_length', DEFAULT_THINKING_LENGTH)
                    _debug_log(f"[DEBUG] sample_tokens: exited continuous (step limit)")
                elif thinking_enabled and state.get('mode') == 'continuous' and last_hidden is not None:
                        if self.latent_vae is not None:
                            self.latent_vae = _align_vae_to_hidden_state(self.latent_vae, last_hidden)
                            vae_dist = self.latent_vae.forward(last_hidden, temperature=1.0)
                            vae_emb = vae_dist.rsample()
                            latent_embedding = vae_emb
                            latent_logprob = vae_dist.log_prob(vae_emb).mean(dim=-1)
                            state['embedding'] = vae_emb
                        else:
                            state['embedding'] = last_hidden

                discrete_latent_carry_embedding = None
                use_discrete_latent_carry = bool(
                    state.get('mode') == 'discrete'
                    and last_hidden is not None
                    and int(sampled) == _get_token_id('latent')
                )
                if use_discrete_latent_carry:
                    discrete_latent_carry_embedding = _build_discrete_latent_carry_embedding(
                        model_runner=self,
                        latent_token_id=_get_token_id('latent'),
                        hidden_state=last_hidden,
                        previous_memory=state.get('delta_memory'),
                    )
                    state['embedding'] = discrete_latent_carry_embedding
                    if _delta_memory_enabled() and last_hidden is not None:
                        updated_memory = last_hidden.detach().clone()
                        previous_memory = state.get('delta_memory')
                        if previous_memory is not None:
                            gamma = _delta_memory_gamma()
                            updated_memory = previous_memory.detach().to(
                                device=updated_memory.device,
                                dtype=updated_memory.dtype,
                            ) + gamma * (updated_memory - previous_memory.detach().to(
                                device=updated_memory.device,
                                dtype=updated_memory.dtype,
                            ))
                        state['delta_memory'] = updated_memory
                    global _DISCRETE_LATENT_CARRY_LOGGED
                    if discrete_latent_carry_embedding is not None and not _DISCRETE_LATENT_CARRY_LOGGED:
                        hidden_norm = float(last_hidden.float().norm().item())
                        carry_norm = float(discrete_latent_carry_embedding.float().norm().item())
                        logger.warning(
                            "[Thinking] discrete latent carry active: req_id=%s token_id=%s mode=add_hidden_to_next_input hidden_norm=%.6f carry_norm=%.6f",
                            req_id,
                            int(sampled),
                            hidden_norm,
                            carry_norm,
                        )
                        _DISCRETE_LATENT_CARRY_LOGGED = True

                use_continuous_embedding = bool(
                    thinking_enabled and state.get('mode') == 'continuous' and last_hidden is not None
                )
                global _CONTINUOUS_AR_CERT_LOGGED
                if use_continuous_embedding and not _CONTINUOUS_AR_CERT_LOGGED:
                    logger.warning(
                        "[Thinking] vLLM continuous AR active: req_id=%s source=%s saved_hidden_states=true saved_latent_embeddings=%s saved_latent_log_probs=%s",
                        req_id,
                        "latent_vae_rsample" if self.latent_vae is not None else "last_hidden",
                        bool(latent_embedding is not None),
                        bool(latent_logprob is not None),
                    )
                    _CONTINUOUS_AR_CERT_LOGGED = True
                # Keep answer-token trace lightweight; continuous tokens still
                # carry hidden/latent tensors for VAE replay.
                token_hidden_state_for_step = last_hidden if store_visualization_data else None
                token_embedding_for_step = None
                all_token_ids_for_step = []
                all_token_logprobs_for_step = []

                if sampled is not None:
                    all_token_ids_for_step = [int(sampled)]
                    all_token_logprobs_for_step = [
                        _sampled_token_logprob(logits_for_trace, i, int(sampled))
                    ]
                    if store_visualization_data:
                        try:
                            embed_tokens = None
                            if (
                                hasattr(self.model, 'language_model')
                                and hasattr(self.model.language_model, 'model')
                                and hasattr(self.model.language_model.model, 'embed_tokens')
                            ):
                                embed_tokens = self.model.language_model.model.embed_tokens.weight
                            elif hasattr(self.model.model, 'embed_tokens'):
                                embed_tokens = self.model.model.embed_tokens.weight
                            elif hasattr(self.model.model, 'language_model'):
                                language_model = self.model.model.language_model
                                if hasattr(language_model, 'model') and hasattr(language_model.model, 'embed_tokens'):
                                    embed_tokens = language_model.model.embed_tokens.weight
                                elif hasattr(language_model, 'embed_tokens'):
                                    embed_tokens = language_model.embed_tokens.weight
                            elif hasattr(self.model.model, 'llm_model') and hasattr(self.model.model.llm_model, 'embed_tokens'):
                                embed_tokens = self.model.model.llm_model.embed_tokens.weight

                            if embed_tokens is not None:
                                token_embedding_for_step = embed_tokens[sampled:sampled + 1, :].detach().clone()
                                if token_embedding_for_step.dtype != torch.bfloat16:
                                    token_embedding_for_step = token_embedding_for_step.to(dtype=torch.bfloat16)
                        except Exception as exc:
                            _debug_log(f"[ERROR] sample_tokens: failed to extract token embedding: {exc}")

                record_request_step(
                    req_id,
                    hidden_state=last_hidden,
                    latent_embedding=latent_embedding,
                    latent_logprob=latent_logprob,
                    use_continuous_embedding=use_continuous_embedding,
                    all_hidden_states=token_hidden_state_for_step,
                    all_token_ids=all_token_ids_for_step,
                    all_token_logprobs=all_token_logprobs_for_step,
                    attention_weights=attention_weights,
                    token_embeddings=token_embedding_for_step,
                )

                # vLLM's next decode step reads slot `num_computed_tokens`, which
                # after sampling corresponds to the just-added output token slot:
                # `num_tokens - 1`. Override that slot so the next forward consumes
                # the hidden-state-derived embedding instead of the sampled token ID.
                token_pos = _get_req_token_pos_after_sample(self, i, req_id)
                if thinking_enabled and state.get('mode') == 'continuous':
                    _set_next_decode_embedding(
                        self,
                        i,
                        token_pos,
                        state.get('embedding'),
                    )
                elif use_discrete_latent_carry:
                    _set_next_decode_embedding(
                        self,
                        i,
                        token_pos,
                        discrete_latent_carry_embedding,
                    )
                else:
                    if state.get('mode') == 'discrete':
                        state['embedding'] = None
                    _set_next_decode_embedding(self, i, token_pos, None)

        return result

    # Apply patches to GPUModelRunner (from gpu.model_runner)
    GPUModelRunner.__init__ = patched_init
    _debug_log("[PATCH] Patched __init__")
    GPUModelRunner.execute_model = patched_execute_model
    _debug_log("[PATCH] Patched execute_model")
    GPUModelRunner._prepare_input_ids = patched_prepare_input_ids
    _debug_log("[PATCH] Patched _prepare_input_ids")
    GPUModelRunner._preprocess = patched_preprocess
    _debug_log("[PATCH] Patched _preprocess")
    GPUModelRunner._model_forward = patched_model_forward
    _debug_log("[PATCH] Patched _model_forward")
    GPUModelRunner.sample_tokens = patched_sample_tokens
    _debug_log("[PATCH] Patched sample_tokens")
    GPUModelRunner._thinking_patch_applied = True
    _debug_log("[PATCH] All patches applied successfully!")

    _patch_applied = True
    logger.info("[Thinking] ✓ Continuous latent AR patch applied")
