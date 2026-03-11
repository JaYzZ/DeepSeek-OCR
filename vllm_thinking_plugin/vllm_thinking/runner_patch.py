"""
Thinking Mode Patch for vLLM - Continuous Latent AR
"""

import functools
import json
import logging
import os
import sys
from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput

import torch
import torch.nn as nn
from safetensors.torch import load_file
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


logger = logging.getLogger(__name__)
_THINKING_DEBUG = os.environ.get("VLLM_THINKING_DEBUG", "0") == "1"
_TARGET_PROB_IO_WARNED = False

# Persistent debug log file
_DEBUG_LOG_FILE = "/tmp/vllm_thinking_debug.log"

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

import time


# Lazy token ID resolution - reads from env vars set by llamafactory integration.py
# This ensures train/infer consistency

def _get_thinking_token_ids():
    """Resolve thinking token IDs.

    Canonical source is QWEN3VL_* IDs exported by training/eval scripts.
    """
    return (
        int(os.environ.get("QWEN3VL_THINKING_START_ID", "151667")),
        int(os.environ.get("QWEN3VL_THINKING_END_ID", "151668")),
        int(os.environ.get("QWEN3VL_LATENT_TOKEN_ID", "151669")),
        int(os.environ.get("QWEN3VL_THINKING_SEP_ID", "151670")),
    )


# Lazy initialization - token IDs resolved at runtime
_THINK_START_ID = None
_THINK_END_ID = None
_LATENT_TOKEN_ID = None
_THINK_SEP_ID = None


def _get_token_id(name):
    """Lazy load token ID on first access."""
    global _THINK_START_ID, _THINK_END_ID, _LATENT_TOKEN_ID, _THINK_SEP_ID
    if _THINK_START_ID is None:
        _THINK_START_ID, _THINK_END_ID, _LATENT_TOKEN_ID, _THINK_SEP_ID = _get_thinking_token_ids()
        logger.info(f"[Thinking] Token IDs resolved: start={_THINK_START_ID}, end={_THINK_END_ID}, latent={_LATENT_TOKEN_ID}, sep={_THINK_SEP_ID}")
    if name == 'think_start':
        return _THINK_START_ID
    elif name == 'think_end':
        return _THINK_END_ID
    elif name == 'latent':
        return _LATENT_TOKEN_ID
    elif name == 'think_sep':
        return _THINK_SEP_ID
    return None


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


# Token IDs now resolved lazily via _get_token_id() - see section above
DEFAULT_THINKING_LENGTH = 1024  # Will be min(max_tokens//2, 1024)
DEFAULT_MIN_CONTINUOUS_STEPS = 0


def _get_min_continuous_steps() -> int:
    """Minimum continuous AR steps before allowing </think> to exit mode."""
    try:
        return max(0, int(os.environ.get("MIN_CONTINUOUS_STEPS", str(DEFAULT_MIN_CONTINUOUS_STEPS))))
    except Exception:
        return DEFAULT_MIN_CONTINUOUS_STEPS

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
        if hasattr(self, 'latent_vae') and self.latent_vae is not None:
            return

        hidden_size = self.model.config.text_config.hidden_size

        # Try LoRA path first
        lora_path = os.environ.get("VLLM_LORA_CHECKPOINT_PATH")
        if lora_path:
            vae_path = os.path.join(lora_path, "vae.safetensors")
            if os.path.exists(vae_path):
                try:
                    vae_state = load_file(vae_path)
                    self.latent_vae = LatentVAE(hidden_size=hidden_size)
                    self.latent_vae.load_state_dict(vae_state, strict=False)
                    self.latent_vae = self.latent_vae.cuda()
                    logger.info(f"[Thinking] Loaded VAE from {vae_path}")
                    return
                except Exception as e:
                    logger.warning(f"[Thinking] Load VAE failed: {e}")

        # Try model path
        model_path = os.environ.get("VLLM_MODEL_PATH", "/share/project/xiyan/huggingface/Qwen/Qwen3-VL-2B-Thinking")
        vae_path = os.path.join(model_path, "vae.safetensors")
        if os.path.exists(vae_path):
            try:
                vae_state = load_file(vae_path)
                self.latent_vae = LatentVAE(hidden_size=hidden_size)
                self.latent_vae.load_state_dict(vae_state, strict=False)
                self.latent_vae = self.latent_vae.cuda()
                logger.info(f"[Thinking] Loaded VAE from {vae_path}")
                return
            except Exception as e:
                logger.warning(f"[Thinking] Load VAE failed: {e}")

        logger.warning("[Thinking] No VAE found")

    def _detect_initial_mode_from_prompt(prompt_ids) -> str:
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

    def _verify_continuous_decode_inputs(self, scheduler_output) -> None:
        if not _THINKING_DEBUG or scheduler_output is None:
            return

        req_data = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if req_data is None or not getattr(req_data, "req_ids", None):
            return

        num_sched_map = getattr(scheduler_output, "num_scheduled_tokens", None) or {}
        req_to_num_computed = {
            req_id: int(req_data.num_computed_tokens[i])
            for i, req_id in enumerate(req_data.req_ids)
        }

        for req_id, decode_pos in req_to_num_computed.items():
            if int(num_sched_map.get(req_id, 0)) <= 0:
                continue
            state = self._thinking_state.get(req_id)
            if not state or state.get("mode") != "continuous":
                continue

            expected = state.get("embedding")
            if expected is None:
                _debug_log(f"[VERIFY] req={req_id} continuous but has no cached embedding")
                continue

            req_idx = self.input_batch.req_id_to_index.get(req_id)
            if req_idx is None:
                raise RuntimeError(f"[VLLM_THINK][VERIFY] req={req_id} missing from req_id_to_index")

            if decode_pos < 0 or decode_pos >= self.max_model_len:
                raise RuntimeError(
                    f"[VLLM_THINK][VERIFY] req={req_id} invalid decode_pos={decode_pos} max_model_len={self.max_model_len}"
                )

            is_token = bool(self.input_batch.is_token_ids[req_idx, decode_pos])
            if is_token:
                raise RuntimeError(
                    f"[VLLM_THINK][VERIFY] req={req_id} decode_pos={decode_pos} unexpectedly marked as token-id input"
                )

            storage = self.input_batch.req_prompt_embeds.get(req_idx)
            if storage is None:
                raise RuntimeError(
                    f"[VLLM_THINK][VERIFY] req={req_id} decode_pos={decode_pos} missing req_prompt_embeds storage"
                )

            actual = storage[decode_pos]
            ref = expected.detach()
            if ref.ndim == 3:
                ref = ref[0, 0]
            elif ref.ndim == 2:
                ref = ref[0]
            if ref.device.type != "cpu":
                ref = ref.to(device="cpu", non_blocking=False)
            if ref.dtype != actual.dtype:
                ref = ref.to(dtype=actual.dtype)

            max_abs_diff = float((actual - ref).abs().max().item())
            actual_norm = float(actual.float().norm().item())
            ref_norm = float(ref.float().norm().item())
            _debug_log(
                f"[VERIFY] req={req_id} mode=continuous decode_pos={decode_pos} "
                f"is_token_ids=False embed_norm={actual_norm:.6f} ref_norm={ref_norm:.6f} "
                f"max_abs_diff={max_abs_diff:.6e}"
            )
            if max_abs_diff > 1e-3:
                raise RuntimeError(
                    f"[VLLM_THINK][VERIFY] req={req_id} decode_pos={decode_pos} "
                    f"embed mismatch max_abs_diff={max_abs_diff:.6e}"
                )

    # ============== VAE Transform Helper ==============
    def _transform_hidden_to_embedding(self, hidden: torch.Tensor) -> torch.Tensor:
        """Transform hidden state to input embedding via VAE. Shape: [batch, hidden] -> [batch, hidden]."""
        if hidden is None or self.latent_vae is None:
            return hidden

        # hidden shape: [batch, hidden] (e.g., [1, 2048])
        try:
            dist = self.latent_vae.forward(hidden, temperature=1.0)
            return dist.mean  # [batch, hidden]
        except Exception as e:
            _debug_log(f"[ERROR] VAE transform failed: {e}")
            return hidden

    # ============== Execute Model ==============
    # Wrap model call to inject inputs_embeds for continuous mode
    _orig_execute_model = GPUModelRunner.execute_model

    @functools.wraps(_orig_execute_model)
    def patched_execute_model(self, scheduler_output, intermediate_tensors=None, dummy_run=False):
        _debug_log(f"[execute_model] CALLED, scheduler_output={type(scheduler_output)}")
        if scheduler_output:
            _debug_log(f"[execute_model] new_reqs={len(getattr(scheduler_output, 'scheduled_new_reqs', []))}, cached={getattr(scheduler_output, 'scheduled_cached_reqs', 'N/A')}")
        if not self._vae_loaded:
            _load_latent_vae_from_checkpoint(self)
            self._vae_loaded = True
            _debug_log(f"[DEBUG] VAE loaded: {self.latent_vae is not None}")

        # Track new requests
        if hasattr(scheduler_output, 'scheduled_new_reqs') and scheduler_output.scheduled_new_reqs:
            for req in scheduler_output.scheduled_new_reqs:
                if req.req_id not in self._thinking_state:
                    # Initial mode detection from prompt
                    prompt_ids = req.prompt_token_ids or []
                    mode = _detect_initial_mode_from_prompt(prompt_ids)

                    max_tokens = getattr(req, 'max_tokens', 2048) or 2048
                    thinking_length = min(max_tokens // 2, DEFAULT_THINKING_LENGTH)

                    self._thinking_state[req.req_id] = {
                        'mode': mode,
                        'step': 0,
                        'hidden': None,
                        'embedding': None,
                        'thinking_length': thinking_length,
                        'max_thinking_length': thinking_length,  # Store max for reset on think_sep
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
                    self._thinking_state[req_id] = {'mode': 'discrete', 'step': 0, 'hidden': None, 'embedding': None}
                    _debug_log(f"[DEBUG] Cached req {req_id}: init as discrete")

        # Clean finished
        if hasattr(scheduler_output, 'finished_req_ids'):
            for req_id in scheduler_output.finished_req_ids:
                self._thinking_state.pop(req_id, None)
                _debug_log(f"[DEBUG] Finished req {req_id}: removed from state")

        _verify_continuous_decode_inputs(self, scheduler_output)
        return _orig_execute_model(self, scheduler_output, intermediate_tensors)

    # ============== Sample Tokens ==============
    # This is where hidden_states is available from execute_model_state
    _orig_sample_tokens = GPUModelRunner.sample_tokens

    @functools.wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output):
        _debug_log(f"[DEBUG] patched_sample_tokens CALLED")
        # Cache ephemeral outputs before calling original sample_tokens().
        # vLLM clears execute_model_state inside sample_tokens.
        hidden_states = None
        logits_for_trace = None
        if hasattr(self, 'execute_model_state') and self.execute_model_state is not None:
            hidden_states = self.execute_model_state.hidden_states
            logits_for_trace = self.execute_model_state.logits
            _debug_log(f"[DEBUG] sample_tokens: hidden_states shape={hidden_states.shape if hidden_states is not None else 'N/A'}")

        # Call original sample_tokens
        result = _orig_sample_tokens(self, grammar_output)

        # Get batch_req_ids to match with thinking state
        batch_req_ids = []
        if hasattr(self, 'input_batch') and hasattr(self.input_batch, 'req_ids'):
            batch_req_ids = self.input_batch.req_ids
        _debug_log(f"[sample_tokens] batch_req_ids={batch_req_ids}, result type={type(result)}")

        # Get sampled tokens from result - use sampled_token_ids instead of outputs
        sampled_tokens_list = []
        if batch_req_ids and hasattr(result, 'sampled_token_ids'):
            sampled_tokens_list = result.sampled_token_ids
            _debug_log(f"[sample_tokens] sampled_token_ids={sampled_tokens_list}")

        if batch_req_ids and sampled_tokens_list:
            for i, req_id in enumerate(batch_req_ids):
                if req_id not in self._thinking_state:
                    continue
                state = self._thinking_state[req_id]
                # sampled_token_ids is list of list, get first token of first (and only) generation
                sampled = sampled_tokens_list[i][0] if i < len(sampled_tokens_list) and sampled_tokens_list[i] else 0
                mode = state.get('mode', 'discrete')
                step = state.get('step', 0)

                _debug_log(f"[DEBUG] sample_tokens req={req_id}: mode={mode}, step={step}, sampled={sampled}")

                # Get hidden states output for preparing next input embedding
                last_hidden = None
                if hidden_states is not None:
                    # Extract the last token's hidden state for this request
                    # Ensure we have the right batch dimension
                    try:
                        if hidden_states.ndim == 3:
                            # [batch, seq, hidden] -> [1, hidden]
                            last_hidden = hidden_states[i:i+1, -1, :].detach().clone()
                        else:
                            # [batch, hidden] -> [1, hidden]
                            last_hidden = hidden_states[i:i+1, :].detach().clone()
                        # Ensure correct dtype and device
                        if last_hidden.dtype != torch.bfloat16:
                            last_hidden = last_hidden.to(dtype=torch.bfloat16)
                        state['step'] = step + 1
                        _debug_log(f"[DEBUG] sample_tokens: got hidden output shape={last_hidden.shape}, dtype={last_hidden.dtype}")
                    except Exception as e:
                        _debug_log(f"[ERROR] sample_tokens: failed to extract hidden: {e}")
                        continue

                # Optional targeted probability tracing:
                # compute exact logprobs/ranks for </think>, <latent>, <think_sep>
                # directly from logits, without requesting full vocab logprobs via vLLM API.
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
                if sampled == _get_token_id('think_start'):
                    state['mode'] = 'continuous'
                    state['step'] = 0  # Reset step when starting thinking
                    _debug_log(f"[DEBUG] sample_tokens: switched to continuous (think_start)")
                elif sampled == _get_token_id('think_end'):
                    min_steps = int(state.get('min_continuous_steps', _get_min_continuous_steps()))
                    if mode == 'continuous' and step < min_steps:
                        # Ignore early </think> exit signal until minimum continuous steps is reached.
                        _debug_log(
                            f"[DEBUG] sample_tokens: ignore early think_end req={req_id} step={step} min_steps={min_steps}"
                        )
                    else:
                        state['mode'] = 'discrete'
                        state['embedding'] = None
                        _debug_log(f"[DEBUG] sample_tokens: switched to discrete (think_end)")
                elif sampled == _get_token_id('think_sep'):
                    # Reset thinking length to limit when think_sep is encountered
                    state['thinking_length'] = state.get('max_thinking_length', DEFAULT_THINKING_LENGTH)
                    state['step'] = 0  # Reset step counter
                    _debug_log(f"[DEBUG] sample_tokens: reset thinking length to {state['thinking_length']} (think_sep)")

                # Check step limit - if reached, force think_end token
                thinking_length = state.get('thinking_length', DEFAULT_THINKING_LENGTH)
                if state.get('mode') == 'continuous' and step >= thinking_length:
                    # Force the think_end token by replacing the sampled token
                    if i < len(sampled_tokens_list) and sampled_tokens_list[i]:
                        sampled_tokens_list[i][0] = _get_token_id('think_end')
                        sampled = sampled_tokens_list[i][0]
                        req_pos = self.input_batch.num_tokens[ i ] - 1
                        if req_pos >= 0:
                            self.input_batch.token_ids_cpu[i, req_pos] = sampled
                            req_state = self.requests.get(req_id)
                            if req_state is not None and req_state.output_token_ids:
                                req_state.output_token_ids[-1] = sampled
                        _debug_log(f"[DEBUG] sample_tokens: forced think_end for req={req_id}")
                    # Switch to discrete mode
                    state['mode'] = 'discrete'
                    state['embedding'] = None
                    # Reset thinking_length to max for potential future thinking phases
                    state['thinking_length'] = state.get('max_thinking_length', DEFAULT_THINKING_LENGTH)
                    _debug_log(f"[DEBUG] sample_tokens: exited continuous (step limit)")
                else:
                    # Apply VAE transformation for continuous mode (only if not forcing think_end)
                    if state.get('mode') == 'continuous' and last_hidden is not None:
                        if self.latent_vae is not None:
                            # VAE: [1, hidden] -> latent distribution -> [1, hidden]
                            vae_dist = self.latent_vae.forward(last_hidden, temperature=1.0)
                            vae_emb = vae_dist.mean  # Use mean for deterministic forward
                            state['embedding'] = vae_emb
                            _debug_log(f"[DEBUG] sample_tokens: VAE embedding shape={vae_emb.shape}")
                        else:
                            # No VAE, use hidden directly as embedding
                            state['embedding'] = last_hidden

                token_pos = self.input_batch.num_tokens[i] - 1
                if state.get('mode') == 'continuous':
                    _set_next_decode_embedding(
                        self,
                        i,
                        token_pos,
                        state.get('embedding'),
                    )
                else:
                    _set_next_decode_embedding(self, i, token_pos, None)

        return result

    # Apply patches to GPUModelRunner (from gpu.model_runner)
    _debug_log("[PATCH] About to patch __init__")
    GPUModelRunner.__init__ = patched_init
    _debug_log("[PATCH] Patched __init__")
    _debug_log("[PATCH] About to patch execute_model")
    GPUModelRunner.execute_model = patched_execute_model
    _debug_log("[PATCH] Patched execute_model")
    _debug_log("[PATCH] About to patch sample_tokens")
    GPUModelRunner.sample_tokens = patched_sample_tokens
    _debug_log("[PATCH] Patched sample_tokens")
    GPUModelRunner._thinking_patch_applied = True
    _debug_log("[PATCH] All patches applied successfully!")

    _patch_applied = True
    logger.info("[Thinking] ✓ Continuous latent AR patch applied")
