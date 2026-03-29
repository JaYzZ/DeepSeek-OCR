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

from vllm_thinking.trace_store import record_request_step, reset_request_trace
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


logger = logging.getLogger(__name__)
_THINKING_DEBUG = os.environ.get("VLLM_THINKING_DEBUG", "0") == "1"
_TARGET_PROB_IO_WARNED = False
_VLLM_VAE_LOGGED = False
_CONTINUOUS_AR_CERT_LOGGED = False

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
        global _VLLM_VAE_LOGGED
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
                    if not _VLLM_VAE_LOGGED:
                        logger.warning("[Thinking] vLLM rollout VAE loaded: path=%s hidden_size=%s", vae_path, hidden_size)
                        _VLLM_VAE_LOGGED = True
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
                if not _VLLM_VAE_LOGGED:
                    logger.warning("[Thinking] vLLM rollout VAE loaded: path=%s hidden_size=%s", vae_path, hidden_size)
                    _VLLM_VAE_LOGGED = True
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
                    thinking_length = min(max_tokens // 2, DEFAULT_THINKING_LENGTH)

                    self._thinking_state[req.req_id] = {
                        'mode': mode,
                        'step': 0,
                        'embedding': None,
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
        # vLLM's async scheduling fast path can overwrite GPU-side `is_token_ids`
        # and force decode inputs back to token IDs. If any scheduled position is
        # marked as a prompt/embed input, bypass that optimization so the mixed
        # token/embed batch reaches `_preprocess` unchanged.
        if self.enable_prompt_embeds:
            req_ids = getattr(self.input_batch, "req_ids", ())
            num_computed_tokens = self.input_batch.num_computed_tokens_cpu
            is_token_ids = self.input_batch.is_token_ids
            num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", {})

            needs_embed_upload = False
            for req_idx, req_id in enumerate(req_ids):
                num_sched = int(num_scheduled_tokens.get(req_id, 0))
                if num_sched <= 0:
                    continue

                start = int(num_computed_tokens[req_idx])
                end = start + num_sched
                if end > start and not bool(is_token_ids[req_idx, start:end].all()):
                    needs_embed_upload = True
                    break

            if needs_embed_upload:
                self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
                self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
                self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
                return

        _orig_prepare_input_ids(
            self,
            scheduler_output,
            total_num_scheduled_tokens,
            cu_num_tokens,
        )
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
                num_tokens = self.input_batch.num_tokens
                flat_idx = 0
                for req_idx, req_id in enumerate(req_ids):
                    state = self._thinking_state.get(req_id)
                    num_sched = int(num_tokens[req_idx] - num_computed[req_idx])
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
        if hasattr(self, 'execute_model_state') and self.execute_model_state is not None:
            hidden_states = self.execute_model_state.hidden_states
            sample_hidden_states = self.execute_model_state.sample_hidden_states
            logits_for_trace = self.execute_model_state.logits
            # Try to extract attention weights if available
            attention_weights = getattr(self.execute_model_state, 'attention_weights', None)

        # Call original sample_tokens
        result = _orig_sample_tokens(self, grammar_output)

        # Get batch_req_ids to match with thinking state
        batch_req_ids = []
        if hasattr(self, 'input_batch') and hasattr(self.input_batch, 'req_ids'):
            batch_req_ids = self.input_batch.req_ids

        # Get sampled tokens from result - use sampled_token_ids instead of outputs
        sampled_tokens_list = []
        if batch_req_ids and hasattr(result, 'sampled_token_ids'):
            sampled_tokens_list = result.sampled_token_ids

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
                if sampled == _get_token_id('think_start'):
                    state['mode'] = 'continuous'
                    state['step'] = 0
                    _debug_log(f"[DEBUG] sample_tokens: switched to continuous (think_start)")
                elif sampled == _get_token_id('think_end'):
                    min_steps = int(state.get('min_continuous_steps', _get_min_continuous_steps()))
                    if mode == 'continuous' and step < min_steps:
                        _debug_log(
                            f"[DEBUG] sample_tokens: ignore early think_end req={req_id} step={step} min_steps={min_steps}"
                        )
                    else:
                        state['mode'] = 'discrete'
                        state['embedding'] = None
                        _debug_log(f"[DEBUG] sample_tokens: switched to discrete (think_end)")
                elif sampled == _get_token_id('think_sep'):
                    state['thinking_length'] = state.get('max_thinking_length', DEFAULT_THINKING_LENGTH)
                    state['step'] = 0
                    _debug_log(f"[DEBUG] sample_tokens: reset thinking length to {state['thinking_length']} (think_sep)")

                thinking_length = state.get('thinking_length', DEFAULT_THINKING_LENGTH)
                latent_embedding = None
                latent_logprob = None
                if state.get('mode') == 'continuous' and step >= thinking_length:
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
                    state['mode'] = 'discrete'
                    state['embedding'] = None
                    state['thinking_length'] = state.get('max_thinking_length', DEFAULT_THINKING_LENGTH)
                    _debug_log(f"[DEBUG] sample_tokens: exited continuous (step limit)")
                else:
                    if state.get('mode') == 'continuous' and last_hidden is not None:
                        if self.latent_vae is not None:
                            vae_dist = self.latent_vae.forward(last_hidden, temperature=1.0)
                            vae_emb = vae_dist.rsample()
                            latent_embedding = vae_emb
                            latent_logprob = vae_dist.log_prob(vae_emb).mean(dim=-1)
                            state['embedding'] = vae_emb
                        else:
                            state['embedding'] = last_hidden

                use_continuous_embedding = bool(state.get('mode') == 'continuous' and last_hidden is not None)
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
                # Prepare all hidden states and token IDs for visualization
                all_hidden_for_step = None
                all_token_ids_for_step = []

                if hidden_states is not None and i < hidden_states.shape[0]:
                    # Extract all hidden states for this request from the batch
                    all_hidden_for_step = hidden_states[i:i+1]  # Keep batch dim

                # Collect token IDs for this step
                if sampled is not None:
                    all_token_ids_for_step = [int(sampled)]

                record_request_step(
                    req_id,
                    hidden_state=last_hidden,
                    latent_embedding=latent_embedding,
                    latent_logprob=latent_logprob,
                    use_continuous_embedding=use_continuous_embedding,
                    all_hidden_states=all_hidden_for_step,
                    all_token_ids=all_token_ids_for_step,
                    attention_weights=attention_weights,
                )

                # vLLM's next decode step reads slot `num_computed_tokens`, which
                # after sampling corresponds to the just-added output token slot:
                # `num_tokens - 1`. Override that slot so the next forward consumes
                # the hidden-state-derived embedding instead of the sampled token ID.
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
