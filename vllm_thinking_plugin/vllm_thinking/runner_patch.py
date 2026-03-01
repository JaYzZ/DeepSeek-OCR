"""
Thinking Mode Patch for vLLM - Continuous Latent AR
"""

import functools
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
_THINKING_DEBUG = os.environ.get("VLLM_THINKING_DEBUG", "1") == "1"

# Persistent debug log file
_DEBUG_LOG_FILE = "/tmp/vllm_thinking_debug.log"

def _debug_log(msg: str):
    """Write debug message to stderr - visible in worker output"""
    timestamp = time.strftime("%H:%M:%S.%f")[:-3]
    log_msg = f"[{timestamp}] [VLLM_THINK] {msg}"
    print(log_msg, flush=True, file=sys.stderr)

import time


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


# Token IDs
THINK_START_ID = 151667  # <think>
THINK_END_ID = 151668   # </think>

LATENT_TOKEN_ID = 151669  # <latent>
THINK_SEP_ID = 151670   # <think_sep>
DEFAULT_THINKING_LENGTH = 1024  # Will be min(max_tokens//2, 1024)

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

    # ============== Init ==============
    _orig_init = GPUModelRunner.__init__

    @functools.wraps(_orig_init)
    def patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        self._thinking_state = {}
        self._vae_loaded = False
        self.latent_vae = None

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
                    has_think = THINK_START_ID in prompt_ids
                    has_end = THINK_END_ID in prompt_ids
                    # If prompt has think_start without think_end, start in continuous mode
                    mode = 'continuous' if (has_think and not has_end) else 'discrete'

                    max_tokens = getattr(req, 'max_tokens', 2048) or 2048
                    thinking_length = min(max_tokens // 2, DEFAULT_THINKING_LENGTH)

                    self._thinking_state[req.req_id] = {
                        'mode': mode,
                        'step': 0,
                        'hidden': None,
                        'embedding': None,
                        'thinking_length': thinking_length,
                        'max_thinking_length': thinking_length,  # Store max for reset on think_sep
                    }
                    _debug_log(f"[DEBUG] New req {req.req_id}: mode={mode}, thinking_length={thinking_length}")

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

        # Link model to runner
        if not hasattr(self.model, '_model_runner'):
            self.model._model_runner = self
            # Wrap the model call to inject inputs_embeds
            _wrap_model_call(self)

        return _orig_execute_model(self, scheduler_output, intermediate_tensors)

    # ============== Wrap Model Call ==============
    def _wrap_model_call(self):
        """Wrap self.model() to inject inputs_embeds for continuous mode."""
        if hasattr(self.model, '_model_call_wrapped'):
            return

        orig_model = self.model.__class__.__call__

        @functools.wraps(orig_model)
        def patched_model_call(*args, **kwargs):
            # Check if we have continuous mode with embedding
            inputs_embeds = None
            if hasattr(self, '_thinking_state'):
                # Get batch info
                batch_size = 1
                batch_req_ids = []
                if hasattr(self, 'input_batch') and hasattr(self.input_batch, 'req_ids'):
                    batch_req_ids = self.input_batch.req_ids
                    batch_size = len(batch_req_ids)

                # Check if any request is in continuous mode
                any_continuous = any(
                    self._thinking_state.get(rid, {}).get('mode') == 'continuous'
                    for rid in batch_req_ids
                ) if batch_req_ids else False

                if any_continuous and batch_req_ids:
                    # Need to build inputs_embeds for ENTIRE batch
                    input_ids = kwargs.get('input_ids')
                    hidden_size = self.model.config.text_config.hidden_size

                    # Get token embeddings from model
                    token_embeds = None
                    embed_found = False
                    if input_ids is not None and hasattr(self.model, 'language_model'):
                        # Get embedding layer from language model
                        lm = self.model.language_model
                        if hasattr(lm, 'embed_tokens'):
                            token_embeds = lm.embed_tokens(input_ids)
                            embed_found = True
                        elif hasattr(lm, 'model') and hasattr(lm.model, 'embed_tokens'):
                            token_embeds = lm.model.embed_tokens(input_ids)
                            embed_found = True

                    _debug_log(f"[DEBUG] patched_model_call: embed_tokens found={embed_found}, token_embeds={token_embeds.shape if token_embeds is not None else None}")

                    inputs_embeds_list = []
                    for idx, rid in enumerate(batch_req_ids):
                        state = self._thinking_state.get(rid, {})
                        mode = state.get('mode', 'discrete')
                        if state.get('mode') == 'continuous' and state.get('embedding') is not None:
                            # Continuous mode: use VAE embedding
                            e = state['embedding']
                            if e.device != torch.device('cuda'):
                                e = e.cuda()
                            if e.dtype != torch.bfloat16:
                                e = e.to(dtype=torch.bfloat16)
                            inputs_embeds_list.append(e)
                            _debug_log(f"[DEBUG] patched_model_call: req={rid} idx={idx} using VAE emb")
                        elif token_embeds is not None:
                            # Discrete mode: use token embedding (last token)
                            inputs_embeds_list.append(token_embeds[idx, -1, :].unsqueeze(0))
                            _debug_log(f"[DEBUG] patched_model_call: req={rid} idx={idx} using token emb")
                        else:
                            # Fallback: cannot handle
                            inputs_embeds_list = None
                            _debug_log(f"[DEBUG] patched_model_call: req={rid} idx={idx} FAILED")
                            break

                    if inputs_embeds_list is not None:
                        # Each element is [1, hidden], stack gives [batch, 1, hidden]
                        inputs_embeds = torch.stack(inputs_embeds_list)
                        _debug_log(f"[DEBUG] patched_model_call: mixed batch embeddings shape={inputs_embeds.shape}")
                    else:
                        _debug_log(f"[DEBUG] patched_model_call: cannot get token embeddings, falling back to discrete")

            if inputs_embeds is not None:
                kwargs['inputs_embeds'] = inputs_embeds
                # Qwen3VL forward requires input_ids as first positional arg, but ignores it when inputs_embeds is provided
                # We need to replace input_ids with a dummy tensor that has the right shape
                if 'input_ids' in kwargs:
                    # Get batch size from inputs_embeds
                    batch_size = inputs_embeds.shape[0]
                    # Create dummy input_ids with the same batch size (1 token per sample)
                    dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=inputs_embeds.device)
                    kwargs['input_ids'] = dummy_input_ids

            return orig_model(*args, **kwargs)

        self.model.__class__.__call__ = patched_model_call
        self.model._model_call_wrapped = True
        _debug_log("[DEBUG] Wrapped model call for VAE injection")

    # ============== Sample Tokens ==============
    # This is where hidden_states is available from execute_model_state
    _orig_sample_tokens = GPUModelRunner.sample_tokens

    @functools.wraps(_orig_sample_tokens)
    def patched_sample_tokens(self, grammar_output):
        _debug_log(f"[DEBUG] patched_sample_tokens CALLED")
        # Get hidden_states from execute_model_state
        hidden_states = None
        if hasattr(self, 'execute_model_state') and self.execute_model_state is not None:
            hidden_states = self.execute_model_state.hidden_states
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

                # Monitor sampled token for mode switching
                if sampled == THINK_START_ID:
                    state['mode'] = 'continuous'
                    state['step'] = 0  # Reset step when starting thinking
                    _debug_log(f"[DEBUG] sample_tokens: switched to continuous (think_start)")
                elif sampled == THINK_END_ID:
                    state['mode'] = 'discrete'
                    state['embedding'] = None
                    _debug_log(f"[DEBUG] sample_tokens: switched to discrete (think_end)")
                elif sampled == THINK_SEP_ID:
                    # Reset thinking length to limit when think_sep is encountered
                    state['thinking_length'] = state.get('max_thinking_length', DEFAULT_THINKING_LENGTH)
                    state['step'] = 0  # Reset step counter
                    _debug_log(f"[DEBUG] sample_tokens: reset thinking length to {state['thinking_length']} (think_sep)")

                # Check step limit - if reached, force think_end token
                thinking_length = state.get('thinking_length', DEFAULT_THINKING_LENGTH)
                if state.get('mode') == 'continuous' and step >= thinking_length:
                    # Force the think_end token by replacing the sampled token
                    if i < len(sampled_tokens_list) and sampled_tokens_list[i]:
                        sampled_tokens_list[i][0] = THINK_END_ID
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
