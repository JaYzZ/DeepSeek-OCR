"""
Local copy of vLLM's NGramPerReqLogitsProcessor with is_argmax_invariant fix.
This ensures compatibility with vLLM's greedy sampling optimization.
"""

import torch
from typing import Iterable
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class NoRepeatNGramLogitsProcessor:
    def __init__(
        self,
        ngram_size: int,
        window_size: int,
        whitelist_token_ids: set[int] | None = None,
    ):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if len(output_ids) < self.ngram_size:
            return logits

        current_prefix = tuple(output_ids[-(self.ngram_size - 1) :])

        search_start = max(0, len(output_ids) - self.window_size)
        search_end = len(output_ids) - self.ngram_size + 1

        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(output_ids[i : i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        banned_tokens = banned_tokens - self.whitelist_token_ids

        if banned_tokens:
            logits[list(banned_tokens)] = -float("inf")

        return logits


class NGramPerReqLogitsProcessor(AdapterLogitsProcessor):
    """N-gram logits processor with is_argmax_invariant returning True.

    This is a local copy from vLLM's deepseek_ocr model with the fix for
    is_argmax_invariant to ensure proper greedy sampling behavior.
    """

    @classmethod
    def validate_params(cls, params: SamplingParams):
        ngram_size = params.extra_args and params.extra_args.get("ngram_size")
        window_size = params.extra_args and params.extra_args.get("window_size", 100)
        whitelist_token_ids = params.extra_args and params.extra_args.get(
            "whitelist_token_ids", None
        )
        # if ngram_size is not provided, skip validation because the processor
        # will not be used.
        if ngram_size is None:
            return None

        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(
                f"`ngram_size` has to be a strictly positive integer, got {ngram_size}."
            )
        if not isinstance(window_size, int) or window_size <= 0:
            raise ValueError(
                "`window_size` has to be a strictly positive integer, "
                f"got {window_size}."
            )
        if whitelist_token_ids is not None and not isinstance(
            whitelist_token_ids, Iterable
        ):
            raise ValueError(
                "`whitelist_token_ids` has to be a sequence of integers, "
                f"got {whitelist_token_ids}."
            )

    def is_argmax_invariant(self) -> bool:
        # CRITICAL FIX: Return True to enable greedy sampling optimization
        # Nov 17 vLLM has this returning False which causes garbage output
        return True

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        ngram_size = params.extra_args and params.extra_args.get("ngram_size")
        window_size = params.extra_args and params.extra_args.get("window_size", 100)
        whitelist_token_ids = params.extra_args and params.extra_args.get(
            "whitelist_token_ids", None
        )
        if ngram_size is None:
            return None

        whitelist_token_ids = set(whitelist_token_ids) if whitelist_token_ids else None
        return NoRepeatNGramLogitsProcessor(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids=whitelist_token_ids,
        )
