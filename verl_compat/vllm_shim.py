"""Repo-local shim for `verl.utils.vllm` compatibility."""

from __future__ import annotations

import inspect
import logging

from msgspec import field
from packaging import version as vs
from vllm.lora.models import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager, WorkerLoRAManager

from verl.third_party.vllm import get_version

logger = logging.getLogger(__name__)


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


def _build_load_adapter():
    def compat_load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
        """Load LoRA adapters from either a path or in-memory tensors."""

        supported_lora_modules = self._adapter_manager.supported_lora_modules
        packed_modules_mapping = self._adapter_manager.packed_modules_mapping
        expected_lora_modules: list[str] = []
        for module in supported_lora_modules:
            if module in packed_modules_mapping:
                expected_lora_modules.extend(packed_modules_mapping[module])
            else:
                expected_lora_modules.append(module)
            if module == "experts":
                expected_lora_modules.append(module)
        expected_lora_modules = list(set(expected_lora_modules))

        model = self._adapter_manager.model
        hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
        extra_vocab_size = getattr(self.lora_config, "lora_extra_vocab_size", 0)

        if isinstance(lora_request, TensorLoRARequest):
            peft_helper = PEFTHelper.from_dict(lora_request.peft_config)
            peft_helper.validate_legal(self.lora_config)
            lora = self._lora_model_cls.from_lora_tensors(
                lora_model_id=lora_request.lora_int_id,
                tensors=lora_request.lora_tensors,
                peft_helper=peft_helper,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.vocab_size + extra_vocab_size,
                embedding_modules=self.embedding_modules,
                embedding_padding_modules=self.embedding_padding_modules,
                weights_mapper=hf_to_vllm_mapper,
            )
        else:
            lora_path = get_adapter_absolute_path(lora_request.lora_path)
            peft_helper = PEFTHelper.from_local_dir(
                lora_path,
                self.max_position_embeddings,
                lora_request.tensorizer_config_dict,
            )
            peft_helper.validate_legal(self.lora_config)
            lora = self._lora_model_cls.from_local_checkpoint(
                lora_path,
                expected_lora_modules,
                peft_helper=peft_helper,
                lora_model_id=lora_request.lora_int_id,
                device="cpu",
                dtype=self.lora_config.lora_dtype,
                target_embedding_padding=self.vocab_size + extra_vocab_size,
                embedding_modules=self.embedding_modules,
                embedding_padding_modules=self.embedding_padding_modules,
                tensorizer_config_dict=lora_request.tensorizer_config_dict,
                weights_mapper=hf_to_vllm_mapper,
            )

        if not hasattr(lora, "extra_vocab_size"):
            lora.extra_vocab_size = 0
        if lora.extra_vocab_size > extra_vocab_size:
            raise ValueError(
                f"LoRA added vocab size {lora.extra_vocab_size} is greater than "
                f"lora_extra_vocab_size {extra_vocab_size}."
            )
        return lora

    compat_load_adapter.__name__ = "compat_load_adapter"
    compat_load_adapter._qwen3vl_compat_patch = True
    return compat_load_adapter


class VLLMHijack:
    @staticmethod
    def hijack() -> None:
        patched = False
        original_from_lora_tensors = LoRAModel.from_lora_tensors.__func__

        if "embeddings" not in inspect.signature(original_from_lora_tensors).parameters:

            def compat_from_lora_tensors(cls, *args, embeddings=None, **kwargs):
                lora = original_from_lora_tensors(cls, *args, **kwargs)
                if not hasattr(lora, "extra_vocab_size"):
                    lora.extra_vocab_size = 0
                return lora

            compat_from_lora_tensors._qwen3vl_compat_patch = True
            LoRAModel.from_lora_tensors = classmethod(compat_from_lora_tensors)
            patched = True

        compat_load_adapter = _build_load_adapter()
        for cls in (WorkerLoRAManager, LRUCacheWorkerLoRAManager):
            current = getattr(cls, "_load_adapter", None)
            if not getattr(current, "_qwen3vl_compat_patch", False):
                setattr(cls, "_load_adapter", compat_load_adapter)
                patched = True

        if patched:
            logger.warning("Installed repo-local vLLM LoRA compatibility patch.")


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3") -> bool:
    """Check if the package version is greater than or equal to the minimum."""

    return vs.parse(get_version(pkg)) >= vs.parse(minver)
