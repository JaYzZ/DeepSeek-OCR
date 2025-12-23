"""OCR-aligned Llava Mistral wrappers

Mirrors llava/model/language_model/llava_mistral.py but uses the
OCRLlavaMetaForCausalLM base so that prepare_inputs_labels_for_multimodal
understands OCR-encoded image-token blocks and can accept pre-encoded OCR
visual features projected to LLM space.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, MistralConfig, MistralModel, MistralForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from OCRVL.model.ocr_llava_arch import OCRLlavaMetaForCausalLM
from llava.model.language_model.llava_mistral import LlavaMistralConfig
from llava.model.llava_arch import LlavaMetaModel


class OCRLlavaMistralModel(LlavaMetaModel, MistralModel):
    # Keep upstream config for compatibility
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(OCRLlavaMistralModel, self).__init__(config)


class OCRLlavaMistralForCausalLM(MistralForCausalLM, OCRLlavaMetaForCausalLM):
    # Override mapping for upstream LlavaMistralConfig so checkpoints load OCR variant
    config_class = LlavaMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = OCRLlavaMistralModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        ocr_image_features: Optional[torch.Tensor] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                ocr_image_features=ocr_image_features,
                vision_scale=vision_scale,
                text_scale=text_scale,
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        ocr_image_features: Optional[torch.Tensor] = None,
        vision_scale: Optional[float] = None,
        text_scale: Optional[float] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or ocr_image_features is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                ocr_image_features=ocr_image_features,
                vision_scale=vision_scale,
                text_scale=text_scale,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        ocr_image_features = kwargs.pop("ocr_image_features", None)
        vision_scale = kwargs.pop("vision_scale", None)
        text_scale = kwargs.pop("text_scale", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        if ocr_image_features is not None:
            inputs['ocr_image_features'] = ocr_image_features
        if vision_scale is not None:
            inputs['vision_scale'] = vision_scale
        if text_scale is not None:
            inputs['text_scale'] = text_scale
        return inputs

# Override AutoModel mapping for upstream Mistral config
AutoModelForCausalLM.register(LlavaMistralConfig, OCRLlavaMistralForCausalLM)

