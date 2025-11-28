"""
Visual Token to Text Decoder using DeepSeek-OCR (MOE 3B)

Utility for transcribing visual tokens back to text using the
DeepSeek-OCR language model decoder.

The DeepSeek-OCR model consists of:
- Vision Encoder: CLIP + SAM → Projector → Visual Tokens [111, 1280]
- Language Decoder: DeepSeek MOE 3B (2B active params)

This module provides a pure decoder interface for vistok → text transcription.

Usage:
    from OCRFlow.utils.vistok_decoder import VistokDecoder

    decoder = VistokDecoder(device="cuda")
    text = decoder.decode(visual_tokens)  # [111, 1280] → text
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class VistokDecoder:
    """
    Decoder for converting visual tokens back to text using DeepSeek-OCR.

    Loads only the LLM decoder portion of DeepSeek-OCR for efficient inference.
    """

    def __init__(
        self,
        model_path: str = "deepseek-ai/DeepSeek-OCR",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 2048,
    ):
        """
        Initialize the vistok decoder.

        Args:
            model_path: HuggingFace model path for DeepSeek-OCR
            device: Device to load model on
            dtype: Data type for model weights
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens

        self.model = None
        self.tokenizer = None
        self._initialized = False

    def _load_model(self):
        """Lazily load the model"""
        if self._initialized:
            return

        logger.info(f"Loading DeepSeek-OCR decoder from {self.model_path}...")

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # Load full model (we need the LLM portion)
        # TODO: Optimize to load only LLM weights without vision encoder
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            device_map=self.device,
        )
        self.model.eval()

        self._initialized = True

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Decoder loaded: {total_params / 1e9:.2f}B parameters")

    @torch.no_grad()
    def decode(
        self,
        visual_tokens: torch.Tensor,
        prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Decode visual tokens to text.

        Args:
            visual_tokens: Visual token embeddings [111, 1280] or [batch, 111, 1280]
            prompt: Optional text prompt to prepend
            max_new_tokens: Override default max tokens
            temperature: Sampling temperature (0 = greedy)
            top_p: Nucleus sampling parameter

        Returns:
            Decoded text string (or list of strings for batch)
        """
        self._load_model()

        if visual_tokens.dim() == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        batch_size = visual_tokens.shape[0]
        visual_tokens = visual_tokens.to(device=self.device, dtype=self.dtype)

        # Build input with visual tokens
        # DeepSeek-OCR uses <image_placeholder> token replaced by visual embeddings
        if prompt:
            input_text = f"<image_placeholder>{prompt}"
        else:
            input_text = "<image_placeholder>"

        # Tokenize (will be modified to inject visual tokens)
        inputs = self.tokenizer(
            [input_text] * batch_size,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate
        max_tokens = max_new_tokens or self.max_new_tokens

        outputs = self.model.generate(
            **inputs,
            images=visual_tokens,  # Visual tokens injected here
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p if temperature > 0 else None,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode outputs
        input_len = inputs.input_ids.shape[1]
        generated = outputs[:, input_len:]
        texts = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

        if single_input:
            return texts[0]
        return texts

    @torch.no_grad()
    def decode_batch(
        self,
        visual_tokens_list: List[torch.Tensor],
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Decode a batch of visual tokens.

        Args:
            visual_tokens_list: List of [111, 1280] tensors
            prompts: Optional list of prompts (same length as visual_tokens_list)
            **kwargs: Additional arguments passed to decode()

        Returns:
            List of decoded text strings
        """
        # Stack into batch
        visual_tokens = torch.stack(visual_tokens_list, dim=0)

        if prompts:
            # Process with individual prompts (slower)
            results = []
            for vt, prompt in zip(visual_tokens_list, prompts):
                text = self.decode(vt, prompt=prompt, **kwargs)
                results.append(text)
            return results
        else:
            return self.decode(visual_tokens, **kwargs)


def create_vistok_decoder(
    model_path: str = "deepseek-ai/DeepSeek-OCR",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> VistokDecoder:
    """
    Create a visual token decoder.

    Args:
        model_path: HuggingFace model path
        device: Device to load on
        dtype: Data type for weights

    Returns:
        VistokDecoder instance
    """
    return VistokDecoder(
        model_path=model_path,
        device=device,
        dtype=dtype,
    )


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print("Testing VistokDecoder...")

    # Create decoder
    decoder = create_vistok_decoder(device="cuda")

    # Test with random visual tokens (would normally come from encoder)
    visual_tokens = torch.randn(111, 1280)

    print("Decoding visual tokens...")
    text = decoder.decode(visual_tokens, prompt="Transcribe the text in this image:")

    print(f"Decoded text: {text[:200]}...")
    print("\nTest complete!")
