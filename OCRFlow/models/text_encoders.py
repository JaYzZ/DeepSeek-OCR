"""
Text Encoders for OCRFlow

Implements dual text encoder setup (T5 + CLIP) similar to UniFlow/Flux.
Provides both sequence-level and pooled text embeddings for conditioning MMDiT.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoConfig,
    T5EncoderModel,
    CLIPTextModel,
    T5Config,
    T5PreTrainedModel
)
from transformers.modeling_outputs import BaseModelOutput
from typing import Tuple, Optional, Dict


class T5EncoderWithProjection(T5PreTrainedModel):
    """
    T5 encoder with additional projection layer for dimension matching.

    This allows using smaller T5 models (e.g., T5-base with 768 dims)
    while projecting to larger dimensions (e.g., 4096 like T5-XXL).
    """

    def __init__(self, config):
        super().__init__(config)
        self.encoder = T5EncoderModel(config)

        # Projection layer (if needed)
        self.project_out_dim = getattr(config, 'project_out_dim', None)
        if self.project_out_dim and self.project_out_dim != config.d_model:
            self.final_projection = nn.Sequential(
                nn.Linear(config.d_model, self.project_out_dim, bias=False),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.project_out_dim, self.project_out_dim, bias=False),
            )
        else:
            self.final_projection = None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> BaseModelOutput:
        """Forward pass through T5 encoder."""
        enc_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        last_hidden = enc_out.last_hidden_state

        # Apply projection if exists
        if self.final_projection is not None:
            last_hidden = self.final_projection(last_hidden)

        return BaseModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=enc_out.hidden_states,
            attentions=enc_out.attentions,
        )


def load_t5_encoder(
    model_path: str = "google/t5-v1_1-base",
    max_length: int = 512,
    project_dim: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
) -> Tuple[AutoTokenizer, nn.Module]:
    """
    Load T5 encoder with optional projection.

    Args:
        model_path: Path to T5 model
        max_length: Maximum sequence length
        project_dim: If specified, project to this dimension
        device: Device to load on
        dtype: Model dtype
        freeze: Whether to freeze parameters

    Returns:
        (tokenizer, model) tuple
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=max_length)

    # Load config and modify if projection needed
    config = T5Config.from_pretrained(model_path)
    if project_dim:
        config.project_out_dim = project_dim

    # Load model
    if project_dim:
        model = T5EncoderWithProjection.from_pretrained(
            model_path,
            config=config,
            dtype=dtype,
        )
    else:
        model = T5EncoderModel.from_pretrained(
            model_path,
            dtype=dtype,
        )

    model = model.to(device)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return tokenizer, model


def load_clip_encoder(
    model_path: str = "openai/clip-vit-large-patch14",
    max_length: int = 77,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
) -> Tuple[AutoTokenizer, CLIPTextModel]:
    """
    Load CLIP text encoder.

    Args:
        model_path: Path to CLIP model
        max_length: Maximum sequence length
        device: Device to load on
        dtype: Model dtype
        freeze: Whether to freeze parameters

    Returns:
        (tokenizer, model) tuple
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    try:
        # Try loading as SigLIP first
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if 'siglip' in config.model_type.lower():
            from transformers.models.siglip2.modeling_siglip2 import Siglip2TextModel
            model = Siglip2TextModel.from_pretrained(
                model_path,
                dtype=dtype,
            )
        else:
            model = CLIPTextModel.from_pretrained(
                model_path,
                dtype=dtype,
            )
    except:
        # Fallback to CLIP
        model = CLIPTextModel.from_pretrained(
            model_path,
            dtype=dtype,
        )

    model = model.to(device)

    if freeze:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    return tokenizer, model


def load_text_encoders(
    t5_model_path: str = "google/t5-v1_1-base",
    clip_model_path: str = "openai/clip-vit-large-patch14",
    t5_max_length: int = 512,
    clip_max_length: int = 77,
    t5_project_dim: Optional[int] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
) -> Dict[str, any]:
    """
    Load dual text encoders (T5 + CLIP).

    Args:
        t5_model_path: Path to T5 model
        clip_model_path: Path to CLIP model
        t5_max_length: Max sequence length for T5
        clip_max_length: Max sequence length for CLIP
        t5_project_dim: Optional projection dimension for T5
        device: Device to load on
        dtype: Model dtype
        freeze: Whether to freeze parameters

    Returns:
        Dictionary with keys:
            - 't5_tokenizer': T5 tokenizer
            - 't5_model': T5 encoder
            - 'clip_tokenizer': CLIP tokenizer
            - 'clip_model': CLIP encoder
            - 't5_dim': T5 output dimension
            - 'clip_dim': CLIP pooled dimension
    """
    # Load T5
    t5_tokenizer, t5_model = load_t5_encoder(
        model_path=t5_model_path,
        max_length=t5_max_length,
        project_dim=t5_project_dim,
        device=device,
        dtype=dtype,
        freeze=freeze,
    )

    # Load CLIP
    clip_tokenizer, clip_model = load_clip_encoder(
        model_path=clip_model_path,
        max_length=clip_max_length,
        device=device,
        dtype=dtype,
        freeze=freeze,
    )

    # Get output dimensions
    t5_dim = t5_project_dim if t5_project_dim else t5_model.config.d_model
    clip_dim = clip_model.config.hidden_size

    return {
        't5_tokenizer': t5_tokenizer,
        't5_model': t5_model,
        'clip_tokenizer': clip_tokenizer,
        'clip_model': clip_model,
        't5_dim': t5_dim,
        'clip_dim': clip_dim,
    }


@torch.no_grad()
def encode_text_dual(
    text: list,
    t5_tokenizer: AutoTokenizer,
    t5_model: nn.Module,
    clip_tokenizer: AutoTokenizer,
    clip_model: nn.Module,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode text using dual encoders (T5 + CLIP).

    Args:
        text: List of text strings
        t5_tokenizer: T5 tokenizer
        t5_model: T5 encoder
        clip_tokenizer: CLIP tokenizer
        clip_model: CLIP encoder
        device: Device for tensors
        dtype: Tensor dtype

    Returns:
        Tuple of (t5_embeds, clip_embeds, t5_attention_mask)
            - t5_embeds: [B, seq_len, t5_dim] sequence embeddings
            - clip_embeds: [B, clip_dim] pooled embeddings
            - t5_attention_mask: [B, seq_len] attention mask for T5
    """
    # Encode with T5
    t5_inputs = t5_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=t5_tokenizer.model_max_length,
    ).to(device)

    t5_outputs = t5_model(
        input_ids=t5_inputs.input_ids,
        attention_mask=t5_inputs.attention_mask,
    )
    t5_embeds = t5_outputs.last_hidden_state.to(dtype)
    t5_mask = t5_inputs.attention_mask

    # Encode with CLIP (get pooled output)
    clip_inputs = clip_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    ).to(device)

    clip_outputs = clip_model(**clip_inputs)
    clip_embeds = clip_outputs.pooler_output.to(dtype)  # [B, clip_dim]

    return t5_embeds, clip_embeds, t5_mask


if __name__ == "__main__":
    # Test text encoders
    print("Testing text encoders...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load encoders
    encoders = load_text_encoders(
        t5_model_path="google/t5-v1_1-base",
        clip_model_path="openai/clip-vit-large-patch14",
        device=device,
        freeze=True,
    )

    print(f"T5 dimension: {encoders['t5_dim']}")
    print(f"CLIP dimension: {encoders['clip_dim']}")

    # Test encoding
    test_texts = [
        "<image>\n<|grounding|>Convert the document to markdown.",
        "<image>\n<|grounding|>OCR this image.",
    ]

    t5_embeds, clip_embeds, t5_mask = encode_text_dual(
        text=test_texts,
        **{k: v for k, v in encoders.items() if 'tokenizer' in k or 'model' in k},
        device=device,
    )

    print(f"\nT5 embeddings shape: {t5_embeds.shape}")
    print(f"CLIP embeddings shape: {clip_embeds.shape}")
    print(f"T5 mask shape: {t5_mask.shape}")
    print("\nTest passed!")
