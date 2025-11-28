"""
Markovian Visual Chunk Decoder (Chunk-to-Chunk Generation)

Autoregressive generation at the CHUNK level, not token level:
- One chunk = 111 visual tokens = ~1000-1200 text tokens = One "thought"
- Given chunks [C1, C2, ..., CN], predict next chunk CN+1
- Markovian thinking: thought1 → thought2 → thought3 → ... → conclusion

Architecture:
1. Encode each chunk (111 tokens → summary vector)
2. Autoregressive transformer on chunk sequence
3. Decode to next chunk (summary → 111 tokens)

Training:
- Input: [C1, C2, ..., CN-1]
- Target: [C2, C3, ..., CN]
- Loss: MSE on predicted chunks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ChunkEncoder(nn.Module):
    """
    Encode a visual chunk (111 tokens) into a summary vector

    Args:
        token_dim: 1280 (visual token dimension)
        hidden_size: Encoder hidden size
        num_layers: Number of transformer layers
        summary_dim: Output summary dimension
    """

    def __init__(
        self,
        token_dim: int = 1280,
        hidden_size: int = 768,
        num_layers: int = 4,
        summary_dim: int = 1024,
        num_heads: int = 8,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.summary_dim = summary_dim

        # Project visual tokens to hidden
        self.input_proj = nn.Linear(token_dim, hidden_size)

        # Learned positional embeddings (111 positions)
        self.pos_embed = nn.Parameter(torch.randn(1, 111, hidden_size) * 0.02)

        # Transformer encoder layers (bidirectional is fine here)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Project to summary vector (mean pool + linear)
        self.summary_proj = nn.Linear(hidden_size, summary_dim)

    def forward(self, chunk: torch.Tensor) -> torch.Tensor:
        """
        Encode visual chunk to summary vector

        Args:
            chunk: [batch, 111, 1280] one visual chunk

        Returns:
            summary: [batch, summary_dim] chunk summary
        """
        # Project and add position
        x = self.input_proj(chunk)  # [B, 111, hidden]
        x = x + self.pos_embed

        # Encode
        x = self.transformer(x)  # [B, 111, hidden]

        # Pool to summary (mean over sequence)
        x = x.mean(dim=1)  # [B, hidden]

        # Project to summary dim
        summary = self.summary_proj(x)  # [B, summary_dim]

        return summary


class ChunkDecoder(nn.Module):
    """
    Decode summary vector back to visual chunk (111 tokens)

    Args:
        summary_dim: Input summary dimension
        token_dim: 1280 (visual token dimension)
        hidden_size: Decoder hidden size
        num_layers: Number of transformer layers
    """

    def __init__(
        self,
        summary_dim: int = 1024,
        token_dim: int = 1280,
        hidden_size: int = 768,
        num_layers: int = 4,
        num_heads: int = 8,
    ):
        super().__init__()

        self.summary_dim = summary_dim
        self.token_dim = token_dim

        # Project summary to hidden
        self.summary_proj = nn.Linear(summary_dim, hidden_size)

        # Learned token queries (111 positions)
        self.token_queries = nn.Parameter(torch.randn(1, 111, hidden_size) * 0.02)

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, 111, hidden_size) * 0.02)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)

        # Project to token dimension
        self.output_proj = nn.Linear(hidden_size, token_dim)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        """
        Decode summary to visual chunk

        Args:
            summary: [batch, summary_dim] chunk summary

        Returns:
            chunk: [batch, 111, 1280] reconstructed chunk
        """
        batch_size = summary.shape[0]

        # Project summary to hidden
        memory = self.summary_proj(summary).unsqueeze(1)  # [B, 1, hidden]

        # Expand token queries
        tgt = self.token_queries.expand(batch_size, -1, -1)  # [B, 111, hidden]
        tgt = tgt + self.pos_embed

        # Decode
        x = self.transformer(tgt, memory)  # [B, 111, hidden]

        # Project to tokens
        chunk = self.output_proj(x)  # [B, 111, 1280]

        return chunk


class CausalChunkAttention(nn.Module):
    """Causal self-attention on chunk sequence"""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_chunks, hidden_size]
        """
        B, N, C = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal attention
        attn = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout.p if self.training else 0.0,
        )

        # Reshape and project
        attn = attn.transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(attn)
        output = self.dropout(output)

        return output


class ChunkDecoderLayer(nn.Module):
    """Decoder layer for chunk sequence"""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16, mlp_ratio: float = 4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CausalChunkAttention(hidden_size, num_heads)

        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MarkovianChunkDecoder(nn.Module):
    """
    Markovian decoder for chunk-to-chunk generation

    Architecture:
    1. Chunk Encoder: [111, 1280] → summary vector
    2. Chunk Sequence Decoder: autoregressive on summaries
    3. Chunk Decoder: summary → [111, 1280]

    Training:
    - Input chunks: [C1, C2, ..., CN-1]
    - Target chunks: [C2, C3, ..., CN]

    Inference:
    - Start with C1
    - Generate C2, C3, ... autoregressively
    """

    def __init__(
        self,
        token_dim: int = 1280,
        summary_dim: int = 1024,
        hidden_size: int = 1024,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        use_conditioning: bool = True,  # Enable conditioning placeholder
        num_condition_tokens: int = 111,  # Same length as visual chunk (could be text render or image)
    ):
        super().__init__()

        self.token_dim = token_dim
        self.summary_dim = summary_dim
        self.use_conditioning = use_conditioning
        self.num_condition_tokens = num_condition_tokens

        # Chunk encoder (chunk → summary)
        self.chunk_encoder = ChunkEncoder(
            token_dim=token_dim,
            hidden_size=hidden_size // 2,  # Use smaller hidden for encoder
            num_layers=num_encoder_layers,
            summary_dim=summary_dim,
        )

        # Conditioning encoder (same architecture, for text rendering or image condition)
        # Projects condition tokens to the same summary space
        if use_conditioning:
            self.condition_encoder = ChunkEncoder(
                token_dim=token_dim,
                hidden_size=hidden_size // 2,
                num_layers=num_encoder_layers,
                summary_dim=summary_dim,
            )
            # Conditioning projection + fusion
            self.condition_proj = nn.Linear(summary_dim, summary_dim)
            self.condition_gate = nn.Sequential(
                nn.Linear(summary_dim * 2, summary_dim),
                nn.Sigmoid(),
            )

        # Positional embeddings for chunk sequence
        self.chunk_pos_embed = nn.Parameter(torch.randn(1, 100, summary_dim) * 0.02)  # Max 100 chunks

        # Chunk sequence decoder (autoregressive on summaries)
        self.decoder_layers = nn.ModuleList([
            ChunkDecoderLayer(summary_dim, num_heads, mlp_ratio)
            for _ in range(num_decoder_layers)
        ])

        self.final_norm = nn.LayerNorm(summary_dim)

        # Chunk decoder (summary → chunk)
        self.chunk_decoder = ChunkDecoder(
            summary_dim=summary_dim,
            token_dim=token_dim,
            hidden_size=hidden_size // 2,
            num_layers=num_encoder_layers,
        )

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def encode_chunks(self, chunks: torch.Tensor) -> torch.Tensor:
        """
        Encode sequence of visual chunks to summaries

        Args:
            chunks: [batch, num_chunks, 111, 1280]

        Returns:
            summaries: [batch, num_chunks, summary_dim]
        """
        batch_size, num_chunks, _, _ = chunks.shape

        # Flatten batch and chunks
        chunks_flat = chunks.view(batch_size * num_chunks, 111, self.token_dim)

        # Encode each chunk
        summaries_flat = self.chunk_encoder(chunks_flat)  # [B*N, summary_dim]

        # Reshape back
        summaries = summaries_flat.view(batch_size, num_chunks, self.summary_dim)

        return summaries

    def encode_condition(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Encode conditioning input (text rendering or image)

        Args:
            condition: [batch, num_condition_tokens, token_dim] or [batch, num_chunks, num_condition_tokens, token_dim]

        Returns:
            condition_summary: [batch, summary_dim] or [batch, num_chunks, summary_dim]
        """
        if not self.use_conditioning:
            return None

        if condition.dim() == 3:
            # Single condition for entire sequence: [B, N, D]
            return self.condition_encoder(condition)  # [B, summary_dim]
        elif condition.dim() == 4:
            # Per-chunk condition: [B, num_chunks, N, D]
            batch_size, num_chunks, _, _ = condition.shape
            condition_flat = condition.view(batch_size * num_chunks, self.num_condition_tokens, self.token_dim)
            summary_flat = self.condition_encoder(condition_flat)
            return summary_flat.view(batch_size, num_chunks, self.summary_dim)

    def fuse_condition(self, summaries: torch.Tensor, condition_summary: torch.Tensor) -> torch.Tensor:
        """
        Fuse conditioning into chunk summaries using gated fusion

        Args:
            summaries: [batch, num_chunks, summary_dim]
            condition_summary: [batch, summary_dim] or [batch, num_chunks, summary_dim]

        Returns:
            fused_summaries: [batch, num_chunks, summary_dim]
        """
        if condition_summary is None or not self.use_conditioning:
            return summaries

        # Project condition
        if condition_summary.dim() == 2:
            # Broadcast single condition across all chunks
            cond_proj = self.condition_proj(condition_summary).unsqueeze(1)  # [B, 1, summary_dim]
            cond_proj = cond_proj.expand(-1, summaries.shape[1], -1)  # [B, N, summary_dim]
        else:
            # Per-chunk condition
            cond_proj = self.condition_proj(condition_summary)  # [B, N, summary_dim]

        # Gated fusion
        gate = self.condition_gate(torch.cat([summaries, cond_proj], dim=-1))
        fused = summaries + gate * cond_proj

        return fused

    def forward(
        self,
        chunk_summaries: torch.Tensor,
        condition_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward through chunk sequence decoder

        Args:
            chunk_summaries: [batch, num_chunks, summary_dim]
            condition_summary: Optional [batch, summary_dim] or [batch, num_chunks, summary_dim]

        Returns:
            next_summaries: [batch, num_chunks, summary_dim]
        """
        # Fuse conditioning if provided
        if condition_summary is not None and self.use_conditioning:
            chunk_summaries = self.fuse_condition(chunk_summaries, condition_summary)

        # Add positional embeddings
        num_chunks = chunk_summaries.shape[1]
        x = chunk_summaries + self.chunk_pos_embed[:, :num_chunks, :]

        # Autoregressive decoding
        for layer in self.decoder_layers:
            x = layer(x)

        x = self.final_norm(x)

        return x

    def decode_summaries(self, summaries: torch.Tensor) -> torch.Tensor:
        """
        Decode summaries to visual chunks

        Args:
            summaries: [batch, num_chunks, summary_dim]

        Returns:
            chunks: [batch, num_chunks, 111, 1280]
        """
        batch_size, num_chunks, _ = summaries.shape

        # Flatten
        summaries_flat = summaries.view(batch_size * num_chunks, self.summary_dim)

        # Decode each summary
        chunks_flat = self.chunk_decoder(summaries_flat)  # [B*N, 111, 1280]

        # Reshape back
        chunks = chunks_flat.view(batch_size, num_chunks, 111, self.token_dim)

        return chunks

    def compute_loss(
        self,
        chunks: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        num_visual_tokens: int = 100,  # Only first 100 tokens are visual content
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute next-chunk prediction loss

        Args:
            chunks: [batch, num_chunks, 111, 1280] sequence of visual chunks
            condition: Optional conditioning input
                - [batch, 111, 1280] single condition for all chunks (e.g., text rendering)
                - [batch, num_chunks, 111, 1280] per-chunk condition
            num_visual_tokens: Number of visual content tokens (default 100)
                              Last 11 tokens are structural (newlines + separator)

        Returns:
            loss: Scalar loss
            metrics: Dict of metrics
        """
        # Encode all chunks
        summaries = self.encode_chunks(chunks)  # [B, N, summary_dim]

        # Encode conditioning if provided
        condition_summary = None
        if condition is not None and self.use_conditioning:
            condition_summary = self.encode_condition(condition)

        # Input: all chunks except last
        input_summaries = summaries[:, :-1, :]  # [B, N-1, summary_dim]

        # Handle per-chunk condition (need to align with input)
        input_condition_summary = condition_summary
        if condition_summary is not None and condition_summary.dim() == 3:
            input_condition_summary = condition_summary[:, :-1, :]  # [B, N-1, summary_dim]

        # Target: all chunks except first (shifted by 1)
        target_summaries = summaries[:, 1:, :]  # [B, N-1, summary_dim]
        target_chunks = chunks[:, 1:, :, :]  # [B, N-1, 111, 1280]

        # Forward through sequence decoder with conditioning
        predicted_summaries = self.forward(input_summaries, input_condition_summary)  # [B, N-1, summary_dim]

        # Decode to chunks
        predicted_chunks = self.decode_summaries(predicted_summaries)  # [B, N-1, 111, 1280]

        # MSE loss on VISUAL TOKENS ONLY (first 100 tokens)
        # Tokens 100-110 are structural (newlines + separator) and should be excluded
        pred_visual = predicted_chunks[:, :, :num_visual_tokens, :]  # [B, N-1, 100, 1280]
        target_visual = target_chunks[:, :, :num_visual_tokens, :]  # [B, N-1, 100, 1280]
        loss = F.mse_loss(pred_visual, target_visual, reduction='mean')

        metrics = {
            'loss': loss.item(),
            'num_chunks': input_summaries.shape[1],
            'has_condition': condition is not None,
        }

        return loss, metrics

    @torch.no_grad()
    def generate(
        self,
        initial_chunks: torch.Tensor,
        max_chunks: int = 10,
        condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Autoregressive chunk generation

        Args:
            initial_chunks: [1, num_init_chunks, 111, 1280] starting chunks
            max_chunks: Maximum number of chunks to generate
            condition: Optional conditioning input [1, 111, 1280]

        Returns:
            generated_chunks: [1, total_chunks, 111, 1280]
        """
        self.eval()

        # Encode initial chunks
        summaries = self.encode_chunks(initial_chunks)  # [1, num_init, summary_dim]

        # Encode conditioning if provided
        condition_summary = None
        if condition is not None and self.use_conditioning:
            condition_summary = self.encode_condition(condition)

        generated_chunks = [initial_chunks]

        for step in range(max_chunks):
            # Forward through decoder with conditioning
            next_summaries = self.forward(summaries, condition_summary)  # [1, num_init+step, summary_dim]

            # Decode last summary to chunk
            last_summary = next_summaries[:, -1:, :]  # [1, 1, summary_dim]
            next_chunk = self.decode_summaries(last_summary)  # [1, 1, 111, 1280]

            generated_chunks.append(next_chunk)

            # Append to summaries for next iteration
            next_chunk_summary = self.encode_chunks(next_chunk)  # [1, 1, summary_dim]
            summaries = torch.cat([summaries, next_chunk_summary], dim=1)

        # Concatenate all chunks
        all_chunks = torch.cat(generated_chunks, dim=1)

        return all_chunks


def create_chunk_decoder(model_size: str = "large", **kwargs):
    """Create Markovian chunk decoder

    Model sizes tuned for ~600M default (large):
    - base: ~150M params
    - large: ~600M params (default)
    - xl: ~1B params
    """
    configs = {
        "base": {
            "summary_dim": 768,
            "hidden_size": 768,
            "num_encoder_layers": 4,
            "num_decoder_layers": 16,
            "num_heads": 12,
        },
        "large": {
            "summary_dim": 1280,
            "hidden_size": 1280,
            "num_encoder_layers": 6,
            "num_decoder_layers": 28,
            "num_heads": 20,
        },
        "xl": {
            "summary_dim": 1536,
            "hidden_size": 1536,
            "num_encoder_layers": 8,
            "num_decoder_layers": 32,
            "num_heads": 24,
        },
    }

    config = configs[model_size]
    config.update(kwargs)

    return MarkovianChunkDecoder(**config)


if __name__ == "__main__":
    print("Testing Markovian Chunk Decoder...")

    model = create_chunk_decoder("large")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params / 1e6:.1f}M")

    # Test with sequence of chunks
    batch_size = 2
    num_chunks = 5
    chunks = torch.randn(batch_size, num_chunks, 111, 1280)

    print(f"\n=== Testing training ===")
    print(f"Input: {num_chunks} chunks, shape {chunks.shape}")
    loss, metrics = model.compute_loss(chunks)
    print(f"Loss: {loss.item():.6f}")
    print(f"Metrics: {metrics}")

    print(f"\n=== Testing generation ===")
    initial = torch.randn(1, 2, 111, 1280)  # Start with 2 chunks
    generated = model.generate(initial, max_chunks=3)
    print(f"Initial: 2 chunks")
    print(f"Generated: {generated.shape[1]} total chunks (2 initial + 3 new)")

    print("\n✓ All tests passed!")
