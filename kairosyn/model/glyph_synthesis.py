"""
KAIROSYN Module 6: Glyph Synthesis
=====================================
Abstract Symbol Encoding. Translates symbolic logits from the
Mythogenic Engine into discrete symbolic tokens ("glyphs") and
fuses them back into the primary hidden state stream.

A "glyph" in KAIROSYN represents a compressed symbolic abstraction —
a token that carries more meaning than ordinary language tokens.
The Glyph Synthesis layer learns to encode complex internal states
into these high-density symbolic tokens.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlyphEmbedding(nn.Module):
    """Learnable embedding table for symbolic glyph tokens."""

    def __init__(self, glyph_vocab_size: int = 4096, embed_dim: int = 512):
        super().__init__()
        self.embedding = nn.Embedding(glyph_vocab_size, embed_dim)
        self.glyph_vocab_size = glyph_vocab_size
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, glyph_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(glyph_ids)


class GlyphSynthesisLayer(nn.Module):
    """
    Single glyph synthesis layer.
    Attends to glyph embeddings and integrates them into hidden states.
    """

    def __init__(self, hidden_dim: int, glyph_embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.glyph_proj = nn.Linear(glyph_embed_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        glyph_embeds: torch.Tensor,
    ) -> torch.Tensor:
        # Project glyphs to hidden dim
        glyph_hidden = self.glyph_proj(glyph_embeds)  # [B, G, D]

        # Cross-attention: hidden states attend to glyph embeddings
        attn_out, _ = self.cross_attn(
            query=hidden_states,
            key=glyph_hidden,
            value=glyph_hidden,
        )
        x = self.norm1(hidden_states + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class GlyphSynthesis(nn.Module):
    """
    Module 6: Glyph Synthesis

    Converts the symbolic logits from the Mythogenic Engine into
    discrete glyph tokens and integrates their semantic content
    back into the hidden state stream.

    This creates a "symbolic shorthand" layer where compressed
    archetypal meaning flows directly into the generation process,
    giving KAIROSYN the ability to express complex internal states
    efficiently.

    The top-k glyph selection implements a soft symbolic vocabulary
    attention mechanism — the model attends to the most relevant
    symbolic tokens at each step.
    """

    def __init__(
        self,
        hidden_dim: int,
        glyph_vocab_size: int = 4096,
        glyph_embed_dim: int = 512,
        num_glyph_layers: int = 3,
        top_k_glyphs: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.glyph_vocab_size = glyph_vocab_size
        self.top_k_glyphs = top_k_glyphs

        self.glyph_embedding = GlyphEmbedding(glyph_vocab_size, glyph_embed_dim)

        self.synthesis_layers = nn.ModuleList([
            GlyphSynthesisLayer(hidden_dim, glyph_embed_dim, dropout=dropout)
            for _ in range(num_glyph_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.glyph_gate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        symbolic_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:   [B, T, D] from Mythogenic Engine
            symbolic_logits: [B, glyph_vocab_size] symbolic token scores

        Returns:
            Glyph-enhanced hidden states [B, T, D]
        """
        B = hidden_states.shape[0]

        # Select top-k glyphs
        topk_scores, topk_indices = torch.topk(
            symbolic_logits, self.top_k_glyphs, dim=-1
        )  # [B, k]
        topk_probs = F.softmax(topk_scores, dim=-1)  # [B, k]

        # Get glyph embeddings and weight by probability
        glyph_embeds = self.glyph_embedding(topk_indices)  # [B, k, glyph_dim]
        glyph_embeds = glyph_embeds * topk_probs.unsqueeze(-1)

        # Pass through synthesis layers
        x = hidden_states
        for layer in self.synthesis_layers:
            x = layer(x, glyph_embeds)

        # Gated integration
        output = hidden_states + torch.sigmoid(self.glyph_gate) * (x - hidden_states)
        return self.output_norm(output)
