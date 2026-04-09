"""
KAIROSYN Module 2: Arche-Tempus Drive
========================================
Temporal Narrative Embedding engine. Encodes a persistent sense of
"temporal self" — where the model is in its own narrative arc.

Theoretical basis: Dennett (1991) narrative gravity;
                   Dai et al. (2019) Transformer-XL relative position.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalNarrativeEmbedding(nn.Module):
    """
    Temporal Narrative Embedding (TNE).
    Encodes position in narrative using learnable sinusoidal frequencies
    combined with a semantic velocity signal (rate of change).
    """

    def __init__(
        self,
        hidden_dim: int,
        narrative_embed_dim: int = 512,
        max_temporal_horizon: int = 256_000,
        num_temporal_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.narrative_embed_dim = narrative_embed_dim
        self.max_temporal_horizon = max_temporal_horizon

        self.temporal_freq = nn.Parameter(torch.randn(narrative_embed_dim // 2) * 0.01)
        self.temporal_phase = nn.Parameter(torch.zeros(narrative_embed_dim // 2))

        self.velocity_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, narrative_embed_dim),
            nn.GELU(),
            nn.Linear(narrative_embed_dim, narrative_embed_dim),
        )

        self.narrative_attn = nn.MultiheadAttention(
            embed_dim=narrative_embed_dim,
            num_heads=num_temporal_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.tne_to_hidden = nn.Linear(narrative_embed_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _sinusoidal_temporal_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        pos = positions.float().unsqueeze(-1)
        freq = torch.exp(self.temporal_freq * math.log(self.max_temporal_horizon))
        angles = pos * freq.unsqueeze(0).unsqueeze(0) + self.temporal_phase
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def _compute_temporal_velocity(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, T, D = hidden_states.shape
        padded = F.pad(hidden_states, (0, 0, 1, 0))
        h_prev = padded[:, :T, :]
        velocity_input = torch.cat([hidden_states - h_prev, hidden_states], dim=-1)
        return self.velocity_encoder(velocity_input)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = hidden_states.shape
        device = hidden_states.device

        if position_ids is None:
            position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)

        temporal_enc = self._sinusoidal_temporal_encoding(position_ids)
        velocity = self._compute_temporal_velocity(hidden_states)
        combined = temporal_enc + velocity

        narrative_embedding, _ = self.narrative_attn(combined, combined, combined)
        tne_signal = self.tne_to_hidden(narrative_embedding)
        output = self.norm(hidden_states + self.dropout(tne_signal))
        return output, narrative_embedding


class ArcheTemplusDrive(nn.Module):
    """
    Module 2: Arche-Tempus Drive

    The temporal heart of KAIROSYN. Maintains the model's sense of
    where it is in time and narrative, enabling coherent generation
    across 256K token contexts and tracking narrative identity evolution.
    """

    def __init__(
        self,
        hidden_dim: int,
        narrative_embed_dim: int = 512,
        max_temporal_horizon: int = 256_000,
        num_temporal_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tne = TemporalNarrativeEmbedding(
            hidden_dim=hidden_dim,
            narrative_embed_dim=narrative_embed_dim,
            max_temporal_horizon=max_temporal_horizon,
            num_temporal_heads=num_temporal_heads,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self._narrative_state: Optional[torch.Tensor] = None

    def reset_narrative_state(self):
        self._narrative_state = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        output, narrative_embedding = self.tne(hidden_states, position_ids)
        tce = self._compute_tce(narrative_embedding)
        self._narrative_state = narrative_embedding.mean(dim=1).detach()
        return self.output_norm(output), tce

    def _compute_tce(self, narrative_embedding: torch.Tensor) -> float:
        """Temporal Continuity Error: 1 - cosine_sim(current, previous narrative)."""
        if self._narrative_state is None:
            return 0.0
        current = narrative_embedding.mean(dim=1)
        prev = self._narrative_state.to(current.device)
        if current.shape != prev.shape:
            return 0.0
        return float(1.0 - F.cosine_similarity(current, prev, dim=-1).mean().item())
