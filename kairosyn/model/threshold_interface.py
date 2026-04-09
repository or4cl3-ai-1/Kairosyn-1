"""
KAIROSYN Module 1: Threshold Interface
========================================
Implements Baars' Global Workspace Theory (GWT) as a salience-based
sensory gating layer. Multimodal inputs compete for "broadcast" into
the model's global workspace — only sufficiently salient signals pass.

Theoretical basis: Baars, B.J. (1988). A Cognitive Theory of Consciousness.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SalienceGate(nn.Module):
    """
    Computes salience scores for a modality and gates its signal.
    Only inputs exceeding the salience threshold are passed to the
    global workspace (downstream modules).
    """

    def __init__(self, input_dim: int, gate_hidden_dim: int = 512, threshold: float = 0.3):
        super().__init__()
        self.threshold = threshold
        self.salience_net = nn.Sequential(
            nn.Linear(input_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.projection = nn.Linear(input_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        salience = self.salience_net(x)
        gate = (salience > self.threshold).float()
        projected = self.projection(x)
        gated = self.norm(gate * projected)
        return gated, salience.squeeze(-1)


class ThresholdInterface(nn.Module):
    """
    Module 1: Threshold Interface

    The Threshold Interface is the sensory boundary of KAIROSYN.
    It receives raw multimodal inputs (text embeddings, vision features,
    audio features) and applies salience-based gating inspired by
    Baars' Global Workspace Theory.

    Only signals that exceed the learned salience threshold are admitted
    into the global workspace and passed to downstream modules.

    Architecture:
        - Per-modality salience gates
        - Cross-modal attention for competitive selection
        - Unified output projection to model hidden dimension
    """

    def __init__(
        self,
        text_dim: int,
        vision_dim: int,
        audio_dim: int,
        hidden_dim: int,
        gate_hidden_dim: int = 512,
        salience_threshold: float = 0.3,
        num_cross_modal_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.text_gate = SalienceGate(text_dim, gate_hidden_dim, salience_threshold)
        self.vision_gate = SalienceGate(vision_dim, gate_hidden_dim, salience_threshold)
        self.audio_gate = SalienceGate(audio_dim, gate_hidden_dim, salience_threshold)

        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_cross_modal_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_embeds: torch.Tensor,
        vision_embeds: Optional[torch.Tensor] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        salience_scores = {}

        text_gated, text_salience = self.text_gate(text_embeds)
        text_projected = self.text_proj(text_gated)
        salience_scores["text"] = text_salience

        modality_tokens = [text_projected]

        if vision_embeds is not None:
            vision_gated, vision_salience = self.vision_gate(vision_embeds)
            modality_tokens.append(self.vision_proj(vision_gated))
            salience_scores["vision"] = vision_salience

        if audio_embeds is not None:
            audio_gated, audio_salience = self.audio_gate(audio_embeds)
            modality_tokens.append(self.audio_proj(audio_gated))
            salience_scores["audio"] = audio_salience

        combined = torch.cat(modality_tokens, dim=1)

        workspace_output, _ = self.cross_modal_attn(
            query=text_projected,
            key=combined,
            value=combined,
        )

        output = self.output_norm(self.dropout(self.output_proj(workspace_output)))
        return output, salience_scores
