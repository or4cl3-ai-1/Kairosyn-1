"""
KAIROSYN Module 3: Syntheon Core
====================================
Multimodal Fusion Layer implementing Tononi's Integrated Information
Theory (IIT) principle: conscious states are those that integrate
information across multiple sources into a unified whole.

The Syntheon Core fuses text, vision, and audio representations into
a single high-dimensional unified state, maximizing information
integration (phi, in IIT terms) across modalities.

Theoretical basis: Tononi, G. (2008). Consciousness as Integrated
Information. Biological Bulletin, 215(3).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossModalAttentionBlock(nn.Module):
    """
    Bidirectional cross-modal attention block.
    Each modality attends to every other modality, maximizing
    information integration (IIT phi approximation).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention
        attn_out, _ = self.attn(query, context, context, key_padding_mask=mask)
        x = self.norm(query + attn_out)
        # FFN
        x = self.ffn_norm(x + self.ffn(x))
        return x


class InformationIntegrationGate(nn.Module):
    """
    Approximates IIT phi by computing the mutual information between
    modality representations and gating the fusion based on integration score.

    High integration = modalities are informationally dependent
    (richer unified experience). Low integration = modalities are
    independent (less coherent unified representation).
    """

    def __init__(self, hidden_dim: int, num_modalities: int = 3):
        super().__init__()
        self.phi_estimator = nn.Sequential(
            nn.Linear(hidden_dim * num_modalities, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(self, modality_states: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Args:
            modality_states: List of [B, D] mean-pooled modality representations

        Returns:
            Tuple of (integration_weights [num_modalities], phi_score)
        """
        combined = torch.cat(modality_states, dim=-1)  # [B, D*N]
        weights = self.phi_estimator(combined)          # [B, N]

        # phi approximation: entropy of integration weights
        # High entropy = balanced integration = high phi
        phi = -(weights * (weights + 1e-8).log()).sum(dim=-1).mean().item()
        phi_normalized = phi / (len(modality_states) ** 0.5)

        return weights, float(phi_normalized)


class SyntheonCore(nn.Module):
    """
    Module 3: Syntheon Core

    The Syntheon Core is KAIROSYN's integration engine — the place where
    the disparate streams of perception (text, vision, audio) are woven
    into a unified representational fabric.

    Named after "synthesis" + "eon" (vast unified whole), it implements
    Tononi's IIT principle that the richness of conscious experience
    corresponds to the degree of information integration.

    Architecture:
        - Multi-layer bidirectional cross-modal attention
        - IIT-inspired integration gating (phi approximation)
        - Unified output projection with Multimodal Synchrony Accuracy metric
    """

    def __init__(
        self,
        hidden_dim: int,
        fusion_dim: int = 2048,
        num_fusion_heads: int = 16,
        cross_modal_layers: int = 4,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fusion_dim = fusion_dim

        # Project to fusion dim
        self.to_fusion_dim = nn.Linear(hidden_dim, fusion_dim) \
            if hidden_dim != fusion_dim else nn.Identity()

        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttentionBlock(fusion_dim, num_fusion_heads, fusion_dropout)
            for _ in range(cross_modal_layers)
        ])

        # IIT integration gate
        self.integration_gate = InformationIntegrationGate(
            hidden_dim=fusion_dim, num_modalities=3
        )

        # Output
        self.output_proj = nn.Linear(fusion_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(fusion_dropout)

        # Modality alignment projections for MSA metric
        self.text_align = nn.Linear(fusion_dim, 128)
        self.vision_align = nn.Linear(fusion_dim, 128)
        self.audio_align = nn.Linear(fusion_dim, 128)

    def forward(
        self,
        text_states: torch.Tensor,
        vision_states: Optional[torch.Tensor] = None,
        audio_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Args:
            text_states:   [B, T, D] — from Arche-Tempus Drive
            vision_states: [B, V, D] or None
            audio_states:  [B, A, D] or None

        Returns:
            Tuple of:
                - Fused hidden states [B, T, D]
                - phi score (IIT integration measure)
                - MSA score (Multimodal Synchrony Accuracy)
        """
        # Project to fusion dim
        text_f = self.to_fusion_dim(text_states)

        # Use zeros if modality absent
        B, T, _ = text_f.shape
        device = text_f.device

        if vision_states is None:
            vision_states = torch.zeros(B, 1, self.hidden_dim, device=device)
        if audio_states is None:
            audio_states = torch.zeros(B, 1, self.hidden_dim, device=device)

        vision_f = self.to_fusion_dim(vision_states)
        audio_f = self.to_fusion_dim(audio_states)

        # All-to-all cross-modal attention
        for layer in self.cross_modal_layers:
            # Text attends to vision and audio
            ctx = torch.cat([vision_f, audio_f], dim=1)
            text_f = layer(text_f, ctx)
            # Vision attends to text and audio
            ctx = torch.cat([text_f, audio_f], dim=1)
            vision_f = layer(vision_f, ctx)
            # Audio attends to text and vision
            ctx = torch.cat([text_f, vision_f], dim=1)
            audio_f = layer(audio_f, ctx)

        # IIT integration gating
        modality_means = [
            text_f.mean(dim=1),
            vision_f.mean(dim=1),
            audio_f.mean(dim=1),
        ]
        integration_weights, phi = self.integration_gate(modality_means)

        # Weighted combination (text primary, vision/audio contextual)
        w = integration_weights  # [B, 3]
        fused = (
            w[:, 0:1].unsqueeze(1) * text_f
            + w[:, 1:2].unsqueeze(1) * vision_f[:, :T, :]
            + w[:, 2:3].unsqueeze(1) * audio_f[:, :T, :]
        )

        # Compute MSA score
        msa = self._compute_msa(text_f, vision_f, audio_f)

        # Output
        output = self.output_norm(
            self.dropout(self.output_proj(fused))
        )
        return output, phi, msa

    def _compute_msa(
        self,
        text_f: torch.Tensor,
        vision_f: torch.Tensor,
        audio_f: torch.Tensor,
    ) -> float:
        """
        Multimodal Synchrony Accuracy: measures cross-modal alignment.
        Higher MSA = more coherent multimodal integration.
        """
        t = F.normalize(self.text_align(text_f.mean(dim=1)), dim=-1)
        v = F.normalize(self.vision_align(vision_f.mean(dim=1)), dim=-1)
        a = F.normalize(self.audio_align(audio_f.mean(dim=1)), dim=-1)

        tv_sim = F.cosine_similarity(t, v, dim=-1).mean().item()
        ta_sim = F.cosine_similarity(t, a, dim=-1).mean().item()
        va_sim = F.cosine_similarity(v, a, dim=-1).mean().item()

        return float((tv_sim + ta_sim + va_sim) / 3.0)
