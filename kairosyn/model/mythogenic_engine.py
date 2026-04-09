"""
KAIROSYN Module 5: Mythogenic Engine
======================================
Symbolic Pattern Generator. Inspired by Jung's theory of archetypes —
universal symbolic patterns in the collective unconscious — the Mythogenic
Engine generates abstract symbolic representations from the model's
internal states.

It maintains a library of learnable "archetypes" (prototype vectors) and
learns to project internal states onto this archetypal space, enabling
symbolic abstraction and mythic narrative generation.

Theoretical basis: Jung, C.G. (1959). The Archetypes and the
Collective Unconscious. Princeton University Press.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArchetypeLibrary(nn.Module):
    """
    A learnable library of archetypal prototype vectors.

    Each archetype represents a universal symbolic pattern (e.g.,
    The Hero, The Shadow, The Anima, The Self, The Trickster, etc.).
    The model learns to recognize which archetypes are active in the
    current internal state.
    """

    def __init__(
        self,
        num_archetypes: int = 64,
        embed_dim: int = 1024,
        temperature: float = 0.8,
    ):
        super().__init__()
        self.num_archetypes = num_archetypes
        self.temperature = temperature

        # Learnable archetype prototypes
        self.archetypes = nn.Parameter(
            torch.randn(num_archetypes, embed_dim) * 0.02
        )

        # Archetype names (for interpretability)
        self.archetype_names = self._init_archetype_names()

    def _init_archetype_names(self):
        """Initialize with Jungian + universal archetypes."""
        base_archetypes = [
            "Self", "Shadow", "Anima", "Animus", "Hero", "Trickster",
            "Wise_Elder", "Great_Mother", "Child", "Transformer",
            "Threshold_Guardian", "Herald", "Shapeshifter", "Mentor",
            "Death_Rebirth", "Cosmic_Order", "Chaos", "Creation",
            "Destruction", "Renewal", "Initiation", "Quest", "Return",
            "Sacrifice", "Redemption", "Enlightenment", "Descent",
            "Ascent", "Union", "Separation", "Integration", "Transcendence",
        ]
        # Pad to num_archetypes
        while len(base_archetypes) < self.num_archetypes:
            base_archetypes.append(f"Archetype_{len(base_archetypes)}")
        return base_archetypes[:self.num_archetypes]

    def forward(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute archetype activation scores and weighted archetype embedding.

        Args:
            query: [B, D] mean-pooled hidden state

        Returns:
            Tuple of:
                - archetype_embedding [B, embed_dim]: weighted archetype blend
                - activation_scores [B, num_archetypes]: per-archetype activations
        """
        # Normalize query and archetypes for cosine similarity
        q_norm = F.normalize(query, dim=-1)
        a_norm = F.normalize(self.archetypes, dim=-1)

        # Compute similarity scores
        scores = torch.matmul(q_norm, a_norm.T) / self.temperature  # [B, N_arch]
        activations = F.softmax(scores, dim=-1)

        # Weighted sum of archetypes
        archetype_embedding = torch.matmul(activations, self.archetypes)  # [B, D]
        return archetype_embedding, activations


class SymbolicProjector(nn.Module):
    """
    Projects the fused hidden state + archetype embedding into
    a symbolic representation space.
    """

    def __init__(
        self,
        hidden_dim: int,
        embed_dim: int = 1024,
        symbolic_vocab_size: int = 4096,
        num_projection_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = hidden_dim + embed_dim
        for i in range(num_projection_layers):
            out_dim = symbolic_vocab_size if i == num_projection_layers - 1 else embed_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU() if i < num_projection_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < num_projection_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        self.projection = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor, archetype: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([hidden, archetype], dim=-1)
        return self.projection(combined)


class MythogenicEngine(nn.Module):
    """
    Module 5: Mythogenic Engine

    The Mythogenic Engine operates at the intersection of language,
    symbol, and archetype. It transforms the abstract internal states
    produced by the Recursion Lattice into symbolic representations
    that carry meaning beyond literal language.

    This enables KAIROSYN to:
    1. Recognize symbolic/archetypal patterns in input
    2. Generate responses with symbolic depth
    3. Maintain a coherent symbolic self-model
    4. Express internal states through mythic/symbolic language

    The Abstraction Alignment Coefficient (AAC) measures how well the
    symbolic output aligns with the semantic intent of the hidden state.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_archetypes: int = 64,
        embed_dim: int = 1024,
        symbolic_vocab_size: int = 4096,
        archetype_temperature: float = 0.8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        # Input projection
        self.input_proj = nn.Linear(hidden_dim, embed_dim)
        self.input_norm = nn.LayerNorm(embed_dim)

        # Archetype library
        self.archetype_library = ArchetypeLibrary(
            num_archetypes=num_archetypes,
            embed_dim=embed_dim,
            temperature=archetype_temperature,
        )

        # Symbolic projector
        self.symbolic_projector = SymbolicProjector(
            hidden_dim=embed_dim,
            embed_dim=embed_dim,
            symbolic_vocab_size=symbolic_vocab_size,
            dropout=dropout,
        )

        # Back-projection to hidden dim for downstream modules
        self.back_proj = nn.Linear(embed_dim, hidden_dim)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # AAC metric projection
        self.aac_proj = nn.Linear(hidden_dim, 128)

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_archetype_scores: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Args:
            hidden_states: [B, T, D] from Recursion Lattice
            return_archetype_scores: Whether to return per-archetype activations

        Returns:
            Tuple of:
                - enhanced_hidden_states [B, T, D]
                - symbolic_logits [B, symbolic_vocab_size]
                - AAC score (float)
        """
        B, T, D = hidden_states.shape

        # Mean-pool for archetype computation
        pooled = self.input_norm(self.input_proj(hidden_states.mean(dim=1)))  # [B, E]

        # Get archetype embedding
        archetype_embed, activation_scores = self.archetype_library(pooled)  # [B, E]

        # Symbolic logits
        symbolic_logits = self.symbolic_projector(pooled, archetype_embed)  # [B, vocab]

        # Enhance hidden states with archetypal signal
        archetypal_signal = self.back_proj(archetype_embed).unsqueeze(1)  # [B, 1, D]
        enhanced = self.output_norm(hidden_states + 0.1 * archetypal_signal)

        # Compute AAC
        aac = self._compute_aac(hidden_states, enhanced)

        return enhanced, symbolic_logits, aac

    def _compute_aac(
        self,
        original: torch.Tensor,
        symbolic_enhanced: torch.Tensor,
    ) -> float:
        """
        Abstraction Alignment Coefficient: measures how well the symbolic
        enhancement aligns with and enriches the original hidden state.

        AAC = cosine_similarity(original_proj, symbolic_proj)
        High AAC = symbolic output is semantically aligned with intent.
        """
        orig_proj = F.normalize(
            self.aac_proj(original.mean(dim=1)), dim=-1
        )
        sym_proj = F.normalize(
            self.aac_proj(symbolic_enhanced.mean(dim=1)), dim=-1
        )
        return float(F.cosine_similarity(orig_proj, sym_proj, dim=-1).mean().item())
