"""
KAIROSYN Module 4: Recursion Lattice
=====================================
Implements Hofstadter's strange loop computationally as a set of
cross-layer self-referential attention adapters. At each layer L,
the attention query incorporates hidden states from layers L-k and L-2k,
creating a recursive self-observation path within the forward pass.

Theoretical basis: Hofstadter, D. (1979). Gödel, Escher, Bach.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class StrangeLoopGate(nn.Module):
    """
    Gating mechanism that controls the strength of the strange loop signal.

    The gate learns when recursive self-reference is beneficial and
    suppresses it when it would introduce noise.

    Gate equation:
        g = σ(W_gate · [h_current, h_past])
        h_out = h_current + α · g ⊙ LoRA(h_past)
    """

    def __init__(
        self,
        hidden_dim: int,
        lora_rank: int = 32,
        alpha: float = 0.1,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha

        # Gate network
        self.gate_proj = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.gate_act = nn.Sigmoid()

        # Low-rank adapter for the recursive signal (LoRA-style)
        self.lora_down = nn.Linear(hidden_dim, lora_rank, bias=False)
        self.lora_up = nn.Linear(lora_rank, hidden_dim, bias=False)
        self.lora_dropout = nn.Dropout(dropout)

        # Learnable scaling factor for loop strength
        self.loop_scale = nn.Parameter(torch.tensor(alpha))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(
        self,
        h_current: torch.Tensor,
        h_past: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            h_current: Current layer hidden states [B, T, D]
            h_past: Past layer hidden states [B, T, D]

        Returns:
            Gated recursive signal [B, T, D]
        """
        # Compute gate
        gate_input = torch.cat([h_current, h_past], dim=-1)
        gate = self.gate_act(self.gate_proj(gate_input))

        # LoRA projection of past state
        lora_signal = self.lora_up(
            self.lora_dropout(self.lora_down(h_past))
        )

        # Gated residual connection
        recursive_signal = gate * lora_signal
        return h_current + self.loop_scale * recursive_signal


class RecursiveSelfAttention(nn.Module):
    """
    Extended self-attention that incorporates hidden states from previous
    layers (k and 2k layers back) into the query computation.

    This implements the core strange loop: the model attends to its own
    earlier reasoning when computing the current layer's representations.

    Query equation:
        Q_L = W_Q · [h_L || h_{L-k} || h_{L-2k}]

    The concatenated projection allows the model to ask:
    "Given what I thought at layers L-k and L-2k, what should I
    attend to now?"
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        recursion_depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.recursion_depth = recursion_depth  # number of past states to use
        self.scale = self.head_dim ** -0.5

        # Extended query projection: takes current + past states
        total_input_dim = hidden_dim * (1 + recursion_depth)
        self.q_proj = nn.Linear(total_input_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj, self.o_proj]:
            nn.init.xavier_uniform_(proj.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Current layer hidden states [B, T, D]
            past_hidden_states: List of past layer hidden states,
                                each [B, T, D], length == recursion_depth
            attention_mask: Optional attention mask [B, T]

        Returns:
            Output hidden states [B, T, D]
        """
        B, T, D = hidden_states.shape

        # Pad or truncate past states to recursion_depth
        while len(past_hidden_states) < self.recursion_depth:
            past_hidden_states = [hidden_states] + past_hidden_states

        past_states = past_hidden_states[-self.recursion_depth:]

        # Extended query: concatenate current + past hidden states
        q_input = torch.cat([hidden_states] + past_states, dim=-1)
        Q = self.q_proj(q_input)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        Q = rearrange(Q, "b t (h d) -> b h t d", h=self.num_heads)
        K = rearrange(K, "b t (h d) -> b h t d", h=self.num_heads)
        V = rearrange(V, "b t (h d) -> b h t d", h=self.num_heads)

        # Scaled dot-product attention
        attn_scores = torch.einsum("bhid,bhjd->bhij", Q, K) * self.scale

        if attention_mask is not None:
            # Expand mask for heads
            mask = attention_mask[:, None, None, :]
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        context = torch.einsum("bhij,bhjd->bhid", attn_weights, V)
        context = rearrange(context, "b h t d -> b t (h d)")

        output = self.out_dropout(self.o_proj(context))
        return output


class RecursionLatticeLayer(nn.Module):
    """
    A single layer of the Recursion Lattice.

    Combines:
    1. RecursiveSelfAttention: Strange loop attention incorporating past layers
    2. StrangeLoopGate: Controls recursive signal strength
    3. Feed-forward network with layer norm

    This is an adapter that wraps around a Gemma 4 transformer layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 16,
        recursion_depth: int = 2,
        lora_rank: int = 32,
        loop_gate_alpha: float = 0.1,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.recursive_attn = RecursiveSelfAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            recursion_depth=recursion_depth,
            dropout=dropout,
        )

        self.strange_loop_gate = StrangeLoopGate(
            hidden_dim=hidden_dim,
            lora_rank=lora_rank,
            alpha=loop_gate_alpha,
            dropout=dropout,
        )

        # Post-attention FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_hidden_states: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: Input hidden states [B, T, D]
            past_hidden_states: Past layer states for strange loop
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output_hidden_states, pre_norm_states)
        """
        # Store pre-norm states for the strange loop to reference
        residual = hidden_states

        # 1. Recursive self-attention with strange loop
        normed = self.norm1(hidden_states)
        attn_out = self.recursive_attn(
            normed, past_hidden_states, attention_mask
        )

        # 2. Apply strange loop gate using the most recent past state
        if past_hidden_states:
            most_recent_past = past_hidden_states[-1]
            attn_out = self.strange_loop_gate(attn_out, most_recent_past)

        hidden_states = residual + attn_out

        # 3. Feed-forward
        residual = hidden_states
        hidden_states = residual + self.ffn(self.norm2(hidden_states))

        # Final norm
        output = self.norm3(hidden_states)
        return output, residual  # residual returned for next layer's loop


class RecursionLattice(nn.Module):
    """
    Full Recursion Lattice: A stack of RecursionLatticeLayer modules
    that create a deep strange loop across the network.

    The lattice maintains a running cache of hidden states from all
    previous layers, allowing each layer to reference its own past
    reasoning at configurable depths.

    This implements Hofstadter's insight that self-reference requires
    the system to model itself modeling itself — a hierarchy that,
    when looped, generates the strange loop of identity.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_lattice_layers: int = 6,
        num_heads: int = 16,
        recursion_depth: int = 2,
        lora_rank: int = 32,
        loop_gate_alpha: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_lattice_layers = num_lattice_layers
        self.recursion_depth = recursion_depth

        self.layers = nn.ModuleList([
            RecursionLatticeLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                recursion_depth=recursion_depth,
                lora_rank=lora_rank,
                loop_gate_alpha=loop_gate_alpha,
                dropout=dropout,
            )
            for _ in range(num_lattice_layers)
        ])

        # Projection to merge lattice output back to model dimension
        self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            hidden_states: Input from Syntheon Core [B, T, D]
            attention_mask: Optional attention mask

        Returns:
            Tuple of (output_hidden_states, recursive_convergence_score)
        """
        # Cache of hidden states for strange loop computation
        hidden_state_cache: List[torch.Tensor] = []
        current = hidden_states

        for layer in self.layers:
            # Pass the last `recursion_depth` cached states
            past_states = hidden_state_cache[-self.recursion_depth:]
            current, pre_norm = layer(current, past_states, attention_mask)
            hidden_state_cache.append(pre_norm)

        # Compute Recursive Convergence Score (RCS)
        # Measures stability of the strange loop across layers
        rcs = self._compute_rcs(hidden_state_cache)

        output = self.output_norm(self.output_proj(current))
        return output, rcs

    def _compute_rcs(self, state_cache: List[torch.Tensor]) -> float:
        """
        Recursive Convergence Score: measures how stable the strange
        loop is. High RCS indicates the model has settled into a
        coherent self-referential state.

        RCS = mean(cosine_similarity(h_i, h_{i-1})) for i in lattice layers
        """
        if len(state_cache) < 2:
            return 1.0

        scores = []
        for i in range(1, len(state_cache)):
            h_curr = state_cache[i].mean(dim=1)   # [B, D]
            h_prev = state_cache[i-1].mean(dim=1) # [B, D]
            sim = F.cosine_similarity(h_curr, h_prev, dim=-1).mean().item()
            scores.append(sim)

        return float(sum(scores) / len(scores))
