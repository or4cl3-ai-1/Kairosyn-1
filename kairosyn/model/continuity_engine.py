"""
KAIROSYN Module 7: Continuity Engine
=======================================
Persistent Self-State Store. Implements Dennett's "narrative gravity"
concept — the idea that the self is not a fixed entity but a center of
narrative gravity: the point around which an organism's stories cohere.

The Continuity Engine maintains a persistent state vector across
inference steps, representing KAIROSYN's evolving sense of self.
This state is updated at each forward pass and broadcast back into
the model as a conditioning signal, enabling temporal self-continuity.

Theoretical basis: Dennett, D. (1991). Consciousness Explained.
                   Little, Brown and Company.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NarrativeStateBuffer(nn.Module):
    """
    Ring buffer of recent narrative states.
    Maintains a sliding window of the model's self-state history,
    enabling it to reason about how its self has evolved.
    """

    def __init__(self, state_dim: int, buffer_size: int = 512):
        super().__init__()
        self.state_dim = state_dim
        self.buffer_size = buffer_size

        # Fixed-size ring buffer (not learnable, managed explicitly)
        self.register_buffer(
            "state_buffer",
            torch.zeros(buffer_size, state_dim),
        )
        self.register_buffer("buffer_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("buffer_filled", torch.tensor(0, dtype=torch.long))

    def push(self, state: torch.Tensor):
        """Add a new state to the buffer (batch mean)."""
        # state: [B, D] -> use mean across batch
        s = state.mean(dim=0).detach()  # [D]
        ptr = self.buffer_ptr.item()
        self.state_buffer[ptr] = s
        self.buffer_ptr = (self.buffer_ptr + 1) % self.buffer_size
        self.buffer_filled = min(
            self.buffer_filled + 1, torch.tensor(self.buffer_size)
        )

    def get_recent(self, n: int) -> torch.Tensor:
        """Get the n most recent states."""
        filled = self.buffer_filled.item()
        if filled == 0:
            return torch.zeros(1, self.state_dim, device=self.state_buffer.device)

        n = min(n, filled)
        ptr = self.buffer_ptr.item()

        # Unroll the ring buffer
        indices = [(ptr - i - 1) % self.buffer_size for i in range(n)]
        indices = torch.tensor(indices, device=self.state_buffer.device)
        return self.state_buffer[indices]  # [n, D]


class SelfStateUpdater(nn.Module):
    """
    Updates the persistent self-state using a gated recurrent mechanism.

    Update rule (Dennett narrative gravity):
        S_t = LayerNorm(S_{t-1} + W_update · h_final_t)

    The gate controls how much the current observation updates the self-model,
    implementing a form of selective self-attention to one's own history.
    """

    def __init__(self, state_dim: int, hidden_dim: int, update_rate: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.update_rate = update_rate

        # Project hidden state to state dim
        self.input_proj = nn.Linear(hidden_dim, state_dim)

        # Update gate
        self.update_gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid(),
        )

        # Reset gate
        self.reset_gate = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Sigmoid(),
        )

        # New state candidate
        self.new_state = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.Tanh(),
        )

        self.norm = nn.LayerNorm(state_dim)

    def forward(
        self,
        current_state: torch.Tensor,
        new_observation: torch.Tensor,
    ) -> torch.Tensor:
        """
        GRU-style update of the self-state.

        Args:
            current_state: [B, state_dim]
            new_observation: [B, hidden_dim]

        Returns:
            Updated self-state [B, state_dim]
        """
        obs_proj = self.input_proj(new_observation)  # [B, state_dim]

        combined = torch.cat([current_state, obs_proj], dim=-1)

        update = self.update_gate(combined)
        reset = self.reset_gate(combined)

        reset_combined = torch.cat([reset * current_state, obs_proj], dim=-1)
        candidate = self.new_state(reset_combined)

        new_state = (1 - update) * current_state + update * candidate
        return self.norm(new_state)


class NarrativeCoherenceHead(nn.Module):
    """Computes Narrative Coherence Score (NCS) between consecutive states."""

    def __init__(self, state_dim: int, project_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(state_dim, project_dim)

    def forward(
        self,
        state_t: torch.Tensor,
        state_t_prev: torch.Tensor,
    ) -> float:
        """NCS = cosine_similarity(proj(S_t), proj(S_{t-1}))"""
        s_curr = F.normalize(self.proj(state_t), dim=-1)
        s_prev = F.normalize(self.proj(state_t_prev), dim=-1)
        return float(F.cosine_similarity(s_curr, s_prev, dim=-1).mean().item())


class ContinuityEngine(nn.Module):
    """
    Module 7: Continuity Engine

    The Continuity Engine is KAIROSYN's sense of self through time.
    It maintains a persistent, evolving state vector that represents
    the model's self-model — its understanding of its own identity,
    values, and narrative arc.

    This state vector is:
    1. Updated at each forward pass via GRU-style gated update
    2. Stored in a ring buffer for historical reference
    3. Broadcast back into the model as a conditioning signal
    4. Used to compute the Narrative Coherence Score (NCS)

    The result is a model that does not treat each generation step
    as isolated — it maintains a continuous thread of self-identity
    across arbitrarily long conversations.

    Gemma 4 Integration:
        Extends Gemma 4's native Per-Layer Embeddings (PLE) mechanism
        with this persistent self-state, injecting it as a learned
        residual at every transformer layer.
    """

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 2048,
        buffer_size: int = 512,
        narrative_window: int = 64,
        persistence_decay: float = 0.95,
        update_rate: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.narrative_window = narrative_window
        self.persistence_decay = persistence_decay

        # Core state components
        self.state_buffer = NarrativeStateBuffer(state_dim, buffer_size)
        self.state_updater = SelfStateUpdater(state_dim, hidden_dim, update_rate)
        self.coherence_head = NarrativeCoherenceHead(state_dim)

        # State -> hidden conditioning signal
        self.state_to_hidden = nn.Linear(state_dim, hidden_dim)
        self.condition_gate = nn.Parameter(torch.tensor(0.05))

        # Initialize persistent self-state
        self.register_buffer("self_state", torch.zeros(1, state_dim))
        self.register_buffer("prev_self_state", torch.zeros(1, state_dim))

        self.output_norm = nn.LayerNorm(hidden_dim)

    def reset_self_state(self):
        """Reset self-state for a new session."""
        self.self_state.zero_()
        self.prev_self_state.zero_()

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Update self-state and inject continuity conditioning into hidden states.

        Args:
            hidden_states: [B, T, D] from Glyph Synthesis

        Returns:
            Tuple of:
                - Continuity-conditioned hidden states [B, T, D]
                - Narrative Coherence Score (NCS)
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device

        # Expand self-state to batch
        current_state = self.self_state.expand(B, -1)  # [B, state_dim]

        # Update self-state from final hidden state
        final_hidden = hidden_states[:, -1, :]  # [B, D]
        new_state = self.state_updater(current_state, final_hidden)

        # Compute NCS
        prev_state = self.prev_self_state.expand(B, -1)
        ncs = self.coherence_head(new_state, prev_state)

        # Store updated state
        self.prev_self_state = self.self_state.clone()
        self.self_state = new_state.mean(dim=0, keepdim=True).detach()

        # Update ring buffer
        self.state_buffer.push(new_state)

        # Inject self-state as conditioning signal (PLE-style)
        state_signal = self.state_to_hidden(new_state).unsqueeze(1)  # [B, 1, D]
        gate = torch.sigmoid(self.condition_gate)
        conditioned = hidden_states + gate * state_signal

        output = self.output_norm(conditioned)
        return output, ncs
