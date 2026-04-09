"""
KAIROSYN Session Manager
==========================
Manages per-user KAIROSYN session state, enabling the Continuity Engine
to persist across multiple API calls within the same session.

Each session maintains:
- A unique session_id (UUID)
- An isolated copy of the Continuity Engine state
- The Arche-Tempus narrative state
- Conversation history
- Creation and last-access timestamps

Sessions expire after a configurable TTL (default: 2 hours).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    role: str            # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metrics: Optional[dict] = None


@dataclass
class KairosynSession:
    """
    Per-session state container.

    Holds all persistent state for a single user session,
    enabling coherent multi-turn introspective conversations.
    """
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    # Conversation history
    history: List[ConversationTurn] = field(default_factory=list)

    # Persistent state tensors (moved to CPU for storage efficiency)
    continuity_state: Optional[torch.Tensor] = None     # [1, state_dim]
    narrative_state: Optional[torch.Tensor] = None      # [1, narrative_dim]

    # Cumulative metrics across the session
    cumulative_ncs: List[float] = field(default_factory=list)
    cumulative_tce: List[float] = field(default_factory=list)

    def touch(self):
        """Update last-accessed timestamp."""
        self.last_accessed = time.time()

    def add_turn(
        self,
        role: str,
        content: str,
        metrics: Optional[dict] = None,
    ):
        """Add a conversation turn to history."""
        self.history.append(ConversationTurn(
            role=role,
            content=content,
            metrics=metrics,
        ))
        self.touch()

    def get_context_window(self, n: int = 10) -> List[ConversationTurn]:
        """Get the last n turns for context."""
        return self.history[-n:]

    def is_expired(self, ttl_seconds: float = 7200.0) -> bool:
        """Check if session has exceeded TTL."""
        return (time.time() - self.last_accessed) > ttl_seconds

    @property
    def turn_count(self) -> int:
        return len(self.history)

    @property
    def mean_ncs(self) -> float:
        if not self.cumulative_ncs:
            return 0.0
        return sum(self.cumulative_ncs) / len(self.cumulative_ncs)

    @property
    def mean_tce(self) -> float:
        if not self.cumulative_tce:
            return 0.0
        return sum(self.cumulative_tce) / len(self.cumulative_tce)


class SessionManager:
    """
    Thread-safe session manager for KAIROSYN multi-user deployments.

    Maintains an in-memory store of active sessions with automatic
    TTL-based expiration. For production deployments with multiple
    workers, replace with Redis-backed session store.
    """

    def __init__(
        self,
        session_ttl: float = 7200.0,    # 2 hours
        max_sessions: int = 1000,
        cleanup_interval: int = 300,    # Clean up every 5 minutes
    ):
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions
        self.cleanup_interval = cleanup_interval

        self._sessions: Dict[str, KairosynSession] = {}
        self._last_cleanup: float = time.time()

        logger.info(
            f"SessionManager initialized "
            f"(TTL={session_ttl}s, max_sessions={max_sessions})"
        )

    def create_session(self, session_id: Optional[str] = None) -> KairosynSession:
        """
        Create a new KAIROSYN session.

        Args:
            session_id: Optional custom session ID. Auto-generates UUID if None.

        Returns:
            New KairosynSession instance.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Evict oldest session if at capacity
        if len(self._sessions) >= self.max_sessions:
            self._evict_oldest()

        session = KairosynSession(session_id=session_id)
        self._sessions[session_id] = session

        logger.debug(f"Session created: {session_id}")
        return session

    def get_session(self, session_id: str) -> Optional[KairosynSession]:
        """
        Retrieve an existing session.

        Args:
            session_id: Session identifier.

        Returns:
            KairosynSession if found and not expired, else None.
        """
        self._maybe_cleanup()

        session = self._sessions.get(session_id)
        if session is None:
            return None

        if session.is_expired(self.session_ttl):
            self.delete_session(session_id)
            logger.debug(f"Session expired and removed: {session_id}")
            return None

        session.touch()
        return session

    def get_or_create(self, session_id: Optional[str]) -> Tuple[KairosynSession, bool]:
        """
        Get existing session or create a new one.

        Returns:
            Tuple of (session, created) where created=True if new session.
        """
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session, False

        new_session = self.create_session(session_id)
        return new_session, True

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and free its state tensors."""
        session = self._sessions.pop(session_id, None)
        if session:
            # Explicitly free tensors
            session.continuity_state = None
            session.narrative_state = None
            logger.debug(f"Session deleted: {session_id}")
            return True
        return False

    def update_session_state(
        self,
        session_id: str,
        continuity_state: Optional[torch.Tensor],
        narrative_state: Optional[torch.Tensor],
        ncs: float,
        tce: float,
    ):
        """
        Update the persistent state tensors for a session after a forward pass.

        Args:
            session_id: Session to update.
            continuity_state: New Continuity Engine self-state.
            narrative_state: New Arche-Tempus narrative state.
            ncs: Narrative Coherence Score for this turn.
            tce: Temporal Continuity Error for this turn.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return

        # Store tensors on CPU to free GPU memory
        if continuity_state is not None:
            session.continuity_state = continuity_state.detach().cpu()
        if narrative_state is not None:
            session.narrative_state = narrative_state.detach().cpu()

        session.cumulative_ncs.append(ncs)
        session.cumulative_tce.append(tce)
        session.touch()

    def restore_session_state(
        self,
        model,
        session: KairosynSession,
        device: torch.device,
    ):
        """
        Restore session state into the model's Continuity Engine
        and Arche-Tempus Drive before a forward pass.

        Args:
            model: KairosynModel instance
            session: Session whose state to restore
            device: Target device for tensors
        """
        if session.continuity_state is not None:
            model.continuity_engine.self_state = (
                session.continuity_state.to(device)
            )

        if session.narrative_state is not None:
            model.arche_tempus._narrative_state = (
                session.narrative_state.to(device)
            )

    def active_session_count(self) -> int:
        return len(self._sessions)

    def _evict_oldest(self):
        """Evict the oldest (least recently accessed) session."""
        if not self._sessions:
            return
        oldest_id = min(
            self._sessions,
            key=lambda sid: self._sessions[sid].last_accessed,
        )
        self.delete_session(oldest_id)
        logger.warning(f"Evicted oldest session: {oldest_id}")

    def _maybe_cleanup(self):
        """Periodically clean up expired sessions."""
        now = time.time()
        if (now - self._last_cleanup) < self.cleanup_interval:
            return

        expired = [
            sid for sid, s in self._sessions.items()
            if s.is_expired(self.session_ttl)
        ]
        for sid in expired:
            self.delete_session(sid)

        if expired:
            logger.info(f"Session cleanup: removed {len(expired)} expired sessions.")

        self._last_cleanup = now
