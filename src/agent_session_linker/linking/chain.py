"""Ordered session chain for linear conversation continuations.

Represents a linearly ordered sequence of sessions where each session is a
direct continuation of the previous one.  Provides helpers for appending,
iterating, and extracting aggregated context from the most recent sessions
in the chain.

Classes
-------
- SessionChain  — ordered chain of session IDs with context extraction
"""
from __future__ import annotations

from agent_session_linker.session.manager import SessionManager
from agent_session_linker.session.state import ContextSegment, SessionState


class SessionChain:
    """An ordered sequence of session IDs representing a conversation chain.

    Sessions are stored by ID only; ``SessionState`` objects are loaded on
    demand via the supplied ``SessionManager``.

    Parameters
    ----------
    manager:
        The session manager used to load session states when needed.
    initial_session_ids:
        Optional initial sequence of session IDs (oldest first).
    """

    def __init__(
        self,
        manager: SessionManager,
        initial_session_ids: list[str] | None = None,
    ) -> None:
        self._manager = manager
        self._chain: list[str] = list(initial_session_ids or [])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, session_id: str) -> None:
        """Append a session ID to the end of the chain.

        Parameters
        ----------
        session_id:
            The session to add.  Duplicate IDs are allowed (the chain
            records insertion order faithfully).
        """
        self._chain.append(session_id)

    def prepend(self, session_id: str) -> None:
        """Prepend a session ID at the beginning of the chain.

        Parameters
        ----------
        session_id:
            The session to add at position 0.
        """
        self._chain.insert(0, session_id)

    def remove(self, session_id: str) -> None:
        """Remove the first occurrence of ``session_id`` from the chain.

        Parameters
        ----------
        session_id:
            Session to remove.

        Raises
        ------
        ValueError
            If ``session_id`` is not in the chain.
        """
        self._chain.remove(session_id)

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get_chain(self) -> list[str]:
        """Return the ordered list of session IDs (oldest first).

        Returns
        -------
        list[str]
            A shallow copy of the internal chain.
        """
        return list(self._chain)

    def get_sessions(self) -> list[SessionState]:
        """Load and return all sessions in chain order.

        Sessions that cannot be loaded (e.g. deleted from the backend) are
        silently skipped.

        Returns
        -------
        list[SessionState]
            Loaded sessions in chain order (oldest first).
        """
        states: list[SessionState] = []
        for session_id in self._chain:
            try:
                states.append(self._manager.load_session(session_id))
            except Exception:  # noqa: BLE001
                continue
        return states

    def get_context_from_chain(self, n_recent: int) -> str:
        """Build a formatted context string from the ``n_recent`` newest sessions.

        Loads the last ``n_recent`` sessions in the chain and concatenates
        their segments into a single readable string, separated by session
        boundaries.  Sessions that cannot be loaded are skipped.

        Parameters
        ----------
        n_recent:
            Number of most-recent sessions to include.  Must be >= 1.

        Returns
        -------
        str
            Formatted context string.  Empty string when the chain is empty
            or no sessions could be loaded.

        Raises
        ------
        ValueError
            If ``n_recent < 1``.
        """
        if n_recent < 1:
            raise ValueError(f"n_recent must be >= 1, got {n_recent!r}.")

        recent_ids = self._chain[-n_recent:]
        if not recent_ids:
            return ""

        parts: list[str] = []

        for session_id in recent_ids:
            try:
                session = self._manager.load_session(session_id)
            except Exception:  # noqa: BLE001
                continue

            if not session.segments:
                continue

            parts.append(f"[Session {session.session_id[:8]}]")
            parts.append(self._format_segments(session.segments))

        return "\n\n".join(parts)

    def get_all_segments(self, n_recent: int | None = None) -> list[ContextSegment]:
        """Return all context segments from the chain (or the ``n_recent`` newest sessions).

        Parameters
        ----------
        n_recent:
            When provided, only the last ``n_recent`` sessions are included.
            When None, all sessions in the chain are included.

        Returns
        -------
        list[ContextSegment]
            Segments in document order (oldest session first, oldest segment
            within each session first).
        """
        session_ids = self._chain if n_recent is None else self._chain[-n_recent:]
        segments: list[ContextSegment] = []
        for session_id in session_ids:
            try:
                session = self._manager.load_session(session_id)
                segments.extend(session.segments)
            except Exception:  # noqa: BLE001
                continue
        return segments

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._chain)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._chain

    def __repr__(self) -> str:
        return f"SessionChain(length={len(self._chain)})"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_segments(segments: list[ContextSegment]) -> str:
        """Render a list of segments as a readable string.

        Parameters
        ----------
        segments:
            Segments to render.

        Returns
        -------
        str
            Formatted string with role labels.
        """
        lines: list[str] = []
        for segment in segments:
            role_label = segment.role.upper()
            lines.append(f"{role_label}: {segment.content}")
        return "\n".join(lines)
