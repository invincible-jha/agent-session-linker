"""Session lifecycle management.

Provides ``SessionManager``, the primary facade for creating, persisting,
loading, listing, and deleting sessions via a pluggable storage backend.

Classes
-------
- SessionManager  — CRUD facade over a StorageBackend
"""
from __future__ import annotations

from datetime import datetime, timezone

from agent_session_linker.session.serializer import SessionSerializer
from agent_session_linker.session.state import SessionState
from agent_session_linker.storage.base import StorageBackend


class SessionNotFoundError(KeyError):
    """Raised when a requested session does not exist in the backend."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session {session_id!r} not found.")


class SessionManager:
    """Create, save, load, list, and delete sessions.

    All persistence operations are delegated to the supplied
    ``StorageBackend``.  Serialization is handled by ``SessionSerializer``.

    Parameters
    ----------
    backend:
        The storage backend to use for persistence.
    serializer:
        Optional custom serializer.  Defaults to a ``SessionSerializer``
        with checksum validation enabled.
    default_agent_id:
        Agent identifier used when creating new sessions without an explicit
        ``agent_id``.
    """

    def __init__(
        self,
        backend: StorageBackend,
        serializer: SessionSerializer | None = None,
        default_agent_id: str = "default",
    ) -> None:
        self._backend = backend
        self._serializer = serializer or SessionSerializer()
        self._default_agent_id = default_agent_id

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    def create_session(
        self,
        agent_id: str | None = None,
        *,
        parent_session_id: str | None = None,
        preferences: dict[str, str] | None = None,
    ) -> SessionState:
        """Create and return a new in-memory session (not yet persisted).

        Call ``save_session`` to persist the returned state.

        Parameters
        ----------
        agent_id:
            Override the manager's ``default_agent_id`` for this session.
        parent_session_id:
            Link this session to a prior session for chain navigation.
        preferences:
            Initial preference key-value pairs.

        Returns
        -------
        SessionState
            A freshly initialised, unpersisted session.
        """
        state = SessionState(
            agent_id=agent_id or self._default_agent_id,
            parent_session_id=parent_session_id,
            preferences=preferences or {},
        )
        return state

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_session(self, state: SessionState) -> str:
        """Persist a session to the storage backend.

        ``updated_at`` is refreshed and a fresh checksum is embedded before
        serialisation.

        Parameters
        ----------
        state:
            The session state to persist.

        Returns
        -------
        str
            The ``session_id`` under which the data was saved.
        """
        state.updated_at = datetime.now(timezone.utc)
        raw = self._serializer.to_json(state)
        self._backend.save(state.session_id, raw)
        return state.session_id

    def load_session(self, session_id: str) -> SessionState:
        """Load and return a session from the storage backend.

        Parameters
        ----------
        session_id:
            The session to retrieve.

        Returns
        -------
        SessionState
            The deserialised and checksum-verified session.

        Raises
        ------
        SessionNotFoundError
            If no session with ``session_id`` exists in the backend.
        """
        if not self._backend.exists(session_id):
            raise SessionNotFoundError(session_id)
        raw = self._backend.load(session_id)
        return self._serializer.from_json(raw)

    def delete_session(self, session_id: str) -> None:
        """Remove a session from the storage backend.

        Parameters
        ----------
        session_id:
            The session to delete.

        Raises
        ------
        SessionNotFoundError
            If no session with ``session_id`` exists.
        """
        if not self._backend.exists(session_id):
            raise SessionNotFoundError(session_id)
        self._backend.delete(session_id)

    def session_exists(self, session_id: str) -> bool:
        """Return True if ``session_id`` is present in the backend.

        Parameters
        ----------
        session_id:
            The session identifier to check.

        Returns
        -------
        bool
        """
        return self._backend.exists(session_id)

    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[str]:
        """Return the IDs of all sessions in the backend.

        Returns
        -------
        list[str]
            Sorted list of session IDs.
        """
        return sorted(self._backend.list())

    def list_sessions_for_agent(self, agent_id: str) -> list[str]:
        """Return session IDs belonging to ``agent_id``.

        This performs a full scan: each session is loaded and filtered.
        For large collections prefer a backend that supports indexed queries.

        Parameters
        ----------
        agent_id:
            The agent identifier to filter by.

        Returns
        -------
        list[str]
            Session IDs (sorted) whose ``agent_id`` matches.
        """
        matching: list[str] = []
        for session_id in self._backend.list():
            try:
                state = self.load_session(session_id)
                if state.agent_id == agent_id:
                    matching.append(session_id)
            except Exception:  # noqa: BLE001 — skip corrupt / unreadable entries
                continue
        return sorted(matching)

    # ------------------------------------------------------------------
    # Continuation
    # ------------------------------------------------------------------

    def continue_session(self, parent_session_id: str) -> SessionState:
        """Load ``parent_session_id`` and create a child continuation session.

        The child session has ``parent_session_id`` set and inherits the
        parent's ``agent_id``, ``preferences``, active tasks, and entities.

        Parameters
        ----------
        parent_session_id:
            The session to continue from.

        Returns
        -------
        SessionState
            A new unpersisted session linked to the parent.
        """
        parent = self.load_session(parent_session_id)
        child = SessionState(
            agent_id=parent.agent_id,
            parent_session_id=parent_session_id,
            preferences=dict(parent.preferences),
            entities=list(parent.entities),
            tasks=[t for t in parent.tasks if t.status.value in ("pending", "in_progress")],
        )
        return child

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, object]:
        """Return aggregate statistics across all sessions in the backend.

        The returned dict includes:
        - ``total_sessions`` — count of stored sessions
        - ``total_segments`` — sum of segment counts
        - ``total_tokens`` — sum of token counts
        - ``total_tasks`` — sum of task counts
        - ``total_entities`` — sum of entity counts
        - ``total_cost_usd`` — summed cost across all sessions
        - ``agents`` — list of unique agent IDs

        Returns
        -------
        dict[str, object]
            Aggregate statistics dictionary.
        """
        total_sessions = 0
        total_segments = 0
        total_tokens = 0
        total_tasks = 0
        total_entities = 0
        total_cost: float = 0.0
        agent_ids: set[str] = set()

        for session_id in self._backend.list():
            try:
                state = self.load_session(session_id)
                total_sessions += 1
                total_segments += len(state.segments)
                total_tokens += state.total_tokens()
                total_tasks += len(state.tasks)
                total_entities += len(state.entities)
                total_cost += state.total_cost_usd
                agent_ids.add(state.agent_id)
            except Exception:  # noqa: BLE001
                continue

        return {
            "total_sessions": total_sessions,
            "total_segments": total_segments,
            "total_tokens": total_tokens,
            "total_tasks": total_tasks,
            "total_entities": total_entities,
            "total_cost_usd": round(total_cost, 6),
            "agents": sorted(agent_ids),
        }
