"""Convenience API for agent-session-linker — 3-line quickstart.

Example
-------
::

    from agent_session_linker import Session
    session = Session()
    session.save()
    loaded = session.load(session.session_id)

"""
from __future__ import annotations

from typing import Any


class Session:
    """Zero-config session manager for the 80% use case.

    Uses in-memory storage so no filesystem configuration is required.
    State is created immediately and can be saved/loaded by session ID.

    Parameters
    ----------
    agent_id:
        Agent identifier for the session.

    Example
    -------
    ::

        from agent_session_linker import Session
        session = Session()
        session.save()
        restored = session.load(session.session_id)
        print(restored.session_id)
    """

    def __init__(self, agent_id: str = "default-agent") -> None:
        from agent_session_linker.storage.memory import InMemoryBackend
        from agent_session_linker.session.manager import SessionManager

        self._backend = InMemoryBackend()
        self._manager = SessionManager(
            backend=self._backend,
            default_agent_id=agent_id,
        )
        self._state = self._manager.create_session(agent_id=agent_id)
        self.agent_id = agent_id

    @property
    def session_id(self) -> str:
        """The current session's unique identifier."""
        return self._state.session_id

    @property
    def state(self) -> Any:
        """The underlying SessionState object."""
        return self._state

    def save(self) -> str:
        """Persist the current session state.

        Returns
        -------
        str
            The session ID of the saved session.
        """
        self._manager.save_session(self._state)
        return self._state.session_id

    def load(self, session_id: str) -> Any:
        """Load a session by its ID.

        Parameters
        ----------
        session_id:
            ID of the session to load.

        Returns
        -------
        SessionState
            The loaded session state.

        Raises
        ------
        SessionNotFoundError
            If no session with the given ID exists.
        """
        return self._manager.load_session(session_id)

    def add_context(self, key: str, value: str) -> None:
        """Add a key-value pair to the session preferences.

        Parameters
        ----------
        key:
            Context key.
        value:
            Context value string.
        """
        self._state.preferences[key] = value

    def __repr__(self) -> str:
        return f"Session(id={self.session_id!r}, agent_id={self.agent_id!r})"
