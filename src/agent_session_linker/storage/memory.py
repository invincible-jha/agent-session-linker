"""In-memory storage backend.

Stores sessions in a plain Python dict.  All data is lost when the process
exits.  This backend is primarily useful for tests and local prototyping.

Classes
-------
- InMemoryBackend  — dict-backed ephemeral storage
"""
from __future__ import annotations

from agent_session_linker.storage.base import StorageBackend


class InMemoryBackend(StorageBackend):
    """Ephemeral, in-process storage backend backed by a Python dict.

    All stored sessions live only for the lifetime of this object.

    Parameters
    ----------
    initial_data:
        Optional pre-populated mapping of session IDs to raw payloads.
        A shallow copy is taken so the caller's dict is not mutated.
    """

    def __init__(self, initial_data: dict[str, str] | None = None) -> None:
        self._store: dict[str, str] = dict(initial_data or {})

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, session_id: str, payload: str) -> None:
        """Store ``payload`` under ``session_id``, overwriting if present."""
        self._store[session_id] = payload

    def load(self, session_id: str) -> str:
        """Return the payload for ``session_id``.

        Raises
        ------
        KeyError
            If ``session_id`` is not in the store.
        """
        try:
            return self._store[session_id]
        except KeyError:
            raise KeyError(f"Session {session_id!r} not found in InMemoryBackend.") from None

    def list(self) -> list[str]:
        """Return all stored session IDs in insertion order."""
        return list(self._store)

    def delete(self, session_id: str) -> None:
        """Remove ``session_id`` from the store.

        Raises
        ------
        KeyError
            If ``session_id`` is not in the store.
        """
        try:
            del self._store[session_id]
        except KeyError:
            raise KeyError(f"Session {session_id!r} not found in InMemoryBackend.") from None

    def exists(self, session_id: str) -> bool:
        """Return True if ``session_id`` is present."""
        return session_id in self._store

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all stored sessions."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"InMemoryBackend(sessions={len(self._store)})"
