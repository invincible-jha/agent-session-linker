"""Async in-memory session storage backend.

Stores sessions in a plain Python dict guarded by ``asyncio.Lock``.
All data is lost when the process exits.  This backend is primarily
useful for tests and local prototyping.

Classes
-------
- AsyncInMemoryBackend  — dict-backed ephemeral async storage
"""

from __future__ import annotations

import asyncio
from typing import Sequence

from agent_session_linker.storage.async_base import AsyncStorageBackend


class AsyncInMemoryBackend(AsyncStorageBackend):
    """Ephemeral async in-process storage backend backed by a Python dict.

    An ``asyncio.Lock`` guards all mutations so that concurrent coroutines
    do not race on the internal dict.

    Parameters
    ----------
    initial_data:
        Optional pre-populated mapping of session IDs to raw payloads.
        A shallow copy is taken so the caller's dict is not mutated.
    """

    def __init__(self, initial_data: dict[str, str] | None = None) -> None:
        self._store: dict[str, str] = dict(initial_data or {})
        self._lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # AsyncStorageBackend interface
    # ------------------------------------------------------------------

    async def save(self, session_id: str, payload: str) -> None:
        """Store ``payload`` under ``session_id``, overwriting if present."""
        async with self._lock:
            self._store[session_id] = payload

    async def load(self, session_id: str) -> str:
        """Return the payload for ``session_id``.

        Raises
        ------
        KeyError
            If ``session_id`` is not in the store.
        """
        async with self._lock:
            if session_id not in self._store:
                raise KeyError(
                    f"Session {session_id!r} not found in AsyncInMemoryBackend."
                )
            return self._store[session_id]

    async def list_sessions(self) -> Sequence[str]:
        """Return all stored session IDs in insertion order."""
        async with self._lock:
            return list(self._store)

    async def delete(self, session_id: str) -> bool:
        """Remove ``session_id`` from the store.

        Returns
        -------
        bool
            True if the session existed and was deleted, False otherwise.
        """
        async with self._lock:
            if session_id not in self._store:
                return False
            del self._store[session_id]
            return True

    async def exists(self, session_id: str) -> bool:
        """Return True if ``session_id`` is present."""
        async with self._lock:
            return session_id in self._store

    # ------------------------------------------------------------------
    # Extras
    # ------------------------------------------------------------------

    async def clear(self) -> None:
        """Remove all stored sessions."""
        async with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return f"AsyncInMemoryBackend(sessions={len(self._store)})"


__all__ = ["AsyncInMemoryBackend"]
