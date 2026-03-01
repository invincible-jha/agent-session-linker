"""Abstract base class for async session storage backends.

All concrete async backends must implement the operations defined here.
The raw payload exchanged with the backend is always a UTF-8 string
(typically JSON-encoded ``SessionState``).

Classes
-------
- AsyncStorageBackend  — abstract base for all async backends
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class AsyncStorageBackend(ABC):
    """Protocol for async reading and writing of raw session payloads.

    All methods are coroutines (``async def``).  Implementations should use
    ``asyncio.Lock`` for in-process thread safety where needed.
    """

    @abstractmethod
    async def save(self, session_id: str, payload: str) -> None:
        """Persist ``payload`` under ``session_id``.

        If a document already exists for ``session_id`` it is overwritten.

        Parameters
        ----------
        session_id:
            Unique session identifier used as the storage key.
        payload:
            UTF-8 string to persist (typically JSON).
        """

    @abstractmethod
    async def load(self, session_id: str) -> str:
        """Return the raw payload stored under ``session_id``.

        Parameters
        ----------
        session_id:
            The session to retrieve.

        Returns
        -------
        str
            The previously saved payload string.

        Raises
        ------
        KeyError
            If no entry exists for ``session_id``.
        """

    @abstractmethod
    async def list_sessions(self) -> Sequence[str]:
        """Return a sequence of all stored session IDs.

        Returns
        -------
        Sequence[str]
            All session IDs currently in this backend.  Order is
            implementation-defined.
        """

    @abstractmethod
    async def delete(self, session_id: str) -> bool:
        """Remove the entry for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to remove.

        Returns
        -------
        bool
            True if the entry existed and was deleted, False otherwise.
        """

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        """Return True if an entry for ``session_id`` exists.

        Parameters
        ----------
        session_id:
            The session identifier to test.

        Returns
        -------
        bool
        """


__all__ = ["AsyncStorageBackend"]
