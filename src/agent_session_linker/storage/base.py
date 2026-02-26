"""Abstract base class for session storage backends.

All concrete backends must implement the five operations defined here.
The raw payload exchanged with the backend is always a UTF-8 string
(typically JSON-encoded ``SessionState``).

Classes
-------
- StorageBackend  — abstract base for all backends
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """Protocol for reading and writing raw session payloads.

    Backend implementations must be safe for sequential (single-threaded)
    use.  Thread-safety is the responsibility of the caller when used from
    concurrent code.
    """

    @abstractmethod
    def save(self, session_id: str, payload: str) -> None:
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
    def load(self, session_id: str) -> str:
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
    def list(self) -> list[str]:
        """Return a list of all stored session IDs.

        Returns
        -------
        list[str]
            All session IDs currently in this backend.  Order is
            implementation-defined.
        """

    @abstractmethod
    def delete(self, session_id: str) -> None:
        """Remove the entry for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to remove.

        Raises
        ------
        KeyError
            If no entry exists for ``session_id``.
        """

    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Return True if an entry for ``session_id`` exists.

        Parameters
        ----------
        session_id:
            The session identifier to test.

        Returns
        -------
        bool
        """
