"""Filesystem storage backend.

Persists each session as an individual JSON file under a configurable
directory.  Defaults to ``~/.agent-sessions/``.

Classes
-------
- FilesystemBackend  — JSON-file-per-session storage
"""
from __future__ import annotations

import os
from pathlib import Path

from agent_session_linker.storage.base import StorageBackend

_DEFAULT_STORAGE_DIR: Path = Path.home() / ".agent-sessions"
_FILE_EXTENSION = ".json"


class FilesystemBackend(StorageBackend):
    """Stores sessions as individual JSON files.

    Each session is stored as ``<storage_dir>/<session_id>.json``.

    Parameters
    ----------
    storage_dir:
        Root directory for session files.  Defaults to
        ``~/.agent-sessions/``.  Created on first use if absent.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        self._storage_dir: Path = (
            Path(storage_dir) if storage_dir is not None else _DEFAULT_STORAGE_DIR
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create the storage directory tree if it does not already exist."""
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> Path:
        """Return the file path for ``session_id``.

        Parameters
        ----------
        session_id:
            The session identifier.

        Returns
        -------
        Path
            Absolute path to the JSON file.
        """
        # Guard against path traversal attacks.
        safe_name = os.path.basename(session_id)
        return self._storage_dir / f"{safe_name}{_FILE_EXTENSION}"

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, session_id: str, payload: str) -> None:
        """Write ``payload`` to ``<storage_dir>/<session_id>.json``.

        The directory is created if it does not yet exist.

        Parameters
        ----------
        session_id:
            Storage key.
        payload:
            UTF-8 string to write.
        """
        self._ensure_dir()
        path = self._path_for(session_id)
        path.write_text(payload, encoding="utf-8")

    def load(self, session_id: str) -> str:
        """Read and return the payload for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to retrieve.

        Returns
        -------
        str
            File contents as a UTF-8 string.

        Raises
        ------
        KeyError
            If the file does not exist.
        """
        path = self._path_for(session_id)
        if not path.exists():
            raise KeyError(
                f"Session {session_id!r} not found at {path}"
            )
        return path.read_text(encoding="utf-8")

    def list(self) -> list[str]:
        """Return all session IDs present in the storage directory.

        Returns
        -------
        list[str]
            Session IDs derived from file stems.  Empty list if the
            directory does not yet exist.
        """
        if not self._storage_dir.exists():
            return []
        return [
            path.stem
            for path in self._storage_dir.glob(f"*{_FILE_EXTENSION}")
            if path.is_file()
        ]

    def delete(self, session_id: str) -> None:
        """Remove the file for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to delete.

        Raises
        ------
        KeyError
            If the file does not exist.
        """
        path = self._path_for(session_id)
        if not path.exists():
            raise KeyError(
                f"Session {session_id!r} not found at {path}"
            )
        path.unlink()

    def exists(self, session_id: str) -> bool:
        """Return True if the file for ``session_id`` exists."""
        return self._path_for(session_id).exists()

    def __repr__(self) -> str:
        return f"FilesystemBackend(storage_dir={str(self._storage_dir)!r})"
