"""SQLite storage backend.

Stores sessions in a single SQLite database file using the Python standard
library ``sqlite3`` module — no third-party dependencies required.

Classes
-------
- SQLiteBackend  — SQLite-backed session storage
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from agent_session_linker.storage.base import StorageBackend

_DEFAULT_DB_PATH: Path = Path.home() / ".agent-sessions" / "sessions.db"
_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    saved_at   TEXT NOT NULL DEFAULT (datetime('now'))
)
"""
_UPSERT_SQL = """
INSERT INTO sessions (session_id, payload, saved_at)
VALUES (?, ?, datetime('now'))
ON CONFLICT(session_id) DO UPDATE SET
    payload  = excluded.payload,
    saved_at = excluded.saved_at
"""


class SQLiteBackend(StorageBackend):
    """Persists sessions in a local SQLite database.

    Each session occupies one row with the ``session_id`` as the primary key
    and the raw JSON payload stored as TEXT.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Defaults to
        ``~/.agent-sessions/sessions.db``.  The parent directory and table
        are created automatically on first use.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path: Path = (
            Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        )
        self._initialized = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Open (or reopen) a connection and ensure the table exists.

        Returns
        -------
        sqlite3.Connection
            A ready-to-use connection with row_factory set.
        """
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
        return conn

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, session_id: str, payload: str) -> None:
        """Upsert ``payload`` for ``session_id`` in the sessions table.

        Parameters
        ----------
        session_id:
            Storage key.
        payload:
            UTF-8 JSON string.
        """
        with self._get_connection() as conn:
            conn.execute(_UPSERT_SQL, (session_id, payload))

    def load(self, session_id: str) -> str:
        """Return the payload row for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to retrieve.

        Returns
        -------
        str
            Raw payload string.

        Raises
        ------
        KeyError
            If no row exists for ``session_id``.
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT payload FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Session {session_id!r} not found in SQLiteBackend.")
        return str(row["payload"])

    def list(self) -> list[str]:
        """Return all session IDs stored in the database.

        Returns
        -------
        list[str]
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT session_id FROM sessions ORDER BY saved_at DESC"
            ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def delete(self, session_id: str) -> None:
        """Remove the row for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to delete.

        Raises
        ------
        KeyError
            If no row exists for ``session_id``.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
        if cursor.rowcount == 0:
            raise KeyError(f"Session {session_id!r} not found in SQLiteBackend.")

    def exists(self, session_id: str) -> bool:
        """Return True if a row for ``session_id`` exists."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def __repr__(self) -> str:
        return f"SQLiteBackend(db_path={str(self._db_path)!r})"
