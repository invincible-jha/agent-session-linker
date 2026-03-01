"""Async SQLite session storage backend — requires aiosqlite (guarded import).

Uses the same schema as the synchronous ``SQLiteBackend`` so both backends
can operate against the same database file.

Classes
-------
- AsyncSQLiteBackend  — aiosqlite-backed async session storage
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from agent_session_linker.storage.async_base import AsyncStorageBackend

_AIOSQLITE_IMPORT_ERROR = (
    "AsyncSQLiteBackend requires the 'aiosqlite' package. "
    "Install it with: pip install aiosqlite  or  pip install 'agent-session-linker[async]'"
)

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


class AsyncSQLiteBackend(AsyncStorageBackend):
    """Persists sessions in a local SQLite database using aiosqlite.

    Parameters
    ----------
    db_path:
        Path to the SQLite file.  Defaults to
        ``~/.agent-sessions/sessions.db``.  The parent directory and table
        are created automatically on first use.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        try:
            import aiosqlite as _aiosqlite  # noqa: F401
        except ImportError as exc:
            raise ImportError(_AIOSQLITE_IMPORT_ERROR) from exc

        self._db_path: Path = (
            Path(db_path) if db_path is not None else _DEFAULT_DB_PATH
        )
        self._schema_initialised = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _ensure_schema(self) -> None:
        """Create the sessions table on first use."""
        if self._schema_initialised:
            return
        import aiosqlite

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self._db_path)) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(_CREATE_TABLE_SQL)
            await conn.commit()
        self._schema_initialised = True

    # ------------------------------------------------------------------
    # AsyncStorageBackend interface
    # ------------------------------------------------------------------

    async def save(self, session_id: str, payload: str) -> None:
        """Upsert ``payload`` for ``session_id`` in the sessions table."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(str(self._db_path)) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(_UPSERT_SQL, (session_id, payload))
            await conn.commit()

    async def load(self, session_id: str) -> str:
        """Return the payload row for ``session_id``.

        Raises
        ------
        KeyError
            If no row exists for ``session_id``.
        """
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(str(self._db_path)) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT payload FROM sessions WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            raise KeyError(f"Session {session_id!r} not found in AsyncSQLiteBackend.")
        return str(row["payload"])

    async def list_sessions(self) -> Sequence[str]:
        """Return all session IDs stored in the database."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(str(self._db_path)) as conn:
            async with conn.execute(
                "SELECT session_id FROM sessions ORDER BY saved_at DESC"
            ) as cursor:
                rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]

    async def delete(self, session_id: str) -> bool:
        """Remove the row for ``session_id``.

        Returns
        -------
        bool
            True if the session existed and was deleted, False otherwise.
        """
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(str(self._db_path)) as conn:
            await conn.execute("PRAGMA journal_mode=WAL")
            cursor = await conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            await conn.commit()
        return cursor.rowcount > 0

    async def exists(self, session_id: str) -> bool:
        """Return True if a row for ``session_id`` exists."""
        import aiosqlite

        await self._ensure_schema()
        async with aiosqlite.connect(str(self._db_path)) as conn:
            async with conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
        return row is not None

    def __repr__(self) -> str:
        return f"AsyncSQLiteBackend(db_path={str(self._db_path)!r})"


__all__ = ["AsyncSQLiteBackend"]
