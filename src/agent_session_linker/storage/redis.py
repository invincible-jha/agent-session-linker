"""Redis storage backend.

Import-guarded: ``redis`` is an optional dependency.  Attempting to
instantiate ``RedisBackend`` without the ``redis`` package installed will
raise ``ImportError`` with a helpful message.

Classes
-------
- RedisBackend  — Redis key-value session storage
"""
from __future__ import annotations

from agent_session_linker.storage.base import StorageBackend

_REDIS_IMPORT_ERROR = (
    "The 'redis' package is required for RedisBackend. "
    "Install it with: pip install redis"
)


class RedisBackend(StorageBackend):
    """Persists sessions in a Redis instance.

    Each session is stored as a Redis string under the key
    ``<key_prefix><session_id>``.

    Parameters
    ----------
    host:
        Redis server hostname. Defaults to ``"localhost"``.
    port:
        Redis server port. Defaults to ``6379``.
    db:
        Redis logical database index. Defaults to ``0``.
    password:
        Optional authentication password.
    key_prefix:
        String prepended to all session keys.  Defaults to
        ``"agent_session:"``.
    ttl_seconds:
        Optional TTL for session keys.  When ``None`` (default) keys
        persist until explicitly deleted.
    url:
        If supplied, overrides host/port/db/password and is used as a
        Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        key_prefix: str = "agent_session:",
        ttl_seconds: int | None = None,
        url: str | None = None,
    ) -> None:
        try:
            import redis as redis_module  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(_REDIS_IMPORT_ERROR) from exc

        if url is not None:
            self._client = redis_module.Redis.from_url(url)
        else:
            self._client = redis_module.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=True,
            )
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _key(self, session_id: str) -> str:
        """Return the full Redis key for ``session_id``.

        Parameters
        ----------
        session_id:
            Raw session identifier.

        Returns
        -------
        str
            Prefixed Redis key.
        """
        return f"{self._key_prefix}{session_id}"

    # ------------------------------------------------------------------
    # StorageBackend interface
    # ------------------------------------------------------------------

    def save(self, session_id: str, payload: str) -> None:
        """Write ``payload`` to Redis under the prefixed key.

        Parameters
        ----------
        session_id:
            Storage key.
        payload:
            UTF-8 string payload.
        """
        key = self._key(session_id)
        if self._ttl_seconds is not None:
            self._client.setex(key, self._ttl_seconds, payload)
        else:
            self._client.set(key, payload)

    def load(self, session_id: str) -> str:
        """Return the payload for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to retrieve.

        Returns
        -------
        str
            Stored payload.

        Raises
        ------
        KeyError
            If no key exists for ``session_id``.
        """
        value = self._client.get(self._key(session_id))
        if value is None:
            raise KeyError(f"Session {session_id!r} not found in RedisBackend.")
        return str(value)

    def list(self) -> list[str]:
        """Return all session IDs stored under the configured prefix.

        Uses a Redis SCAN to avoid blocking the server.

        Returns
        -------
        list[str]
            Session IDs with the prefix stripped.
        """
        prefix = self._key_prefix
        prefix_len = len(prefix)
        session_ids: list[str] = []
        cursor: int = 0
        while True:
            cursor, keys = self._client.scan(cursor=cursor, match=f"{prefix}*", count=100)
            for key in keys:
                session_ids.append(str(key)[prefix_len:])
            if cursor == 0:
                break
        return session_ids

    def delete(self, session_id: str) -> None:
        """Remove the key for ``session_id``.

        Parameters
        ----------
        session_id:
            The session to delete.

        Raises
        ------
        KeyError
            If no key exists for ``session_id``.
        """
        deleted = self._client.delete(self._key(session_id))
        if deleted == 0:
            raise KeyError(f"Session {session_id!r} not found in RedisBackend.")

    def exists(self, session_id: str) -> bool:
        """Return True if the key for ``session_id`` exists in Redis."""
        return bool(self._client.exists(self._key(session_id)))

    def __repr__(self) -> str:
        return (
            f"RedisBackend(key_prefix={self._key_prefix!r}, "
            f"ttl_seconds={self._ttl_seconds!r})"
        )
