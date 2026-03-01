"""Async Redis session storage backend — requires redis[asyncio] (guarded import).

Uses the same key structure as the synchronous ``RedisBackend`` for namespace
compatibility.

Classes
-------
- AsyncRedisBackend  — redis.asyncio-backed async session storage
"""

from __future__ import annotations

from typing import Sequence

from agent_session_linker.storage.async_base import AsyncStorageBackend

_REDIS_IMPORT_ERROR = (
    "AsyncRedisBackend requires the 'redis' package with asyncio support. "
    "Install it with: pip install redis  or  "
    "pip install 'agent-session-linker[async-redis]'"
)


class AsyncRedisBackend(AsyncStorageBackend):
    """Persists sessions in a Redis instance using ``redis.asyncio``.

    Each session is stored as a Redis string under the key
    ``<key_prefix><session_id>``.

    Parameters
    ----------
    host:
        Redis server hostname.  Defaults to ``"localhost"``.
    port:
        Redis server port.  Defaults to ``6379``.
    db:
        Redis logical database index.  Defaults to ``0``.
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
            import redis.asyncio as redis_asyncio
        except ImportError as exc:
            raise ImportError(_REDIS_IMPORT_ERROR) from exc

        if url is not None:
            self._client = redis_asyncio.Redis.from_url(url, decode_responses=True)
        else:
            self._client = redis_asyncio.Redis(
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
        """Return the full Redis key for ``session_id``."""
        return f"{self._key_prefix}{session_id}"

    # ------------------------------------------------------------------
    # AsyncStorageBackend interface
    # ------------------------------------------------------------------

    async def save(self, session_id: str, payload: str) -> None:
        """Write ``payload`` to Redis under the prefixed key."""
        key = self._key(session_id)
        if self._ttl_seconds is not None:
            await self._client.setex(key, self._ttl_seconds, payload)
        else:
            await self._client.set(key, payload)

    async def load(self, session_id: str) -> str:
        """Return the payload for ``session_id``.

        Raises
        ------
        KeyError
            If no key exists for ``session_id``.
        """
        value: str | None = await self._client.get(self._key(session_id))
        if value is None:
            raise KeyError(f"Session {session_id!r} not found in AsyncRedisBackend.")
        return str(value)

    async def list_sessions(self) -> Sequence[str]:
        """Return all session IDs stored under the configured prefix.

        Uses Redis SCAN to avoid blocking the server.
        """
        prefix = self._key_prefix
        prefix_len = len(prefix)
        session_ids: list[str] = []
        cursor: int = 0
        while True:
            cursor, keys = await self._client.scan(
                cursor=cursor, match=f"{prefix}*", count=100
            )
            for key in keys:
                session_ids.append(str(key)[prefix_len:])
            if cursor == 0:
                break
        return session_ids

    async def delete(self, session_id: str) -> bool:
        """Remove the key for ``session_id``.

        Returns
        -------
        bool
            True if the session existed and was deleted, False otherwise.
        """
        deleted: int = await self._client.delete(self._key(session_id))
        return deleted > 0

    async def exists(self, session_id: str) -> bool:
        """Return True if the key for ``session_id`` exists in Redis."""
        return bool(await self._client.exists(self._key(session_id)))

    def __repr__(self) -> str:
        return (
            f"AsyncRedisBackend(key_prefix={self._key_prefix!r}, "
            f"ttl_seconds={self._ttl_seconds!r})"
        )


__all__ = ["AsyncRedisBackend"]
