"""Storage backend subpackage.

All sync backends implement the ``StorageBackend`` ABC.
All async backends implement the ``AsyncStorageBackend`` ABC.
Import only what you need; optional backends guard their third-party
imports so that the package remains installable without those extras.

Public surface (sync)
---------------------
- StorageBackend    — abstract base class
- FilesystemBackend — persist sessions as JSON files
- SQLiteBackend     — persist sessions in a local SQLite database
- InMemoryBackend   — in-process dict (useful for testing)
- RedisBackend      — Redis backend (requires ``redis`` package)
- S3Backend         — AWS S3 backend  (requires ``boto3`` package)

Public surface (async)
----------------------
- AsyncStorageBackend  — abstract base class for async backends
- AsyncInMemoryBackend — async dict-based backend with asyncio.Lock
- AsyncSQLiteBackend   — async aiosqlite backend (requires aiosqlite)
- AsyncRedisBackend    — async redis.asyncio backend (requires redis>=5)
"""
from __future__ import annotations

from agent_session_linker.storage.async_base import AsyncStorageBackend
from agent_session_linker.storage.async_memory import AsyncInMemoryBackend
from agent_session_linker.storage.base import StorageBackend
from agent_session_linker.storage.filesystem import FilesystemBackend
from agent_session_linker.storage.memory import InMemoryBackend
from agent_session_linker.storage.sqlite import SQLiteBackend

__all__ = [
    "FilesystemBackend",
    "InMemoryBackend",
    "SQLiteBackend",
    "StorageBackend",
    "AsyncStorageBackend",
    "AsyncInMemoryBackend",
]

# AsyncSQLiteBackend — guarded by the aiosqlite dependency
try:
    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    __all__ = [*__all__, "AsyncSQLiteBackend"]
except ImportError:
    pass

# AsyncRedisBackend — guarded by the redis[asyncio] dependency
try:
    from agent_session_linker.storage.async_redis import AsyncRedisBackend

    __all__ = [*__all__, "AsyncRedisBackend"]
except ImportError:
    pass
