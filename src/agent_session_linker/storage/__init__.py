"""Storage backend subpackage.

All backends implement the ``StorageBackend`` ABC.  Import only what you
need; optional backends (Redis, S3) guard their third-party imports so
that the package remains installable without those extras.

Public surface
--------------
- StorageBackend    — abstract base class
- FilesystemBackend — persist sessions as JSON files
- SQLiteBackend     — persist sessions in a local SQLite database
- InMemoryBackend   — in-process dict (useful for testing)
- RedisBackend      — Redis backend (requires ``redis`` package)
- S3Backend         — AWS S3 backend  (requires ``boto3`` package)
"""
from __future__ import annotations

from agent_session_linker.storage.base import StorageBackend
from agent_session_linker.storage.filesystem import FilesystemBackend
from agent_session_linker.storage.memory import InMemoryBackend
from agent_session_linker.storage.sqlite import SQLiteBackend

__all__ = [
    "FilesystemBackend",
    "InMemoryBackend",
    "SQLiteBackend",
    "StorageBackend",
]
