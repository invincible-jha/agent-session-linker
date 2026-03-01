"""Tests for async storage backends in agent-session-linker.

Coverage:
- AsyncInMemoryBackend: save/load/delete/list_sessions/exists (6 tests)
- AsyncSQLiteBackend: same operations (6 tests, skipped if aiosqlite absent)
- Concurrent operations (4 tests)
- KeyError on missing session (2 tests)
- Backward compatibility: sync API unchanged (3 tests)
Total: 21+ tests
"""

from __future__ import annotations

import asyncio
import importlib

import pytest

from agent_session_linker.storage.async_memory import AsyncInMemoryBackend

_aiosqlite_available = importlib.util.find_spec("aiosqlite") is not None

# ---------------------------------------------------------------------------
# AsyncInMemoryBackend — core CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_inmemory_save_and_load() -> None:
    """save stores a payload; load retrieves it by session_id."""
    backend = AsyncInMemoryBackend()
    await backend.save("sess1", '{"key": "value"}')
    payload = await backend.load("sess1")
    assert payload == '{"key": "value"}'


@pytest.mark.asyncio
async def test_async_inmemory_load_missing_raises_keyerror() -> None:
    """load raises KeyError when the session_id is not present."""
    backend = AsyncInMemoryBackend()
    with pytest.raises(KeyError):
        await backend.load("ghost")


@pytest.mark.asyncio
async def test_async_inmemory_delete_existing_returns_true() -> None:
    """delete returns True and removes the entry."""
    backend = AsyncInMemoryBackend()
    await backend.save("del_sess", "data")
    result = await backend.delete("del_sess")
    assert result is True
    assert await backend.exists("del_sess") is False


@pytest.mark.asyncio
async def test_async_inmemory_delete_missing_returns_false() -> None:
    """delete returns False when the session does not exist."""
    backend = AsyncInMemoryBackend()
    result = await backend.delete("absent")
    assert result is False


@pytest.mark.asyncio
async def test_async_inmemory_list_sessions() -> None:
    """list_sessions returns all stored session IDs."""
    backend = AsyncInMemoryBackend()
    ids = {"a1", "b2", "c3"}
    for session_id in ids:
        await backend.save(session_id, "payload")
    sessions = await backend.list_sessions()
    assert set(sessions) == ids


@pytest.mark.asyncio
async def test_async_inmemory_exists() -> None:
    """exists returns True when present, False otherwise."""
    backend = AsyncInMemoryBackend()
    assert await backend.exists("missing") is False
    await backend.save("present", "data")
    assert await backend.exists("present") is True


@pytest.mark.asyncio
async def test_async_inmemory_overwrite() -> None:
    """Saving with the same session_id overwrites the existing payload."""
    backend = AsyncInMemoryBackend()
    await backend.save("ow", "original")
    await backend.save("ow", "updated")
    payload = await backend.load("ow")
    assert payload == "updated"


@pytest.mark.asyncio
async def test_async_inmemory_clear() -> None:
    """clear() removes all sessions."""
    backend = AsyncInMemoryBackend()
    await backend.save("s1", "d1")
    await backend.save("s2", "d2")
    await backend.clear()
    sessions = await backend.list_sessions()
    assert list(sessions) == []


# ---------------------------------------------------------------------------
# AsyncSQLiteBackend — core CRUD
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_save_and_load(tmp_path: object) -> None:
    """AsyncSQLiteBackend: save stores payload; load retrieves it."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "test.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    await backend.save("sq_sess1", '{"x": 1}')
    loaded = await backend.load("sq_sess1")
    assert loaded == '{"x": 1}'


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_load_missing_raises_keyerror(tmp_path: object) -> None:
    """AsyncSQLiteBackend: load raises KeyError for missing session."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "test.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    with pytest.raises(KeyError):
        await backend.load("nonexistent")


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_delete(tmp_path: object) -> None:
    """AsyncSQLiteBackend: delete returns True for existing, False for missing."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "test.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    await backend.save("sq_del", "data")
    assert await backend.delete("sq_del") is True
    assert await backend.delete("sq_del") is False


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_list_sessions(tmp_path: object) -> None:
    """AsyncSQLiteBackend: list_sessions returns all stored session IDs."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "test.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    ids = {"sq_a", "sq_b", "sq_c"}
    for session_id in ids:
        await backend.save(session_id, "payload")
    sessions = await backend.list_sessions()
    assert set(sessions) == ids


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_exists(tmp_path: object) -> None:
    """AsyncSQLiteBackend: exists returns True/False correctly."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "test.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    assert await backend.exists("nope") is False
    await backend.save("yes", "payload")
    assert await backend.exists("yes") is True


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_async_sqlite_overwrite(tmp_path: object) -> None:
    """AsyncSQLiteBackend: saving same id again overwrites the payload."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "test.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    await backend.save("ow_sq", "first")
    await backend.save("ow_sq", "second")
    loaded = await backend.load("ow_sq")
    assert loaded == "second"


# ---------------------------------------------------------------------------
# KeyError on missing session (covers both in-memory and sqlite paths)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_keyerror_on_missing_inmemory() -> None:
    """AsyncInMemoryBackend raises KeyError with session_id in message."""
    backend = AsyncInMemoryBackend()
    with pytest.raises(KeyError, match="no_session"):
        await backend.load("no_session")


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_keyerror_on_missing_sqlite(tmp_path: object) -> None:
    """AsyncSQLiteBackend raises KeyError with session_id in message."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "test.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    with pytest.raises(KeyError, match="missing_id"):
        await backend.load("missing_id")


# ---------------------------------------------------------------------------
# Concurrent operations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_saves_inmemory() -> None:
    """Three concurrent saves to AsyncInMemoryBackend all land correctly."""
    backend = AsyncInMemoryBackend()
    await asyncio.gather(
        backend.save("t1", "payload1"),
        backend.save("t2", "payload2"),
        backend.save("t3", "payload3"),
    )
    sessions = await backend.list_sessions()
    assert set(sessions) == {"t1", "t2", "t3"}


@pytest.mark.asyncio
async def test_concurrent_reads_inmemory() -> None:
    """Three concurrent loads from AsyncInMemoryBackend return correct values."""
    backend = AsyncInMemoryBackend()
    await backend.save("r1", "one")
    await backend.save("r2", "two")
    await backend.save("r3", "three")
    p1, p2, p3 = await asyncio.gather(
        backend.load("r1"),
        backend.load("r2"),
        backend.load("r3"),
    )
    assert p1 == "one"
    assert p2 == "two"
    assert p3 == "three"


@pytest.mark.asyncio
async def test_concurrent_saves_and_reads_inmemory() -> None:
    """Mixed concurrent saves and reads do not raise or corrupt state."""
    backend = AsyncInMemoryBackend()
    await backend.save("pre", "existing")

    save_tasks = [backend.save(f"n{i}", f"data{i}") for i in range(5)]
    read_tasks = [backend.load("pre") for _ in range(5)]

    results = await asyncio.gather(*save_tasks, *read_tasks)
    # No exception — all reads return the pre-existing value
    loaded_values = results[5:]  # first 5 are None from save
    assert all(v == "existing" for v in loaded_values)


@pytest.mark.asyncio
@pytest.mark.skipif(not _aiosqlite_available, reason="aiosqlite not installed")
async def test_concurrent_saves_sqlite(tmp_path: object) -> None:
    """Three concurrent saves to AsyncSQLiteBackend all succeed."""
    import pathlib

    from agent_session_linker.storage.async_sqlite import AsyncSQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "concurrent.db"
    backend = AsyncSQLiteBackend(db_path=db_path)
    await asyncio.gather(
        backend.save("c1", "data1"),
        backend.save("c2", "data2"),
        backend.save("c3", "data3"),
    )
    sessions = await backend.list_sessions()
    assert set(sessions) == {"c1", "c2", "c3"}


# ---------------------------------------------------------------------------
# Backward compatibility — sync API unchanged
# ---------------------------------------------------------------------------


def test_sync_inmemory_save_load_still_works() -> None:
    """InMemoryBackend (sync) continues to work after adding async variant."""
    from agent_session_linker.storage.memory import InMemoryBackend

    backend = InMemoryBackend()
    backend.save("sync1", "payload")
    loaded = backend.load("sync1")
    assert loaded == "payload"


def test_sync_inmemory_delete_still_works() -> None:
    """InMemoryBackend.delete still raises KeyError for missing session."""
    from agent_session_linker.storage.memory import InMemoryBackend

    backend = InMemoryBackend()
    backend.save("s", "data")
    backend.delete("s")
    with pytest.raises(KeyError):
        backend.delete("s")


def test_sync_sqlite_still_works(tmp_path: object) -> None:
    """SQLiteBackend (sync) continues to work after adding async variant."""
    import pathlib

    from agent_session_linker.storage.sqlite import SQLiteBackend

    db_path = pathlib.Path(str(tmp_path)) / "sync_test.db"
    backend = SQLiteBackend(db_path=db_path)
    backend.save("sync_sq", "value")
    loaded = backend.load("sync_sq")
    assert loaded == "value"
    assert backend.exists("sync_sq") is True
