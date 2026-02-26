"""Unit tests for agent_session_linker.storage.sqlite.SQLiteBackend.

Uses tmp_path so every test gets an isolated, ephemeral SQLite file.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_session_linker.storage.sqlite import SQLiteBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_sessions.db"


@pytest.fixture()
def backend(db_path: Path) -> SQLiteBackend:
    return SQLiteBackend(db_path=db_path)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSQLiteBackendConstruction:
    def test_accepts_string_path(self, tmp_path: Path) -> None:
        backend = SQLiteBackend(db_path=str(tmp_path / "str.db"))
        assert isinstance(backend._db_path, Path)

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        p = tmp_path / "path.db"
        backend = SQLiteBackend(db_path=p)
        assert backend._db_path == p

    def test_default_none_uses_home_based_path(self) -> None:
        backend = SQLiteBackend(db_path=None)
        assert "sessions.db" in str(backend._db_path)

    def test_repr_contains_db_path(self, backend: SQLiteBackend, db_path: Path) -> None:
        assert "test_sessions.db" in repr(backend)


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSQLiteBackendSaveLoad:
    def test_save_and_load_roundtrip(self, backend: SQLiteBackend) -> None:
        backend.save("s1", '{"key": "value"}')
        assert backend.load("s1") == '{"key": "value"}'

    def test_save_creates_parent_directory(
        self, tmp_path: Path
    ) -> None:
        nested_db = tmp_path / "nested" / "deep" / "sessions.db"
        backend = SQLiteBackend(db_path=nested_db)
        backend.save("s1", "data")
        assert nested_db.exists()

    def test_save_upserts_existing_session(self, backend: SQLiteBackend) -> None:
        backend.save("s1", "original")
        backend.save("s1", "updated")
        assert backend.load("s1") == "updated"

    def test_load_missing_raises_key_error(self, backend: SQLiteBackend) -> None:
        with pytest.raises(KeyError):
            backend.load("nonexistent")

    def test_load_error_message_contains_session_id(
        self, backend: SQLiteBackend
    ) -> None:
        with pytest.raises(KeyError, match="ghost"):
            backend.load("ghost")

    def test_save_multiple_sessions(self, backend: SQLiteBackend) -> None:
        backend.save("s1", "payload-1")
        backend.save("s2", "payload-2")
        assert backend.load("s1") == "payload-1"
        assert backend.load("s2") == "payload-2"


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestSQLiteBackendExists:
    def test_exists_false_before_save(self, backend: SQLiteBackend) -> None:
        assert backend.exists("s1") is False

    def test_exists_true_after_save(self, backend: SQLiteBackend) -> None:
        backend.save("s1", "payload")
        assert backend.exists("s1") is True

    def test_exists_false_after_delete(self, backend: SQLiteBackend) -> None:
        backend.save("s1", "payload")
        backend.delete("s1")
        assert backend.exists("s1") is False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestSQLiteBackendList:
    def test_list_empty_on_new_database(self, backend: SQLiteBackend) -> None:
        assert backend.list() == []

    def test_list_returns_saved_ids(self, backend: SQLiteBackend) -> None:
        backend.save("alpha", "a")
        backend.save("beta", "b")
        ids = backend.list()
        assert "alpha" in ids
        assert "beta" in ids

    def test_list_count_matches_saves(self, backend: SQLiteBackend) -> None:
        for i in range(5):
            backend.save(f"sess-{i}", "data")
        assert len(backend.list()) == 5

    def test_list_excludes_deleted(self, backend: SQLiteBackend) -> None:
        backend.save("keep", "k")
        backend.save("remove", "r")
        backend.delete("remove")
        result = backend.list()
        assert "keep" in result
        assert "remove" not in result

    def test_list_returns_list_type(self, backend: SQLiteBackend) -> None:
        backend.save("s1", "data")
        assert isinstance(backend.list(), list)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestSQLiteBackendDelete:
    def test_delete_removes_session(self, backend: SQLiteBackend) -> None:
        backend.save("s1", "payload")
        backend.delete("s1")
        assert not backend.exists("s1")

    def test_delete_nonexistent_raises_key_error(
        self, backend: SQLiteBackend
    ) -> None:
        with pytest.raises(KeyError):
            backend.delete("ghost")

    def test_delete_error_message_contains_session_id(
        self, backend: SQLiteBackend
    ) -> None:
        with pytest.raises(KeyError, match="ghost"):
            backend.delete("ghost")

    def test_delete_only_removes_target_session(
        self, backend: SQLiteBackend
    ) -> None:
        backend.save("s1", "keep")
        backend.save("s2", "remove")
        backend.delete("s2")
        assert backend.exists("s1")
        assert not backend.exists("s2")
