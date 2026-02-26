"""Unit tests for agent_session_linker.storage.filesystem.FilesystemBackend.

Uses pytest's tmp_path fixture to isolate all file I/O.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_session_linker.storage.filesystem import FilesystemBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions"


@pytest.fixture()
def backend(storage_dir: Path) -> FilesystemBackend:
    return FilesystemBackend(storage_dir=storage_dir)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestFilesystemBackendConstruction:
    def test_accepts_string_path(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=str(tmp_path / "str-path"))
        assert isinstance(backend._storage_dir, Path)

    def test_accepts_path_object(self, tmp_path: Path) -> None:
        p = tmp_path / "path-obj"
        backend = FilesystemBackend(storage_dir=p)
        assert backend._storage_dir == p

    def test_repr_contains_storage_dir(self, backend: FilesystemBackend) -> None:
        assert "sessions" in repr(backend)

    def test_default_none_uses_home_dir(self) -> None:
        backend = FilesystemBackend(storage_dir=None)
        assert ".agent-sessions" in str(backend._storage_dir)


# ---------------------------------------------------------------------------
# _path_for / path traversal guard
# ---------------------------------------------------------------------------


class TestFilesystemBackendPathFor:
    def test_path_for_returns_json_file(self, backend: FilesystemBackend) -> None:
        p = backend._path_for("session-abc")
        assert p.name == "session-abc.json"

    def test_path_traversal_is_blocked(self, backend: FilesystemBackend) -> None:
        """../../etc/passwd should be reduced to passwd.json inside storage_dir."""
        p = backend._path_for("../../etc/passwd")
        assert p.parent == backend._storage_dir
        assert "etc" not in p.parts[:-1]


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestFilesystemBackendSaveLoad:
    def test_save_creates_directory_if_missing(
        self, backend: FilesystemBackend, storage_dir: Path
    ) -> None:
        assert not storage_dir.exists()
        backend.save("s1", "data")
        assert storage_dir.exists()

    def test_save_and_load_roundtrip(self, backend: FilesystemBackend) -> None:
        backend.save("s1", '{"key": "value"}')
        assert backend.load("s1") == '{"key": "value"}'

    def test_save_overwrites_existing(self, backend: FilesystemBackend) -> None:
        backend.save("s1", "original")
        backend.save("s1", "updated")
        assert backend.load("s1") == "updated"

    def test_load_missing_raises_key_error(self, backend: FilesystemBackend) -> None:
        with pytest.raises(KeyError):
            backend.load("nonexistent")

    def test_load_error_message_contains_session_id(
        self, backend: FilesystemBackend
    ) -> None:
        with pytest.raises(KeyError, match="ghost"):
            backend.load("ghost")

    def test_save_persists_unicode_content(self, backend: FilesystemBackend) -> None:
        payload = "unicode: \u00e9\u00e0\u00fc"
        backend.save("uni", payload)
        assert backend.load("uni") == payload


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestFilesystemBackendExists:
    def test_exists_false_before_save(self, backend: FilesystemBackend) -> None:
        assert backend.exists("s1") is False

    def test_exists_true_after_save(self, backend: FilesystemBackend) -> None:
        backend.save("s1", "payload")
        assert backend.exists("s1") is True

    def test_exists_false_after_delete(self, backend: FilesystemBackend) -> None:
        backend.save("s1", "payload")
        backend.delete("s1")
        assert backend.exists("s1") is False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestFilesystemBackendList:
    def test_list_empty_when_dir_does_not_exist(
        self, backend: FilesystemBackend
    ) -> None:
        assert backend.list() == []

    def test_list_returns_saved_sessions(self, backend: FilesystemBackend) -> None:
        backend.save("alpha", "a")
        backend.save("beta", "b")
        ids = backend.list()
        assert "alpha" in ids
        assert "beta" in ids

    def test_list_count_matches_saves(self, backend: FilesystemBackend) -> None:
        for i in range(4):
            backend.save(f"sess-{i}", "data")
        assert len(backend.list()) == 4

    def test_list_excludes_deleted(self, backend: FilesystemBackend) -> None:
        backend.save("keep", "k")
        backend.save("remove", "r")
        backend.delete("remove")
        result = backend.list()
        assert "keep" in result
        assert "remove" not in result

    def test_list_ignores_non_json_files(
        self, backend: FilesystemBackend, storage_dir: Path
    ) -> None:
        backend.save("sess-a", "data")
        storage_dir.mkdir(parents=True, exist_ok=True)
        (storage_dir / "readme.txt").write_text("not a session")
        ids = backend.list()
        assert "sess-a" in ids
        assert "readme" not in ids


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestFilesystemBackendDelete:
    def test_delete_removes_file(self, backend: FilesystemBackend) -> None:
        backend.save("s1", "payload")
        backend.delete("s1")
        assert not backend.exists("s1")

    def test_delete_nonexistent_raises_key_error(
        self, backend: FilesystemBackend
    ) -> None:
        with pytest.raises(KeyError):
            backend.delete("ghost")

    def test_delete_error_message_contains_session_id(
        self, backend: FilesystemBackend
    ) -> None:
        with pytest.raises(KeyError, match="ghost"):
            backend.delete("ghost")
