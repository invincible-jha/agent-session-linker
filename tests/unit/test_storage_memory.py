"""Unit tests for agent_session_linker.storage.memory.InMemoryBackend.

Tests cover all StorageBackend interface methods plus extras
(clear, __len__, __repr__).
"""
from __future__ import annotations

import pytest

from agent_session_linker.storage.memory import InMemoryBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend() -> InMemoryBackend:
    return InMemoryBackend()


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestInMemoryBackendSaveLoad:
    def test_save_and_load_roundtrip(self, backend: InMemoryBackend) -> None:
        backend.save("s1", '{"key": "value"}')
        assert backend.load("s1") == '{"key": "value"}'

    def test_save_overwrites_existing(self, backend: InMemoryBackend) -> None:
        backend.save("s1", "original")
        backend.save("s1", "updated")
        assert backend.load("s1") == "updated"

    def test_load_missing_raises_key_error(self, backend: InMemoryBackend) -> None:
        with pytest.raises(KeyError):
            backend.load("nonexistent")

    def test_load_error_message_contains_id(self, backend: InMemoryBackend) -> None:
        with pytest.raises(KeyError, match="ghost"):
            backend.load("ghost")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestInMemoryBackendExists:
    def test_exists_false_before_save(self, backend: InMemoryBackend) -> None:
        assert backend.exists("s1") is False

    def test_exists_true_after_save(self, backend: InMemoryBackend) -> None:
        backend.save("s1", "payload")
        assert backend.exists("s1") is True

    def test_exists_false_after_delete(self, backend: InMemoryBackend) -> None:
        backend.save("s1", "payload")
        backend.delete("s1")
        assert backend.exists("s1") is False


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestInMemoryBackendList:
    def test_list_empty(self, backend: InMemoryBackend) -> None:
        assert backend.list() == []

    def test_list_returns_saved_ids(self, backend: InMemoryBackend) -> None:
        backend.save("alpha", "a")
        backend.save("beta", "b")
        ids = backend.list()
        assert "alpha" in ids
        assert "beta" in ids

    def test_list_count_matches_saves(self, backend: InMemoryBackend) -> None:
        for i in range(5):
            backend.save(f"sess-{i}", "data")
        assert len(backend.list()) == 5

    def test_list_excludes_deleted(self, backend: InMemoryBackend) -> None:
        backend.save("keep", "k")
        backend.save("remove", "r")
        backend.delete("remove")
        assert "remove" not in backend.list()
        assert "keep" in backend.list()


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestInMemoryBackendDelete:
    def test_delete_removes_key(self, backend: InMemoryBackend) -> None:
        backend.save("s1", "payload")
        backend.delete("s1")
        assert not backend.exists("s1")

    def test_delete_nonexistent_raises_key_error(
        self, backend: InMemoryBackend
    ) -> None:
        with pytest.raises(KeyError):
            backend.delete("ghost")


# ---------------------------------------------------------------------------
# clear / __len__ / __repr__
# ---------------------------------------------------------------------------


class TestInMemoryBackendExtras:
    def test_len_zero_initially(self, backend: InMemoryBackend) -> None:
        assert len(backend) == 0

    def test_len_increments_on_save(self, backend: InMemoryBackend) -> None:
        backend.save("a", "1")
        backend.save("b", "2")
        assert len(backend) == 2

    def test_len_decrements_on_delete(self, backend: InMemoryBackend) -> None:
        backend.save("a", "1")
        backend.delete("a")
        assert len(backend) == 0

    def test_clear_empties_store(self, backend: InMemoryBackend) -> None:
        backend.save("a", "1")
        backend.save("b", "2")
        backend.clear()
        assert len(backend) == 0
        assert backend.list() == []

    def test_repr_contains_session_count(self, backend: InMemoryBackend) -> None:
        backend.save("x", "y")
        assert "1" in repr(backend)

    def test_initial_data_accepted(self) -> None:
        pre_loaded = InMemoryBackend(initial_data={"pre": "data"})
        assert pre_loaded.exists("pre")
        assert pre_loaded.load("pre") == "data"

    def test_initial_data_shallow_copied(self) -> None:
        source: dict[str, str] = {"k": "v"}
        pre_loaded = InMemoryBackend(initial_data=source)
        source["extra"] = "new"
        assert not pre_loaded.exists("extra")
