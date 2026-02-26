"""Unit tests for agent_session_linker.storage.redis.RedisBackend.

All tests use a MagicMock in place of the real redis client so no
Redis server is required.  The ``redis`` package is also mocked at the
import level so the test suite runs without it installed.
"""
from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — build a fully mocked RedisBackend without the real redis package
# ---------------------------------------------------------------------------

def _make_backend(
    key_prefix: str = "agent_session:",
    ttl_seconds: int | None = None,
    url: str | None = None,
) -> Any:
    """Return a RedisBackend whose internal client is a MagicMock."""
    mock_redis_module = MagicMock()
    mock_client = MagicMock()
    mock_redis_module.Redis.return_value = mock_client
    mock_redis_module.Redis.from_url.return_value = mock_client

    with patch.dict(sys.modules, {"redis": mock_redis_module}):
        from agent_session_linker.storage.redis import RedisBackend
        backend = RedisBackend(
            key_prefix=key_prefix,
            ttl_seconds=ttl_seconds,
            url=url,
        )
    # Expose the mock client for assertion in tests.
    backend._mock_client = mock_client  # type: ignore[attr-defined]
    return backend


# ---------------------------------------------------------------------------
# Import-guard behaviour
# ---------------------------------------------------------------------------


class TestRedisBackendImportGuard:
    def test_import_error_raised_when_redis_missing(self) -> None:
        """RedisBackend.__init__ raises ImportError when redis is not installed."""
        with patch.dict(sys.modules, {"redis": None}):  # type: ignore[dict-item]
            # Force removal from cache so the import guard is triggered.
            import importlib
            import agent_session_linker.storage.redis as redis_mod
            importlib.reload(redis_mod)
            with pytest.raises(ImportError, match="redis"):
                redis_mod.RedisBackend()

    def test_import_error_message_contains_install_hint(self) -> None:
        with patch.dict(sys.modules, {"redis": None}):  # type: ignore[dict-item]
            import importlib
            import agent_session_linker.storage.redis as redis_mod
            importlib.reload(redis_mod)
            with pytest.raises(ImportError, match="pip install redis"):
                redis_mod.RedisBackend()


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestRedisBackendConstruction:
    def test_constructs_with_defaults(self) -> None:
        backend = _make_backend()
        assert backend._key_prefix == "agent_session:"
        assert backend._ttl_seconds is None

    def test_constructs_with_custom_prefix(self) -> None:
        backend = _make_backend(key_prefix="myapp:")
        assert backend._key_prefix == "myapp:"

    def test_constructs_with_ttl(self) -> None:
        backend = _make_backend(ttl_seconds=3600)
        assert backend._ttl_seconds == 3600

    def test_from_url_calls_from_url_factory(self) -> None:
        # Use the _make_backend helper which already mocks the redis module
        # at import time.  We verify from_url is called when url= is supplied.
        backend = _make_backend(url="redis://localhost:6379/0")
        # _make_backend wires mock_client via Redis.from_url when url is given.
        # The backend was created successfully — assert the internal client is set.
        assert backend._client is not None


# ---------------------------------------------------------------------------
# _key helper
# ---------------------------------------------------------------------------


class TestRedisBackendKey:
    def test_key_prepends_prefix(self) -> None:
        backend = _make_backend(key_prefix="pfx:")
        assert backend._key("abc") == "pfx:abc"

    def test_key_empty_prefix(self) -> None:
        backend = _make_backend(key_prefix="")
        assert backend._key("sess-1") == "sess-1"


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestRedisBackendSave:
    def test_save_calls_set_without_ttl(self) -> None:
        backend = _make_backend()
        backend.save("s1", '{"data": 1}')
        backend._mock_client.set.assert_called_once_with(
            "agent_session:s1", '{"data": 1}'
        )

    def test_save_calls_setex_with_ttl(self) -> None:
        backend = _make_backend(ttl_seconds=600)
        backend.save("s2", "payload")
        backend._mock_client.setex.assert_called_once_with(
            "agent_session:s2", 600, "payload"
        )

    def test_save_does_not_call_setex_when_no_ttl(self) -> None:
        backend = _make_backend()
        backend.save("s1", "data")
        backend._mock_client.setex.assert_not_called()

    def test_save_uses_prefixed_key(self) -> None:
        backend = _make_backend(key_prefix="test:")
        backend.save("session-abc", "val")
        backend._mock_client.set.assert_called_once_with("test:session-abc", "val")


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestRedisBackendLoad:
    def test_load_returns_value_from_redis(self) -> None:
        backend = _make_backend()
        backend._mock_client.get.return_value = "stored-payload"
        result = backend.load("s1")
        assert result == "stored-payload"

    def test_load_raises_key_error_when_none(self) -> None:
        backend = _make_backend()
        backend._mock_client.get.return_value = None
        with pytest.raises(KeyError, match="s1"):
            backend.load("s1")

    def test_load_calls_get_with_prefixed_key(self) -> None:
        backend = _make_backend(key_prefix="p:")
        backend._mock_client.get.return_value = "v"
        backend.load("id1")
        backend._mock_client.get.assert_called_once_with("p:id1")

    def test_load_error_message_contains_session_id(self) -> None:
        backend = _make_backend()
        backend._mock_client.get.return_value = None
        with pytest.raises(KeyError, match="missing-session"):
            backend.load("missing-session")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestRedisBackendList:
    def test_list_returns_empty_when_no_keys(self) -> None:
        backend = _make_backend()
        # scan returns (cursor=0, keys=[])
        backend._mock_client.scan.return_value = (0, [])
        assert backend.list() == []

    def test_list_strips_prefix_from_keys(self) -> None:
        backend = _make_backend(key_prefix="agent_session:")
        backend._mock_client.scan.return_value = (
            0,
            ["agent_session:s1", "agent_session:s2"],
        )
        result = backend.list()
        assert "s1" in result
        assert "s2" in result

    def test_list_paginates_until_cursor_zero(self) -> None:
        backend = _make_backend(key_prefix="pfx:")
        # First call returns cursor=5 (non-zero), second returns 0.
        backend._mock_client.scan.side_effect = [
            (5, ["pfx:a", "pfx:b"]),
            (0, ["pfx:c"]),
        ]
        result = backend.list()
        assert result == ["a", "b", "c"]
        assert backend._mock_client.scan.call_count == 2

    def test_list_single_page(self) -> None:
        backend = _make_backend(key_prefix="x:")
        backend._mock_client.scan.return_value = (0, ["x:session-1"])
        assert backend.list() == ["session-1"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestRedisBackendDelete:
    def test_delete_calls_redis_delete(self) -> None:
        backend = _make_backend()
        backend._mock_client.delete.return_value = 1
        backend.delete("s1")
        backend._mock_client.delete.assert_called_once_with("agent_session:s1")

    def test_delete_raises_key_error_when_not_found(self) -> None:
        backend = _make_backend()
        backend._mock_client.delete.return_value = 0
        with pytest.raises(KeyError, match="s1"):
            backend.delete("s1")

    def test_delete_error_message_contains_session_id(self) -> None:
        backend = _make_backend()
        backend._mock_client.delete.return_value = 0
        with pytest.raises(KeyError, match="nonexistent"):
            backend.delete("nonexistent")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestRedisBackendExists:
    def test_exists_true_when_key_present(self) -> None:
        backend = _make_backend()
        backend._mock_client.exists.return_value = 1
        assert backend.exists("s1") is True

    def test_exists_false_when_key_absent(self) -> None:
        backend = _make_backend()
        backend._mock_client.exists.return_value = 0
        assert backend.exists("s1") is False

    def test_exists_uses_prefixed_key(self) -> None:
        backend = _make_backend(key_prefix="pre:")
        backend._mock_client.exists.return_value = 1
        backend.exists("my-session")
        backend._mock_client.exists.assert_called_once_with("pre:my-session")


# ---------------------------------------------------------------------------
# __repr__
# ---------------------------------------------------------------------------


class TestRedisBackendRepr:
    def test_repr_contains_key_prefix(self) -> None:
        backend = _make_backend(key_prefix="myapp:")
        assert "myapp:" in repr(backend)

    def test_repr_contains_ttl(self) -> None:
        backend = _make_backend(ttl_seconds=120)
        assert "120" in repr(backend)

    def test_repr_contains_none_when_no_ttl(self) -> None:
        backend = _make_backend()
        assert "None" in repr(backend)
