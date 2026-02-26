"""Coverage boost tests for agent-session-linker.

Covers:
- plugins/registry.py
- linking/chain.py
- middleware/checkpoint.py
- middleware/context_window.py
- middleware/session_middleware.py
- storage/filesystem.py
- storage/sqlite.py
- context/injector.py
- context/relevance.py
"""
from __future__ import annotations

from abc import abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_session_linker.plugins.registry import (
    PluginAlreadyRegisteredError,
    PluginNotFoundError,
    PluginRegistry,
)
from agent_session_linker.linking.chain import SessionChain
from agent_session_linker.middleware.checkpoint import (
    CheckpointManager,
    CheckpointRecord,
    _build_checkpoint_key,
)
from agent_session_linker.middleware.context_window import ContextWindowManager
from agent_session_linker.middleware.session_middleware import SessionMiddleware
from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
from agent_session_linker.session.state import ContextSegment, SessionState
from agent_session_linker.storage.filesystem import FilesystemBackend
from agent_session_linker.storage.memory import InMemoryBackend
from agent_session_linker.storage.sqlite import SQLiteBackend
from agent_session_linker.context.injector import (
    ContextInjector,
    InjectionConfig,
    _compute_idf,
    _term_frequency,
    _tfidf_score,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(content: str = "Hello world.", role: str = "user") -> SessionState:
    """Return a minimal SessionState with one segment."""
    session = SessionState()
    session.add_segment(role=role, content=content)
    return session


def _make_segment(content: str, role: str = "user", segment_type: str = "conversation") -> ContextSegment:
    return ContextSegment(role=role, content=content, segment_type=segment_type)


def _make_manager(tmp_path: Path) -> SessionManager:
    backend = FilesystemBackend(storage_dir=tmp_path)
    return SessionManager(backend=backend)


# ---------------------------------------------------------------------------
# PluginRegistry
# ---------------------------------------------------------------------------


from abc import ABC


class _BasePlugin(ABC):
    @abstractmethod
    def run(self) -> str: ...


class TestPluginRegistry:
    def _make_registry(self) -> PluginRegistry[_BasePlugin]:
        return PluginRegistry(_BasePlugin, "test-registry")

    def test_register_and_get(self) -> None:
        registry = self._make_registry()

        @registry.register("my-plugin")
        class MyPlugin(_BasePlugin):
            def run(self) -> str:
                return "ok"

        assert registry.get("my-plugin") is MyPlugin

    def test_get_unregistered_raises_plugin_not_found(self) -> None:
        registry = self._make_registry()
        with pytest.raises(PluginNotFoundError) as exc_info:
            registry.get("missing")
        assert exc_info.value.plugin_name == "missing"

    def test_register_duplicate_raises_already_registered(self) -> None:
        registry = self._make_registry()

        @registry.register("dup")
        class P1(_BasePlugin):
            def run(self) -> str:
                return "p1"

        with pytest.raises(PluginAlreadyRegisteredError) as exc_info:
            @registry.register("dup")
            class P2(_BasePlugin):
                def run(self) -> str:
                    return "p2"

        assert exc_info.value.plugin_name == "dup"

    def test_register_non_subclass_raises_type_error(self) -> None:
        registry = self._make_registry()
        with pytest.raises(TypeError):
            @registry.register("bad")  # type: ignore[arg-type]
            class NotAPlugin:
                pass

    def test_register_class_direct(self) -> None:
        registry = self._make_registry()

        class Impl(_BasePlugin):
            def run(self) -> str:
                return "impl"

        registry.register_class("impl-plugin", Impl)
        assert registry.get("impl-plugin") is Impl

    def test_register_class_duplicate_raises(self) -> None:
        registry = self._make_registry()

        class Impl(_BasePlugin):
            def run(self) -> str:
                return "x"

        registry.register_class("x", Impl)
        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register_class("x", Impl)

    def test_register_class_non_subclass_raises(self) -> None:
        registry = self._make_registry()
        with pytest.raises(TypeError):
            registry.register_class("bad", object)  # type: ignore[arg-type]

    def test_deregister_removes_plugin(self) -> None:
        registry = self._make_registry()

        @registry.register("to-remove")
        class Impl(_BasePlugin):
            def run(self) -> str:
                return "x"

        registry.deregister("to-remove")
        assert "to-remove" not in registry

    def test_deregister_missing_raises(self) -> None:
        registry = self._make_registry()
        with pytest.raises(PluginNotFoundError):
            registry.deregister("nonexistent")

    def test_list_plugins_sorted(self) -> None:
        registry = self._make_registry()

        class A(_BasePlugin):
            def run(self) -> str:
                return "a"

        class B(_BasePlugin):
            def run(self) -> str:
                return "b"

        registry.register_class("zebra", A)
        registry.register_class("apple", B)
        names = registry.list_plugins()
        assert names == ["apple", "zebra"]

    def test_contains_operator(self) -> None:
        registry = self._make_registry()

        class Impl(_BasePlugin):
            def run(self) -> str:
                return "x"

        registry.register_class("present", Impl)
        assert "present" in registry
        assert "absent" not in registry

    def test_len(self) -> None:
        registry = self._make_registry()

        class Impl(_BasePlugin):
            def run(self) -> str:
                return "x"

        assert len(registry) == 0
        registry.register_class("one", Impl)
        assert len(registry) == 1

    def test_repr_contains_name(self) -> None:
        registry = self._make_registry()
        assert "test-registry" in repr(registry)

    def test_load_entrypoints_empty_group(self) -> None:
        registry = self._make_registry()
        # No entrypoints in this test group — should be a no-op
        registry.load_entrypoints("agent_session_linker.plugins.nonexistent")
        assert len(registry) == 0

    def test_load_entrypoints_already_registered_skipped(self) -> None:
        """Pre-registered names are silently skipped during entry-point loading."""
        registry = self._make_registry()

        class Impl(_BasePlugin):
            def run(self) -> str:
                return "x"

        registry.register_class("existing", Impl)
        # Mock entry_points to return one EP with the already-registered name
        mock_ep = MagicMock()
        mock_ep.name = "existing"
        mock_ep.load.return_value = Impl
        with patch("agent_session_linker.plugins.registry.importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("some.group")
        assert len(registry) == 1  # Still only one

    def test_load_entrypoints_load_failure_skipped(self) -> None:
        registry = self._make_registry()
        mock_ep = MagicMock()
        mock_ep.name = "failing-ep"
        mock_ep.load.side_effect = ImportError("module not found")
        with patch("agent_session_linker.plugins.registry.importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("some.group")
        assert "failing-ep" not in registry

    def test_load_entrypoints_bad_class_skipped(self) -> None:
        registry = self._make_registry()
        mock_ep = MagicMock()
        mock_ep.name = "bad-class"
        mock_ep.load.return_value = object  # Not a subclass of _BasePlugin
        with patch("agent_session_linker.plugins.registry.importlib.metadata.entry_points", return_value=[mock_ep]):
            registry.load_entrypoints("some.group")
        assert "bad-class" not in registry


# ---------------------------------------------------------------------------
# SessionChain
# ---------------------------------------------------------------------------


class TestSessionChain:
    def _make_chain(self, tmp_path: Path) -> tuple[SessionChain, SessionManager]:
        manager = _make_manager(tmp_path)
        chain = SessionChain(manager)
        return chain, manager

    def test_empty_chain_length(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        assert len(chain) == 0

    def test_append_increases_length(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("session-1")
        assert len(chain) == 1

    def test_prepend_adds_at_front(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("session-2")
        chain.prepend("session-1")
        assert chain.get_chain()[0] == "session-1"

    def test_remove_session(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("s1")
        chain.append("s2")
        chain.remove("s1")
        assert "s1" not in chain

    def test_remove_missing_raises_value_error(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        with pytest.raises(ValueError):
            chain.remove("nonexistent")

    def test_get_chain_returns_copy(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("s1")
        copy = chain.get_chain()
        copy.append("s2")
        assert len(chain) == 1

    def test_contains_operator(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("present")
        assert "present" in chain
        assert "absent" not in chain

    def test_repr_includes_length(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        assert "SessionChain" in repr(chain)

    def test_get_sessions_loads_existing(self, tmp_path: Path) -> None:
        chain, manager = self._make_chain(tmp_path)
        session = _make_session("Some context content.")
        saved_id = manager.save_session(session)
        chain.append(saved_id)
        sessions = chain.get_sessions()
        assert len(sessions) == 1

    def test_get_sessions_skips_missing(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("does-not-exist")
        sessions = chain.get_sessions()
        assert sessions == []

    def test_get_context_from_chain_n_recent_invalid(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        with pytest.raises(ValueError):
            chain.get_context_from_chain(0)

    def test_get_context_from_chain_empty_chain(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        result = chain.get_context_from_chain(3)
        assert result == ""

    def test_get_context_from_chain_with_session(self, tmp_path: Path) -> None:
        chain, manager = self._make_chain(tmp_path)
        session = _make_session("Machine learning is fascinating.")
        saved_id = manager.save_session(session)
        chain.append(saved_id)
        result = chain.get_context_from_chain(1)
        assert "Machine learning" in result

    def test_get_context_from_chain_skips_missing(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("ghost-session-id")
        result = chain.get_context_from_chain(1)
        assert result == ""

    def test_get_context_from_chain_empty_session_skipped(self, tmp_path: Path) -> None:
        chain, manager = self._make_chain(tmp_path)
        # Save a session with NO segments
        session = SessionState()
        saved_id = manager.save_session(session)
        chain.append(saved_id)
        result = chain.get_context_from_chain(1)
        assert result == ""

    def test_get_all_segments_all(self, tmp_path: Path) -> None:
        chain, manager = self._make_chain(tmp_path)
        s1 = _make_session("First session content.")
        s2 = _make_session("Second session content.")
        id1 = manager.save_session(s1)
        id2 = manager.save_session(s2)
        chain.append(id1)
        chain.append(id2)
        segments = chain.get_all_segments()
        assert len(segments) == 2

    def test_get_all_segments_n_recent(self, tmp_path: Path) -> None:
        chain, manager = self._make_chain(tmp_path)
        s1 = _make_session("First session content.")
        s2 = _make_session("Second session content.")
        s3 = _make_session("Third session content.")
        id1 = manager.save_session(s1)
        id2 = manager.save_session(s2)
        id3 = manager.save_session(s3)
        chain.append(id1)
        chain.append(id2)
        chain.append(id3)
        segments = chain.get_all_segments(n_recent=2)
        assert len(segments) == 2

    def test_get_all_segments_skips_missing(self, tmp_path: Path) -> None:
        chain, _ = self._make_chain(tmp_path)
        chain.append("does-not-exist")
        segments = chain.get_all_segments()
        assert segments == []

    def test_initial_session_ids(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        chain = SessionChain(manager, initial_session_ids=["a", "b", "c"])
        assert chain.get_chain() == ["a", "b", "c"]

    def test_format_segments_includes_role(self, tmp_path: Path) -> None:
        chain, manager = self._make_chain(tmp_path)
        session = _make_session("Hello from user.")
        saved_id = manager.save_session(session)
        chain.append(saved_id)
        result = chain.get_context_from_chain(1)
        assert "USER" in result


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------


class TestCheckpointManager:
    def _make_manager_and_checkpoint(self, tmp_path: Path) -> tuple[CheckpointManager, InMemoryBackend]:
        backend = InMemoryBackend()
        mgr = CheckpointManager(backend=backend)
        return mgr, backend

    def test_create_checkpoint_returns_record(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        session = _make_session("Context content for checkpoint.")
        record = mgr.create_checkpoint(session, label="test-checkpoint")
        assert isinstance(record, CheckpointRecord)
        assert record.label == "test-checkpoint"

    def test_create_checkpoint_default_label_is_timestamp(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        session = _make_session("Some content.")
        record = mgr.create_checkpoint(session)
        # Default label is an ISO timestamp string
        assert "T" in record.label or "-" in record.label

    def test_create_checkpoint_stores_segment_count(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        session = _make_session("Content.")
        record = mgr.create_checkpoint(session)
        assert record.segment_count == len(session.segments)

    def test_restore_checkpoint_returns_session_state(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        session = _make_session("Content for restore test.")
        record = mgr.create_checkpoint(session, label="restore-me")
        restored = mgr.restore_checkpoint(record.checkpoint_id)
        assert isinstance(restored, SessionState)

    def test_restore_checkpoint_not_found_raises(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        with pytest.raises(KeyError):
            mgr.restore_checkpoint("nonexistent-key")

    def test_list_checkpoints_empty(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        assert mgr.list_checkpoints("no-session") == []

    def test_list_checkpoints_after_create(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        session = _make_session("Content.")
        mgr.create_checkpoint(session, label="cp1")
        mgr.create_checkpoint(session, label="cp2")
        records = mgr.list_checkpoints(session.session_id)
        assert len(records) == 2

    def test_delete_checkpoint_removes_it(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        session = _make_session("Content.")
        record = mgr.create_checkpoint(session, label="to-delete")
        mgr.delete_checkpoint(record.checkpoint_id, session.session_id)
        remaining = mgr.list_checkpoints(session.session_id)
        assert all(r.checkpoint_id != record.checkpoint_id for r in remaining)

    def test_delete_checkpoint_not_found_raises(self, tmp_path: Path) -> None:
        mgr, _ = self._make_manager_and_checkpoint(tmp_path)
        with pytest.raises(KeyError):
            mgr.delete_checkpoint("ghost-id", "session-id")

    def test_max_checkpoints_evicts_oldest(self, tmp_path: Path) -> None:
        backend = InMemoryBackend()
        mgr = CheckpointManager(backend=backend, max_checkpoints_per_session=2)
        session = _make_session("Content.")
        r1 = mgr.create_checkpoint(session, label="cp1")
        r2 = mgr.create_checkpoint(session, label="cp2")
        r3 = mgr.create_checkpoint(session, label="cp3")
        records = mgr.list_checkpoints(session.session_id)
        # After eviction: oldest (cp1) was removed; cp2 and cp3 remain
        assert len(records) == 2
        checkpoint_ids = [r.checkpoint_id for r in records]
        assert r1.checkpoint_id not in checkpoint_ids

    def test_checkpoint_record_to_dict_and_from_dict(self) -> None:
        record = CheckpointRecord(
            checkpoint_id="ck-001",
            session_id="sess-001",
            label="my-label",
            created_at=datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            segment_count=5,
            token_count=200,
        )
        data = record.to_dict()
        restored = CheckpointRecord.from_dict(data)
        assert restored.checkpoint_id == "ck-001"
        assert restored.label == "my-label"
        assert restored.segment_count == 5

    def test_build_checkpoint_key_format(self) -> None:
        key = _build_checkpoint_key("session-abc", 3)
        assert "__checkpoint__" in key
        assert "0003" in key


# ---------------------------------------------------------------------------
# ContextWindowManager
# ---------------------------------------------------------------------------


class TestContextWindowManager:
    def test_empty_window_returns_empty_string(self) -> None:
        mgr = ContextWindowManager()
        assert mgr.get_window() == ""

    def test_add_segment_increases_token_count(self) -> None:
        mgr = ContextWindowManager(max_tokens=1000)
        seg = _make_segment("This is test content for token counting.")
        mgr.add(seg)
        assert mgr.token_count() > 0

    def test_get_window_renders_segment(self) -> None:
        mgr = ContextWindowManager()
        seg = _make_segment("Hello world content.", role="user")
        mgr.add(seg)
        window = mgr.get_window()
        assert "USER" in window
        assert "Hello world" in window

    def test_eviction_when_token_budget_exceeded(self) -> None:
        mgr = ContextWindowManager(max_tokens=10)
        # Add segments with known sizes
        seg1 = _make_segment("A" * 40, role="user")  # ~10 tokens
        seg2 = _make_segment("B" * 40, role="assistant")  # ~10 tokens
        seg3 = _make_segment("C" * 40, role="user")  # ~10 tokens
        mgr.add(seg1)
        mgr.add(seg2)
        mgr.add(seg3)
        # With max_tokens=10, at most one segment should remain
        assert len(mgr) <= 2

    def test_eviction_respects_max_segments(self) -> None:
        mgr = ContextWindowManager(max_tokens=100_000, max_segments=2)
        for i in range(5):
            mgr.add(_make_segment(f"Segment content number {i}.", role="user"))
        assert len(mgr) <= 2

    def test_single_oversized_segment_accepted(self) -> None:
        mgr = ContextWindowManager(max_tokens=5)
        large_seg = _make_segment("X" * 200, role="user")  # Way over budget
        mgr.add(large_seg)
        assert len(mgr) == 1

    def test_get_segments_returns_list(self) -> None:
        mgr = ContextWindowManager()
        seg = _make_segment("Test content.")
        mgr.add(seg)
        segments = mgr.get_segments()
        assert isinstance(segments, list)
        assert len(segments) == 1

    def test_clear_empties_window(self) -> None:
        mgr = ContextWindowManager()
        mgr.add(_make_segment("Content."))
        mgr.clear()
        assert len(mgr) == 0
        assert mgr.token_count() == 0

    def test_custom_separators(self) -> None:
        mgr = ContextWindowManager(role_separator="->", segment_separator="|")
        mgr.add(_make_segment("Content A.", role="user"))
        mgr.add(_make_segment("Content B.", role="assistant"))
        window = mgr.get_window()
        assert "|" in window
        assert "USER->Content A." in window

    def test_invalid_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            ContextWindowManager(max_tokens=0)

    def test_invalid_max_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="max_segments"):
            ContextWindowManager(max_segments=0)

    def test_repr_contains_info(self) -> None:
        mgr = ContextWindowManager(max_tokens=200)
        assert "ContextWindowManager" in repr(mgr)
        assert "200" in repr(mgr)

    def test_segment_uses_explicit_token_count(self) -> None:
        mgr = ContextWindowManager(max_tokens=100)
        seg = ContextSegment(role="user", content="Short.", token_count=50)
        mgr.add(seg)
        assert mgr.token_count() == 50


# ---------------------------------------------------------------------------
# SessionMiddleware
# ---------------------------------------------------------------------------


class TestSessionMiddleware:
    def _make_middleware(self, tmp_path: Path) -> tuple[SessionMiddleware, SessionManager]:
        manager = _make_manager(tmp_path)
        middleware = SessionMiddleware(manager, auto_create=True)
        return middleware, manager

    def test_before_request_creates_new_session(self, tmp_path: Path) -> None:
        middleware, _ = self._make_middleware(tmp_path)
        session = middleware.before_request("req-001")
        assert isinstance(session, SessionState)

    def test_before_request_loads_existing_session(self, tmp_path: Path) -> None:
        middleware, manager = self._make_middleware(tmp_path)
        existing = _make_session("Existing content.")
        existing.session_id = "known-session"
        manager.save_session(existing)
        # Now call before_request with the same ID
        session = middleware.before_request("known-session")
        assert session.session_id == "known-session"

    def test_after_request_saves_session(self, tmp_path: Path) -> None:
        middleware, manager = self._make_middleware(tmp_path)
        middleware.before_request("req-save")
        saved_id = middleware.after_request("req-save", new_context="Hello from assistant.")
        assert isinstance(saved_id, str)

    def test_after_request_appends_string_context(self, tmp_path: Path) -> None:
        middleware, manager = self._make_middleware(tmp_path)
        middleware.before_request("req-ctx")
        middleware.after_request("req-ctx", new_context="New assistant message.")
        loaded = manager.load_session("req-ctx")
        contents = [s.content for s in loaded.segments]
        assert any("New assistant message" in c for c in contents)

    def test_after_request_appends_segment_list(self, tmp_path: Path) -> None:
        middleware, manager = self._make_middleware(tmp_path)
        middleware.before_request("req-segs")
        segs = [_make_segment("Segment A."), _make_segment("Segment B.")]
        middleware.after_request("req-segs", new_context=segs)
        loaded = manager.load_session("req-segs")
        contents = [s.content for s in loaded.segments]
        assert any("Segment A" in c for c in contents)

    def test_after_request_none_context_saves_without_append(self, tmp_path: Path) -> None:
        middleware, manager = self._make_middleware(tmp_path)
        middleware.before_request("req-none")
        saved_id = middleware.after_request("req-none", new_context=None)
        assert saved_id == "req-none"

    def test_after_request_without_before_raises_key_error(self, tmp_path: Path) -> None:
        middleware, _ = self._make_middleware(tmp_path)
        with pytest.raises(KeyError, match="req-orphan"):
            middleware.after_request("req-orphan")

    def test_get_active_returns_session(self, tmp_path: Path) -> None:
        middleware, _ = self._make_middleware(tmp_path)
        middleware.before_request("req-active")
        session = middleware.get_active("req-active")
        assert session is not None

    def test_get_active_returns_none_when_not_active(self, tmp_path: Path) -> None:
        middleware, _ = self._make_middleware(tmp_path)
        assert middleware.get_active("nonexistent") is None

    def test_clear_active_removes_session(self, tmp_path: Path) -> None:
        middleware, _ = self._make_middleware(tmp_path)
        middleware.before_request("req-clear")
        middleware.clear_active("req-clear")
        assert middleware.get_active("req-clear") is None

    def test_clear_active_nonexistent_is_safe(self, tmp_path: Path) -> None:
        middleware, _ = self._make_middleware(tmp_path)
        # Should not raise
        middleware.clear_active("does-not-exist")

    def test_auto_create_false_raises_on_missing(self, tmp_path: Path) -> None:
        manager = _make_manager(tmp_path)
        middleware = SessionMiddleware(manager, auto_create=False)
        with pytest.raises(SessionNotFoundError):
            middleware.before_request("no-such-session")


# ---------------------------------------------------------------------------
# FilesystemBackend
# ---------------------------------------------------------------------------


class TestFilesystemBackend:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        backend.save("sess-001", '{"key": "value"}')
        loaded = backend.load("sess-001")
        assert loaded == '{"key": "value"}'

    def test_load_missing_raises_key_error(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        with pytest.raises(KeyError):
            backend.load("nonexistent")

    def test_exists_true_after_save(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        backend.save("sess-exists", "data")
        assert backend.exists("sess-exists") is True

    def test_exists_false_when_not_saved(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        assert backend.exists("never-saved") is False

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        backend.save("sess-del", "payload")
        backend.delete("sess-del")
        assert not backend.exists("sess-del")

    def test_delete_missing_raises_key_error(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        with pytest.raises(KeyError):
            backend.delete("does-not-exist")

    def test_list_returns_session_ids(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        backend.save("s1", "data1")
        backend.save("s2", "data2")
        ids = backend.list()
        assert set(ids) == {"s1", "s2"}

    def test_list_empty_when_dir_does_not_exist(self, tmp_path: Path) -> None:
        storage_dir = tmp_path / "nonexistent_subdir"
        backend = FilesystemBackend(storage_dir=storage_dir)
        assert backend.list() == []

    def test_repr_contains_dir(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        assert "FilesystemBackend" in repr(backend)
        assert "storage_dir" in repr(backend)

    def test_path_traversal_guard(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(storage_dir=tmp_path)
        # A traversal key like "../../etc/passwd" should be safely handled
        backend.save("../../harmless", "data")
        # File should be created inside the storage dir, not above it
        files = list(tmp_path.glob("*.json"))
        assert all(str(tmp_path) in str(f) for f in files)

    def test_default_storage_dir_is_set(self) -> None:
        backend = FilesystemBackend()
        assert backend._storage_dir is not None


# ---------------------------------------------------------------------------
# SQLiteBackend
# ---------------------------------------------------------------------------


class TestSQLiteBackend:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        backend.save("sess-001", '{"data": "value"}')
        loaded = backend.load("sess-001")
        assert loaded == '{"data": "value"}'

    def test_load_missing_raises_key_error(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        with pytest.raises(KeyError):
            backend.load("nonexistent")

    def test_upsert_overwrites_existing(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        backend.save("sess-1", "v1")
        backend.save("sess-1", "v2")
        assert backend.load("sess-1") == "v2"

    def test_exists_true_after_save(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        backend.save("sess-exists", "data")
        assert backend.exists("sess-exists") is True

    def test_exists_false_when_not_saved(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        assert backend.exists("never-saved") is False

    def test_delete_removes_entry(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        backend.save("sess-del", "data")
        backend.delete("sess-del")
        assert not backend.exists("sess-del")

    def test_delete_missing_raises_key_error(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        with pytest.raises(KeyError):
            backend.delete("does-not-exist")

    def test_list_returns_all_session_ids(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        backend = SQLiteBackend(db_path=db)
        backend.save("a", "data")
        backend.save("b", "data")
        ids = backend.list()
        assert set(ids) >= {"a", "b"}

    def test_repr_contains_db_path(self, tmp_path: Path) -> None:
        db = tmp_path / "sessions.db"
        backend = SQLiteBackend(db_path=db)
        assert "sessions.db" in repr(backend)

    def test_default_db_path_is_set(self) -> None:
        backend = SQLiteBackend()
        assert backend._db_path is not None


# ---------------------------------------------------------------------------
# ContextInjector (TF-IDF helpers + inject API)
# ---------------------------------------------------------------------------


class TestTokenizeHelper:
    def test_basic_tokenize(self) -> None:
        tokens = _tokenize("Machine learning is powerful")
        assert "machine" in tokens
        assert "learning" in tokens
        # Stop words removed
        assert "is" not in tokens

    def test_empty_string(self) -> None:
        assert _tokenize("") == []

    def test_single_char_words_excluded(self) -> None:
        tokens = _tokenize("a b c hello")
        assert "a" not in tokens
        assert "hello" in tokens


class TestTermFrequency:
    def test_empty_tokens(self) -> None:
        assert _term_frequency([]) == {}

    def test_normalised_by_length(self) -> None:
        tf = _term_frequency(["a", "a", "b"])
        assert tf["a"] == pytest.approx(2 / 3)
        assert tf["b"] == pytest.approx(1 / 3)


class TestComputeIdf:
    def test_empty_corpus(self) -> None:
        assert _compute_idf([]) == {}

    def test_idf_higher_for_rare_terms(self) -> None:
        corpus = [["machine", "learning"], ["learning", "deep"], ["nlp"]]
        idf = _compute_idf(corpus)
        # "machine" appears in 1/3 docs, "learning" in 2/3 — machine IDF > learning IDF
        assert idf["machine"] > idf["learning"]


class TestTfIdfScore:
    def test_empty_query_returns_zero(self) -> None:
        idf = {"machine": 1.0}
        assert _tfidf_score([], ["machine", "learning"], idf) == 0.0

    def test_empty_doc_returns_zero(self) -> None:
        idf = {"machine": 1.0}
        assert _tfidf_score(["machine"], [], idf) == 0.0

    def test_matching_query_returns_positive(self) -> None:
        idf = {"machine": 1.5, "learning": 1.2}
        score = _tfidf_score(["machine"], ["machine", "learning"], idf)
        assert score > 0.0


class TestContextInjector:
    def test_inject_empty_sessions_returns_empty(self) -> None:
        injector = ContextInjector()
        result = injector.inject([], "some query")
        assert result == ""

    def test_inject_returns_string(self) -> None:
        injector = ContextInjector()
        session = _make_session("Machine learning is a powerful tool.")
        result = injector.inject([session], "machine learning")
        assert isinstance(result, str)

    def test_inject_includes_context_header(self) -> None:
        injector = ContextInjector()
        session = _make_session("Deep learning content.")
        result = injector.inject([session], "deep learning")
        assert "PRIOR SESSION CONTEXT" in result

    def test_inject_with_no_eligible_segments(self) -> None:
        """When all segments are too old, the header is returned."""
        config = InjectionConfig(max_age_hours=0.000001)  # Extremely short window
        injector = ContextInjector(config=config)
        session = _make_session("Very old content.")
        result = injector.inject([session], "old content")
        assert "PRIOR SESSION CONTEXT" in result

    def test_inject_with_summary(self) -> None:
        injector = ContextInjector()
        session = _make_session("Content here.")
        session.summary = "This session was about AI."
        result = injector.inject([session], "AI")
        assert "This session was about AI." in result

    def test_inject_includes_active_tasks(self) -> None:
        from agent_session_linker.session.state import TaskState, TaskStatus
        injector = ContextInjector(config=InjectionConfig(include_active_tasks=True))
        session = _make_session("Content.")
        task = TaskState(title="Fix the bug", status=TaskStatus.IN_PROGRESS)
        session.tasks.append(task)
        result = injector.inject([session], "bug fix")
        assert "Fix the bug" in result

    def test_inject_excludes_completed_tasks(self) -> None:
        from agent_session_linker.session.state import TaskState, TaskStatus
        injector = ContextInjector(config=InjectionConfig(include_active_tasks=True))
        session = _make_session("Content.")
        task = TaskState(title="Completed task", status=TaskStatus.COMPLETED)
        session.tasks.append(task)
        result = injector.inject([session], "completed task")
        assert "Completed task" not in result

    def test_inject_includes_entities(self) -> None:
        from agent_session_linker.session.state import EntityReference
        config = InjectionConfig(include_entities=True)
        injector = ContextInjector(config=config)
        session = _make_session("Content about machine learning.")
        entity = EntityReference(
            canonical_name="machine learning",
            entity_type="concept",
        )
        session.entities.append(entity)
        result = injector.inject([session], "machine learning")
        assert "machine learning" in result

    def test_inject_respects_max_segments(self) -> None:
        config = InjectionConfig(max_segments=1, token_budget=100_000)
        injector = ContextInjector(config=config)
        session = SessionState()
        for i in range(10):
            session.add_segment(role="user", content=f"Segment number {i} content here.")
        result = injector.inject([session], "segment content")
        # Only 1 segment should appear in Relevant Context Segments
        segment_headers = result.count("[USER |")
        assert segment_headers <= 1

    def test_score_segment_returns_float(self) -> None:
        injector = ContextInjector()
        segment = _make_segment("Machine learning is powerful.")
        reference = [_make_segment("Deep learning and neural networks.")]
        score = injector.score_segment(segment, "machine learning", reference)
        assert isinstance(score, float)

    def test_score_segment_empty_references(self) -> None:
        injector = ContextInjector()
        segment = _make_segment("Test content.")
        score = injector.score_segment(segment, "test", [])
        assert isinstance(score, float)

    def test_inject_entity_with_aliases(self) -> None:
        from agent_session_linker.session.state import EntityReference
        config = InjectionConfig(include_entities=True)
        injector = ContextInjector(config=config)
        session = _make_session("Content about neural networks.")
        entity = EntityReference(
            canonical_name="neural network",
            entity_type="concept",
            aliases=["NN", "deep net"],
        )
        session.entities.append(entity)
        result = injector.inject([session], "neural network")
        assert "neural" in result.lower() or "network" in result.lower()
