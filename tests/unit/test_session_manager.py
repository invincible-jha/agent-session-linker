"""Unit tests for agent_session_linker.session.manager.

Tests cover SessionManager CRUD operations, continuation sessions,
agent filtering, statistics, and error paths — all using InMemoryBackend
so no I/O is needed.
"""
from __future__ import annotations

import pytest

from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
from agent_session_linker.session.state import SessionState, TaskStatus
from agent_session_linker.storage.memory import InMemoryBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture()
def manager(backend: InMemoryBackend) -> SessionManager:
    return SessionManager(backend=backend, default_agent_id="test-agent")


# ---------------------------------------------------------------------------
# SessionNotFoundError
# ---------------------------------------------------------------------------


class TestSessionNotFoundError:
    def test_message_contains_id(self) -> None:
        err = SessionNotFoundError("abc-123")
        assert "abc-123" in str(err)

    def test_session_id_attribute(self) -> None:
        err = SessionNotFoundError("xyz")
        assert err.session_id == "xyz"

    def test_is_key_error_subclass(self) -> None:
        err = SessionNotFoundError("s1")
        assert isinstance(err, KeyError)


# ---------------------------------------------------------------------------
# create_session
# ---------------------------------------------------------------------------


class TestSessionManagerCreate:
    def test_returns_session_state(self, manager: SessionManager) -> None:
        session = manager.create_session()
        assert isinstance(session, SessionState)

    def test_default_agent_id_applied(self, manager: SessionManager) -> None:
        session = manager.create_session()
        assert session.agent_id == "test-agent"

    def test_override_agent_id(self, manager: SessionManager) -> None:
        session = manager.create_session(agent_id="custom-bot")
        assert session.agent_id == "custom-bot"

    def test_parent_session_id_stored(self, manager: SessionManager) -> None:
        session = manager.create_session(parent_session_id="parent-abc")
        assert session.parent_session_id == "parent-abc"

    def test_preferences_stored(self, manager: SessionManager) -> None:
        session = manager.create_session(preferences={"lang": "en"})
        assert session.preferences["lang"] == "en"

    def test_new_session_not_persisted_yet(
        self, manager: SessionManager, backend: InMemoryBackend
    ) -> None:
        session = manager.create_session()
        assert not backend.exists(session.session_id)


# ---------------------------------------------------------------------------
# save_session / load_session / session_exists
# ---------------------------------------------------------------------------


class TestSessionManagerPersistence:
    def test_save_returns_session_id(self, manager: SessionManager) -> None:
        session = manager.create_session()
        returned_id = manager.save_session(session)
        assert returned_id == session.session_id

    def test_session_exists_after_save(self, manager: SessionManager) -> None:
        session = manager.create_session()
        manager.save_session(session)
        assert manager.session_exists(session.session_id) is True

    def test_session_not_exists_before_save(self, manager: SessionManager) -> None:
        session = manager.create_session()
        assert manager.session_exists(session.session_id) is False

    def test_load_session_returns_correct_id(self, manager: SessionManager) -> None:
        session = manager.create_session()
        manager.save_session(session)
        loaded = manager.load_session(session.session_id)
        assert loaded.session_id == session.session_id

    def test_load_session_preserves_agent_id(self, manager: SessionManager) -> None:
        session = manager.create_session(agent_id="special-bot")
        manager.save_session(session)
        loaded = manager.load_session(session.session_id)
        assert loaded.agent_id == "special-bot"

    def test_load_session_preserves_segments(self, manager: SessionManager) -> None:
        session = manager.create_session()
        session.add_segment("user", "hello", token_count=5)
        manager.save_session(session)
        loaded = manager.load_session(session.session_id)
        assert len(loaded.segments) == 1
        assert loaded.segments[0].content == "hello"

    def test_load_nonexistent_raises_not_found_error(
        self, manager: SessionManager
    ) -> None:
        with pytest.raises(SessionNotFoundError):
            manager.load_session("ghost-id")

    def test_save_overwrites_existing(self, manager: SessionManager) -> None:
        session = manager.create_session()
        manager.save_session(session)
        session.summary = "updated summary"
        manager.save_session(session)
        loaded = manager.load_session(session.session_id)
        assert loaded.summary == "updated summary"


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


class TestSessionManagerDelete:
    def test_delete_removes_session(self, manager: SessionManager) -> None:
        session = manager.create_session()
        manager.save_session(session)
        manager.delete_session(session.session_id)
        assert manager.session_exists(session.session_id) is False

    def test_delete_nonexistent_raises_not_found(
        self, manager: SessionManager
    ) -> None:
        with pytest.raises(SessionNotFoundError):
            manager.delete_session("does-not-exist")


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


class TestSessionManagerList:
    def test_empty_backend_returns_empty_list(
        self, manager: SessionManager
    ) -> None:
        assert manager.list_sessions() == []

    def test_list_returns_sorted_ids(self, manager: SessionManager) -> None:
        s1 = manager.create_session()
        s2 = manager.create_session()
        manager.save_session(s1)
        manager.save_session(s2)
        ids = manager.list_sessions()
        assert ids == sorted(ids)

    def test_list_contains_saved_sessions(self, manager: SessionManager) -> None:
        session = manager.create_session()
        manager.save_session(session)
        assert session.session_id in manager.list_sessions()

    def test_list_count_matches_saved(self, manager: SessionManager) -> None:
        for _ in range(3):
            s = manager.create_session()
            manager.save_session(s)
        assert len(manager.list_sessions()) == 3

    def test_list_sessions_for_agent_filters_correctly(
        self, manager: SessionManager
    ) -> None:
        s_alpha = manager.create_session(agent_id="alpha")
        s_beta = manager.create_session(agent_id="beta")
        manager.save_session(s_alpha)
        manager.save_session(s_beta)
        alpha_ids = manager.list_sessions_for_agent("alpha")
        assert s_alpha.session_id in alpha_ids
        assert s_beta.session_id not in alpha_ids

    def test_list_sessions_for_agent_unknown_agent_empty(
        self, manager: SessionManager
    ) -> None:
        session = manager.create_session()
        manager.save_session(session)
        assert manager.list_sessions_for_agent("unknown-agent") == []


# ---------------------------------------------------------------------------
# continue_session
# ---------------------------------------------------------------------------


class TestSessionManagerContinue:
    def test_continue_creates_child_with_parent_id(
        self, manager: SessionManager
    ) -> None:
        parent = manager.create_session()
        manager.save_session(parent)
        child = manager.continue_session(parent.session_id)
        assert child.parent_session_id == parent.session_id

    def test_continue_inherits_agent_id(self, manager: SessionManager) -> None:
        parent = manager.create_session(agent_id="my-bot")
        manager.save_session(parent)
        child = manager.continue_session(parent.session_id)
        assert child.agent_id == "my-bot"

    def test_continue_inherits_preferences(self, manager: SessionManager) -> None:
        parent = manager.create_session(preferences={"theme": "dark"})
        manager.save_session(parent)
        child = manager.continue_session(parent.session_id)
        assert child.preferences.get("theme") == "dark"

    def test_continue_inherits_entities(self, manager: SessionManager) -> None:
        parent = manager.create_session()
        parent.track_entity("OpenAI", "org")
        manager.save_session(parent)
        child = manager.continue_session(parent.session_id)
        assert any(e.canonical_name == "OpenAI" for e in child.entities)

    def test_continue_inherits_only_active_tasks(
        self, manager: SessionManager
    ) -> None:
        parent = manager.create_session()
        active_task = parent.add_task("Active work")
        done_task = parent.add_task("Finished work")
        parent.update_task(done_task.task_id, status=TaskStatus.COMPLETED)
        manager.save_session(parent)
        child = manager.continue_session(parent.session_id)
        child_task_ids = [t.task_id for t in child.tasks]
        assert active_task.task_id in child_task_ids
        assert done_task.task_id not in child_task_ids

    def test_continue_nonexistent_parent_raises(
        self, manager: SessionManager
    ) -> None:
        with pytest.raises(SessionNotFoundError):
            manager.continue_session("ghost-parent")

    def test_continue_child_has_new_session_id(
        self, manager: SessionManager
    ) -> None:
        parent = manager.create_session()
        manager.save_session(parent)
        child = manager.continue_session(parent.session_id)
        assert child.session_id != parent.session_id


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestSessionManagerStats:
    def test_stats_empty_backend(self, manager: SessionManager) -> None:
        stats = manager.get_stats()
        assert stats["total_sessions"] == 0
        assert stats["total_segments"] == 0
        assert stats["total_tokens"] == 0
        assert stats["agents"] == []

    def test_stats_counts_sessions(self, manager: SessionManager) -> None:
        for _ in range(2):
            s = manager.create_session()
            manager.save_session(s)
        stats = manager.get_stats()
        assert stats["total_sessions"] == 2

    def test_stats_counts_segments(self, manager: SessionManager) -> None:
        session = manager.create_session()
        session.add_segment("user", "a", token_count=10)
        session.add_segment("assistant", "b", token_count=20)
        manager.save_session(session)
        stats = manager.get_stats()
        assert stats["total_segments"] == 2
        assert stats["total_tokens"] == 30

    def test_stats_collects_unique_agents(self, manager: SessionManager) -> None:
        s1 = manager.create_session(agent_id="alpha")
        s2 = manager.create_session(agent_id="beta")
        s3 = manager.create_session(agent_id="alpha")
        for s in (s1, s2, s3):
            manager.save_session(s)
        stats = manager.get_stats()
        assert "alpha" in stats["agents"]
        assert "beta" in stats["agents"]
        assert len(stats["agents"]) == 2
