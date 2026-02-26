"""Unit tests for agent_session_linker.linking.chain.SessionChain.

All session loading is isolated via InMemoryBackend and SessionManager so
no external I/O is required.
"""
from __future__ import annotations

import pytest

from agent_session_linker.linking.chain import SessionChain
from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
from agent_session_linker.session.state import SessionState
from agent_session_linker.storage.memory import InMemoryBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def manager() -> SessionManager:
    backend = InMemoryBackend()
    return SessionManager(backend=backend, default_agent_id="test-agent")


@pytest.fixture()
def saved_session(manager: SessionManager) -> SessionState:
    session = manager.create_session()
    session.add_segment("user", "hello", token_count=10)
    session.add_segment("assistant", "world", token_count=15)
    manager.save_session(session)
    return session


@pytest.fixture()
def chain(manager: SessionManager) -> SessionChain:
    return SessionChain(manager=manager)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSessionChainConstruction:
    def test_empty_chain_has_zero_len(self, chain: SessionChain) -> None:
        assert len(chain) == 0

    def test_initial_ids_accepted(self, manager: SessionManager) -> None:
        chain = SessionChain(manager=manager, initial_session_ids=["a", "b"])
        assert len(chain) == 2

    def test_get_chain_returns_copy(self, manager: SessionManager) -> None:
        chain = SessionChain(manager=manager, initial_session_ids=["a"])
        copy = chain.get_chain()
        copy.append("b")
        assert len(chain) == 1

    def test_repr_contains_length(self, chain: SessionChain) -> None:
        assert "0" in repr(chain)


# ---------------------------------------------------------------------------
# append / prepend / remove
# ---------------------------------------------------------------------------


class TestSessionChainMutation:
    def test_append_increases_len(self, chain: SessionChain) -> None:
        chain.append("s1")
        assert len(chain) == 1

    def test_append_order_preserved(self, chain: SessionChain) -> None:
        chain.append("s1")
        chain.append("s2")
        assert chain.get_chain() == ["s1", "s2"]

    def test_prepend_adds_at_front(self, chain: SessionChain) -> None:
        chain.append("s2")
        chain.prepend("s1")
        assert chain.get_chain()[0] == "s1"

    def test_prepend_increases_len(self, chain: SessionChain) -> None:
        chain.prepend("s1")
        assert len(chain) == 1

    def test_remove_decreases_len(self, chain: SessionChain) -> None:
        chain.append("s1")
        chain.remove("s1")
        assert len(chain) == 0

    def test_remove_only_first_occurrence(self, chain: SessionChain) -> None:
        chain.append("s1")
        chain.append("s1")
        chain.remove("s1")
        assert len(chain) == 1

    def test_remove_missing_raises_value_error(self, chain: SessionChain) -> None:
        with pytest.raises(ValueError):
            chain.remove("ghost")

    def test_duplicate_ids_allowed(self, chain: SessionChain) -> None:
        chain.append("s1")
        chain.append("s1")
        assert len(chain) == 2


# ---------------------------------------------------------------------------
# __contains__
# ---------------------------------------------------------------------------


class TestSessionChainContains:
    def test_contains_true_after_append(self, chain: SessionChain) -> None:
        chain.append("s1")
        assert "s1" in chain

    def test_contains_false_before_append(self, chain: SessionChain) -> None:
        assert "s1" not in chain

    def test_contains_false_after_remove(self, chain: SessionChain) -> None:
        chain.append("s1")
        chain.remove("s1")
        assert "s1" not in chain


# ---------------------------------------------------------------------------
# get_sessions
# ---------------------------------------------------------------------------


class TestSessionChainGetSessions:
    def test_get_sessions_loads_existing(
        self, chain: SessionChain, saved_session: SessionState
    ) -> None:
        chain.append(saved_session.session_id)
        sessions = chain.get_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == saved_session.session_id

    def test_get_sessions_skips_missing(self, chain: SessionChain) -> None:
        chain.append("nonexistent-session")
        sessions = chain.get_sessions()
        assert sessions == []

    def test_get_sessions_returns_in_order(
        self, manager: SessionManager
    ) -> None:
        s1 = manager.create_session()
        s2 = manager.create_session()
        manager.save_session(s1)
        manager.save_session(s2)
        chain = SessionChain(manager=manager, initial_session_ids=[s1.session_id, s2.session_id])
        sessions = chain.get_sessions()
        assert sessions[0].session_id == s1.session_id
        assert sessions[1].session_id == s2.session_id


# ---------------------------------------------------------------------------
# get_context_from_chain
# ---------------------------------------------------------------------------


class TestSessionChainGetContext:
    def test_raises_value_error_for_n_recent_zero(
        self, chain: SessionChain
    ) -> None:
        with pytest.raises(ValueError, match="n_recent"):
            chain.get_context_from_chain(0)

    def test_raises_value_error_for_negative_n_recent(
        self, chain: SessionChain
    ) -> None:
        with pytest.raises(ValueError):
            chain.get_context_from_chain(-1)

    def test_returns_empty_string_for_empty_chain(
        self, chain: SessionChain
    ) -> None:
        assert chain.get_context_from_chain(1) == ""

    def test_returns_context_for_valid_session(
        self, chain: SessionChain, saved_session: SessionState
    ) -> None:
        chain.append(saved_session.session_id)
        context = chain.get_context_from_chain(1)
        assert "hello" in context
        assert "world" in context

    def test_respects_n_recent_limit(self, manager: SessionManager) -> None:
        sessions = []
        for i in range(3):
            s = manager.create_session()
            s.add_segment("user", f"message-{i}", token_count=5)
            manager.save_session(s)
            sessions.append(s)
        chain = SessionChain(
            manager=manager,
            initial_session_ids=[s.session_id for s in sessions],
        )
        context = chain.get_context_from_chain(1)
        # Only last session's content should appear.
        assert "message-2" in context
        assert "message-0" not in context

    def test_skips_sessions_without_segments(self, manager: SessionManager) -> None:
        empty_session = manager.create_session()
        manager.save_session(empty_session)
        chain = SessionChain(
            manager=manager, initial_session_ids=[empty_session.session_id]
        )
        assert chain.get_context_from_chain(1) == ""

    def test_skips_missing_sessions_in_context(
        self, chain: SessionChain, saved_session: SessionState
    ) -> None:
        chain.append("nonexistent")
        chain.append(saved_session.session_id)
        context = chain.get_context_from_chain(2)
        assert "hello" in context

    def test_context_contains_role_labels(
        self, chain: SessionChain, saved_session: SessionState
    ) -> None:
        chain.append(saved_session.session_id)
        context = chain.get_context_from_chain(1)
        assert "USER" in context
        assert "ASSISTANT" in context


# ---------------------------------------------------------------------------
# get_all_segments
# ---------------------------------------------------------------------------


class TestSessionChainGetAllSegments:
    def test_returns_empty_for_empty_chain(self, chain: SessionChain) -> None:
        assert chain.get_all_segments() == []

    def test_returns_all_segments_in_order(self, manager: SessionManager) -> None:
        s1 = manager.create_session()
        s1.add_segment("user", "first", token_count=5)
        manager.save_session(s1)
        s2 = manager.create_session()
        s2.add_segment("assistant", "second", token_count=5)
        manager.save_session(s2)
        chain = SessionChain(
            manager=manager, initial_session_ids=[s1.session_id, s2.session_id]
        )
        segments = chain.get_all_segments()
        assert len(segments) == 2
        assert segments[0].content == "first"
        assert segments[1].content == "second"

    def test_n_recent_limits_sessions(self, manager: SessionManager) -> None:
        sessions = []
        for i in range(3):
            s = manager.create_session()
            s.add_segment("user", f"msg-{i}", token_count=5)
            manager.save_session(s)
            sessions.append(s)
        chain = SessionChain(
            manager=manager,
            initial_session_ids=[s.session_id for s in sessions],
        )
        segments = chain.get_all_segments(n_recent=1)
        assert len(segments) == 1
        assert segments[0].content == "msg-2"

    def test_none_n_recent_returns_all(self, manager: SessionManager) -> None:
        sessions = []
        for i in range(3):
            s = manager.create_session()
            s.add_segment("user", f"item-{i}", token_count=5)
            manager.save_session(s)
            sessions.append(s)
        chain = SessionChain(
            manager=manager,
            initial_session_ids=[s.session_id for s in sessions],
        )
        segments = chain.get_all_segments(n_recent=None)
        assert len(segments) == 3

    def test_skips_sessions_that_fail_to_load(
        self, chain: SessionChain, saved_session: SessionState
    ) -> None:
        chain.append("nonexistent")
        chain.append(saved_session.session_id)
        segments = chain.get_all_segments()
        assert len(segments) == 2


# ---------------------------------------------------------------------------
# _format_segments (static)
# ---------------------------------------------------------------------------


class TestSessionChainFormatSegments:
    def test_format_produces_role_label(self, saved_session: SessionState) -> None:
        formatted = SessionChain._format_segments(saved_session.segments)
        assert "USER" in formatted
        assert "ASSISTANT" in formatted

    def test_format_includes_content(self, saved_session: SessionState) -> None:
        formatted = SessionChain._format_segments(saved_session.segments)
        assert "hello" in formatted
        assert "world" in formatted

    def test_format_empty_list_returns_empty_string(self) -> None:
        assert SessionChain._format_segments([]) == ""
