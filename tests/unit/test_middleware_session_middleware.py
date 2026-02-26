"""Unit tests for agent_session_linker.middleware.session_middleware.SessionMiddleware."""
from __future__ import annotations

import pytest

from agent_session_linker.middleware.session_middleware import SessionMiddleware
from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
from agent_session_linker.session.state import ContextSegment, SessionState
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


@pytest.fixture()
def middleware(manager: SessionManager) -> SessionMiddleware:
    return SessionMiddleware(manager=manager)


@pytest.fixture()
def saved_session(manager: SessionManager) -> SessionState:
    session = manager.create_session()
    manager.save_session(session)
    return session


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestSessionMiddlewareConstruction:
    def test_auto_create_defaults_true(self, middleware: SessionMiddleware) -> None:
        assert middleware.auto_create is True

    def test_auto_create_can_be_disabled(self, manager: SessionManager) -> None:
        mw = SessionMiddleware(manager=manager, auto_create=False)
        assert mw.auto_create is False

    def test_no_active_sessions_initially(self, middleware: SessionMiddleware) -> None:
        assert middleware.get_active("any-id") is None


# ---------------------------------------------------------------------------
# before_request
# ---------------------------------------------------------------------------


class TestSessionMiddlewareBeforeRequest:
    def test_loads_existing_session(
        self,
        middleware: SessionMiddleware,
        saved_session: SessionState,
    ) -> None:
        loaded = middleware.before_request(saved_session.session_id)
        assert loaded.session_id == saved_session.session_id

    def test_caches_loaded_session(
        self,
        middleware: SessionMiddleware,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        assert middleware.get_active(saved_session.session_id) is not None

    def test_creates_new_session_when_auto_create_is_true(
        self, middleware: SessionMiddleware
    ) -> None:
        session = middleware.before_request("brand-new-id")
        assert session.session_id == "brand-new-id"

    def test_before_request_returns_session_state(
        self, middleware: SessionMiddleware, saved_session: SessionState
    ) -> None:
        result = middleware.before_request(saved_session.session_id)
        assert isinstance(result, SessionState)

    def test_raises_not_found_when_auto_create_disabled(
        self, manager: SessionManager
    ) -> None:
        mw = SessionMiddleware(manager=manager, auto_create=False)
        with pytest.raises(SessionNotFoundError):
            mw.before_request("nonexistent-session")

    def test_created_session_id_matches_requested(
        self, middleware: SessionMiddleware
    ) -> None:
        session = middleware.before_request("my-custom-id")
        assert session.session_id == "my-custom-id"


# ---------------------------------------------------------------------------
# after_request
# ---------------------------------------------------------------------------


class TestSessionMiddlewareAfterRequest:
    def test_after_request_returns_session_id(
        self,
        middleware: SessionMiddleware,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        saved_id = middleware.after_request(saved_session.session_id)
        assert saved_id == saved_session.session_id

    def test_after_request_persists_session(
        self,
        middleware: SessionMiddleware,
        manager: SessionManager,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        middleware.after_request(saved_session.session_id)
        assert manager.session_exists(saved_session.session_id)

    def test_after_request_removes_from_active_cache(
        self,
        middleware: SessionMiddleware,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        middleware.after_request(saved_session.session_id)
        assert middleware.get_active(saved_session.session_id) is None

    def test_after_request_appends_string_context(
        self,
        middleware: SessionMiddleware,
        manager: SessionManager,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        middleware.after_request(saved_session.session_id, new_context="assistant reply")
        loaded = manager.load_session(saved_session.session_id)
        contents = [s.content for s in loaded.segments]
        assert "assistant reply" in contents

    def test_after_request_appends_segment_list(
        self,
        middleware: SessionMiddleware,
        manager: SessionManager,
        saved_session: SessionState,
    ) -> None:
        from datetime import datetime, timezone

        segment = ContextSegment(
            role="user",
            content="extra segment",
            token_count=5,
            turn_index=0,
            segment_type="conversation",
            timestamp=datetime.now(timezone.utc),
        )
        middleware.before_request(saved_session.session_id)
        middleware.after_request(saved_session.session_id, new_context=[segment])
        loaded = manager.load_session(saved_session.session_id)
        assert any(s.content == "extra segment" for s in loaded.segments)

    def test_after_request_with_none_context_saves_unchanged(
        self,
        middleware: SessionMiddleware,
        manager: SessionManager,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        middleware.after_request(saved_session.session_id, new_context=None)
        loaded = manager.load_session(saved_session.session_id)
        assert loaded.session_id == saved_session.session_id

    def test_after_request_without_before_request_raises_key_error(
        self, middleware: SessionMiddleware
    ) -> None:
        with pytest.raises(KeyError, match="no-prior-before"):
            middleware.after_request("no-prior-before")

    def test_full_request_cycle(
        self,
        middleware: SessionMiddleware,
        manager: SessionManager,
        saved_session: SessionState,
    ) -> None:
        session = middleware.before_request(saved_session.session_id)
        assert isinstance(session, SessionState)
        saved_id = middleware.after_request(
            saved_session.session_id, new_context="done"
        )
        assert saved_id == saved_session.session_id
        loaded = manager.load_session(saved_id)
        assert any(s.content == "done" for s in loaded.segments)


# ---------------------------------------------------------------------------
# get_active
# ---------------------------------------------------------------------------


class TestSessionMiddlewareGetActive:
    def test_get_active_none_before_before_request(
        self, middleware: SessionMiddleware
    ) -> None:
        assert middleware.get_active("any") is None

    def test_get_active_returns_session_after_before_request(
        self,
        middleware: SessionMiddleware,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        active = middleware.get_active(saved_session.session_id)
        assert active is not None
        assert active.session_id == saved_session.session_id


# ---------------------------------------------------------------------------
# clear_active
# ---------------------------------------------------------------------------


class TestSessionMiddlewareClearActive:
    def test_clear_active_removes_from_cache(
        self,
        middleware: SessionMiddleware,
        saved_session: SessionState,
    ) -> None:
        middleware.before_request(saved_session.session_id)
        middleware.clear_active(saved_session.session_id)
        assert middleware.get_active(saved_session.session_id) is None

    def test_clear_active_on_missing_id_does_not_raise(
        self, middleware: SessionMiddleware
    ) -> None:
        # Should be a no-op, not an error.
        middleware.clear_active("nonexistent")

    def test_clear_active_does_not_save_session(
        self,
        middleware: SessionMiddleware,
        manager: SessionManager,
    ) -> None:
        """Clearing should not persist any changes to the backend."""
        mw = SessionMiddleware(manager=manager, auto_create=True)
        mw.before_request("ephemeral-session")
        # The auto-created session hasn't been saved to backend yet.
        mw.clear_active("ephemeral-session")
        assert not manager.session_exists("ephemeral-session")
