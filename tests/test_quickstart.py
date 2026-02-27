"""Test that the 3-line quickstart API works for agent-session-linker."""
from __future__ import annotations


def test_quickstart_import() -> None:
    from agent_session_linker import Session

    session = Session()
    assert session is not None


def test_quickstart_session_id() -> None:
    from agent_session_linker import Session

    session = Session()
    assert isinstance(session.session_id, str)
    assert len(session.session_id) > 0


def test_quickstart_save_and_load() -> None:
    from agent_session_linker import Session

    session = Session()
    session_id = session.save()
    assert isinstance(session_id, str)

    restored = session.load(session_id)
    assert restored is not None
    assert restored.session_id == session_id


def test_quickstart_add_context() -> None:
    from agent_session_linker import Session

    session = Session()
    session.add_context("user_language", "python")
    assert session.state.preferences.get("user_language") == "python"


def test_quickstart_state_accessible() -> None:
    from agent_session_linker import Session
    from agent_session_linker.session.state import SessionState

    session = Session()
    assert isinstance(session.state, SessionState)


def test_quickstart_repr() -> None:
    from agent_session_linker import Session

    session = Session(agent_id="my-agent")
    text = repr(session)
    assert "Session" in text
    assert "my-agent" in text
