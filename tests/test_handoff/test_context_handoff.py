"""Tests for agent_session_linker.handoff.context_handoff."""
from __future__ import annotations

import pytest

from agent_session_linker.handoff.context_handoff import (
    HandoffBuilder,
    HandoffConfig,
    HandoffPayload,
)
from agent_session_linker.session.state import SessionState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session(agent_id: str = "agent_a") -> SessionState:
    s = SessionState(agent_id=agent_id)
    s.add_segment("user", "Hello, I need help with deployment.")
    s.add_segment("assistant", "Sure, let me help you deploy the service.")
    s.add_segment("user", "Please use the staging environment first.")
    s.add_task(title="Deploy to staging")
    s.add_task(title="Verify health checks")
    s.track_entity("staging-env", entity_type="environment")
    s.preferences["language"] = "en"
    s.summary = "User wants to deploy service to staging."
    return s


# ===========================================================================
# HandoffConfig
# ===========================================================================


class TestHandoffConfig:
    def test_defaults(self) -> None:
        config = HandoffConfig()
        assert config.max_segments is None
        assert config.include_entities is True
        assert config.include_tasks is True
        assert config.include_preferences is True
        assert config.include_summary is True
        assert config.segment_types == ()

    def test_frozen(self) -> None:
        config = HandoffConfig()
        with pytest.raises(Exception):
            config.max_segments = 5  # type: ignore[misc]

    def test_negative_max_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="max_segments"):
            HandoffConfig(max_segments=-1)

    def test_zero_max_segments_allowed(self) -> None:
        config = HandoffConfig(max_segments=0)
        assert config.max_segments == 0


# ===========================================================================
# HandoffPayload
# ===========================================================================


class TestHandoffPayload:
    def _make_payload(self) -> HandoffPayload:
        session = _make_session()
        builder = HandoffBuilder()
        return builder.build(session, target_agent_id="agent_b")

    def test_frozen(self) -> None:
        payload = self._make_payload()
        with pytest.raises(Exception):
            payload.source_agent_id = "other"  # type: ignore[misc]

    def test_has_handoff_id(self) -> None:
        payload = self._make_payload()
        assert payload.handoff_id
        assert len(payload.handoff_id) > 0

    def test_source_agent_id(self) -> None:
        payload = self._make_payload()
        assert payload.source_agent_id == "agent_a"

    def test_target_agent_id(self) -> None:
        payload = self._make_payload()
        assert payload.target_agent_id == "agent_b"

    def test_segment_count(self) -> None:
        payload = self._make_payload()
        assert payload.segment_count == 3

    def test_task_count(self) -> None:
        payload = self._make_payload()
        assert payload.task_count == 2

    def test_entity_count(self) -> None:
        payload = self._make_payload()
        assert payload.entity_count == 1

    def test_summary_line(self) -> None:
        payload = self._make_payload()
        line = payload.summary_line()
        assert "agent_a" in line
        assert "agent_b" in line

    def test_to_json_and_from_json_roundtrip(self) -> None:
        payload = self._make_payload()
        json_str = payload.to_json()
        restored = HandoffPayload.from_json(json_str)
        assert restored.handoff_id == payload.handoff_id
        assert restored.target_agent_id == payload.target_agent_id
        assert restored.segment_count == payload.segment_count

    def test_handoff_reason_preserved(self) -> None:
        session = _make_session()
        builder = HandoffBuilder()
        payload = builder.build(session, target_agent_id="b", handoff_reason="escalation")
        assert payload.handoff_reason == "escalation"

    def test_created_at_is_utc(self) -> None:
        from datetime import timezone
        payload = self._make_payload()
        assert payload.created_at.tzinfo is not None


# ===========================================================================
# HandoffBuilder — filtering
# ===========================================================================


class TestHandoffBuilderFiltering:
    def test_max_segments_limits_count(self) -> None:
        session = _make_session()
        config = HandoffConfig(max_segments=2)
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.segment_count == 2

    def test_max_segments_takes_most_recent(self) -> None:
        session = _make_session()
        config = HandoffConfig(max_segments=1)
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        # Should be the last segment
        assert "staging" in payload.segments[0]["content"]

    def test_max_segments_zero_includes_none(self) -> None:
        session = _make_session()
        config = HandoffConfig(max_segments=0)
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.segment_count == 0

    def test_exclude_entities(self) -> None:
        session = _make_session()
        config = HandoffConfig(include_entities=False)
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.entity_count == 0

    def test_exclude_tasks(self) -> None:
        session = _make_session()
        config = HandoffConfig(include_tasks=False)
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.task_count == 0

    def test_exclude_preferences(self) -> None:
        session = _make_session()
        config = HandoffConfig(include_preferences=False)
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.preferences == {}

    def test_exclude_summary(self) -> None:
        session = _make_session()
        config = HandoffConfig(include_summary=False)
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.summary == ""

    def test_segment_type_filter(self) -> None:
        session = SessionState(agent_id="a")
        session.add_segment("user", "hello")
        session.add_segment("user", "code here", segment_type="code")
        config = HandoffConfig(segment_types=("code",))
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.segment_count == 1
        assert payload.segments[0]["segment_type"] == "code"

    def test_no_segment_type_filter_includes_all(self) -> None:
        session = _make_session()
        config = HandoffConfig(segment_types=())
        builder = HandoffBuilder(config)
        payload = builder.build(session, target_agent_id="b")
        assert payload.segment_count == 3

    def test_extra_metadata_passed_through(self) -> None:
        session = _make_session()
        builder = HandoffBuilder()
        payload = builder.build(
            session,
            target_agent_id="b",
            extra_metadata={"priority": "high"},
        )
        assert payload.metadata["priority"] == "high"

    def test_summary_included_by_default(self) -> None:
        session = _make_session()
        builder = HandoffBuilder()
        payload = builder.build(session, target_agent_id="b")
        assert "staging" in payload.summary

    def test_preferences_copied(self) -> None:
        session = _make_session()
        builder = HandoffBuilder()
        payload = builder.build(session, target_agent_id="b")
        assert payload.preferences.get("language") == "en"
