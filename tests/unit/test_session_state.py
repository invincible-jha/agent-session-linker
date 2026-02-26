"""Unit tests for agent_session_linker.session.state.

Tests cover SessionState, ContextSegment, EntityReference, TaskState,
ToolContext, and TaskStatus — creation, mutation, helpers, and validation.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from agent_session_linker.session.state import (
    ContextSegment,
    EntityReference,
    SessionState,
    TaskState,
    TaskStatus,
    ToolContext,
)


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_all_values_are_strings(self) -> None:
        for status in TaskStatus:
            assert isinstance(status.value, str)

    def test_pending_value(self) -> None:
        assert TaskStatus.PENDING.value == "pending"

    def test_in_progress_value(self) -> None:
        assert TaskStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_value(self) -> None:
        assert TaskStatus.COMPLETED.value == "completed"

    def test_failed_value(self) -> None:
        assert TaskStatus.FAILED.value == "failed"

    def test_cancelled_value(self) -> None:
        assert TaskStatus.CANCELLED.value == "cancelled"

    def test_five_statuses_defined(self) -> None:
        assert len(list(TaskStatus)) == 5


# ---------------------------------------------------------------------------
# ContextSegment
# ---------------------------------------------------------------------------


class TestContextSegment:
    def test_required_fields_only(self) -> None:
        seg = ContextSegment(role="user", content="Hello")
        assert seg.role == "user"
        assert seg.content == "Hello"

    def test_defaults(self) -> None:
        seg = ContextSegment(role="user", content="Hi")
        assert seg.token_count == 0
        assert seg.segment_type == "conversation"
        assert seg.turn_index == 0
        assert seg.metadata == {}

    def test_segment_id_is_auto_generated_uuid(self) -> None:
        seg = ContextSegment(role="user", content="x")
        assert len(seg.segment_id) == 36
        assert seg.segment_id.count("-") == 4

    def test_two_segments_have_different_ids(self) -> None:
        s1 = ContextSegment(role="user", content="a")
        s2 = ContextSegment(role="user", content="b")
        assert s1.segment_id != s2.segment_id

    def test_timestamp_is_utc(self) -> None:
        seg = ContextSegment(role="user", content="t")
        assert seg.timestamp.tzinfo is not None

    def test_explicit_token_count(self) -> None:
        seg = ContextSegment(role="assistant", content="reply", token_count=42)
        assert seg.token_count == 42

    def test_metadata_stored(self) -> None:
        seg = ContextSegment(role="tool", content="result", metadata={"key": "val"})
        assert seg.metadata["key"] == "val"

    def test_mutable_fields(self) -> None:
        seg = ContextSegment(role="user", content="original")
        seg.content = "updated"
        assert seg.content == "updated"


# ---------------------------------------------------------------------------
# EntityReference
# ---------------------------------------------------------------------------


class TestEntityReference:
    def test_required_field(self) -> None:
        entity = EntityReference(canonical_name="OpenAI")
        assert entity.canonical_name == "OpenAI"

    def test_defaults(self) -> None:
        entity = EntityReference(canonical_name="Acme")
        assert entity.entity_type == "concept"
        assert entity.aliases == []
        assert entity.attributes == {}
        assert entity.first_seen_session == ""
        assert entity.last_seen_session == ""
        assert entity.confidence == 1.0

    def test_confidence_validation_upper(self) -> None:
        with pytest.raises(Exception):
            EntityReference(canonical_name="x", confidence=1.1)

    def test_confidence_validation_lower(self) -> None:
        with pytest.raises(Exception):
            EntityReference(canonical_name="x", confidence=-0.1)

    def test_aliases_stored(self) -> None:
        entity = EntityReference(canonical_name="AI", aliases=["Artificial Intelligence"])
        assert "Artificial Intelligence" in entity.aliases

    def test_auto_id_generated(self) -> None:
        e1 = EntityReference(canonical_name="A")
        e2 = EntityReference(canonical_name="B")
        assert e1.entity_id != e2.entity_id


# ---------------------------------------------------------------------------
# TaskState
# ---------------------------------------------------------------------------


class TestTaskState:
    def test_required_field(self) -> None:
        task = TaskState(title="Write tests")
        assert task.title == "Write tests"

    def test_default_status_is_pending(self) -> None:
        task = TaskState(title="T")
        assert task.status == TaskStatus.PENDING

    def test_default_priority(self) -> None:
        task = TaskState(title="T")
        assert task.priority == 5

    def test_priority_validation_upper(self) -> None:
        with pytest.raises(Exception):
            TaskState(title="T", priority=11)

    def test_priority_validation_lower(self) -> None:
        with pytest.raises(Exception):
            TaskState(title="T", priority=0)

    def test_mark_in_progress(self) -> None:
        task = TaskState(title="T")
        before = task.updated_at
        time.sleep(0.001)
        task.mark_in_progress()
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.updated_at >= before

    def test_mark_completed(self) -> None:
        task = TaskState(title="T")
        task.mark_completed()
        assert task.status == TaskStatus.COMPLETED

    def test_mark_failed(self) -> None:
        task = TaskState(title="T")
        task.mark_failed()
        assert task.status == TaskStatus.FAILED

    def test_parent_task_id_optional(self) -> None:
        task = TaskState(title="Child", parent_task_id="parent-uuid")
        assert task.parent_task_id == "parent-uuid"

    def test_tags_stored(self) -> None:
        task = TaskState(title="Tagged", tags=["urgent", "backend"])
        assert "urgent" in task.tags


# ---------------------------------------------------------------------------
# ToolContext
# ---------------------------------------------------------------------------


class TestToolContext:
    def test_required_field(self) -> None:
        tool = ToolContext(tool_name="read_file")
        assert tool.tool_name == "read_file"

    def test_defaults(self) -> None:
        tool = ToolContext(tool_name="search")
        assert tool.duration_ms == 0.0
        assert tool.success is True
        assert tool.error_message == ""
        assert tool.token_cost == 0

    def test_failed_invocation(self) -> None:
        tool = ToolContext(tool_name="bad_tool", success=False, error_message="timeout")
        assert tool.success is False
        assert tool.error_message == "timeout"


# ---------------------------------------------------------------------------
# SessionState — creation and schema
# ---------------------------------------------------------------------------


class TestSessionStateCreation:
    def test_default_agent_id(self) -> None:
        session = SessionState()
        assert session.agent_id == "default"

    def test_schema_version_set(self) -> None:
        session = SessionState()
        assert session.schema_version == "1.0"

    def test_empty_collections_on_creation(self) -> None:
        session = SessionState()
        assert session.segments == []
        assert session.entities == []
        assert session.tasks == []
        assert session.tools_used == []

    def test_session_id_is_uuid(self) -> None:
        session = SessionState()
        assert len(session.session_id) == 36

    def test_two_sessions_have_different_ids(self) -> None:
        s1 = SessionState()
        s2 = SessionState()
        assert s1.session_id != s2.session_id

    def test_parent_session_id_stored(self) -> None:
        parent = SessionState()
        child = SessionState(parent_session_id=parent.session_id)
        assert child.parent_session_id == parent.session_id


# ---------------------------------------------------------------------------
# SessionState — checksum
# ---------------------------------------------------------------------------


class TestSessionStateChecksum:
    def test_compute_checksum_returns_64_char_hex(self) -> None:
        session = SessionState()
        digest = session.compute_checksum()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_compute_checksum_stored_on_state(self) -> None:
        session = SessionState()
        digest = session.compute_checksum()
        assert session.checksum == digest

    def test_verify_checksum_passes_after_compute(self) -> None:
        session = SessionState()
        session.compute_checksum()
        assert session.verify_checksum() is True

    def test_verify_checksum_fails_after_tampering(self) -> None:
        session = SessionState()
        session.compute_checksum()
        session.agent_id = "tampered"
        assert session.verify_checksum() is False

    def test_checksum_deterministic(self) -> None:
        session = SessionState(session_id="fixed-id", agent_id="bot")
        d1 = session.compute_checksum()
        d2 = session.compute_checksum()
        assert d1 == d2

    def test_different_sessions_different_checksums(self) -> None:
        s1 = SessionState(agent_id="alpha")
        s2 = SessionState(agent_id="beta")
        assert s1.compute_checksum() != s2.compute_checksum()


# ---------------------------------------------------------------------------
# SessionState — add_segment
# ---------------------------------------------------------------------------


class TestSessionStateAddSegment:
    def test_add_segment_returns_segment(self) -> None:
        session = SessionState()
        seg = session.add_segment("user", "Hello")
        assert isinstance(seg, ContextSegment)

    def test_add_segment_appends_to_list(self) -> None:
        session = SessionState()
        session.add_segment("user", "Hello")
        assert len(session.segments) == 1

    def test_add_segment_turn_index_increments(self) -> None:
        session = SessionState()
        s0 = session.add_segment("user", "first")
        s1 = session.add_segment("assistant", "second")
        assert s0.turn_index == 0
        assert s1.turn_index == 1

    def test_add_segment_with_token_count(self) -> None:
        session = SessionState()
        seg = session.add_segment("user", "text", token_count=100)
        assert seg.token_count == 100

    def test_add_segment_with_metadata(self) -> None:
        session = SessionState()
        seg = session.add_segment("system", "instructions", metadata={"src": "config"})
        assert seg.metadata["src"] == "config"

    def test_add_segment_updates_updated_at(self) -> None:
        session = SessionState()
        before = session.updated_at
        time.sleep(0.001)
        session.add_segment("user", "ping")
        assert session.updated_at >= before


# ---------------------------------------------------------------------------
# SessionState — track_entity
# ---------------------------------------------------------------------------


class TestSessionStateTrackEntity:
    def test_track_entity_creates_new(self) -> None:
        session = SessionState()
        entity = session.track_entity("OpenAI", "org")
        assert entity.canonical_name == "OpenAI"
        assert len(session.entities) == 1

    def test_track_entity_deduplicates_case_insensitive(self) -> None:
        session = SessionState()
        e1 = session.track_entity("OpenAI", "org")
        e2 = session.track_entity("openai", "org")
        assert e1 is e2
        assert len(session.entities) == 1

    def test_track_entity_sets_first_seen_session(self) -> None:
        session = SessionState()
        entity = session.track_entity("Acme", "org")
        assert entity.first_seen_session == session.session_id

    def test_track_entity_updates_last_seen_on_revisit(self) -> None:
        session = SessionState()
        entity = session.track_entity("Acme", "org")
        _ = session.track_entity("Acme", "org")
        assert entity.last_seen_session == session.session_id

    def test_track_entity_with_aliases_and_confidence(self) -> None:
        session = SessionState()
        entity = session.track_entity(
            "ML", "concept", aliases=["machine learning"], confidence=0.9
        )
        assert "machine learning" in entity.aliases
        assert entity.confidence == 0.9


# ---------------------------------------------------------------------------
# SessionState — add_task / update_task
# ---------------------------------------------------------------------------


class TestSessionStateTaskManagement:
    def test_add_task_returns_task_state(self) -> None:
        session = SessionState()
        task = session.add_task("Implement feature")
        assert isinstance(task, TaskState)

    def test_add_task_appends_to_list(self) -> None:
        session = SessionState()
        session.add_task("Task A")
        session.add_task("Task B")
        assert len(session.tasks) == 2

    def test_add_task_with_priority_and_tags(self) -> None:
        session = SessionState()
        task = session.add_task("Urgent", priority=1, tags=["critical"])
        assert task.priority == 1
        assert "critical" in task.tags

    def test_update_task_changes_status(self) -> None:
        session = SessionState()
        task = session.add_task("Do work")
        session.update_task(task.task_id, status=TaskStatus.COMPLETED)
        assert task.status == TaskStatus.COMPLETED

    def test_update_task_appends_notes(self) -> None:
        # add_task() does not accept a notes keyword argument; create the task
        # first and then set the initial note via update_task.
        session = SessionState()
        task = session.add_task("Investigate")
        session.update_task(task.task_id, notes="initial note")
        session.update_task(task.task_id, notes="additional note")
        assert "additional note" in task.notes

    def test_update_task_not_found_raises_key_error(self) -> None:
        session = SessionState()
        with pytest.raises(KeyError):
            session.update_task("nonexistent-id", status=TaskStatus.FAILED)

    def test_update_task_changes_priority(self) -> None:
        session = SessionState()
        task = session.add_task("Task", priority=5)
        session.update_task(task.task_id, priority=2)
        assert task.priority == 2


# ---------------------------------------------------------------------------
# SessionState — total_tokens
# ---------------------------------------------------------------------------


class TestSessionStateTotalTokens:
    def test_empty_session_returns_zero(self) -> None:
        session = SessionState()
        assert session.total_tokens() == 0

    def test_sums_segment_token_counts(self) -> None:
        session = SessionState()
        session.add_segment("user", "a", token_count=10)
        session.add_segment("assistant", "b", token_count=20)
        assert session.total_tokens() == 30

    def test_zero_token_count_segments_not_counted(self) -> None:
        session = SessionState()
        session.add_segment("user", "no tokens")
        assert session.total_tokens() == 0
