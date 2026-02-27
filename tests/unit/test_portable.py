"""Unit tests for the Universal Session Format (USF) portable package.

Tests cover:
- USF dataclass creation, validation, and immutability
- UniversalSession Pydantic model creation, field defaults, and validators
- Checksum computation, storage, and verification
- JSON serialisation (to_json) and deserialisation (from_json)
- LangChainExporter, CrewAIExporter, OpenAIExporter — happy paths and edge cases
- LangChainImporter, CrewAIImporter, OpenAIImporter — happy paths and edge cases
- Round-trip: export → re-import consistency
- Cross-format conversion via USF as the intermediate layer
- CLI commands: portable export, import, convert
- Error handling: invalid inputs, missing fields, corrupt checksums
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_session_linker.cli.main import cli
from agent_session_linker.portable.exporters import (
    CrewAIExporter,
    LangChainExporter,
    OpenAIExporter,
)
from agent_session_linker.portable.importers import (
    CrewAIImporter,
    LangChainImporter,
    OpenAIImporter,
)
from agent_session_linker.portable.usf import (
    USFEntity,
    USFMessage,
    USFTaskState,
    USFVersion,
    UniversalSession,
)


# ---------------------------------------------------------------------------
# Helpers / shared factories
# ---------------------------------------------------------------------------

_UTC = timezone.utc


def _now() -> datetime:
    return datetime.now(_UTC)


def _make_message(
    role: str = "user",
    content: str = "Hello",
    metadata: dict[str, Any] | None = None,
) -> USFMessage:
    return USFMessage(
        role=role,
        content=content,
        timestamp=_now(),
        metadata=metadata or {},
    )


def _make_entity(
    name: str = "Acme",
    entity_type: str = "org",
    value: str = "Acme Corp",
    confidence: float = 0.9,
) -> USFEntity:
    return USFEntity(name=name, entity_type=entity_type, value=value, confidence=confidence)


def _make_task(
    task_id: str = "t-001",
    status: str = "completed",
    progress: float = 1.0,
    result: str | None = "Done",
) -> USFTaskState:
    return USFTaskState(task_id=task_id, status=status, progress=progress, result=result)


def _make_session(**kwargs: Any) -> UniversalSession:
    defaults: dict[str, Any] = {
        "framework_source": "test",
        "messages": [_make_message()],
        "entities": [_make_entity()],
        "task_state": [_make_task()],
        "working_memory": {"key": "value"},
    }
    defaults.update(kwargs)
    return UniversalSession(**defaults)


# ---------------------------------------------------------------------------
# USFVersion
# ---------------------------------------------------------------------------


class TestUSFVersion:
    def test_is_string(self) -> None:
        assert isinstance(USFVersion, str)

    def test_value(self) -> None:
        assert USFVersion == "1.0"

    def test_non_empty(self) -> None:
        assert len(USFVersion) > 0


# ---------------------------------------------------------------------------
# USFMessage
# ---------------------------------------------------------------------------


class TestUSFMessage:
    def test_valid_user_role(self) -> None:
        msg = _make_message(role="user")
        assert msg.role == "user"

    def test_valid_assistant_role(self) -> None:
        msg = _make_message(role="assistant")
        assert msg.role == "assistant"

    def test_valid_system_role(self) -> None:
        msg = _make_message(role="system")
        assert msg.role == "system"

    def test_valid_tool_role(self) -> None:
        msg = _make_message(role="tool")
        assert msg.role == "tool"

    def test_invalid_role_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="role"):
            USFMessage(role="robot", content="hi", timestamp=_now(), metadata={})

    def test_content_stored(self) -> None:
        msg = _make_message(content="Hello world")
        assert msg.content == "Hello world"

    def test_timestamp_stored(self) -> None:
        ts = _now()
        msg = USFMessage(role="user", content="x", timestamp=ts, metadata={})
        assert msg.timestamp == ts

    def test_metadata_stored(self) -> None:
        msg = _make_message(metadata={"src": "config"})
        assert msg.metadata["src"] == "config"

    def test_frozen_immutability(self) -> None:
        msg = _make_message()
        with pytest.raises((AttributeError, TypeError)):
            msg.role = "system"  # type: ignore[misc]

    def test_two_messages_can_be_equal(self) -> None:
        ts = _now()
        m1 = USFMessage(role="user", content="hi", timestamp=ts, metadata={})
        m2 = USFMessage(role="user", content="hi", timestamp=ts, metadata={})
        assert m1 == m2

    def test_empty_content_is_allowed(self) -> None:
        msg = _make_message(content="")
        assert msg.content == ""

    def test_empty_metadata_default(self) -> None:
        msg = _make_message()
        assert isinstance(msg.metadata, dict)


# ---------------------------------------------------------------------------
# USFEntity
# ---------------------------------------------------------------------------


class TestUSFEntity:
    def test_valid_entity(self) -> None:
        entity = _make_entity()
        assert entity.name == "Acme"

    def test_confidence_zero_is_valid(self) -> None:
        entity = USFEntity(name="X", entity_type="t", value="v", confidence=0.0)
        assert entity.confidence == 0.0

    def test_confidence_one_is_valid(self) -> None:
        entity = USFEntity(name="X", entity_type="t", value="v", confidence=1.0)
        assert entity.confidence == 1.0

    def test_confidence_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            USFEntity(name="X", entity_type="t", value="v", confidence=1.1)

    def test_confidence_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="confidence"):
            USFEntity(name="X", entity_type="t", value="v", confidence=-0.01)

    def test_entity_type_stored(self) -> None:
        entity = _make_entity(entity_type="project")
        assert entity.entity_type == "project"

    def test_value_stored(self) -> None:
        entity = _make_entity(value="some value")
        assert entity.value == "some value"

    def test_frozen_immutability(self) -> None:
        entity = _make_entity()
        with pytest.raises((AttributeError, TypeError)):
            entity.name = "Changed"  # type: ignore[misc]

    def test_two_entities_equal_when_same(self) -> None:
        e1 = USFEntity(name="A", entity_type="t", value="v", confidence=0.5)
        e2 = USFEntity(name="A", entity_type="t", value="v", confidence=0.5)
        assert e1 == e2

    def test_empty_value_is_allowed(self) -> None:
        entity = USFEntity(name="E", entity_type="t", value="", confidence=1.0)
        assert entity.value == ""


# ---------------------------------------------------------------------------
# USFTaskState
# ---------------------------------------------------------------------------


class TestUSFTaskState:
    def test_valid_pending_status(self) -> None:
        task = USFTaskState(task_id="t1", status="pending", progress=0.0, result=None)
        assert task.status == "pending"

    def test_valid_in_progress_status(self) -> None:
        task = USFTaskState(task_id="t1", status="in_progress", progress=0.5, result=None)
        assert task.status == "in_progress"

    def test_valid_completed_status(self) -> None:
        task = _make_task(status="completed")
        assert task.status == "completed"

    def test_valid_failed_status(self) -> None:
        task = _make_task(status="failed", result="error msg")
        assert task.status == "failed"

    def test_invalid_status_raises(self) -> None:
        with pytest.raises(ValueError, match="status"):
            USFTaskState(task_id="t1", status="running", progress=0.0, result=None)

    def test_progress_zero_is_valid(self) -> None:
        task = USFTaskState(task_id="t1", status="pending", progress=0.0, result=None)
        assert task.progress == 0.0

    def test_progress_one_is_valid(self) -> None:
        task = USFTaskState(task_id="t1", status="completed", progress=1.0, result=None)
        assert task.progress == 1.0

    def test_progress_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="progress"):
            USFTaskState(task_id="t1", status="pending", progress=1.01, result=None)

    def test_progress_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="progress"):
            USFTaskState(task_id="t1", status="pending", progress=-0.1, result=None)

    def test_result_none_is_allowed(self) -> None:
        task = USFTaskState(task_id="t1", status="pending", progress=0.0, result=None)
        assert task.result is None

    def test_result_stored(self) -> None:
        task = _make_task(result="output text")
        assert task.result == "output text"

    def test_frozen_immutability(self) -> None:
        task = _make_task()
        with pytest.raises((AttributeError, TypeError)):
            task.status = "failed"  # type: ignore[misc]

    def test_task_id_stored(self) -> None:
        task = _make_task(task_id="xyz-999")
        assert task.task_id == "xyz-999"


# ---------------------------------------------------------------------------
# UniversalSession — creation and defaults
# ---------------------------------------------------------------------------


class TestUniversalSessionCreation:
    def test_default_version_is_usf_version(self) -> None:
        session = UniversalSession()
        assert session.version == USFVersion

    def test_session_id_auto_generated(self) -> None:
        session = UniversalSession()
        assert len(session.session_id) == 36  # UUID4 format
        assert session.session_id.count("-") == 4

    def test_two_sessions_have_different_ids(self) -> None:
        s1 = UniversalSession()
        s2 = UniversalSession()
        assert s1.session_id != s2.session_id

    def test_created_at_is_utc(self) -> None:
        session = UniversalSession()
        assert session.created_at.tzinfo is not None

    def test_updated_at_is_utc(self) -> None:
        session = UniversalSession()
        assert session.updated_at.tzinfo is not None

    def test_framework_source_default_empty(self) -> None:
        session = UniversalSession()
        assert session.framework_source == ""

    def test_framework_source_stored(self) -> None:
        session = UniversalSession(framework_source="langchain")
        assert session.framework_source == "langchain"

    def test_empty_messages_default(self) -> None:
        session = UniversalSession()
        assert session.messages == []

    def test_empty_entities_default(self) -> None:
        session = UniversalSession()
        assert session.entities == []

    def test_empty_task_state_default(self) -> None:
        session = UniversalSession()
        assert session.task_state == []

    def test_empty_working_memory_default(self) -> None:
        session = UniversalSession()
        assert session.working_memory == {}

    def test_empty_metadata_default(self) -> None:
        session = UniversalSession()
        assert session.metadata == {}

    def test_checksum_computed_on_creation(self) -> None:
        session = UniversalSession()
        assert len(session.checksum) == 64

    def test_messages_stored(self) -> None:
        msg = _make_message()
        session = UniversalSession(messages=[msg])
        assert len(session.messages) == 1
        assert session.messages[0].content == msg.content

    def test_entities_stored(self) -> None:
        entity = _make_entity()
        session = UniversalSession(entities=[entity])
        assert len(session.entities) == 1
        assert session.entities[0].name == "Acme"

    def test_task_state_stored(self) -> None:
        task = _make_task()
        session = UniversalSession(task_state=[task])
        assert len(session.task_state) == 1

    def test_working_memory_stored(self) -> None:
        session = UniversalSession(working_memory={"foo": "bar"})
        assert session.working_memory["foo"] == "bar"

    def test_metadata_stored(self) -> None:
        session = UniversalSession(metadata={"tag": "v1"})
        assert session.metadata["tag"] == "v1"

    def test_invalid_empty_version_raises(self) -> None:
        with pytest.raises(Exception):
            UniversalSession(version="")


# ---------------------------------------------------------------------------
# UniversalSession — checksum
# ---------------------------------------------------------------------------


class TestUniversalSessionChecksum:
    def test_compute_checksum_returns_64_char_hex(self) -> None:
        session = UniversalSession()
        digest = session.compute_checksum()
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_compute_checksum_stored_on_model(self) -> None:
        session = UniversalSession()
        digest = session.compute_checksum()
        assert session.checksum == digest

    def test_verify_checksum_passes_after_compute(self) -> None:
        session = UniversalSession()
        session.compute_checksum()
        assert session.verify_checksum() is True

    def test_verify_checksum_fails_on_corrupt_checksum(self) -> None:
        session = UniversalSession()
        session.checksum = "deadbeef" * 8
        assert session.verify_checksum() is False

    def test_verify_checksum_does_not_mutate_stored_checksum(self) -> None:
        session = UniversalSession()
        original = session.checksum
        session.verify_checksum()
        assert session.checksum == original

    def test_checksum_is_deterministic(self) -> None:
        session = UniversalSession(session_id="fixed-id", framework_source="test")
        d1 = session.compute_checksum()
        d2 = session.compute_checksum()
        assert d1 == d2

    def test_different_framework_sources_yield_different_checksums(self) -> None:
        s1 = UniversalSession(session_id="same", framework_source="langchain")
        s2 = UniversalSession(session_id="same", framework_source="crewai")
        assert s1.compute_checksum() != s2.compute_checksum()

    def test_different_messages_yield_different_checksums(self) -> None:
        s1 = UniversalSession(messages=[_make_message(content="hello")])
        s2 = UniversalSession(messages=[_make_message(content="world")])
        assert s1.compute_checksum() != s2.compute_checksum()

    def test_checksum_changes_when_working_memory_changes(self) -> None:
        session = UniversalSession()
        d1 = session.compute_checksum()
        session.working_memory["new_key"] = "new_value"
        d2 = session.compute_checksum()
        assert d1 != d2

    def test_checksum_auto_set_at_creation(self) -> None:
        session = UniversalSession()
        assert session.checksum != ""


# ---------------------------------------------------------------------------
# UniversalSession — JSON serialisation
# ---------------------------------------------------------------------------


class TestUniversalSessionSerialization:
    def test_to_json_returns_string(self) -> None:
        session = _make_session()
        result = session.to_json()
        assert isinstance(result, str)

    def test_to_json_is_valid_json(self) -> None:
        session = _make_session()
        parsed = json.loads(session.to_json())
        assert isinstance(parsed, dict)

    def test_to_json_contains_version(self) -> None:
        session = _make_session()
        parsed = json.loads(session.to_json())
        assert parsed["version"] == USFVersion

    def test_to_json_contains_session_id(self) -> None:
        session = _make_session()
        parsed = json.loads(session.to_json())
        assert parsed["session_id"] == session.session_id

    def test_to_json_contains_checksum(self) -> None:
        session = _make_session()
        parsed = json.loads(session.to_json())
        assert "checksum" in parsed
        assert len(parsed["checksum"]) == 64

    def test_to_json_contains_messages(self) -> None:
        session = _make_session()
        parsed = json.loads(session.to_json())
        assert len(parsed["messages"]) == 1

    def test_to_json_contains_entities(self) -> None:
        session = _make_session()
        parsed = json.loads(session.to_json())
        assert len(parsed["entities"]) == 1

    def test_to_json_contains_task_state(self) -> None:
        session = _make_session()
        parsed = json.loads(session.to_json())
        assert len(parsed["task_state"]) == 1

    def test_from_json_round_trip_session_id(self) -> None:
        session = _make_session()
        restored = UniversalSession.from_json(session.to_json())
        assert restored.session_id == session.session_id

    def test_from_json_round_trip_framework_source(self) -> None:
        session = _make_session(framework_source="openai")
        restored = UniversalSession.from_json(session.to_json())
        assert restored.framework_source == "openai"

    def test_from_json_round_trip_messages(self) -> None:
        session = _make_session()
        restored = UniversalSession.from_json(session.to_json())
        assert len(restored.messages) == 1
        assert restored.messages[0].role == "user"

    def test_from_json_round_trip_entities(self) -> None:
        session = _make_session()
        restored = UniversalSession.from_json(session.to_json())
        assert len(restored.entities) == 1
        assert restored.entities[0].name == "Acme"

    def test_from_json_round_trip_task_state(self) -> None:
        session = _make_session()
        restored = UniversalSession.from_json(session.to_json())
        assert len(restored.task_state) == 1
        assert restored.task_state[0].status == "completed"

    def test_from_json_round_trip_working_memory(self) -> None:
        session = _make_session(working_memory={"alpha": "beta"})
        restored = UniversalSession.from_json(session.to_json())
        assert restored.working_memory["alpha"] == "beta"

    def test_from_json_preserves_checksum(self) -> None:
        session = _make_session()
        json_str = session.to_json()
        restored = UniversalSession.from_json(json_str)
        assert restored.checksum == session.checksum

    def test_from_json_verify_checksum_passes(self) -> None:
        session = _make_session()
        restored = UniversalSession.from_json(session.to_json())
        assert restored.verify_checksum() is True

    def test_from_json_invalid_json_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            UniversalSession.from_json("{not valid json")

    def test_from_json_empty_messages_list(self) -> None:
        session = UniversalSession(framework_source="test")
        restored = UniversalSession.from_json(session.to_json())
        assert restored.messages == []

    def test_from_json_multiple_messages(self) -> None:
        session = UniversalSession(
            framework_source="test",
            messages=[
                _make_message(role="user", content="question"),
                _make_message(role="assistant", content="answer"),
            ],
        )
        restored = UniversalSession.from_json(session.to_json())
        assert len(restored.messages) == 2
        assert restored.messages[1].role == "assistant"

    def test_from_json_timestamp_is_timezone_aware(self) -> None:
        session = _make_session()
        restored = UniversalSession.from_json(session.to_json())
        assert restored.created_at.tzinfo is not None

    def test_to_json_refreshes_checksum(self) -> None:
        session = _make_session()
        session.working_memory["new"] = "data"
        session.checksum = "stale" * 10  # intentionally stale
        json_str = session.to_json()
        parsed = json.loads(json_str)
        # The checksum in JSON must be a valid 64-char hex, not the stale value
        assert len(parsed["checksum"]) == 64
        assert parsed["checksum"] != "stale" * 10


# ---------------------------------------------------------------------------
# LangChainExporter
# ---------------------------------------------------------------------------


class TestLangChainExporter:
    def setup_method(self) -> None:
        self.exporter = LangChainExporter()

    def test_export_returns_dict(self) -> None:
        session = _make_session()
        result = self.exporter.export(session)
        assert isinstance(result, dict)

    def test_export_has_messages_key(self) -> None:
        session = _make_session()
        result = self.exporter.export(session)
        assert "messages" in result

    def test_export_has_memory_variables_key(self) -> None:
        session = _make_session()
        result = self.exporter.export(session)
        assert "memory_variables" in result

    def test_export_user_role_becomes_human(self) -> None:
        session = UniversalSession(messages=[_make_message(role="user")])
        result = self.exporter.export(session)
        assert result["messages"][0]["type"] == "human"

    def test_export_assistant_role_becomes_ai(self) -> None:
        session = UniversalSession(messages=[_make_message(role="assistant")])
        result = self.exporter.export(session)
        assert result["messages"][0]["type"] == "ai"

    def test_export_system_role_preserved(self) -> None:
        session = UniversalSession(messages=[_make_message(role="system")])
        result = self.exporter.export(session)
        assert result["messages"][0]["type"] == "system"

    def test_export_tool_role_becomes_function(self) -> None:
        session = UniversalSession(messages=[_make_message(role="tool")])
        result = self.exporter.export(session)
        assert result["messages"][0]["type"] == "function"

    def test_export_content_preserved(self) -> None:
        session = UniversalSession(messages=[_make_message(content="hello there")])
        result = self.exporter.export(session)
        assert result["messages"][0]["content"] == "hello there"

    def test_export_metadata_in_additional_kwargs(self) -> None:
        session = UniversalSession(messages=[_make_message(metadata={"src": "test"})])
        result = self.exporter.export(session)
        assert result["messages"][0]["additional_kwargs"]["src"] == "test"

    def test_export_working_memory_to_memory_variables(self) -> None:
        session = UniversalSession(working_memory={"chat_history": "..."})
        result = self.exporter.export(session)
        assert result["memory_variables"]["chat_history"] == "..."

    def test_export_empty_session(self) -> None:
        session = UniversalSession()
        result = self.exporter.export(session)
        assert result["messages"] == []
        assert result["memory_variables"] == {}

    def test_export_multiple_messages(self) -> None:
        session = UniversalSession(
            messages=[
                _make_message(role="user", content="Q"),
                _make_message(role="assistant", content="A"),
            ]
        )
        result = self.exporter.export(session)
        assert len(result["messages"]) == 2

    def test_export_message_count_matches(self) -> None:
        msgs = [_make_message() for _ in range(5)]
        session = UniversalSession(messages=msgs)
        result = self.exporter.export(session)
        assert len(result["messages"]) == 5


# ---------------------------------------------------------------------------
# CrewAIExporter
# ---------------------------------------------------------------------------


class TestCrewAIExporter:
    def setup_method(self) -> None:
        self.exporter = CrewAIExporter()

    def test_export_returns_dict(self) -> None:
        result = self.exporter.export(_make_session())
        assert isinstance(result, dict)

    def test_export_has_context_key(self) -> None:
        result = self.exporter.export(_make_session())
        assert "context" in result

    def test_export_has_task_results_key(self) -> None:
        result = self.exporter.export(_make_session())
        assert "task_results" in result

    def test_export_context_has_session_id(self) -> None:
        session = _make_session()
        result = self.exporter.export(session)
        assert result["context"]["session_id"] == session.session_id

    def test_export_context_has_framework_source(self) -> None:
        session = _make_session(framework_source="crewai")
        result = self.exporter.export(session)
        assert result["context"]["framework_source"] == "crewai"

    def test_export_context_messages(self) -> None:
        session = UniversalSession(messages=[_make_message(role="user", content="go")])
        result = self.exporter.export(session)
        assert result["context"]["messages"][0]["role"] == "user"
        assert result["context"]["messages"][0]["content"] == "go"

    def test_export_context_working_memory(self) -> None:
        session = UniversalSession(working_memory={"info": "42"})
        result = self.exporter.export(session)
        assert result["context"]["working_memory"]["info"] == "42"

    def test_export_context_entities(self) -> None:
        session = UniversalSession(entities=[_make_entity(name="Robot")])
        result = self.exporter.export(session)
        assert result["context"]["entities"][0]["name"] == "Robot"

    def test_export_task_results(self) -> None:
        session = UniversalSession(task_state=[_make_task(task_id="t1", status="completed")])
        result = self.exporter.export(session)
        assert result["task_results"][0]["task_id"] == "t1"
        assert result["task_results"][0]["status"] == "completed"

    def test_export_task_result_field(self) -> None:
        session = UniversalSession(task_state=[_make_task(result="output")])
        result = self.exporter.export(session)
        assert result["task_results"][0]["result"] == "output"

    def test_export_task_progress(self) -> None:
        session = UniversalSession(task_state=[_make_task(progress=0.75)])
        result = self.exporter.export(session)
        assert result["task_results"][0]["progress"] == 0.75

    def test_export_empty_session(self) -> None:
        result = self.exporter.export(UniversalSession())
        assert result["context"]["messages"] == []
        assert result["task_results"] == []

    def test_export_entity_confidence_preserved(self) -> None:
        session = UniversalSession(entities=[_make_entity(confidence=0.75)])
        result = self.exporter.export(session)
        assert result["context"]["entities"][0]["confidence"] == 0.75


# ---------------------------------------------------------------------------
# OpenAIExporter
# ---------------------------------------------------------------------------


class TestOpenAIExporter:
    def setup_method(self) -> None:
        self.exporter = OpenAIExporter()

    def test_export_returns_dict(self) -> None:
        result = self.exporter.export(_make_session())
        assert isinstance(result, dict)

    def test_export_has_thread_id_key(self) -> None:
        result = self.exporter.export(_make_session())
        assert "thread_id" in result

    def test_export_has_messages_key(self) -> None:
        result = self.exporter.export(_make_session())
        assert "messages" in result

    def test_export_thread_id_is_session_id(self) -> None:
        session = _make_session()
        result = self.exporter.export(session)
        assert result["thread_id"] == session.session_id

    def test_export_message_role_preserved(self) -> None:
        session = UniversalSession(messages=[_make_message(role="user")])
        result = self.exporter.export(session)
        assert result["messages"][0]["role"] == "user"

    def test_export_message_content_preserved(self) -> None:
        session = UniversalSession(messages=[_make_message(content="test content")])
        result = self.exporter.export(session)
        assert result["messages"][0]["content"] == "test content"

    def test_export_assistant_role_preserved(self) -> None:
        session = UniversalSession(messages=[_make_message(role="assistant")])
        result = self.exporter.export(session)
        assert result["messages"][0]["role"] == "assistant"

    def test_export_empty_messages(self) -> None:
        result = self.exporter.export(UniversalSession())
        assert result["messages"] == []

    def test_export_message_count(self) -> None:
        session = UniversalSession(
            messages=[_make_message(), _make_message(role="assistant")]
        )
        result = self.exporter.export(session)
        assert len(result["messages"]) == 2

    def test_export_system_role_preserved(self) -> None:
        session = UniversalSession(messages=[_make_message(role="system")])
        result = self.exporter.export(session)
        assert result["messages"][0]["role"] == "system"


# ---------------------------------------------------------------------------
# LangChainImporter
# ---------------------------------------------------------------------------


class TestLangChainImporter:
    def setup_method(self) -> None:
        self.importer = LangChainImporter()

    def _lc_data(
        self,
        messages: list[dict[str, Any]] | None = None,
        memory_variables: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "messages": messages or [],
            "memory_variables": memory_variables or {},
        }

    def test_import_returns_universal_session(self) -> None:
        result = self.importer.import_session(self._lc_data())
        assert isinstance(result, UniversalSession)

    def test_import_framework_source_is_langchain(self) -> None:
        result = self.importer.import_session(self._lc_data())
        assert result.framework_source == "langchain"

    def test_import_human_message_becomes_user(self) -> None:
        data = self._lc_data(messages=[{"type": "human", "content": "hi"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "user"

    def test_import_ai_message_becomes_assistant(self) -> None:
        data = self._lc_data(messages=[{"type": "ai", "content": "hello"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "assistant"

    def test_import_system_message_preserved(self) -> None:
        data = self._lc_data(messages=[{"type": "system", "content": "prompt"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "system"

    def test_import_function_message_becomes_tool(self) -> None:
        data = self._lc_data(messages=[{"type": "function", "content": "result"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "tool"

    def test_import_content_preserved(self) -> None:
        data = self._lc_data(messages=[{"type": "human", "content": "what is AI?"}])
        result = self.importer.import_session(data)
        assert result.messages[0].content == "what is AI?"

    def test_import_memory_variables(self) -> None:
        data = self._lc_data(memory_variables={"history": "prev context"})
        result = self.importer.import_session(data)
        assert result.working_memory["history"] == "prev context"

    def test_import_additional_kwargs_as_metadata(self) -> None:
        data = self._lc_data(
            messages=[{"type": "human", "content": "hi", "additional_kwargs": {"k": "v"}}]
        )
        result = self.importer.import_session(data)
        assert result.messages[0].metadata["k"] == "v"

    def test_import_empty_data(self) -> None:
        result = self.importer.import_session({})
        assert result.messages == []
        assert result.working_memory == {}

    def test_import_missing_messages_key(self) -> None:
        result = self.importer.import_session({"memory_variables": {}})
        assert result.messages == []

    def test_import_session_id_auto_generated(self) -> None:
        result = self.importer.import_session(self._lc_data())
        assert len(result.session_id) == 36

    def test_import_invalid_message_raises(self) -> None:
        with pytest.raises(ValueError):
            self.importer.import_session({"messages": ["not a dict"]})

    def test_import_multiple_messages(self) -> None:
        data = self._lc_data(
            messages=[
                {"type": "human", "content": "Q"},
                {"type": "ai", "content": "A"},
            ]
        )
        result = self.importer.import_session(data)
        assert len(result.messages) == 2

    def test_import_timestamp_is_utc(self) -> None:
        data = self._lc_data(messages=[{"type": "human", "content": "x"}])
        result = self.importer.import_session(data)
        assert result.messages[0].timestamp.tzinfo is not None

    def test_import_with_timestamp_field(self) -> None:
        ts = "2025-01-15T10:00:00+00:00"
        data = self._lc_data(messages=[{"type": "human", "content": "hi", "timestamp": ts}])
        result = self.importer.import_session(data)
        assert result.messages[0].timestamp.year == 2025

    def test_import_unknown_type_defaults_to_user(self) -> None:
        data = self._lc_data(messages=[{"type": "unknown_type", "content": "hi"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "user"


# ---------------------------------------------------------------------------
# CrewAIImporter
# ---------------------------------------------------------------------------


class TestCrewAIImporter:
    def setup_method(self) -> None:
        self.importer = CrewAIImporter()

    def _crew_data(
        self,
        context: dict[str, Any] | None = None,
        task_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "context": context or {},
            "task_results": task_results or [],
        }

    def test_import_returns_universal_session(self) -> None:
        result = self.importer.import_session(self._crew_data())
        assert isinstance(result, UniversalSession)

    def test_import_framework_source_is_crewai(self) -> None:
        result = self.importer.import_session(self._crew_data())
        assert result.framework_source == "crewai"

    def test_import_session_id_from_context(self) -> None:
        data = self._crew_data(context={"session_id": "crew-abc"})
        result = self.importer.import_session(data)
        assert result.session_id == "crew-abc"

    def test_import_session_id_auto_generated_when_absent(self) -> None:
        result = self.importer.import_session(self._crew_data())
        assert len(result.session_id) == 36

    def test_import_messages_from_context(self) -> None:
        data = self._crew_data(
            context={"messages": [{"role": "user", "content": "hello"}]}
        )
        result = self.importer.import_session(data)
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "hello"

    def test_import_working_memory_from_context(self) -> None:
        data = self._crew_data(context={"working_memory": {"k": "v"}})
        result = self.importer.import_session(data)
        assert result.working_memory["k"] == "v"

    def test_import_entities_from_context(self) -> None:
        data = self._crew_data(
            context={
                "entities": [
                    {"name": "Bot", "entity_type": "agent", "value": "Bot v1", "confidence": 0.8}
                ]
            }
        )
        result = self.importer.import_session(data)
        assert result.entities[0].name == "Bot"
        assert result.entities[0].confidence == 0.8

    def test_import_task_results(self) -> None:
        data = self._crew_data(
            task_results=[{"task_id": "t1", "status": "completed", "progress": 1.0, "result": "ok"}]
        )
        result = self.importer.import_session(data)
        assert result.task_state[0].task_id == "t1"
        assert result.task_state[0].status == "completed"

    def test_import_task_result_none(self) -> None:
        data = self._crew_data(
            task_results=[{"task_id": "t1", "status": "pending", "progress": 0.0, "result": None}]
        )
        result = self.importer.import_session(data)
        assert result.task_state[0].result is None

    def test_import_invalid_task_status_defaults_to_pending(self) -> None:
        data = self._crew_data(
            task_results=[{"task_id": "t1", "status": "running", "progress": 0.5}]
        )
        result = self.importer.import_session(data)
        assert result.task_state[0].status == "pending"

    def test_import_invalid_role_defaults_to_user(self) -> None:
        data = self._crew_data(
            context={"messages": [{"role": "agent", "content": "hi"}]}
        )
        result = self.importer.import_session(data)
        assert result.messages[0].role == "user"

    def test_import_empty_data(self) -> None:
        result = self.importer.import_session({})
        assert result.messages == []
        assert result.task_state == []

    def test_import_invalid_message_raises(self) -> None:
        with pytest.raises(ValueError):
            self.importer.import_session({"context": {"messages": ["bad"]}})

    def test_import_invalid_entity_raises(self) -> None:
        with pytest.raises(ValueError):
            self.importer.import_session({"context": {"entities": ["bad"]}})

    def test_import_invalid_task_result_raises(self) -> None:
        with pytest.raises(ValueError):
            self.importer.import_session({"task_results": ["bad"]})


# ---------------------------------------------------------------------------
# OpenAIImporter
# ---------------------------------------------------------------------------


class TestOpenAIImporter:
    def setup_method(self) -> None:
        self.importer = OpenAIImporter()

    def _oai_data(
        self,
        thread_id: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {"messages": messages or []}
        if thread_id is not None:
            data["thread_id"] = thread_id
        return data

    def test_import_returns_universal_session(self) -> None:
        result = self.importer.import_session(self._oai_data())
        assert isinstance(result, UniversalSession)

    def test_import_framework_source_is_openai(self) -> None:
        result = self.importer.import_session(self._oai_data())
        assert result.framework_source == "openai"

    def test_import_thread_id_used_as_session_id(self) -> None:
        data = self._oai_data(thread_id="thread_xyz")
        result = self.importer.import_session(data)
        assert result.session_id == "thread_xyz"

    def test_import_session_id_auto_generated_when_absent(self) -> None:
        result = self.importer.import_session(self._oai_data())
        assert len(result.session_id) == 36

    def test_import_user_role_preserved(self) -> None:
        data = self._oai_data(messages=[{"role": "user", "content": "hello"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "user"

    def test_import_assistant_role_preserved(self) -> None:
        data = self._oai_data(messages=[{"role": "assistant", "content": "hi"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "assistant"

    def test_import_content_preserved(self) -> None:
        data = self._oai_data(messages=[{"role": "user", "content": "what?"}])
        result = self.importer.import_session(data)
        assert result.messages[0].content == "what?"

    def test_import_unix_timestamp(self) -> None:
        ts = 1700000000
        data = self._oai_data(messages=[{"role": "user", "content": "x", "created_at": ts}])
        result = self.importer.import_session(data)
        assert result.messages[0].timestamp.tzinfo is not None

    def test_import_iso_timestamp(self) -> None:
        ts = "2025-06-01T12:00:00+00:00"
        data = self._oai_data(messages=[{"role": "user", "content": "x", "created_at": ts}])
        result = self.importer.import_session(data)
        assert result.messages[0].timestamp.year == 2025

    def test_import_metadata_from_message(self) -> None:
        data = self._oai_data(
            messages=[{"role": "user", "content": "x", "metadata": {"model": "gpt-4"}}]
        )
        result = self.importer.import_session(data)
        assert result.messages[0].metadata["model"] == "gpt-4"

    def test_import_unknown_role_defaults_to_user(self) -> None:
        data = self._oai_data(messages=[{"role": "moderator", "content": "x"}])
        result = self.importer.import_session(data)
        assert result.messages[0].role == "user"

    def test_import_empty_data(self) -> None:
        result = self.importer.import_session({})
        assert result.messages == []

    def test_import_invalid_message_raises(self) -> None:
        with pytest.raises(ValueError):
            self.importer.import_session({"messages": ["not a dict"]})

    def test_import_multiple_messages(self) -> None:
        data = self._oai_data(
            messages=[
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]
        )
        result = self.importer.import_session(data)
        assert len(result.messages) == 2


# ---------------------------------------------------------------------------
# Round-trip consistency tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """Verify that import → export preserves message content faithfully."""

    def test_langchain_export_reimport_message_count(self) -> None:
        session = _make_session(
            messages=[_make_message(role="user", content="ping")],
            framework_source="langchain",
        )
        exported = LangChainExporter().export(session)
        restored = LangChainImporter().import_session(exported)
        assert len(restored.messages) == len(session.messages)

    def test_langchain_export_reimport_content(self) -> None:
        session = UniversalSession(
            messages=[_make_message(role="user", content="round trip content")]
        )
        exported = LangChainExporter().export(session)
        restored = LangChainImporter().import_session(exported)
        assert restored.messages[0].content == "round trip content"

    def test_langchain_working_memory_round_trip(self) -> None:
        session = UniversalSession(working_memory={"history": "some context"})
        exported = LangChainExporter().export(session)
        restored = LangChainImporter().import_session(exported)
        assert restored.working_memory["history"] == "some context"

    def test_crewai_export_reimport_message_count(self) -> None:
        session = _make_session(
            messages=[_make_message(role="user", content="task run")],
            framework_source="crewai",
        )
        exported = CrewAIExporter().export(session)
        restored = CrewAIImporter().import_session(exported)
        assert len(restored.messages) == len(session.messages)

    def test_crewai_export_reimport_task_state(self) -> None:
        session = UniversalSession(
            task_state=[_make_task(task_id="t42", status="completed", progress=1.0)]
        )
        exported = CrewAIExporter().export(session)
        restored = CrewAIImporter().import_session(exported)
        assert restored.task_state[0].task_id == "t42"
        assert restored.task_state[0].status == "completed"

    def test_crewai_export_reimport_entities(self) -> None:
        session = UniversalSession(entities=[_make_entity(name="ProjectX")])
        exported = CrewAIExporter().export(session)
        restored = CrewAIImporter().import_session(exported)
        assert restored.entities[0].name == "ProjectX"

    def test_openai_export_reimport_message_content(self) -> None:
        session = UniversalSession(
            messages=[
                _make_message(role="user", content="hi"),
                _make_message(role="assistant", content="hello"),
            ]
        )
        exported = OpenAIExporter().export(session)
        restored = OpenAIImporter().import_session(exported)
        assert restored.messages[0].content == "hi"
        assert restored.messages[1].content == "hello"

    def test_openai_thread_id_preserved(self) -> None:
        session = UniversalSession(session_id="thread-123")
        exported = OpenAIExporter().export(session)
        restored = OpenAIImporter().import_session(exported)
        assert restored.session_id == "thread-123"

    def test_json_round_trip_full_session(self) -> None:
        original = _make_session()
        json_str = original.to_json()
        restored = UniversalSession.from_json(json_str)
        assert restored.session_id == original.session_id
        assert restored.framework_source == original.framework_source
        assert len(restored.messages) == len(original.messages)
        assert len(restored.entities) == len(original.entities)
        assert len(restored.task_state) == len(original.task_state)


# ---------------------------------------------------------------------------
# Cross-format conversion tests
# ---------------------------------------------------------------------------


class TestCrossFormatConversion:
    """Verify that converting A → USF → B preserves information correctly."""

    def test_langchain_to_openai_message_content(self) -> None:
        lc_data = {
            "messages": [{"type": "human", "content": "convert me"}],
            "memory_variables": {},
        }
        session = LangChainImporter().import_session(lc_data)
        oai_data = OpenAIExporter().export(session)
        assert oai_data["messages"][0]["content"] == "convert me"

    def test_langchain_to_openai_role_mapping(self) -> None:
        lc_data = {
            "messages": [
                {"type": "human", "content": "Q"},
                {"type": "ai", "content": "A"},
            ],
            "memory_variables": {},
        }
        session = LangChainImporter().import_session(lc_data)
        oai_data = OpenAIExporter().export(session)
        assert oai_data["messages"][0]["role"] == "user"
        assert oai_data["messages"][1]["role"] == "assistant"

    def test_openai_to_langchain_message_content(self) -> None:
        oai_data = {
            "thread_id": "t1",
            "messages": [{"role": "user", "content": "openai input"}],
        }
        session = OpenAIImporter().import_session(oai_data)
        lc_data = LangChainExporter().export(session)
        assert lc_data["messages"][0]["content"] == "openai input"

    def test_openai_to_langchain_role_mapping(self) -> None:
        oai_data = {
            "thread_id": "t1",
            "messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ],
        }
        session = OpenAIImporter().import_session(oai_data)
        lc_data = LangChainExporter().export(session)
        assert lc_data["messages"][0]["type"] == "human"
        assert lc_data["messages"][1]["type"] == "ai"

    def test_crewai_to_openai_message_content(self) -> None:
        crew_data = {
            "context": {"messages": [{"role": "user", "content": "crew task"}]},
            "task_results": [],
        }
        session = CrewAIImporter().import_session(crew_data)
        oai_data = OpenAIExporter().export(session)
        assert oai_data["messages"][0]["content"] == "crew task"

    def test_crewai_to_langchain_working_memory(self) -> None:
        crew_data = {
            "context": {"working_memory": {"memo": "note"}},
            "task_results": [],
        }
        session = CrewAIImporter().import_session(crew_data)
        lc_data = LangChainExporter().export(session)
        assert lc_data["memory_variables"]["memo"] == "note"

    def test_langchain_to_crewai_preserves_messages(self) -> None:
        lc_data = {
            "messages": [
                {"type": "human", "content": "start"},
                {"type": "ai", "content": "done"},
            ],
            "memory_variables": {},
        }
        session = LangChainImporter().import_session(lc_data)
        crew_data = CrewAIExporter().export(session)
        assert len(crew_data["context"]["messages"]) == 2

    def test_openai_to_crewai_thread_id_in_session_id(self) -> None:
        oai_data = {"thread_id": "thread-oai-99", "messages": []}
        session = OpenAIImporter().import_session(oai_data)
        crew_data = CrewAIExporter().export(session)
        assert crew_data["context"]["session_id"] == "thread-oai-99"


# ---------------------------------------------------------------------------
# CLI — portable export
# ---------------------------------------------------------------------------


class TestCLIPortableExport:
    def test_export_langchain(self, tmp_path: Path) -> None:
        session = _make_session(framework_source="langchain")
        input_file = tmp_path / "session.json"
        input_file.write_text(session.to_json(), encoding="utf-8")
        output_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "export",
                "--format",
                "langchain",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        exported = json.loads(output_file.read_text())
        assert "messages" in exported

    def test_export_crewai(self, tmp_path: Path) -> None:
        session = _make_session()
        input_file = tmp_path / "session.json"
        input_file.write_text(session.to_json(), encoding="utf-8")
        output_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "export",
                "--format",
                "crewai",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        exported = json.loads(output_file.read_text())
        assert "context" in exported
        assert "task_results" in exported

    def test_export_openai(self, tmp_path: Path) -> None:
        session = _make_session()
        input_file = tmp_path / "session.json"
        input_file.write_text(session.to_json(), encoding="utf-8")
        output_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "export",
                "--format",
                "openai",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        exported = json.loads(output_file.read_text())
        assert "thread_id" in exported
        assert "messages" in exported

    def test_export_invalid_input_exits_nonzero(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid", encoding="utf-8")
        output_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "export",
                "--format",
                "langchain",
                "--input",
                str(bad_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI — portable import
# ---------------------------------------------------------------------------


class TestCLIPortableImport:
    def test_import_langchain(self, tmp_path: Path) -> None:
        lc_data = {
            "messages": [{"type": "human", "content": "hi"}],
            "memory_variables": {},
        }
        input_file = tmp_path / "lc.json"
        input_file.write_text(json.dumps(lc_data), encoding="utf-8")
        output_file = tmp_path / "session.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "import",
                "--format",
                "langchain",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        assert output_file.exists()
        session = UniversalSession.from_json(output_file.read_text())
        assert session.framework_source == "langchain"

    def test_import_crewai(self, tmp_path: Path) -> None:
        crew_data: dict[str, Any] = {"context": {}, "task_results": []}
        input_file = tmp_path / "crew.json"
        input_file.write_text(json.dumps(crew_data), encoding="utf-8")
        output_file = tmp_path / "session.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "import",
                "--format",
                "crewai",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        session = UniversalSession.from_json(output_file.read_text())
        assert session.framework_source == "crewai"

    def test_import_openai(self, tmp_path: Path) -> None:
        oai_data = {"thread_id": "t-abc", "messages": []}
        input_file = tmp_path / "oai.json"
        input_file.write_text(json.dumps(oai_data), encoding="utf-8")
        output_file = tmp_path / "session.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "import",
                "--format",
                "openai",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        session = UniversalSession.from_json(output_file.read_text())
        assert session.framework_source == "openai"
        assert session.session_id == "t-abc"

    def test_import_invalid_json_exits_nonzero(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{bad json", encoding="utf-8")
        output_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "import",
                "--format",
                "langchain",
                "--input",
                str(bad_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI — portable convert
# ---------------------------------------------------------------------------


class TestCLIPortableConvert:
    def test_convert_langchain_to_openai(self, tmp_path: Path) -> None:
        lc_data = {
            "messages": [
                {"type": "human", "content": "convert this"},
                {"type": "ai", "content": "done"},
            ],
            "memory_variables": {},
        }
        input_file = tmp_path / "lc.json"
        input_file.write_text(json.dumps(lc_data), encoding="utf-8")
        output_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "convert",
                "--from",
                "langchain",
                "--to",
                "openai",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        out = json.loads(output_file.read_text())
        assert "thread_id" in out
        assert out["messages"][0]["content"] == "convert this"

    def test_convert_openai_to_crewai(self, tmp_path: Path) -> None:
        oai_data = {
            "thread_id": "t1",
            "messages": [{"role": "user", "content": "crew task"}],
        }
        input_file = tmp_path / "oai.json"
        input_file.write_text(json.dumps(oai_data), encoding="utf-8")
        output_file = tmp_path / "crew_out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "convert",
                "--from",
                "openai",
                "--to",
                "crewai",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        out = json.loads(output_file.read_text())
        assert "context" in out
        assert out["context"]["messages"][0]["content"] == "crew task"

    def test_convert_crewai_to_langchain(self, tmp_path: Path) -> None:
        crew_data = {
            "context": {"messages": [{"role": "assistant", "content": "response"}]},
            "task_results": [],
        }
        input_file = tmp_path / "crew.json"
        input_file.write_text(json.dumps(crew_data), encoding="utf-8")
        output_file = tmp_path / "lc_out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "convert",
                "--from",
                "crewai",
                "--to",
                "langchain",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        out = json.loads(output_file.read_text())
        assert "messages" in out
        assert out["messages"][0]["type"] == "ai"

    def test_convert_invalid_input_exits_nonzero(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{bad", encoding="utf-8")
        output_file = tmp_path / "out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "convert",
                "--from",
                "langchain",
                "--to",
                "openai",
                "--input",
                str(bad_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code != 0

    def test_convert_langchain_to_crewai(self, tmp_path: Path) -> None:
        lc_data = {
            "messages": [{"type": "human", "content": "plan step"}],
            "memory_variables": {"ctx": "info"},
        }
        input_file = tmp_path / "lc.json"
        input_file.write_text(json.dumps(lc_data), encoding="utf-8")
        output_file = tmp_path / "crew_out.json"

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "portable",
                "convert",
                "--from",
                "langchain",
                "--to",
                "crewai",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0, result.output
        out = json.loads(output_file.read_text())
        assert out["context"]["working_memory"]["ctx"] == "info"


# ---------------------------------------------------------------------------
# Error handling — miscellaneous edge cases
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_from_json_missing_version_uses_default(self) -> None:
        payload = {
            "session_id": "abc",
            "framework_source": "test",
            "messages": [],
            "entities": [],
            "task_state": [],
            "working_memory": {},
            "metadata": {},
            "created_at": datetime.now(_UTC).isoformat(),
            "updated_at": datetime.now(_UTC).isoformat(),
            "checksum": "",
        }
        session = UniversalSession.from_json(json.dumps(payload))
        assert session.version == USFVersion

    def test_from_json_missing_session_id_auto_generated(self) -> None:
        payload = {
            "version": "1.0",
            "framework_source": "test",
            "messages": [],
            "entities": [],
            "task_state": [],
            "working_memory": {},
            "metadata": {},
            "created_at": datetime.now(_UTC).isoformat(),
            "updated_at": datetime.now(_UTC).isoformat(),
            "checksum": "",
        }
        session = UniversalSession.from_json(json.dumps(payload))
        assert len(session.session_id) == 36

    def test_usf_message_role_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            USFMessage(role="", content="hi", timestamp=_now(), metadata={})

    def test_usf_entity_confidence_exactly_boundary(self) -> None:
        e0 = USFEntity(name="x", entity_type="t", value="v", confidence=0.0)
        e1 = USFEntity(name="x", entity_type="t", value="v", confidence=1.0)
        assert e0.confidence == 0.0
        assert e1.confidence == 1.0

    def test_usf_task_progress_exactly_boundary(self) -> None:
        t0 = USFTaskState(task_id="t", status="pending", progress=0.0, result=None)
        t1 = USFTaskState(task_id="t", status="completed", progress=1.0, result=None)
        assert t0.progress == 0.0
        assert t1.progress == 1.0

    def test_verify_checksum_after_modifying_working_memory(self) -> None:
        session = _make_session()
        session.compute_checksum()
        original_checksum = session.checksum
        session.working_memory["injected"] = "tamper"
        # checksum still refers to old state
        assert session.verify_checksum() is False

    def test_public_api_exports_portable_types(self) -> None:
        import agent_session_linker as pkg

        assert hasattr(pkg, "UniversalSession")
        assert hasattr(pkg, "LangChainExporter")
        assert hasattr(pkg, "CrewAIImporter")

    def test_portable_package_init_exports_all(self) -> None:
        from agent_session_linker.portable import (
            UniversalSession,
            LangChainExporter,
            CrewAIExporter,
            OpenAIExporter,
            LangChainImporter,
            CrewAIImporter,
            OpenAIImporter,
        )
        assert UniversalSession is not None
        assert LangChainExporter is not None
        assert CrewAIExporter is not None
        assert OpenAIExporter is not None
        assert LangChainImporter is not None
        assert CrewAIImporter is not None
        assert OpenAIImporter is not None
