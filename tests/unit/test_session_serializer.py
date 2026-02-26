"""Unit tests for agent_session_linker.session.serializer.

Tests cover JSON and YAML round-trips, schema version enforcement,
checksum validation, and the serialize/deserialize dispatch helpers.
"""
from __future__ import annotations

import json

import pytest

from agent_session_linker.session.serializer import SchemaVersionError, SessionSerializer
from agent_session_linker.session.state import SessionState, TaskStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(**kwargs: object) -> SessionState:
    """Return a populated SessionState for serialization tests."""
    session = SessionState(**kwargs)  # type: ignore[arg-type]
    session.add_segment("user", "Hello, agent!")
    session.add_segment("assistant", "How can I help?", token_count=8)
    session.add_task("Write unit tests", priority=2)
    session.track_entity("pytest", "tool")
    return session


# ---------------------------------------------------------------------------
# SchemaVersionError
# ---------------------------------------------------------------------------


class TestSchemaVersionError:
    def test_message_contains_version(self) -> None:
        err = SchemaVersionError("99.0")
        assert "99.0" in str(err)

    def test_version_attribute(self) -> None:
        err = SchemaVersionError("2.0")
        assert err.version == "2.0"

    def test_message_contains_supported_list(self) -> None:
        err = SchemaVersionError("bad")
        assert "1.0" in str(err)


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


class TestSessionSerializerJSON:
    def test_to_json_returns_string(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        result = serializer.to_json(session)
        assert isinstance(result, str)

    def test_to_json_is_valid_json(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_json(session)
        data = json.loads(raw)
        assert isinstance(data, dict)

    def test_to_json_embeds_schema_version(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        data = json.loads(serializer.to_json(session))
        assert data["schema_version"] == "1.0"

    def test_to_json_embeds_checksum(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        data = json.loads(serializer.to_json(session))
        assert len(data["checksum"]) == 64

    def test_from_json_round_trip_session_id(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_json(session)
        restored = serializer.from_json(raw)
        assert restored.session_id == session.session_id

    def test_from_json_round_trip_agent_id(self) -> None:
        serializer = SessionSerializer()
        session = _make_session(agent_id="test-bot")
        raw = serializer.to_json(session)
        restored = serializer.from_json(raw)
        assert restored.agent_id == "test-bot"

    def test_from_json_round_trip_segments(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_json(session)
        restored = serializer.from_json(raw)
        assert len(restored.segments) == len(session.segments)

    def test_from_json_round_trip_tasks(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_json(session)
        restored = serializer.from_json(raw)
        assert len(restored.tasks) == len(session.tasks)

    def test_from_json_round_trip_entities(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_json(session)
        restored = serializer.from_json(raw)
        assert len(restored.entities) == len(session.entities)

    def test_from_json_raises_on_unsupported_version(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        data = json.loads(serializer.to_json(session))
        data["schema_version"] = "99.0"
        with pytest.raises(SchemaVersionError):
            serializer.from_json(json.dumps(data))

    def test_from_json_raises_on_invalid_json(self) -> None:
        serializer = SessionSerializer()
        with pytest.raises(json.JSONDecodeError):
            serializer.from_json("not json!")

    def test_from_json_checksum_validation_passes(self) -> None:
        serializer = SessionSerializer(validate_checksum=True)
        session = _make_session()
        raw = serializer.to_json(session)
        restored = serializer.from_json(raw)
        assert restored.session_id == session.session_id

    def test_from_json_checksum_validation_fails_on_tamper(self) -> None:
        serializer = SessionSerializer(validate_checksum=True)
        session = _make_session()
        raw = serializer.to_json(session)
        data = json.loads(raw)
        data["agent_id"] = "tampered"
        with pytest.raises(ValueError, match="[Cc]hecksum"):
            serializer.from_json(json.dumps(data))

    def test_from_json_skip_checksum_validation(self) -> None:
        serializer = SessionSerializer(validate_checksum=False)
        session = _make_session()
        raw = serializer.to_json(session)
        data = json.loads(raw)
        data["agent_id"] = "tampered"
        # Should not raise when validation is disabled.
        restored = serializer.from_json(json.dumps(data))
        assert restored.agent_id == "tampered"

    def test_from_json_empty_checksum_skips_validation(self) -> None:
        serializer = SessionSerializer(validate_checksum=True)
        session = _make_session()
        raw = serializer.to_json(session)
        data = json.loads(raw)
        data["checksum"] = ""
        # An empty stored checksum skips the validation gate.
        restored = serializer.from_json(json.dumps(data))
        assert restored.session_id == session.session_id

    def test_to_json_default_indent(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_json(session)
        # Default indent=2 produces multi-line JSON.
        assert "\n" in raw


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------


class TestSessionSerializerYAML:
    def test_to_yaml_returns_string(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_yaml(session)
        assert isinstance(raw, str)

    def test_to_yaml_contains_session_id(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_yaml(session)
        assert session.session_id in raw

    def test_from_yaml_round_trip_session_id(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_yaml(session)
        restored = serializer.from_yaml(raw)
        assert restored.session_id == session.session_id

    def test_from_yaml_round_trip_segments(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.to_yaml(session)
        restored = serializer.from_yaml(raw)
        assert len(restored.segments) == len(session.segments)

    def test_from_yaml_raises_on_bad_schema_version(self) -> None:
        import yaml

        serializer = SessionSerializer()
        session = _make_session()
        data = yaml.safe_load(serializer.to_yaml(session))
        data["schema_version"] = "0.0"
        with pytest.raises(SchemaVersionError):
            serializer.from_yaml(yaml.dump(data))


# ---------------------------------------------------------------------------
# serialize / deserialize dispatch
# ---------------------------------------------------------------------------


class TestSessionSerializerDispatch:
    def test_serialize_json_default(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        result = serializer.serialize(session)
        data = json.loads(result)
        assert data["session_id"] == session.session_id

    def test_serialize_yaml_explicit(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        result = serializer.serialize(session, format="yaml")
        assert "session_id:" in result

    def test_deserialize_json(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.serialize(session, format="json")
        restored = serializer.deserialize(raw, format="json")
        assert restored.session_id == session.session_id

    def test_deserialize_yaml(self) -> None:
        serializer = SessionSerializer()
        session = _make_session()
        raw = serializer.serialize(session, format="yaml")
        restored = serializer.deserialize(raw, format="yaml")
        assert restored.session_id == session.session_id
