"""Universal Session Format (USF) domain models.

This module defines the canonical, framework-agnostic representation of an
agent session.  All exporters and importers convert to/from these types.

Constants
---------
USFVersion
    Current format version string.  Bump this on breaking schema changes.

Classes
-------
USFMessage
    A single conversation turn with role, content, timestamp, and metadata.
USFEntity
    A named entity extracted from the session.
USFTaskState
    A tracked task and its lifecycle state.
UniversalSession
    The top-level Pydantic v2 model that aggregates all session data and
    provides checksum, JSON serialization, and class-method deserialization.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Version constant
# ---------------------------------------------------------------------------

USFVersion: str = "1.0"

# ---------------------------------------------------------------------------
# Frozen dataclasses — lightweight value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class USFMessage:
    """A single conversation message in the Universal Session Format.

    Parameters
    ----------
    role:
        The speaker role: ``"user"``, ``"assistant"``, ``"system"``, or
        ``"tool"``.
    content:
        The raw text content of the message.
    timestamp:
        When this message was produced (UTC-aware).
    metadata:
        Arbitrary key-value annotations attached to the message.
    """

    role: str
    content: str
    timestamp: datetime
    metadata: dict[str, object]

    def __post_init__(self) -> None:
        valid_roles = {"user", "assistant", "system", "tool"}
        if self.role not in valid_roles:
            raise ValueError(
                f"USFMessage.role must be one of {sorted(valid_roles)!r}, got {self.role!r}"
            )


@dataclass(frozen=True)
class USFEntity:
    """A named entity captured within a session.

    Parameters
    ----------
    name:
        The canonical display name of the entity.
    entity_type:
        A categorical label (e.g. ``"person"``, ``"project"``, ``"tool"``).
    value:
        The resolved or representative value for this entity.
    confidence:
        Extraction confidence in the closed interval [0.0, 1.0].
    """

    name: str
    entity_type: str
    value: str
    confidence: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"USFEntity.confidence must be in [0.0, 1.0], got {self.confidence!r}"
            )


@dataclass(frozen=True)
class USFTaskState:
    """A tracked task and its current lifecycle state.

    Parameters
    ----------
    task_id:
        Unique identifier for this task.
    status:
        Current lifecycle status: ``"pending"``, ``"in_progress"``,
        ``"completed"``, or ``"failed"``.
    progress:
        Completion fraction in the closed interval [0.0, 1.0].
    result:
        Optional free-text result produced when the task completed or failed.
    """

    task_id: str
    status: str
    progress: float
    result: str | None

    def __post_init__(self) -> None:
        valid_statuses = {"pending", "in_progress", "completed", "failed"}
        if self.status not in valid_statuses:
            raise ValueError(
                f"USFTaskState.status must be one of {sorted(valid_statuses)!r}, "
                f"got {self.status!r}"
            )
        if not (0.0 <= self.progress <= 1.0):
            raise ValueError(
                f"USFTaskState.progress must be in [0.0, 1.0], got {self.progress!r}"
            )


# ---------------------------------------------------------------------------
# Helper — convert dataclasses to JSON-serialisable dicts
# ---------------------------------------------------------------------------


def _message_to_dict(msg: USFMessage) -> dict[str, object]:
    return {
        "role": msg.role,
        "content": msg.content,
        "timestamp": msg.timestamp.isoformat(),
        "metadata": msg.metadata,
    }


def _message_from_dict(data: dict[str, object]) -> USFMessage:
    ts_raw = data["timestamp"]
    if isinstance(ts_raw, datetime):
        timestamp = ts_raw
    else:
        timestamp = datetime.fromisoformat(str(ts_raw))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return USFMessage(
        role=data["role"],
        content=data["content"],
        timestamp=timestamp,
        metadata=dict(data.get("metadata") or {}),
    )


def _entity_to_dict(entity: USFEntity) -> dict[str, object]:
    return {
        "name": entity.name,
        "entity_type": entity.entity_type,
        "value": entity.value,
        "confidence": entity.confidence,
    }


def _entity_from_dict(data: dict[str, object]) -> USFEntity:
    return USFEntity(
        name=data["name"],
        entity_type=data["entity_type"],
        value=data["value"],
        confidence=float(data["confidence"]),
    )


def _task_state_to_dict(task: USFTaskState) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "result": task.result,
    }


def _task_state_from_dict(data: dict[str, object]) -> USFTaskState:
    return USFTaskState(
        task_id=data["task_id"],
        status=data["status"],
        progress=float(data["progress"]),
        result=data.get("result"),
    )


# ---------------------------------------------------------------------------
# UniversalSession — top-level Pydantic v2 model
# ---------------------------------------------------------------------------


class UniversalSession(BaseModel):
    """Framework-agnostic snapshot of an agent session.

    This is the central exchange object.  Import data from a source framework
    using the appropriate :class:`SessionImporter`, operate on this model,
    then export using the target :class:`SessionExporter`.

    Parameters
    ----------
    version:
        USF schema version.  Defaults to :data:`USFVersion`.
    session_id:
        Globally unique session identifier (auto-generated UUID).
    created_at:
        Session creation timestamp (UTC).
    updated_at:
        Last modification timestamp (UTC).
    framework_source:
        The originating framework, e.g. ``"langchain"``, ``"crewai"``,
        ``"openai"``.
    messages:
        Ordered list of conversation messages.
    working_memory:
        Arbitrary key-value store representing in-context working memory.
    entities:
        Named entities extracted from the session.
    task_state:
        Tasks tracked within this session.
    metadata:
        Additional free-form annotations.
    checksum:
        SHA-256 of the serialised session content (excluding this field).
        Computed automatically on model creation.
    """

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    version: str = Field(default=USFVersion)
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    framework_source: str = Field(default="")
    messages: list[USFMessage] = Field(default_factory=list)
    working_memory: dict[str, object] = Field(default_factory=dict)
    entities: list[USFEntity] = Field(default_factory=list)
    task_state: list[USFTaskState] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    checksum: str = Field(default="")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        if not value:
            raise ValueError("version must not be empty")
        return value

    @model_validator(mode="after")
    def _auto_checksum(self) -> "UniversalSession":
        """Compute checksum on initial creation when it has not been set."""
        if not self.checksum:
            self.checksum = self.compute_checksum()
        return self

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------

    def _canonical_dict(self) -> dict[str, object]:
        """Return a stable, JSON-serialisable dict for checksum computation.

        The ``checksum`` field is excluded to avoid circularity.
        """
        return {
            "version": self.version,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "framework_source": self.framework_source,
            "messages": [_message_to_dict(m) for m in self.messages],
            "working_memory": self.working_memory,
            "entities": [_entity_to_dict(e) for e in self.entities],
            "task_state": [_task_state_to_dict(t) for t in self.task_state],
            "metadata": self.metadata,
        }

    def compute_checksum(self) -> str:
        """Compute and store a SHA-256 checksum of this session's content.

        The checksum is deterministic: given the same field values in the same
        order, the same digest is always produced.

        Returns
        -------
        str
            64-character lowercase hex SHA-256 digest.
        """
        canonical_json = json.dumps(self._canonical_dict(), sort_keys=True)
        digest = hashlib.sha256(canonical_json.encode()).hexdigest()
        self.checksum = digest
        return digest

    def verify_checksum(self) -> bool:
        """Return True when the stored checksum matches the recomputed one.

        Returns
        -------
        bool
            True if the session has not been modified since the last
            call to :meth:`compute_checksum`.
        """
        stored = self.checksum
        recomputed = self.compute_checksum()
        # Restore original so calling verify_checksum does not mutate state
        self.checksum = stored
        return stored == recomputed

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise the session to a JSON string.

        The checksum is refreshed before serialisation to ensure it reflects
        the current content.

        Returns
        -------
        str
            A valid JSON string representing the full session.
        """
        self.compute_checksum()
        payload: dict[str, object] = self._canonical_dict()
        payload["checksum"] = self.checksum
        return json.dumps(payload, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "UniversalSession":
        """Deserialise a session from a JSON string produced by :meth:`to_json`.

        Parameters
        ----------
        json_str:
            JSON string previously produced by :meth:`to_json`.

        Returns
        -------
        UniversalSession
            The reconstructed session.

        Raises
        ------
        ValueError
            If ``json_str`` is not valid JSON or is missing required fields.
        """
        try:
            data: dict[str, object] = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON: {exc}") from exc

        stored_checksum = data.pop("checksum", "")

        messages = [_message_from_dict(m) for m in data.get("messages", [])]
        entities = [_entity_from_dict(e) for e in data.get("entities", [])]
        task_state = [_task_state_from_dict(t) for t in data.get("task_state", [])]

        created_raw = data.get("created_at", "")
        updated_raw = data.get("updated_at", "")

        created_at = (
            datetime.fromisoformat(str(created_raw))
            if created_raw
            else datetime.now(timezone.utc)
        )
        updated_at = (
            datetime.fromisoformat(str(updated_raw))
            if updated_raw
            else datetime.now(timezone.utc)
        )
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)

        session = cls(
            version=data.get("version", USFVersion),
            session_id=data.get("session_id", str(uuid4())),
            created_at=created_at,
            updated_at=updated_at,
            framework_source=data.get("framework_source", ""),
            messages=messages,
            working_memory=dict(data.get("working_memory") or {}),
            entities=entities,
            task_state=task_state,
            metadata=dict(data.get("metadata") or {}),
            checksum=stored_checksum,
        )
        # Override the auto-computed checksum with the stored one so that
        # verify_checksum() can detect post-deserialisation tampering.
        session.checksum = stored_checksum
        return session
