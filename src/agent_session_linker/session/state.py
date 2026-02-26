"""Session state domain models.

All types are Pydantic BaseModel subclasses to enable runtime validation,
JSON serialisation, and schema versioning.

Classes
-------
- TaskStatus      — enum for task lifecycle states
- ContextSegment  — a slice of conversation context with metadata
- EntityReference — a pointer to a tracked entity
- TaskState       — a tracked task with its current status
- ToolContext     — a record of a single tool invocation
- SessionState    — top-level session snapshot
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class TaskStatus(str, Enum):
    """Lifecycle states for a tracked task within a session."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContextSegment(BaseModel):
    """A discrete unit of context captured from a conversation turn.

    Parameters
    ----------
    segment_id:
        Unique identifier for this segment.
    role:
        The message role: "user", "assistant", "system", or "tool".
    content:
        The raw text content of the segment.
    token_count:
        Estimated token count for this segment.
    segment_type:
        Categorical label such as "conversation", "reasoning", "code",
        "plan", "output", or "metadata".
    timestamp:
        When this segment was captured (UTC).
    turn_index:
        Zero-based index of the conversation turn this segment belongs to.
    metadata:
        Arbitrary additional key-value data attached to this segment.
    """

    segment_id: str = Field(default_factory=lambda: str(uuid4()))
    role: str
    content: str
    token_count: int = 0
    segment_type: str = "conversation"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    turn_index: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": False}


class EntityReference(BaseModel):
    """A cross-session pointer to a tracked domain entity.

    Parameters
    ----------
    entity_id:
        Unique identifier for this entity record.
    canonical_name:
        The primary normalised name used for matching.
    entity_type:
        Categorical label such as "person", "project", "file",
        "concept", "tool", or "organisation".
    aliases:
        Alternative names or spellings for this entity.
    attributes:
        Key-value attributes describing the entity.
    first_seen_session:
        Session ID where this entity was first observed.
    last_seen_session:
        Session ID where this entity was most recently observed.
    confidence:
        Match/extraction confidence in the range [0.0, 1.0].
    """

    entity_id: str = Field(default_factory=lambda: str(uuid4()))
    canonical_name: str
    entity_type: str = "concept"
    aliases: list[str] = Field(default_factory=list)
    attributes: dict[str, str] = Field(default_factory=dict)
    first_seen_session: str = ""
    last_seen_session: str = ""
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    model_config = {"frozen": False}


class TaskState(BaseModel):
    """A tracked task with its current lifecycle status.

    Parameters
    ----------
    task_id:
        Unique identifier for this task.
    title:
        Short human-readable title.
    description:
        Detailed description of what the task requires.
    status:
        Current lifecycle status (see ``TaskStatus``).
    priority:
        Integer priority where 1 is highest. Defaults to 5.
    created_at:
        When the task was first recorded (UTC).
    updated_at:
        When the task was last modified (UTC).
    parent_task_id:
        Optional reference to a parent task for sub-task hierarchies.
    tags:
        Free-form labels for categorisation and filtering.
    notes:
        Additional free-text notes about the task.
    """

    task_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    priority: int = Field(default=5, ge=1, le=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    parent_task_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    notes: str = ""

    model_config = {"frozen": False}

    def mark_in_progress(self) -> None:
        """Transition status to IN_PROGRESS and refresh updated_at."""
        self.status = TaskStatus.IN_PROGRESS
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Transition status to COMPLETED and refresh updated_at."""
        self.status = TaskStatus.COMPLETED
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self) -> None:
        """Transition status to FAILED and refresh updated_at."""
        self.status = TaskStatus.FAILED
        self.updated_at = datetime.now(timezone.utc)


class ToolContext(BaseModel):
    """A record of a single tool invocation within a session.

    Parameters
    ----------
    invocation_id:
        Unique identifier for this invocation.
    tool_name:
        Name of the tool that was called.
    input_summary:
        Brief summary of the input arguments (not the full payload).
    output_summary:
        Brief summary of the output returned.
    duration_ms:
        Wall-clock execution time in milliseconds.
    success:
        Whether the invocation completed without error.
    error_message:
        Error description when ``success`` is False.
    timestamp:
        When the invocation started (UTC).
    token_cost:
        Tokens consumed by the tool call, if measurable.
    """

    invocation_id: str = Field(default_factory=lambda: str(uuid4()))
    tool_name: str
    input_summary: str = ""
    output_summary: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error_message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_cost: int = 0

    model_config = {"frozen": False}


class SessionState(BaseModel):
    """Complete snapshot of an agent session.

    This is the central domain object.  It aggregates all captured context,
    entity references, task states, and tool invocations for one session.

    Parameters
    ----------
    session_id:
        Globally unique session identifier.
    agent_id:
        Identifier for the agent or agent-type that owns this session.
    schema_version:
        Schema version string used for forward/backward compatibility.
    segments:
        Ordered list of conversation context segments.
    entities:
        Named entities tracked within this session.
    tasks:
        Tasks created or referenced during this session.
    tools_used:
        Chronological log of tool invocations.
    preferences:
        Agent or user preferences captured during the session.
    summary:
        Optional compressed summary of the session's key content.
    parent_session_id:
        If this is a continuation, the ID of the preceding session.
    total_cost_usd:
        Accumulated LLM API cost in USD for this session.
    created_at:
        Session creation timestamp (UTC).
    updated_at:
        Last modification timestamp (UTC).
    checksum:
        SHA-256 of the session's canonical JSON (excluding this field).
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    agent_id: str = "default"
    schema_version: str = "1.0"
    segments: list[ContextSegment] = Field(default_factory=list)
    entities: list[EntityReference] = Field(default_factory=list)
    tasks: list[TaskState] = Field(default_factory=list)
    tools_used: list[ToolContext] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    parent_session_id: str | None = None
    total_cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: str = ""

    model_config = {"frozen": False}

    # ------------------------------------------------------------------
    # Checksum
    # ------------------------------------------------------------------

    def _canonical_dict(self) -> dict[str, object]:
        """Return a stable dict suitable for checksum computation.

        The ``checksum`` field itself is excluded to avoid circularity.
        """
        data = self.model_dump(mode="json")
        data.pop("checksum", None)
        return data  # type: ignore[return-value]

    def compute_checksum(self) -> str:
        """Compute and return the SHA-256 checksum of this session's content.

        The result is stored in ``self.checksum`` as a side-effect and also
        returned as a hex-encoded string.

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
        """Return True if the stored checksum matches the computed one.

        Returns
        -------
        bool
            True when the session has not been tampered with after the
            last call to ``compute_checksum``.
        """
        return self.checksum == self.compute_checksum()

    # ------------------------------------------------------------------
    # Segment helpers
    # ------------------------------------------------------------------

    def add_segment(
        self,
        role: str,
        content: str,
        *,
        token_count: int = 0,
        segment_type: str = "conversation",
        metadata: dict[str, str] | None = None,
    ) -> ContextSegment:
        """Append a new context segment and return it.

        Parameters
        ----------
        role:
            Message role: "user", "assistant", "system", or "tool".
        content:
            The text content of the segment.
        token_count:
            Estimated tokens consumed by this segment.
        segment_type:
            Categorical label (e.g. "conversation", "code", "plan").
        metadata:
            Optional additional key-value data.

        Returns
        -------
        ContextSegment
            The newly created and appended segment.
        """
        turn_index = len(self.segments)
        segment = ContextSegment(
            role=role,
            content=content,
            token_count=token_count,
            segment_type=segment_type,
            turn_index=turn_index,
            metadata=metadata or {},
        )
        self.segments.append(segment)
        self.updated_at = datetime.now(timezone.utc)
        return segment

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def track_entity(
        self,
        canonical_name: str,
        entity_type: str = "concept",
        *,
        aliases: list[str] | None = None,
        attributes: dict[str, str] | None = None,
        confidence: float = 1.0,
    ) -> EntityReference:
        """Add or update an entity reference in this session.

        If an entity with the same ``canonical_name`` already exists it is
        returned unchanged.  Otherwise a new ``EntityReference`` is created.

        Parameters
        ----------
        canonical_name:
            Primary name used for deduplication and matching.
        entity_type:
            Categorical label.
        aliases:
            Alternative names for this entity.
        attributes:
            Key-value descriptive attributes.
        confidence:
            Extraction confidence in [0.0, 1.0].

        Returns
        -------
        EntityReference
            The existing or newly created entity reference.
        """
        for existing in self.entities:
            if existing.canonical_name.lower() == canonical_name.lower():
                existing.last_seen_session = self.session_id
                return existing

        entity = EntityReference(
            canonical_name=canonical_name,
            entity_type=entity_type,
            aliases=aliases or [],
            attributes=attributes or {},
            first_seen_session=self.session_id,
            last_seen_session=self.session_id,
            confidence=confidence,
        )
        self.entities.append(entity)
        self.updated_at = datetime.now(timezone.utc)
        return entity

    # ------------------------------------------------------------------
    # Task helpers
    # ------------------------------------------------------------------

    def update_task(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        notes: str | None = None,
        priority: int | None = None,
    ) -> TaskState:
        """Update an existing task by ID.

        Parameters
        ----------
        task_id:
            The ``task_id`` of the task to update.
        status:
            New status value, if changing.
        notes:
            Appended to existing notes separated by a newline.
        priority:
            New priority value.

        Returns
        -------
        TaskState
            The updated task.

        Raises
        ------
        KeyError
            If no task with ``task_id`` exists in this session.
        """
        for task in self.tasks:
            if task.task_id == task_id:
                if status is not None:
                    task.status = status
                if notes is not None:
                    task.notes = f"{task.notes}\n{notes}".strip()
                if priority is not None:
                    task.priority = priority
                task.updated_at = datetime.now(timezone.utc)
                self.updated_at = datetime.now(timezone.utc)
                return task
        raise KeyError(f"Task {task_id!r} not found in session {self.session_id!r}")

    def add_task(
        self,
        title: str,
        *,
        description: str = "",
        priority: int = 5,
        tags: list[str] | None = None,
        parent_task_id: str | None = None,
    ) -> TaskState:
        """Create and append a new task to this session.

        Parameters
        ----------
        title:
            Short human-readable title.
        description:
            Detailed description.
        priority:
            Priority level 1 (highest) to 10 (lowest).
        tags:
            Optional categorisation labels.
        parent_task_id:
            Optional parent task for hierarchy.

        Returns
        -------
        TaskState
            The newly created task.
        """
        task = TaskState(
            title=title,
            description=description,
            priority=priority,
            tags=tags or [],
            parent_task_id=parent_task_id,
        )
        self.tasks.append(task)
        self.updated_at = datetime.now(timezone.utc)
        return task

    # ------------------------------------------------------------------
    # Token accounting
    # ------------------------------------------------------------------

    def total_tokens(self) -> int:
        """Return the sum of token_count across all segments.

        Returns
        -------
        int
            Total estimated token count for the session.
        """
        return sum(segment.token_count for segment in self.segments)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def _ensure_schema_version(self) -> "SessionState":
        if not self.schema_version:
            self.schema_version = self.SCHEMA_VERSION
        return self
