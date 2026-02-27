"""Cross-agent context handoff — package segments, entities, and tasks
for transfer from one agent to another.

Design
------
:class:`HandoffPayload` is a frozen snapshot of the context being
transferred.  It is serialisable to JSON for transport over any medium.

:class:`HandoffBuilder` constructs a payload from a :class:`SessionState`,
applying filters defined in :class:`HandoffConfig`.

Usage
-----
::

    from agent_session_linker.handoff import HandoffBuilder, HandoffConfig
    from agent_session_linker.session.state import SessionState

    source = SessionState(agent_id="agent_a")
    source.add_segment("user", "Please help me deploy service X")
    source.add_task(title="Deploy service X")

    config = HandoffConfig(max_segments=5, include_tasks=True)
    builder = HandoffBuilder(config)
    payload = builder.build(
        source_session=source,
        target_agent_id="agent_b",
        handoff_reason="escalation",
    )
    json_str = payload.to_json()
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_session_linker.session.state import (
    ContextSegment,
    EntityReference,
    SessionState,
    TaskState,
)


# ---------------------------------------------------------------------------
# HandoffConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HandoffConfig:
    """Configuration that controls what context is included in a handoff.

    Parameters
    ----------
    max_segments:
        Maximum number of context segments to include.  The most recent
        segments are preferred.  ``None`` means include all.
    include_entities:
        Whether to include entity references.  Default True.
    include_tasks:
        Whether to include task states.  Default True.
    include_preferences:
        Whether to copy over session preferences.  Default True.
    include_summary:
        Whether to include the session summary string.  Default True.
    segment_types:
        If non-empty, only segments with matching ``segment_type`` are
        included.  Empty tuple means all types.
    """

    max_segments: int | None = None
    include_entities: bool = True
    include_tasks: bool = True
    include_preferences: bool = True
    include_summary: bool = True
    segment_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_segments is not None and self.max_segments < 0:
            raise ValueError(
                f"max_segments must be non-negative or None, "
                f"got {self.max_segments!r}."
            )


# ---------------------------------------------------------------------------
# HandoffPayload
# ---------------------------------------------------------------------------


class HandoffPayload(BaseModel):
    """Immutable snapshot of context prepared for agent handoff.

    Parameters
    ----------
    handoff_id:
        Unique identifier for this handoff.
    source_session_id:
        The session ID of the originating agent.
    source_agent_id:
        The agent that is handing off.
    target_agent_id:
        The agent that will receive the context.
    handoff_reason:
        Human-readable reason for the handoff.
    segments:
        Context segments selected for transfer.
    entities:
        Entity references selected for transfer.
    tasks:
        Task states selected for transfer.
    preferences:
        Session preferences to propagate.
    summary:
        Optional session summary.
    created_at:
        UTC timestamp when the payload was built.
    metadata:
        Arbitrary annotations.
    """

    model_config = {"frozen": True}

    handoff_id: str = Field(default_factory=lambda: str(uuid4()))
    source_session_id: str
    source_agent_id: str
    target_agent_id: str
    handoff_reason: str = ""
    segments: list[dict[str, object]] = Field(default_factory=list)
    entities: list[dict[str, object]] = Field(default_factory=list)
    tasks: list[dict[str, object]] = Field(default_factory=list)
    preferences: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        """Number of segments included."""
        return len(self.segments)

    @property
    def entity_count(self) -> int:
        """Number of entities included."""
        return len(self.entities)

    @property
    def task_count(self) -> int:
        """Number of tasks included."""
        return len(self.tasks)

    def to_json(self) -> str:
        """Serialise the payload to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, json_str: str) -> "HandoffPayload":
        """Deserialise from a JSON string produced by :meth:`to_json`.

        Parameters
        ----------
        json_str:
            JSON string previously produced by :meth:`to_json`.

        Returns
        -------
        HandoffPayload
        """
        return cls.model_validate_json(json_str)

    def summary_line(self) -> str:
        """Return a one-line human-readable description."""
        return (
            f"HandoffPayload {self.handoff_id[:8]} | "
            f"{self.source_agent_id} -> {self.target_agent_id} | "
            f"{self.segment_count} segments, {self.task_count} tasks, "
            f"{self.entity_count} entities | reason={self.handoff_reason!r}"
        )


# ---------------------------------------------------------------------------
# HandoffBuilder
# ---------------------------------------------------------------------------


class HandoffBuilder:
    """Build a :class:`HandoffPayload` from a :class:`SessionState`.

    Parameters
    ----------
    config:
        Optional :class:`HandoffConfig`.  Defaults to including everything.

    Example
    -------
    ::

        builder = HandoffBuilder()
        payload = builder.build(source_session, target_agent_id="agent_b")
    """

    def __init__(self, config: HandoffConfig | None = None) -> None:
        self._config = config or HandoffConfig()

    def build(
        self,
        source_session: SessionState,
        target_agent_id: str,
        handoff_reason: str = "",
        extra_metadata: dict[str, object] | None = None,
    ) -> HandoffPayload:
        """Construct a :class:`HandoffPayload` from *source_session*.

        Parameters
        ----------
        source_session:
            The session to extract context from.
        target_agent_id:
            The receiving agent.
        handoff_reason:
            Why this handoff is happening.
        extra_metadata:
            Additional key-value annotations to include.

        Returns
        -------
        HandoffPayload
        """
        cfg = self._config

        # Segments
        segments = list(source_session.segments)
        if cfg.segment_types:
            segments = [s for s in segments if s.segment_type in cfg.segment_types]
        if cfg.max_segments is not None:
            if cfg.max_segments == 0:
                segments = []
            else:
                segments = segments[-cfg.max_segments:]

        # Entities
        entities: list[EntityReference] = []
        if cfg.include_entities:
            entities = list(source_session.entities)

        # Tasks
        tasks: list[TaskState] = []
        if cfg.include_tasks:
            tasks = list(source_session.tasks)

        # Preferences
        preferences: dict[str, str] = {}
        if cfg.include_preferences:
            preferences = dict(source_session.preferences)

        # Summary
        summary = ""
        if cfg.include_summary:
            summary = source_session.summary

        return HandoffPayload(
            source_session_id=source_session.session_id,
            source_agent_id=source_session.agent_id,
            target_agent_id=target_agent_id,
            handoff_reason=handoff_reason,
            segments=[_segment_to_dict(s) for s in segments],
            entities=[_entity_to_dict(e) for e in entities],
            tasks=[_task_to_dict(t) for t in tasks],
            preferences=preferences,
            summary=summary,
            metadata=extra_metadata or {},
        )


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------


def _segment_to_dict(seg: ContextSegment) -> dict[str, object]:
    return {
        "segment_id": seg.segment_id,
        "role": seg.role,
        "content": seg.content,
        "token_count": seg.token_count,
        "segment_type": seg.segment_type,
        "timestamp": seg.timestamp.isoformat(),
        "turn_index": seg.turn_index,
        "metadata": seg.metadata,
    }


def _entity_to_dict(entity: EntityReference) -> dict[str, object]:
    return {
        "entity_id": entity.entity_id,
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
        "aliases": entity.aliases,
        "attributes": entity.attributes,
        "confidence": entity.confidence,
    }


def _task_to_dict(task: TaskState) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "title": task.title,
        "description": task.description,
        "status": task.status.value,
        "priority": task.priority,
        "tags": task.tags,
        "notes": task.notes,
    }
