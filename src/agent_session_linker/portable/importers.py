"""Session importers — convert framework-native dicts to :class:`UniversalSession`.

Each importer implements the :class:`SessionImporter` Protocol and normalises
incoming data into a validated :class:`UniversalSession` instance.

Classes
-------
SessionImporter
    Protocol that every importer must satisfy.
LangChainImporter
    Converts LangChain memory format to USF.
CrewAIImporter
    Converts CrewAI context format to USF.
OpenAIImporter
    Converts OpenAI thread format to USF.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol
from uuid import uuid4

from agent_session_linker.portable.usf import (
    USFEntity,
    USFMessage,
    USFTaskState,
    USFVersion,
    UniversalSession,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SessionImporter(Protocol):
    """Protocol for all USF importers.

    Implementors must accept a plain ``dict`` in the source framework's native
    format and return a validated :class:`UniversalSession`.
    """

    def import_session(self, data: dict[str, object]) -> UniversalSession:
        """Convert framework-native *data* to a :class:`UniversalSession`.

        Parameters
        ----------
        data:
            Framework-native session dict.

        Returns
        -------
        UniversalSession
            A fully validated USF session.

        Raises
        ------
        ValueError
            If *data* is missing required fields or contains invalid values.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: object) -> datetime:
    """Parse *value* into a timezone-aware UTC datetime.

    Parameters
    ----------
    value:
        A datetime object, an ISO-8601 string, or ``None``.

    Returns
    -------
    datetime
        UTC-aware datetime; defaults to UTC now when *value* is falsy.
    """
    if not value:
        return _utc_now()
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _langchain_type_to_usf_role(msg_type: str) -> str:
    """Map a LangChain message type to a USF role.

    Parameters
    ----------
    msg_type:
        LangChain type string, e.g. ``"human"``, ``"ai"``, ``"system"``.

    Returns
    -------
    str
        USF role string.
    """
    mapping: dict[str, str] = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "function": "tool",
        "tool": "tool",
    }
    return mapping.get(msg_type, "user")


# ---------------------------------------------------------------------------
# LangChain importer
# ---------------------------------------------------------------------------


class LangChainImporter:
    """Import a LangChain memory dict as a :class:`UniversalSession`.

    Expected input structure::

        {
            "messages": [
                {"type": "human", "content": "..."},
                {"type": "ai",    "content": "..."},
                ...
            ],
            "memory_variables": {
                "key": "value",
                ...
            }
        }

    The ``messages`` list may be absent (treated as empty).
    Each message may optionally carry a ``"timestamp"`` key and an
    ``"additional_kwargs"`` dict that becomes the message metadata.
    """

    def import_session(self, data: dict[str, object]) -> UniversalSession:
        """Convert a LangChain memory dict to a :class:`UniversalSession`.

        Parameters
        ----------
        data:
            LangChain memory dict.

        Returns
        -------
        UniversalSession
            A validated USF session with ``framework_source="langchain"``.

        Raises
        ------
        ValueError
            If a message entry is missing required fields.
        """
        raw_messages: list[object] = data.get("messages") or []
        messages: list[USFMessage] = []
        for entry in raw_messages:
            if not isinstance(entry, dict):
                raise ValueError(f"Each message must be a dict, got {type(entry).__name__!r}")
            msg_type = entry.get("type") or "human"
            content = entry.get("content", "")
            timestamp = _parse_timestamp(entry.get("timestamp"))
            metadata: dict[str, object] = dict(entry.get("additional_kwargs") or {})
            messages.append(
                USFMessage(
                    role=_langchain_type_to_usf_role(msg_type),
                    content=str(content),
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )

        working_memory: dict[str, object] = dict(data.get("memory_variables") or {})

        return UniversalSession(
            version=USFVersion,
            session_id=str(uuid4()),
            created_at=_utc_now(),
            updated_at=_utc_now(),
            framework_source="langchain",
            messages=messages,
            working_memory=working_memory,
            entities=[],
            task_state=[],
            metadata={},
        )


# ---------------------------------------------------------------------------
# CrewAI importer
# ---------------------------------------------------------------------------


class CrewAIImporter:
    """Import a CrewAI context dict as a :class:`UniversalSession`.

    Expected input structure::

        {
            "context": {
                "session_id":       "...",       # optional
                "framework_source": "crewai",    # optional
                "messages":         [...],       # optional
                "working_memory":   {...},       # optional
                "entities":         [...],       # optional
            },
            "task_results": [                    # optional
                {
                    "task_id":  "...",
                    "status":   "completed",
                    "progress": 1.0,
                    "result":   "...",
                },
                ...
            ]
        }

    Every sub-list is optional and defaults to empty.
    """

    def import_session(self, data: dict[str, object]) -> UniversalSession:
        """Convert a CrewAI context dict to a :class:`UniversalSession`.

        Parameters
        ----------
        data:
            CrewAI context dict.

        Returns
        -------
        UniversalSession
            A validated USF session with ``framework_source="crewai"``.

        Raises
        ------
        ValueError
            If nested structures contain invalid values.
        """
        context: dict[str, object] = dict(data.get("context") or {})
        task_results: list[object] = data.get("task_results") or []

        session_id: str = str(context.get("session_id") or uuid4())

        raw_messages: list[object] = context.get("messages") or []
        messages: list[USFMessage] = []
        for entry in raw_messages:
            if not isinstance(entry, dict):
                raise ValueError(f"Each message must be a dict, got {type(entry).__name__!r}")
            role = str(entry.get("role") or "user")
            if role not in {"user", "assistant", "system", "tool"}:
                role = "user"
            timestamp = _parse_timestamp(entry.get("timestamp"))
            metadata = dict(entry.get("metadata") or {})
            messages.append(
                USFMessage(
                    role=role,
                    content=str(entry.get("content", "")),
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )

        working_memory: dict[str, object] = dict(context.get("working_memory") or {})

        raw_entities: list[object] = context.get("entities") or []
        entities: list[USFEntity] = []
        for entry in raw_entities:
            if not isinstance(entry, dict):
                raise ValueError(f"Each entity must be a dict, got {type(entry).__name__!r}")
            entities.append(
                USFEntity(
                    name=str(entry.get("name", "")),
                    entity_type=str(entry.get("entity_type", "concept")),
                    value=str(entry.get("value", "")),
                    confidence=float(entry.get("confidence", 1.0)),
                )
            )

        task_state: list[USFTaskState] = []
        valid_statuses = {"pending", "in_progress", "completed", "failed"}
        for entry in task_results:
            if not isinstance(entry, dict):
                raise ValueError(f"Each task result must be a dict, got {type(entry).__name__!r}")
            status = str(entry.get("status", "pending"))
            if status not in valid_statuses:
                status = "pending"
            task_state.append(
                USFTaskState(
                    task_id=str(entry.get("task_id") or uuid4()),
                    status=status,
                    progress=float(entry.get("progress", 0.0)),
                    result=entry.get("result"),
                )
            )

        return UniversalSession(
            version=USFVersion,
            session_id=session_id,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            framework_source="crewai",
            messages=messages,
            working_memory=working_memory,
            entities=entities,
            task_state=task_state,
            metadata={},
        )


# ---------------------------------------------------------------------------
# OpenAI importer
# ---------------------------------------------------------------------------


class OpenAIImporter:
    """Import an OpenAI thread dict as a :class:`UniversalSession`.

    Expected input structure::

        {
            "thread_id": "thread_abc123",       # optional
            "messages": [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }

    Each message may optionally include a ``"created_at"`` Unix timestamp
    (integer) or an ISO-8601 string, and a ``"metadata"`` dict.
    """

    def import_session(self, data: dict[str, object]) -> UniversalSession:
        """Convert an OpenAI thread dict to a :class:`UniversalSession`.

        Parameters
        ----------
        data:
            OpenAI thread dict.

        Returns
        -------
        UniversalSession
            A validated USF session with ``framework_source="openai"``.

        Raises
        ------
        ValueError
            If a message entry is missing required fields.
        """
        thread_id: str = str(data.get("thread_id") or uuid4())
        raw_messages: list[object] = data.get("messages") or []

        messages: list[USFMessage] = []
        for entry in raw_messages:
            if not isinstance(entry, dict):
                raise ValueError(f"Each message must be a dict, got {type(entry).__name__!r}")
            role = str(entry.get("role") or "user")
            if role not in {"user", "assistant", "system", "tool"}:
                role = "user"

            raw_ts = entry.get("created_at")
            if isinstance(raw_ts, (int, float)):
                timestamp = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
            else:
                timestamp = _parse_timestamp(raw_ts)

            metadata: dict[str, object] = dict(entry.get("metadata") or {})
            messages.append(
                USFMessage(
                    role=role,
                    content=str(entry.get("content", "")),
                    timestamp=timestamp,
                    metadata=metadata,
                )
            )

        return UniversalSession(
            version=USFVersion,
            session_id=thread_id,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            framework_source="openai",
            messages=messages,
            working_memory={},
            entities=[],
            task_state=[],
            metadata={},
        )
