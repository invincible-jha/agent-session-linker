"""Session exporters — convert a :class:`UniversalSession` to framework-native dicts.

Each exporter implements the :class:`SessionExporter` Protocol and produces a
``dict`` that can be passed directly to the target framework's session or
memory APIs.

All export methods accept an optional *encryptor* keyword argument of type
:class:`~agent_session_linker.portable.encryption.SessionEncryptor`.  When
provided, the JSON representation of the session is encrypted before being
embedded in the output dict under the key ``"_encrypted"``.  Callers that do
not need encryption simply omit the argument; existing call sites are fully
backward-compatible.

Classes
-------
SessionExporter
    Protocol that every exporter must satisfy.
LangChainExporter
    Converts USF to LangChain memory format
    (``{"messages": [...], "memory_variables": {...}}``).
CrewAIExporter
    Converts USF to CrewAI context format
    (``{"context": {...}, "task_results": [...]}}``).
OpenAIExporter
    Converts USF to OpenAI thread format
    (``{"thread_id": ..., "messages": [...]}``).
"""
from __future__ import annotations

import base64
import json
from typing import TYPE_CHECKING, Protocol

from agent_session_linker.portable.usf import UniversalSession, USFMessage, USFTaskState

if TYPE_CHECKING:
    from agent_session_linker.portable.encryption import SessionEncryptor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_encryption(
    output: dict[str, object],
    session: UniversalSession,
    encryptor: SessionEncryptor | None,
) -> dict[str, object]:
    """Optionally encrypt the session JSON and embed it into *output*.

    When *encryptor* is ``None`` the original *output* dict is returned
    unchanged.  When *encryptor* is provided the full session is encrypted
    and the result is returned as::

        {
            "_encrypted": "<base64-encoded wire bytes>",
            "_encryption_version": "1.0",
        }

    Parameters
    ----------
    output:
        The plain export dict produced by an exporter's ``export`` method.
    session:
        The source :class:`UniversalSession` (used to obtain the JSON payload
        for encryption).
    encryptor:
        An optional :class:`SessionEncryptor` instance.

    Returns
    -------
    dict[str, object]
        Either *output* unchanged, or an encrypted envelope dict.
    """
    if encryptor is None:
        return output
    payload: dict[str, object] = json.loads(session.to_json())
    encrypted = encryptor.encrypt(payload)
    wire_bytes = encrypted.to_bytes()
    return {
        "_encrypted": base64.b64encode(wire_bytes).decode("ascii"),
        "_encryption_version": encrypted.version,
    }


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class SessionExporter(Protocol):
    """Protocol for all USF exporters.

    Implementors must accept a :class:`UniversalSession` and return a plain
    ``dict`` that is ready to be consumed by the target framework.
    """

    def export(
        self,
        session: UniversalSession,
        *,
        encryptor: SessionEncryptor | None = None,
    ) -> dict[str, object]:
        """Convert *session* to the target framework's native dict format.

        Parameters
        ----------
        session:
            A :class:`UniversalSession` to export.
        encryptor:
            Optional :class:`~agent_session_linker.portable.encryption.SessionEncryptor`.
            When provided the output is an encrypted envelope instead of a
            plain framework dict.

        Returns
        -------
        dict[str, object]
            Framework-native representation of the session, or an encrypted
            envelope dict when *encryptor* is supplied.
        """
        ...  # pragma: no cover


# ---------------------------------------------------------------------------
# LangChain exporter
# ---------------------------------------------------------------------------


def _usf_role_to_langchain_type(role: str) -> str:
    """Map a USF role string to the corresponding LangChain message type.

    Parameters
    ----------
    role:
        One of ``"user"``, ``"assistant"``, ``"system"``, ``"tool"``.

    Returns
    -------
    str
        LangChain message type string.
    """
    mapping: dict[str, str] = {
        "user": "human",
        "assistant": "ai",
        "system": "system",
        "tool": "function",
    }
    return mapping.get(role, role)


class LangChainExporter:
    """Export a :class:`UniversalSession` to LangChain memory format.

    The output dict has the following structure::

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

    ``memory_variables`` is populated from ``session.working_memory``.
    """

    def export(
        self,
        session: UniversalSession,
        *,
        encryptor: SessionEncryptor | None = None,
    ) -> dict[str, object]:
        """Convert *session* to LangChain memory format.

        Parameters
        ----------
        session:
            Source :class:`UniversalSession`.
        encryptor:
            Optional :class:`~agent_session_linker.portable.encryption.SessionEncryptor`.
            When supplied the return value is an encrypted envelope dict.

        Returns
        -------
        dict[str, object]
            LangChain-compatible memory dict, or an encrypted envelope.
        """
        messages: list[dict[str, object]] = [
            self._convert_message(msg) for msg in session.messages
        ]
        output: dict[str, object] = {
            "messages": messages,
            "memory_variables": dict(session.working_memory),
        }
        return _apply_encryption(output, session, encryptor)

    def _convert_message(self, msg: USFMessage) -> dict[str, object]:
        return {
            "type": _usf_role_to_langchain_type(msg.role),
            "content": msg.content,
            "additional_kwargs": dict(msg.metadata),
        }


# ---------------------------------------------------------------------------
# CrewAI exporter
# ---------------------------------------------------------------------------


class CrewAIExporter:
    """Export a :class:`UniversalSession` to CrewAI context format.

    The output dict has the following structure::

        {
            "context": {
                "session_id":       "...",
                "framework_source": "...",
                "messages":         [...],
                "working_memory":   {...},
                "entities":         [...],
            },
            "task_results": [
                {
                    "task_id":  "...",
                    "status":   "completed",
                    "progress": 1.0,
                    "result":   "...",
                },
                ...
            ]
        }
    """

    def export(
        self,
        session: UniversalSession,
        *,
        encryptor: SessionEncryptor | None = None,
    ) -> dict[str, object]:
        """Convert *session* to CrewAI context format.

        Parameters
        ----------
        session:
            Source :class:`UniversalSession`.
        encryptor:
            Optional :class:`~agent_session_linker.portable.encryption.SessionEncryptor`.
            When supplied the return value is an encrypted envelope dict.

        Returns
        -------
        dict[str, object]
            CrewAI-compatible context dict, or an encrypted envelope.
        """
        context: dict[str, object] = {
            "session_id": session.session_id,
            "framework_source": session.framework_source,
            "messages": [self._convert_message(msg) for msg in session.messages],
            "working_memory": dict(session.working_memory),
            "entities": [
                {
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "value": entity.value,
                    "confidence": entity.confidence,
                }
                for entity in session.entities
            ],
        }
        task_results: list[dict[str, object]] = [
            self._convert_task(task) for task in session.task_state
        ]
        output: dict[str, object] = {
            "context": context,
            "task_results": task_results,
        }
        return _apply_encryption(output, session, encryptor)

    def _convert_message(self, msg: USFMessage) -> dict[str, object]:
        return {
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat(),
            "metadata": dict(msg.metadata),
        }

    def _convert_task(self, task: USFTaskState) -> dict[str, object]:
        return {
            "task_id": task.task_id,
            "status": task.status,
            "progress": task.progress,
            "result": task.result,
        }


# ---------------------------------------------------------------------------
# OpenAI exporter
# ---------------------------------------------------------------------------


class OpenAIExporter:
    """Export a :class:`UniversalSession` to OpenAI Assistants thread format.

    The output dict has the following structure::

        {
            "thread_id": "...",
            "messages": [
                {"role": "user",      "content": "..."},
                {"role": "assistant", "content": "..."},
                ...
            ]
        }

    The ``thread_id`` is taken from ``session.session_id``.  The ``role``
    field uses OpenAI's two-value vocabulary (``"user"`` / ``"assistant"``);
    ``"system"`` and ``"tool"`` roles are preserved as-is since the
    Assistants API accepts them in message-level metadata when needed.
    """

    def export(
        self,
        session: UniversalSession,
        *,
        encryptor: SessionEncryptor | None = None,
    ) -> dict[str, object]:
        """Convert *session* to OpenAI thread format.

        Parameters
        ----------
        session:
            Source :class:`UniversalSession`.
        encryptor:
            Optional :class:`~agent_session_linker.portable.encryption.SessionEncryptor`.
            When supplied the return value is an encrypted envelope dict.

        Returns
        -------
        dict[str, object]
            OpenAI-compatible thread dict, or an encrypted envelope.
        """
        messages: list[dict[str, object]] = [
            self._convert_message(msg) for msg in session.messages
        ]
        output: dict[str, object] = {
            "thread_id": session.session_id,
            "messages": messages,
        }
        return _apply_encryption(output, session, encryptor)

    def _convert_message(self, msg: USFMessage) -> dict[str, object]:
        return {
            "role": msg.role,
            "content": msg.content,
        }
