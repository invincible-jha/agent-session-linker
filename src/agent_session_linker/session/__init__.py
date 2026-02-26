"""Session management subpackage.

Provides the core domain objects and management logic for capturing,
persisting, and restoring agent session state across conversation turns.

Public surface
--------------
- SessionState       — full session snapshot dataclass
- ContextSegment     — individual context unit with metadata
- EntityReference    — cross-session entity pointer
- TaskState          — tracked task with status
- TaskStatus         — enum: PENDING, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
- ToolContext        — tool invocation record
- SessionManager    — create / save / load / list / delete sessions
- SessionSerializer  — JSON/YAML round-trip with schema versioning
"""
from __future__ import annotations

from agent_session_linker.session.state import (
    ContextSegment,
    EntityReference,
    SessionState,
    TaskState,
    TaskStatus,
    ToolContext,
)
from agent_session_linker.session.manager import SessionManager
from agent_session_linker.session.serializer import SessionSerializer

__all__ = [
    "ContextSegment",
    "EntityReference",
    "SessionManager",
    "SessionSerializer",
    "SessionState",
    "TaskState",
    "TaskStatus",
    "ToolContext",
]
