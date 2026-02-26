"""Middleware subpackage for session lifecycle management.

Provides hooks, sliding context windows, and periodic checkpointing that
sit between the raw storage layer and application-level session usage.

Public surface
--------------
- SessionMiddleware     — before/after request hooks for auto save/load
- ContextWindowManager  — sliding token-budget context window
- CheckpointManager     — periodic session snapshotting and restoration
- CheckpointRecord      — dataclass describing a checkpoint
"""
from __future__ import annotations

from agent_session_linker.middleware.checkpoint import CheckpointManager, CheckpointRecord
from agent_session_linker.middleware.context_window import ContextWindowManager
from agent_session_linker.middleware.session_middleware import SessionMiddleware

__all__ = [
    "CheckpointManager",
    "CheckpointRecord",
    "ContextWindowManager",
    "SessionMiddleware",
]
