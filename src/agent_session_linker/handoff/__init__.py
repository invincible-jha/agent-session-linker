"""Cross-agent context handoff.

Transfer session context from one agent to another with optional filtering
and transformation of segments, entities, and tasks.

Classes
-------
HandoffPayload
    Immutable snapshot of context selected for handoff.
HandoffConfig
    Configuration controlling what is included in the handoff.
HandoffBuilder
    Constructs a HandoffPayload from a SessionState.
"""
from __future__ import annotations

from agent_session_linker.handoff.context_handoff import (
    HandoffBuilder,
    HandoffConfig,
    HandoffPayload,
)

__all__ = [
    "HandoffBuilder",
    "HandoffConfig",
    "HandoffPayload",
]
