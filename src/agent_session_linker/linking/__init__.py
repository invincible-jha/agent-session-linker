"""Session linking subpackage.

Provides tools for creating, querying, and traversing relationships
between sessions.

Public surface
--------------
- LinkedSession  — dataclass describing a directed session relationship
- SessionLinker  — create and query session relationship graphs
- SessionChain   — ordered chain of sessions for linear continuations
"""
from __future__ import annotations

from agent_session_linker.linking.chain import SessionChain
from agent_session_linker.linking.session_linker import LinkedSession, SessionLinker

__all__ = [
    "LinkedSession",
    "SessionChain",
    "SessionLinker",
]
