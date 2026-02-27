"""Session branching for A/B testing and experimental divergence.

Create independent branches of a session to compare different strategies
or model configurations without polluting the canonical session.

Classes
-------
SessionBranch
    An independent branch derived from a parent session.
BranchManager
    Creates and manages branches for a parent session.
BranchConfig
    Configuration for branch creation.
"""
from __future__ import annotations

from agent_session_linker.branching.branch_manager import (
    BranchConfig,
    BranchManager,
    SessionBranch,
)

__all__ = [
    "BranchConfig",
    "BranchManager",
    "SessionBranch",
]
