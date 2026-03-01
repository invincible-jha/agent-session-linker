"""Shared bootstrap for agent-session-linker benchmarks."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_SRC = _REPO_ROOT / "src"
_BENCHMARKS = _REPO_ROOT / "benchmarks"

for _path in [str(_SRC), str(_BENCHMARKS)]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from agent_session_linker.session.manager import SessionManager
from agent_session_linker.session.state import SessionState
from agent_session_linker.storage.memory import InMemoryBackend
from agent_session_linker.linking.session_linker import SessionLinker

__all__ = ["SessionManager", "SessionState", "InMemoryBackend", "SessionLinker"]
