#!/usr/bin/env python3
"""Example: Storage Backends

Demonstrates saving and restoring sessions using the in-memory,
filesystem, and SQLite storage backends.

Usage:
    python examples/02_storage_backends.py

Requirements:
    pip install agent-session-linker
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import agent_session_linker
from agent_session_linker import (
    ContextSegment,
    FilesystemBackend,
    InMemoryBackend,
    SQLiteBackend,
    SessionManager,
    SessionState,
)


def create_state(session_id: str) -> SessionState:
    return SessionState(
        session_id=session_id,
        segments=[
            ContextSegment(text="User asked about deployment status."),
            ContextSegment(text="Agent retrieved pipeline logs."),
        ],
    )


def demo_backend(label: str, manager: SessionManager, session_id: str) -> None:
    state = create_state(session_id)
    manager.save(state)
    loaded = manager.resume(session_id=session_id)
    print(f"  [{label}] saved + resumed: {len(loaded.segments)} segments")


def main() -> None:
    print(f"agent-session-linker version: {agent_session_linker.__version__}")

    print("\nIn-memory backend:")
    demo_backend("memory", SessionManager(backend=InMemoryBackend()), "s-mem-001")

    print("\nFilesystem backend:")
    with tempfile.TemporaryDirectory() as tmpdir:
        fs_backend = FilesystemBackend(root=Path(tmpdir))
        demo_backend("filesystem", SessionManager(backend=fs_backend), "s-fs-001")
        files = list(Path(tmpdir).iterdir())
        print(f"  Files written: {len(files)}")

    print("\nSQLite backend:")
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    try:
        sq_backend = SQLiteBackend(db_path=db_path)
        demo_backend("sqlite", SessionManager(backend=sq_backend), "s-sq-001")
        print(f"  DB size: {db_path.stat().st_size} bytes")
    finally:
        db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
