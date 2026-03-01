#!/usr/bin/env python3
"""Example: Session Linking and Checkpointing

Demonstrates linking sessions across conversation turns, building
session chains, and creating restore checkpoints.

Usage:
    python examples/05_session_linking.py

Requirements:
    pip install agent-session-linker
"""
from __future__ import annotations

import agent_session_linker
from agent_session_linker import (
    CheckpointManager,
    CheckpointRecord,
    ContextSegment,
    InMemoryBackend,
    LinkedSession,
    SessionChain,
    SessionLinker,
    SessionManager,
    SessionState,
)


def main() -> None:
    print(f"agent-session-linker version: {agent_session_linker.__version__}")

    backend = InMemoryBackend()
    manager = SessionManager(backend=backend)

    # Create three sessions representing consecutive conversations
    for i in range(1, 4):
        state = SessionState(
            session_id=f"session-{i:03d}",
            segments=[
                ContextSegment(text=f"Conversation {i}: topic introduced."),
                ContextSegment(text=f"Conversation {i}: action taken."),
            ],
        )
        manager.save(state)

    print(f"Sessions saved: {manager.count()}")

    # Link sessions into a chain
    linker = SessionLinker(manager=manager)
    linked: LinkedSession = linker.link(
        session_ids=["session-001", "session-002", "session-003"]
    )
    print(f"Linked session: {linked.session_id}")
    print(f"  Linked from: {linked.linked_from}")
    print(f"  Total segments: {linked.total_segment_count()}")

    # Build a session chain for ordered traversal
    chain = SessionChain(session_ids=["session-001", "session-002", "session-003"])
    chain.append("session-004")  # hypothetical next session
    print(f"\nSession chain: {chain.length()} sessions")
    for session_id in chain.iter():
        print(f"  {session_id}")

    # Checkpoint the current state
    cp_manager = CheckpointManager(backend=backend)
    checkpoint: CheckpointRecord = cp_manager.create(
        session_id="session-003",
        label="before-major-action",
    )
    print(f"\nCheckpoint created: {checkpoint.checkpoint_id}")
    print(f"  Label: {checkpoint.label}")

    # Restore from checkpoint
    restored = cp_manager.restore(checkpoint.checkpoint_id)
    print(f"  Restored to session: {restored.session_id} "
          f"({len(restored.segments)} segments)")


if __name__ == "__main__":
    main()
