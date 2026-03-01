#!/usr/bin/env python3
"""Example: Quickstart — agent-session-linker

Minimal working example: create a session, add context segments,
save, and resume from an in-memory backend.

Usage:
    python examples/01_quickstart.py

Requirements:
    pip install agent-session-linker
"""
from __future__ import annotations

import agent_session_linker
from agent_session_linker import (
    Session,
    SessionManager,
    InMemoryBackend,
    ContextSegment,
    TaskState,
    TaskStatus,
)


def main() -> None:
    print(f"agent-session-linker version: {agent_session_linker.__version__}")

    # Step 1: Create a session using the convenience class
    backend = InMemoryBackend()
    session = Session(session_id="session-001", backend=backend)
    session.add_context("user asked about Q3 revenue figures")
    session.add_context("assistant retrieved financial report")
    print(f"Session '{session.session_id}': {session.context_count()} segments")

    # Step 2: Attach a task state
    task = TaskState(
        task_id="task-revenue-lookup",
        description="Look up Q3 revenue",
        status=TaskStatus.IN_PROGRESS,
    )
    session.add_task(task)
    print(f"Task added: '{task.task_id}' ({task.status.value})")

    # Step 3: Save and resume with SessionManager
    manager = SessionManager(backend=backend)
    session.save()
    resumed = manager.resume(session_id="session-001")
    print(f"\nResumed session: {resumed.session_id}")
    print(f"  Context segments: {resumed.context_count()}")
    print(f"  Tasks: {len(resumed.tasks())}")

    # Step 4: Mark task complete
    session.complete_task("task-revenue-lookup")
    completed_tasks = [t for t in session.tasks() if t.status == TaskStatus.DONE]
    print(f"  Completed tasks: {len(completed_tasks)}")


if __name__ == "__main__":
    main()
