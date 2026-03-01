#!/usr/bin/env python3
"""Example: LangChain Session Integration

Demonstrates exporting sessions to LangChain memory format and
using session context as chat history for LangChain chains.

Usage:
    python examples/07_langchain_sessions.py

Requirements:
    pip install agent-session-linker
    pip install langchain   # optional — example degrades gracefully
"""
from __future__ import annotations

import agent_session_linker
from agent_session_linker import (
    ContextSegment,
    InMemoryBackend,
    LangChainExporter,
    LangChainImporter,
    SessionManager,
    SessionState,
)

try:
    from langchain.schema import HumanMessage, AIMessage
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False


def main() -> None:
    print(f"agent-session-linker version: {agent_session_linker.__version__}")

    if not _LANGCHAIN_AVAILABLE:
        print("LangChain not installed — demonstrating export only.")
        print("Install with: pip install langchain")

    # Build a session with conversation history
    backend = InMemoryBackend()
    manager = SessionManager(backend=backend)
    state = SessionState(
        session_id="lc-session-001",
        segments=[
            ContextSegment(
                text="User: What are the Q3 highlights?",
                role="human",
                importance=0.8,
            ),
            ContextSegment(
                text="Assistant: Revenue grew 18% year-over-year.",
                role="assistant",
                importance=0.9,
            ),
            ContextSegment(
                text="User: Which product drove the most growth?",
                role="human",
                importance=0.8,
            ),
            ContextSegment(
                text="Assistant: The enterprise tier accounted for 62% of growth.",
                role="assistant",
                importance=0.9,
            ),
        ],
    )
    manager.save(state)
    print(f"Session '{state.session_id}': {len(state.segments)} segments saved")

    # Export to LangChain message format
    exporter = LangChainExporter()
    lc_messages = exporter.export(state)
    print(f"\nExported {len(lc_messages)} LangChain messages:")
    for msg in lc_messages:
        role = type(msg).__name__
        print(f"  [{role}] {str(msg.content)[:60]}")

    # Import from LangChain messages back to a session
    if _LANGCHAIN_AVAILABLE:
        from langchain.schema import HumanMessage, AIMessage
        external_messages = [
            HumanMessage(content="Summarise the main risks."),
            AIMessage(content="The primary risk is supply chain disruption."),
        ]
        importer = LangChainImporter()
        imported = importer.import_messages(
            messages=external_messages,
            session_id="lc-import-001",
        )
        print(f"\nImported from LangChain: {len(imported.segments)} segments")
        for seg in imported.segments:
            print(f"  [{seg.role}] {seg.text[:60]}")
    else:
        print("\n(LangChain import skipped — not installed)")


if __name__ == "__main__":
    main()
