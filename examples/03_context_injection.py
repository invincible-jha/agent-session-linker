#!/usr/bin/env python3
"""Example: Context Injection and Freshness Decay

Demonstrates injecting session context into a prompt window and
applying freshness decay to age out stale context segments.

Usage:
    python examples/03_context_injection.py

Requirements:
    pip install agent-session-linker
"""
from __future__ import annotations

import time

import agent_session_linker
from agent_session_linker import (
    ContextInjector,
    ContextSegment,
    ContextSummarizer,
    DecayCurve,
    FreshnessDecay,
    InjectionConfig,
    RelevanceScorer,
    SessionState,
)


def main() -> None:
    print(f"agent-session-linker version: {agent_session_linker.__version__}")

    # Build a session state with several context segments
    state = SessionState(
        session_id="demo-inject-001",
        segments=[
            ContextSegment(text="User is analysing Q3 revenue data.", importance=0.9),
            ContextSegment(text="Database connection established.", importance=0.4),
            ContextSegment(text="Retrieved 2,400 rows from finance table.", importance=0.7),
            ContextSegment(text="Agent generated bar chart for revenue.", importance=0.8),
            ContextSegment(text="Session started 10 minutes ago.", importance=0.2),
        ],
    )
    print(f"Session segments: {len(state.segments)}")

    # Inject relevant context into a prompt (token budget)
    config = InjectionConfig(max_tokens=200, top_k=3)
    injector = ContextInjector(config=config)
    query = "What data has been analysed so far?"
    injected = injector.inject(state=state, query=query)
    print(f"\nInjected {len(injected.selected_segments)} segments for query:")
    for seg in injected.selected_segments:
        print(f"  [{seg.importance:.1f}] {seg.text[:60]}")

    # Relevance scoring
    scorer = RelevanceScorer()
    scores = scorer.score(query=query, segments=state.segments)
    print(f"\nRelevance scores (top 3):")
    for seg, score in sorted(scores, key=lambda x: x[1], reverse=True)[:3]:
        print(f"  [{score:.3f}] {seg.text[:55]}")

    # Freshness decay — reduce importance of older segments
    decay = FreshnessDecay(curve=DecayCurve.EXPONENTIAL, half_life_seconds=60)
    decayed = decay.apply(state=state)
    print(f"\nAfter freshness decay:")
    for seg in decayed.segments:
        print(f"  importance={seg.importance:.3f} | {seg.text[:50]}")

    # Summarise context
    summarizer = ContextSummarizer()
    summary = summarizer.summarize(state=state, max_words=30)
    print(f"\nContext summary ({len(summary.split())} words):")
    print(f"  {summary}")


if __name__ == "__main__":
    main()
