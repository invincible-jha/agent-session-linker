#!/usr/bin/env python3
"""Example: Entity Extraction and Tracking

Demonstrates extracting named entities from conversation text,
tracking them across turns, and linking co-references.

Usage:
    python examples/04_entity_tracking.py

Requirements:
    pip install agent-session-linker
"""
from __future__ import annotations

import agent_session_linker
from agent_session_linker import (
    Entity,
    EntityExtractor,
    EntityLinker,
    EntityTracker,
    TrackedEntity,
)


def main() -> None:
    print(f"agent-session-linker version: {agent_session_linker.__version__}")

    # Step 1: Extract entities from conversation turns
    extractor = EntityExtractor()
    turns = [
        "The project is called DarkEnergy and is managed by Alice.",
        "Alice confirmed that DarkEnergy will launch in March.",
        "The launch date for the project is March 15.",
    ]

    all_entities: list[Entity] = []
    for i, turn in enumerate(turns):
        entities = extractor.extract(text=turn)
        print(f"Turn {i + 1}: {len(entities)} entities — {[e.text for e in entities]}")
        all_entities.extend(entities)

    # Step 2: Track entities across turns
    tracker = EntityTracker()
    for entity in all_entities:
        tracker.update(entity)

    tracked: list[TrackedEntity] = tracker.list()
    print(f"\nTracked entities ({len(tracked)} unique):")
    for te in tracked:
        print(f"  [{te.entity_type}] '{te.canonical}' — seen {te.mention_count}x")

    # Step 3: Link co-references ("the project" -> "DarkEnergy")
    linker = EntityLinker(tracker=tracker)
    text = "the project will be announced by her next week"
    links = linker.link(text=text)
    print(f"\nCo-reference links in: '{text}'")
    for link in links:
        print(f"  '{link.surface}' -> '{link.resolved}'")


if __name__ == "__main__":
    main()
