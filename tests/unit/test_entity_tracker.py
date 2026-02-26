"""Tests for EntityTracker and TrackedEntity."""
from __future__ import annotations

import pytest

from agent_session_linker.entity.extractor import Entity
from agent_session_linker.entity.tracker import EntityTracker, TrackedEntity


def _entity(text: str, entity_type: str = "PERSON", confidence: float = 1.0) -> Entity:
    return Entity(text=text, entity_type=entity_type, start=0, end=len(text), confidence=confidence)


class TestTrackedEntityRepr:
    def test_repr_contains_text(self) -> None:
        te = TrackedEntity(text="alice", entity_type="PERSON")
        assert "alice" in repr(te)

    def test_repr_contains_type(self) -> None:
        te = TrackedEntity(text="openai", entity_type="ORG")
        assert "ORG" in repr(te)


class TestEntityTrackerUpdate:
    def test_new_entity_added(self) -> None:
        tracker = EntityTracker()
        tracker.update([_entity("John Smith")])
        assert len(tracker) == 1

    def test_existing_entity_increments_frequency(self) -> None:
        tracker = EntityTracker()
        tracker.update([_entity("John Smith")])
        tracker.update([_entity("John Smith")])
        te = tracker.get("John Smith", "PERSON")
        assert te is not None
        assert te.frequency == 2

    def test_case_insensitive_normalisation(self) -> None:
        tracker = EntityTracker(case_sensitive=False)
        tracker.update([_entity("OpenAI")])
        tracker.update([_entity("openai")])
        te = tracker.get("openai", "ORG")
        # Both map to the same key.
        te_person = tracker.get("openai", "PERSON")
        assert te_person is not None
        assert te_person.frequency == 2

    def test_case_sensitive_keeps_separate(self) -> None:
        tracker = EntityTracker(case_sensitive=True)
        tracker.update([_entity("OpenAI", "ORG")])
        tracker.update([_entity("openai", "ORG")])
        assert len(tracker) == 2

    def test_confidence_averaged(self) -> None:
        tracker = EntityTracker()
        tracker.update([_entity("Alice", confidence=1.0)])
        tracker.update([_entity("Alice", confidence=0.5)])
        te = tracker.get("alice", "PERSON")
        assert te is not None
        assert 0.5 < te.confidence < 1.0

    def test_surface_forms_tracked(self) -> None:
        tracker = EntityTracker(case_sensitive=False)
        tracker.update([_entity("OpenAI")])
        tracker.update([_entity("openai")])
        te = tracker.get("openai", "PERSON")
        assert te is not None
        assert "OpenAI" in te.surface_forms

    def test_duplicate_surface_form_not_added_twice(self) -> None:
        tracker = EntityTracker()
        tracker.update([_entity("Alice")])
        tracker.update([_entity("Alice")])
        te = tracker.get("alice", "PERSON")
        assert te is not None
        assert te.surface_forms.count("Alice") == 1

    def test_multiple_entities_in_batch(self) -> None:
        tracker = EntityTracker()
        tracker.update([
            _entity("Alice", "PERSON"),
            _entity("OpenAI", "ORG"),
        ])
        assert len(tracker) == 2


class TestEntityTrackerQuery:
    def _populated_tracker(self) -> EntityTracker:
        tracker = EntityTracker()
        tracker.update([
            _entity("Alice", "PERSON"),
            _entity("Bob", "PERSON"),
            _entity("OpenAI", "ORG"),
        ])
        tracker.update([_entity("Alice", "PERSON")])  # Alice freq = 2
        return tracker

    def test_get_top_returns_most_frequent(self) -> None:
        tracker = self._populated_tracker()
        top = tracker.get_top(1)
        assert top[0].text == "alice"
        assert top[0].frequency == 2

    def test_get_top_with_type_filter(self) -> None:
        tracker = self._populated_tracker()
        top = tracker.get_top(5, entity_type="ORG")
        assert all(e.entity_type == "ORG" for e in top)

    def test_get_top_respects_limit(self) -> None:
        tracker = self._populated_tracker()
        top = tracker.get_top(1)
        assert len(top) == 1

    def test_get_by_type_filters(self) -> None:
        tracker = self._populated_tracker()
        orgs = tracker.get_by_type("ORG")
        assert all(e.entity_type == "ORG" for e in orgs)
        assert len(orgs) == 1

    def test_get_by_type_sorted_by_frequency(self) -> None:
        tracker = EntityTracker()
        tracker.update([_entity("Alice", "PERSON")])
        tracker.update([_entity("Alice", "PERSON")])
        tracker.update([_entity("Bob", "PERSON")])
        persons = tracker.get_by_type("PERSON")
        assert persons[0].frequency >= persons[-1].frequency

    def test_get_all_returns_all(self) -> None:
        tracker = self._populated_tracker()
        all_entities = tracker.get_all()
        assert len(all_entities) == 3

    def test_get_returns_none_for_unknown(self) -> None:
        tracker = EntityTracker()
        assert tracker.get("ghost", "PERSON") is None

    def test_reset_clears_all(self) -> None:
        tracker = self._populated_tracker()
        tracker.reset()
        assert len(tracker) == 0

    def test_repr_contains_count(self) -> None:
        tracker = self._populated_tracker()
        assert "3" in repr(tracker)
