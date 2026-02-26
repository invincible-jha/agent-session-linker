"""Entity frequency tracking across conversation turns.

Maintains a running tally of entities observed across multiple calls to
``update()``, recording first/last seen timestamps and per-type indices
for fast retrieval.

Classes
-------
- TrackedEntity  — dataclass capturing frequency and temporal metadata
- EntityTracker  — aggregate and query entity statistics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_session_linker.entity.extractor import Entity


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class TrackedEntity:
    """An entity whose occurrences have been counted across turns.

    Parameters
    ----------
    text:
        The canonical surface form (normalised to lowercase).
    entity_type:
        Category label: PERSON, ORG, EMAIL, URL, DATE, NUMBER, or MONEY.
    frequency:
        Total number of times this entity has been observed.
    first_seen:
        UTC timestamp of the first observation.
    last_seen:
        UTC timestamp of the most recent observation.
    confidence:
        Average confidence score across all observations.
    surface_forms:
        All unique raw text forms that have been mapped to this entry.
    """

    text: str
    entity_type: str
    frequency: int = field(default=1)
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = field(default=1.0)
    surface_forms: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"TrackedEntity(text={self.text!r}, type={self.entity_type!r}, "
            f"frequency={self.frequency}, confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class EntityTracker:
    """Track entities observed across multiple conversation turns.

    Entities are keyed by ``(normalised_text, entity_type)``.  Each call to
    ``update`` increments frequency counters and refreshes timestamps.

    Parameters
    ----------
    case_sensitive:
        When False (default), entity texts are lowercased before keying so
        that "OpenAI" and "openai" are treated as the same entity.
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive
        self._entities: dict[tuple[str, str], TrackedEntity] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, entities: list[Entity]) -> None:
        """Record a batch of newly observed entities.

        Each entity is merged into the tracker.  If it has been seen before
        (same normalised text + type), its frequency is incremented and
        ``last_seen`` is refreshed.  Otherwise a new ``TrackedEntity`` is
        created.

        Parameters
        ----------
        entities:
            Entities extracted from a single turn or text block.
        """
        now = datetime.now(timezone.utc)
        for entity in entities:
            normalised = entity.text if self.case_sensitive else entity.text.lower()
            key = (normalised, entity.entity_type)

            if key in self._entities:
                tracked = self._entities[key]
                tracked.frequency += 1
                tracked.last_seen = now
                # Update running average confidence.
                old_total = tracked.confidence * (tracked.frequency - 1)
                tracked.confidence = (old_total + entity.confidence) / tracked.frequency
                if entity.text not in tracked.surface_forms:
                    tracked.surface_forms.append(entity.text)
            else:
                self._entities[key] = TrackedEntity(
                    text=normalised,
                    entity_type=entity.entity_type,
                    frequency=1,
                    first_seen=now,
                    last_seen=now,
                    confidence=entity.confidence,
                    surface_forms=[entity.text],
                )

    def get_top(self, n: int, entity_type: str | None = None) -> list[TrackedEntity]:
        """Return the top-``n`` most frequently observed entities.

        Parameters
        ----------
        n:
            Maximum number of results to return.
        entity_type:
            When provided, restrict results to this entity type.

        Returns
        -------
        list[TrackedEntity]
            Entities sorted by frequency descending, then by ``last_seen``
            descending as a tie-breaker.
        """
        candidates = list(self._entities.values())
        if entity_type is not None:
            candidates = [e for e in candidates if e.entity_type == entity_type]
        candidates.sort(key=lambda e: (e.frequency, e.last_seen), reverse=True)
        return candidates[:n]

    def get_by_type(self, entity_type: str) -> list[TrackedEntity]:
        """Return all tracked entities of a given type, most frequent first.

        Parameters
        ----------
        entity_type:
            Category label to filter by (e.g. "PERSON", "ORG").

        Returns
        -------
        list[TrackedEntity]
            Matching entities sorted by frequency descending.
        """
        matching = [
            e for e in self._entities.values()
            if e.entity_type == entity_type
        ]
        matching.sort(key=lambda e: e.frequency, reverse=True)
        return matching

    def get_all(self) -> list[TrackedEntity]:
        """Return all tracked entities sorted by frequency descending.

        Returns
        -------
        list[TrackedEntity]
            All tracked entities.
        """
        return sorted(self._entities.values(), key=lambda e: e.frequency, reverse=True)

    def get(self, text: str, entity_type: str) -> TrackedEntity | None:
        """Look up a specific tracked entity by text and type.

        Parameters
        ----------
        text:
            Surface form (normalised according to ``case_sensitive`` setting).
        entity_type:
            Category label.

        Returns
        -------
        TrackedEntity | None
            The tracked entity if found, else None.
        """
        normalised = text if self.case_sensitive else text.lower()
        return self._entities.get((normalised, entity_type))

    def reset(self) -> None:
        """Clear all tracked entities."""
        self._entities.clear()

    def __len__(self) -> int:
        return len(self._entities)

    def __repr__(self) -> str:
        return f"EntityTracker(tracked={len(self._entities)}, case_sensitive={self.case_sensitive})"
