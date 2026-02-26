"""Regex-based named entity recognition.

Extracts typed entities from free text using handcrafted patterns.
No external NLP libraries are required.

Supported entity types
----------------------
- PERSON   — names matching a two-or-more capitalised word pattern
- ORG      — organisation suffixes (Inc, Ltd, Corp, LLC, Co, Group, etc.)
- EMAIL    — RFC-5321 local-part + domain
- URL      — http / https / ftp URLs
- DATE     — common date formats (ISO, US, European, relative)
- NUMBER   — integers and decimals
- MONEY    — currency-prefixed or -suffixed numeric values

Classes
-------
- Entity          — dataclass representing a single extracted entity
- EntityExtractor — extract entities from text
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A single named entity extracted from text.

    Parameters
    ----------
    text:
        The exact matched surface form in the source text.
    entity_type:
        Category label: PERSON, ORG, EMAIL, URL, DATE, NUMBER, or MONEY.
    start:
        Character offset of the first character in the source text.
    end:
        Character offset one past the last character (exclusive).
    confidence:
        Extraction confidence in [0.0, 1.0].  Pattern-based extractors
        typically assign 1.0; heuristic matches may use lower values.
    """

    text: str
    entity_type: str
    start: int
    end: int
    confidence: float = field(default=1.0)

    def __repr__(self) -> str:
        return (
            f"Entity(text={self.text!r}, type={self.entity_type!r}, "
            f"span=({self.start}, {self.end}), confidence={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Email: local@domain.tld
_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b"
)

# URLs: http/https/ftp with optional path/query
_URL_RE = re.compile(
    r"https?://[^\s<>\"{}|\\^`\[\]]+|ftp://[^\s<>\"{}|\\^`\[\]]+"
)

# Money: optional currency symbol, integer or decimal, optional suffix
_MONEY_RE = re.compile(
    r"(?:"
    r"[$€£¥₹₽]\s*\d[\d,]*(?:\.\d+)?[KMBTkmbt]?"  # prefix-symbol
    r"|"
    r"\d[\d,]*(?:\.\d+)?\s*(?:USD|EUR|GBP|JPY|INR|CAD|AUD|CHF|CNY|BTC|ETH)"  # suffix ISO
    r"|"
    r"\d[\d,]*(?:\.\d+)?\s*(?:dollars?|euros?|pounds?|yen|rupees?|cents?)"  # suffix words
    r")"
)

# Dates: ISO (2024-01-15), US (01/15/2024), EU (15.01.2024), written, relative
_DATE_RE = re.compile(
    r"\b(?:"
    # ISO date
    r"\d{4}-\d{2}-\d{2}"
    r"|"
    # US date with slashes or dashes: MM/DD/YYYY or MM-DD-YYYY
    r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}"
    r"|"
    # EU date with dots: DD.MM.YYYY
    r"\d{1,2}\.\d{1,2}\.\d{2,4}"
    r"|"
    # Written: January 15, 2024 or 15 January 2024
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r"|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+\d{4}"
    r"|"
    # Relative dates
    r"(?:yesterday|today|tomorrow|last\s+(?:week|month|year)|next\s+(?:week|month|year))"
    r")\b",
    re.IGNORECASE,
)

# Numbers: stand-alone integers or decimals (not already part of money/date)
_NUMBER_RE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?(?:[KMBTkmbt](?=\b))?\b"
)

# Organisation: word(s) followed by a legal suffix
_ORG_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9&\-]*(?:\s+[A-Z][A-Za-z0-9&\-]*)*"
    r"\s+(?:Inc\.?|Ltd\.?|LLC\.?|Corp\.?|Co\.?|Group|Holdings|Partners|"
    r"Technologies|Solutions|Systems|Services|Foundation|Institute|"
    r"Association|Enterprises|Ventures|Capital|Consulting)\b"
)

# Person: two or more capitalised words (strict pattern to reduce false positives)
_PERSON_RE = re.compile(
    r"\b(?:(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+)?"
    r"[A-Z][a-z]{1,20}"
    r"(?:\s+[A-Z][a-z]{1,20}){1,3}\b"
)

# Titles that look like person names but are not (common false-positive seeds)
_PERSON_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "United States", "New York", "Los Angeles", "San Francisco",
        "North America", "South America", "Latin America", "Middle East",
        "East Asia", "South Asia", "New Zealand", "Great Britain",
    }
)


# ---------------------------------------------------------------------------
# Helper: merge and de-duplicate overlapping spans
# ---------------------------------------------------------------------------


def _remove_overlaps(entities: list[Entity]) -> list[Entity]:
    """Remove lower-priority entities that overlap with higher-priority ones.

    Priority order (highest first): URL, EMAIL, MONEY, DATE, ORG, PERSON, NUMBER.

    Parameters
    ----------
    entities:
        Raw extracted entities, possibly overlapping.

    Returns
    -------
    list[Entity]
        Non-overlapping entities sorted by start offset.
    """
    priority: dict[str, int] = {
        "URL": 7,
        "EMAIL": 6,
        "MONEY": 5,
        "DATE": 4,
        "ORG": 3,
        "PERSON": 2,
        "NUMBER": 1,
    }
    sorted_entities = sorted(
        entities,
        key=lambda e: (-(priority.get(e.entity_type, 0)), e.start),
    )
    result: list[Entity] = []
    occupied: list[tuple[int, int]] = []

    for entity in sorted_entities:
        overlaps = any(
            entity.start < end and entity.end > start
            for start, end in occupied
        )
        if not overlaps:
            result.append(entity)
            occupied.append((entity.start, entity.end))

    result.sort(key=lambda e: e.start)
    return result


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Extract typed named entities from text using regex patterns.

    Each of the seven supported types is matched by a dedicated compiled
    pattern.  Overlapping matches are resolved by entity-type priority.

    Parameters
    ----------
    types:
        Set of entity types to extract.  Defaults to all supported types.
        Valid values: ``{"PERSON", "ORG", "EMAIL", "URL", "DATE", "NUMBER", "MONEY"}``.
    min_confidence:
        Minimum confidence threshold; matches below this value are excluded.
        Default: 0.5.
    """

    SUPPORTED_TYPES: frozenset[str] = frozenset(
        {"PERSON", "ORG", "EMAIL", "URL", "DATE", "NUMBER", "MONEY"}
    )

    def __init__(
        self,
        types: frozenset[str] | set[str] | None = None,
        min_confidence: float = 0.5,
    ) -> None:
        self.types: frozenset[str] = (
            frozenset(types) if types is not None else self.SUPPORTED_TYPES
        )
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> list[Entity]:
        """Extract all named entities from ``text``.

        Parameters
        ----------
        text:
            The source text to analyse.

        Returns
        -------
        list[Entity]
            Non-overlapping entities in document order.  When two patterns
            match the same span, the higher-priority type wins.
        """
        if not text:
            return []

        raw: list[Entity] = []

        if "URL" in self.types:
            raw.extend(self._extract_by_pattern(text, _URL_RE, "URL", confidence=1.0))

        if "EMAIL" in self.types:
            raw.extend(self._extract_by_pattern(text, _EMAIL_RE, "EMAIL", confidence=1.0))

        if "MONEY" in self.types:
            raw.extend(self._extract_by_pattern(text, _MONEY_RE, "MONEY", confidence=1.0))

        if "DATE" in self.types:
            raw.extend(self._extract_by_pattern(text, _DATE_RE, "DATE", confidence=0.95))

        if "ORG" in self.types:
            raw.extend(self._extract_by_pattern(text, _ORG_RE, "ORG", confidence=0.85))

        if "PERSON" in self.types:
            person_entities = self._extract_by_pattern(text, _PERSON_RE, "PERSON", confidence=0.75)
            person_entities = [
                e for e in person_entities
                if e.text not in _PERSON_EXCLUSIONS
            ]
            raw.extend(person_entities)

        if "NUMBER" in self.types:
            raw.extend(self._extract_by_pattern(text, _NUMBER_RE, "NUMBER", confidence=0.9))

        # Remove overlapping spans (higher-priority type wins).
        deduplicated = _remove_overlaps(raw)

        # Apply minimum confidence filter.
        return [e for e in deduplicated if e.confidence >= self.min_confidence]

    def extract_by_type(self, text: str, entity_type: str) -> list[Entity]:
        """Extract only entities of a specific type.

        Parameters
        ----------
        text:
            Source text.
        entity_type:
            One of the supported type labels.

        Returns
        -------
        list[Entity]
            Matching entities in document order.
        """
        return [e for e in self.extract(text) if e.entity_type == entity_type]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_by_pattern(
        text: str,
        pattern: re.Pattern[str],
        entity_type: str,
        confidence: float,
    ) -> list[Entity]:
        """Run a compiled regex over ``text`` and return Entity objects.

        Parameters
        ----------
        text:
            Source text.
        pattern:
            Compiled regex pattern.
        entity_type:
            Type label for matched spans.
        confidence:
            Confidence to assign all matches.

        Returns
        -------
        list[Entity]
            Matched entities.
        """
        entities: list[Entity] = []
        for match in pattern.finditer(text):
            matched_text = match.group().strip()
            if matched_text:
                entities.append(
                    Entity(
                        text=matched_text,
                        entity_type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                    )
                )
        return entities
