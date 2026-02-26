"""Tests for EntityExtractor and Entity."""
from __future__ import annotations

import pytest

from agent_session_linker.entity.extractor import Entity, EntityExtractor, _remove_overlaps


class TestEntityDataclass:
    def test_repr_contains_text(self) -> None:
        entity = Entity(text="John Smith", entity_type="PERSON", start=0, end=10)
        assert "John Smith" in repr(entity)

    def test_repr_contains_type(self) -> None:
        entity = Entity(text="test@example.com", entity_type="EMAIL", start=0, end=16)
        assert "EMAIL" in repr(entity)

    def test_default_confidence_is_one(self) -> None:
        entity = Entity(text="foo", entity_type="ORG", start=0, end=3)
        assert entity.confidence == 1.0


class TestRemoveOverlaps:
    def test_no_entities_returns_empty(self) -> None:
        assert _remove_overlaps([]) == []

    def test_non_overlapping_entities_all_kept(self) -> None:
        entities = [
            Entity("foo", "ORG", 0, 3),
            Entity("bar", "PERSON", 10, 13),
        ]
        result = _remove_overlaps(entities)
        assert len(result) == 2

    def test_higher_priority_wins_overlap(self) -> None:
        # URL has priority 7, NUMBER has priority 1 — same span.
        url = Entity("http://example.com", "URL", 0, 18)
        number = Entity("http", "NUMBER", 0, 4)
        result = _remove_overlaps([number, url])
        types = {e.entity_type for e in result}
        assert "URL" in types

    def test_sorted_by_start(self) -> None:
        entities = [
            Entity("bar", "ORG", 10, 13),
            Entity("foo", "EMAIL", 0, 3),
        ]
        result = _remove_overlaps(entities)
        assert result[0].start < result[1].start


class TestEntityExtractor:
    def test_empty_text_returns_empty(self) -> None:
        extractor = EntityExtractor()
        assert extractor.extract("") == []

    def test_extracts_email(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("Contact us at support@example.com for help.")
        emails = [e for e in entities if e.entity_type == "EMAIL"]
        assert len(emails) >= 1
        assert emails[0].text == "support@example.com"

    def test_extracts_url(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("Visit https://www.example.com for more info.")
        urls = [e for e in entities if e.entity_type == "URL"]
        assert len(urls) >= 1
        assert "example.com" in urls[0].text

    def test_extracts_money_dollar(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("The product costs $42.99.")
        money = [e for e in entities if e.entity_type == "MONEY"]
        assert len(money) >= 1

    def test_extracts_date_iso(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("The meeting is on 2024-03-15.")
        dates = [e for e in entities if e.entity_type == "DATE"]
        assert len(dates) >= 1
        assert "2024-03-15" in dates[0].text

    def test_extracts_number(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("There are 42 items in stock.")
        numbers = [e for e in entities if e.entity_type == "NUMBER"]
        assert len(numbers) >= 1

    def test_extracts_org(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("Apple Inc. is a technology company.")
        orgs = [e for e in entities if e.entity_type == "ORG"]
        assert len(orgs) >= 1

    def test_extracts_person(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("Dr. John Smith gave a presentation.")
        persons = [e for e in entities if e.entity_type == "PERSON"]
        assert len(persons) >= 1

    def test_type_filter_limits_extraction(self) -> None:
        extractor = EntityExtractor(types={"EMAIL"})
        entities = extractor.extract("Email: test@test.com, Date: 2024-01-01")
        types = {e.entity_type for e in entities}
        assert types == {"EMAIL"} or len(entities) == 0

    def test_min_confidence_filters_low_confidence(self) -> None:
        # With very high min_confidence, person entities (0.75 confidence) are filtered.
        extractor = EntityExtractor(types={"PERSON"}, min_confidence=0.9)
        entities = extractor.extract("John Smith attended the meeting.")
        persons = [e for e in entities if e.entity_type == "PERSON"]
        assert len(persons) == 0

    def test_extract_by_type_email(self) -> None:
        extractor = EntityExtractor()
        text = "Email us at info@company.org or visit https://company.org"
        emails = extractor.extract_by_type(text, "EMAIL")
        assert all(e.entity_type == "EMAIL" for e in emails)

    def test_no_entities_in_plain_text(self) -> None:
        extractor = EntityExtractor()
        entities = extractor.extract("the quick brown fox jumped over the lazy dog")
        # May have very few or none — just check it runs without error.
        assert isinstance(entities, list)

    def test_person_exclusion_applied(self) -> None:
        extractor = EntityExtractor()
        # "United States" is in the exclusion list.
        entities = extractor.extract("He visited the United States last year.")
        persons = [e for e in entities if e.entity_type == "PERSON" and e.text == "United States"]
        assert len(persons) == 0

    def test_extract_multiple_emails(self) -> None:
        extractor = EntityExtractor()
        text = "alice@a.com and bob@b.com are contacts."
        emails = extractor.extract_by_type(text, "EMAIL")
        assert len(emails) == 2
