"""Tests for EntityLinker and _normalised_edit_distance."""
from __future__ import annotations

import pytest

from agent_session_linker.entity.extractor import Entity
from agent_session_linker.entity.linker import EntityLinker, _normalised_edit_distance


def _entity(text: str, entity_type: str = "PERSON") -> Entity:
    return Entity(text=text, entity_type=entity_type, start=0, end=len(text))


class TestNormalisedEditDistance:
    def test_identical_strings_returns_zero(self) -> None:
        assert _normalised_edit_distance("abc", "abc") == pytest.approx(0.0)

    def test_empty_source_returns_one(self) -> None:
        assert _normalised_edit_distance("", "abc") == pytest.approx(1.0)

    def test_empty_target_returns_one(self) -> None:
        assert _normalised_edit_distance("abc", "") == pytest.approx(1.0)

    def test_both_empty_returns_zero(self) -> None:
        assert _normalised_edit_distance("", "") == pytest.approx(0.0)

    def test_single_char_difference(self) -> None:
        dist = _normalised_edit_distance("cat", "bat")
        # One substitution out of 3 chars.
        assert dist == pytest.approx(1 / 3)

    def test_symmetry(self) -> None:
        a = "kitten"
        b = "sitting"
        assert _normalised_edit_distance(a, b) == pytest.approx(
            _normalised_edit_distance(b, a)
        )

    def test_completely_different_strings(self) -> None:
        dist = _normalised_edit_distance("abc", "xyz")
        assert dist > 0.5

    def test_shorter_source_is_padded(self) -> None:
        dist = _normalised_edit_distance("a", "abc")
        assert 0.0 < dist <= 1.0


class TestEntityLinker:
    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            EntityLinker(similarity_threshold=0.0)

    def test_threshold_above_one_raises(self) -> None:
        with pytest.raises(ValueError):
            EntityLinker(similarity_threshold=1.1)

    def test_link_empty_catalogue_returns_none(self) -> None:
        linker = EntityLinker()
        entity = _entity("John Smith")
        assert linker.link(entity, []) is None

    def test_link_exact_match(self) -> None:
        linker = EntityLinker(similarity_threshold=0.9)
        mention = _entity("John Smith")
        known = [_entity("John Smith"), _entity("Jane Doe")]
        result = linker.link(mention, known)
        assert result is not None
        assert result.text == "John Smith"

    def test_link_fuzzy_match(self) -> None:
        linker = EntityLinker(similarity_threshold=0.7)
        mention = _entity("Jon Smith")  # typo
        known = [_entity("John Smith")]
        result = linker.link(mention, known)
        assert result is not None

    def test_link_below_threshold_returns_none(self) -> None:
        linker = EntityLinker(similarity_threshold=0.99)
        mention = _entity("completely different person")
        known = [_entity("John Smith")]
        result = linker.link(mention, known)
        assert result is None

    def test_require_same_type_filters_cross_type(self) -> None:
        linker = EntityLinker(require_same_type=True)
        mention = _entity("Google", entity_type="ORG")
        known = [_entity("Google", entity_type="PERSON")]  # wrong type
        result = linker.link(mention, known)
        assert result is None

    def test_require_same_type_false_accepts_cross_type(self) -> None:
        linker = EntityLinker(require_same_type=False, similarity_threshold=0.8)
        mention = _entity("Google", entity_type="ORG")
        known = [_entity("Google", entity_type="PERSON")]
        result = linker.link(mention, known)
        assert result is not None

    def test_case_insensitive_by_default(self) -> None:
        linker = EntityLinker(similarity_threshold=0.9, case_sensitive=False)
        mention = _entity("john smith")
        known = [_entity("John Smith")]
        result = linker.link(mention, known)
        assert result is not None

    def test_case_sensitive_does_not_match_different_case(self) -> None:
        linker = EntityLinker(similarity_threshold=0.95, case_sensitive=True)
        mention = _entity("john smith")
        known = [_entity("John Smith")]
        # Different case, should fail at high threshold.
        result = linker.link(mention, known)
        assert result is None

    def test_link_all_returns_pairs(self) -> None:
        linker = EntityLinker()
        entities = [_entity("Alice"), _entity("Bob")]
        known = [_entity("Alice Smith"), _entity("Bobby")]
        results = linker.link_all(entities, known)
        assert len(results) == 2
        assert all(isinstance(pair, tuple) for pair in results)

    def test_similarity_method(self) -> None:
        linker = EntityLinker()
        score = linker.similarity("hello", "hello")
        assert score == pytest.approx(1.0)

    def test_similarity_different_strings(self) -> None:
        linker = EntityLinker()
        score = linker.similarity("abc", "xyz")
        assert 0.0 <= score <= 1.0

    def test_no_matching_type_returns_none(self) -> None:
        linker = EntityLinker(require_same_type=True)
        mention = _entity("OpenAI", entity_type="ORG")
        known = [_entity("Alice", entity_type="PERSON")]
        result = linker.link(mention, known)
        assert result is None
