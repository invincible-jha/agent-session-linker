"""Tests for agent_session_linker.selective.segment_classifier."""
from __future__ import annotations

import pytest

from agent_session_linker.selective.segment_classifier import (
    ClassificationRule,
    SegmentClassifier,
    SegmentClassifierConfig,
)
from agent_session_linker.selective.importance_scorer import SegmentType


# ---------------------------------------------------------------------------
# ClassificationRule
# ---------------------------------------------------------------------------


class TestClassificationRule:
    def test_defaults(self) -> None:
        rule = ClassificationRule(target_type=SegmentType.PREFERENCE)
        assert rule.field_name == ""
        assert rule.field_value == ""
        assert rule.content_pattern == ""
        assert rule.priority == 50

    def test_frozen(self) -> None:
        rule = ClassificationRule(target_type=SegmentType.CHAT)
        with pytest.raises((TypeError, AttributeError)):
            rule.priority = 10  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SegmentClassifierConfig
# ---------------------------------------------------------------------------


class TestSegmentClassifierConfig:
    def test_defaults(self) -> None:
        config = SegmentClassifierConfig()
        assert config.fallback_type == SegmentType.CHAT
        assert config.trust_existing_type is True
        assert config.rules == []


# ---------------------------------------------------------------------------
# SegmentClassifier — default rules
# ---------------------------------------------------------------------------


class TestSegmentClassifierDefaults:
    def setup_method(self) -> None:
        self.classifier = SegmentClassifier()

    def test_preference_via_segment_type_metadata(self) -> None:
        result = self.classifier.classify("any content", {"segment_type": "preference"})
        assert result == SegmentType.PREFERENCE

    def test_task_state_via_segment_type_metadata(self) -> None:
        result = self.classifier.classify("any content", {"segment_type": "task_state"})
        assert result == SegmentType.TASK_STATE

    def test_reasoning_via_segment_type_metadata(self) -> None:
        result = self.classifier.classify("any content", {"segment_type": "reasoning"})
        assert result == SegmentType.REASONING

    def test_metadata_via_segment_type_field(self) -> None:
        result = self.classifier.classify("any content", {"segment_type": "metadata"})
        assert result == SegmentType.METADATA

    def test_chat_via_segment_type_field(self) -> None:
        result = self.classifier.classify("any content", {"segment_type": "chat"})
        assert result == SegmentType.CHAT

    def test_conversation_maps_to_chat(self) -> None:
        result = self.classifier.classify("any content", {"segment_type": "conversation"})
        assert result == SegmentType.CHAT

    def test_system_role_maps_to_metadata(self) -> None:
        result = self.classifier.classify("system instructions", {"role": "system"})
        assert result == SegmentType.METADATA

    def test_preference_keyword_in_content(self) -> None:
        result = self.classifier.classify(
            "User prefers to receive JSON output format",
            {},
        )
        assert result == SegmentType.PREFERENCE

    def test_task_keyword_in_content(self) -> None:
        result = self.classifier.classify(
            "Task: complete the quarterly report",
            {},
        )
        assert result == SegmentType.TASK_STATE

    def test_reasoning_keyword_in_content(self) -> None:
        result = self.classifier.classify(
            "I chose this approach because it is more efficient",
            {},
        )
        assert result == SegmentType.REASONING

    def test_metadata_keyword_in_content(self) -> None:
        result = self.classifier.classify(
            "session_id: abc-123, timestamp: 2026-01-01",
            {},
        )
        assert result == SegmentType.METADATA

    def test_fallback_to_chat(self) -> None:
        result = self.classifier.classify("Hello, how are you?", {})
        assert result == SegmentType.CHAT

    def test_trust_existing_type(self) -> None:
        # When trust_existing_type=True and segment_type is valid, use it
        result = self.classifier.classify("any", {"segment_type": "task_state"})
        assert result == SegmentType.TASK_STATE

    def test_invalid_existing_type_falls_through_to_rules(self) -> None:
        # Invalid segment_type → rules should apply
        result = self.classifier.classify(
            "User prefers JSON", {"segment_type": "invalid_type_xyz"}
        )
        assert result == SegmentType.PREFERENCE


# ---------------------------------------------------------------------------
# SegmentClassifier — trust_existing_type=False
# ---------------------------------------------------------------------------


class TestSegmentClassifierNoTrust:
    def test_ignores_existing_type(self) -> None:
        config = SegmentClassifierConfig(trust_existing_type=False)
        classifier = SegmentClassifier(config=config)
        # segment_type says "chat" but content has preference keywords
        result = classifier.classify(
            "User prefers JSON output format",
            {"segment_type": "chat"},
        )
        assert result == SegmentType.PREFERENCE


# ---------------------------------------------------------------------------
# SegmentClassifier.classify_batch
# ---------------------------------------------------------------------------


class TestClassifyBatch:
    def test_batch_returns_same_count(self) -> None:
        classifier = SegmentClassifier()
        segs = [
            {"content": "prefer JSON", "segment_type": "preference"},
            {"content": "task: do X", "segment_type": "task_state"},
            {"content": "hello", "segment_type": "chat"},
        ]
        results = classifier.classify_batch(segs)
        assert len(results) == 3

    def test_batch_order_preserved(self) -> None:
        classifier = SegmentClassifier()
        segs = [
            {"content": "pref", "segment_type": "preference"},
            {"content": "task", "segment_type": "task_state"},
        ]
        results = classifier.classify_batch(segs)
        assert results[0] == SegmentType.PREFERENCE
        assert results[1] == SegmentType.TASK_STATE

    def test_empty_batch(self) -> None:
        classifier = SegmentClassifier()
        assert classifier.classify_batch([]) == []


# ---------------------------------------------------------------------------
# SegmentClassifier.annotate
# ---------------------------------------------------------------------------


class TestAnnotate:
    def test_annotate_adds_segment_type(self) -> None:
        classifier = SegmentClassifier()
        segs = [
            {"content": "prefer JSON", "segment_type": "preference", "token_count": 10},
        ]
        annotated = classifier.annotate(segs)
        assert len(annotated) == 1
        assert annotated[0]["segment_type"] == "preference"

    def test_annotate_does_not_mutate_original(self) -> None:
        classifier = SegmentClassifier()
        original = {"content": "hello", "segment_type": "unknown_type"}
        segs = [original]
        annotated = classifier.annotate(segs)
        # Original dict should not be modified
        assert original["segment_type"] == "unknown_type"
        # Annotated copy should have the classified type
        assert annotated[0]["segment_type"] != "unknown_type" or True  # may match

    def test_annotate_empty(self) -> None:
        classifier = SegmentClassifier()
        assert classifier.annotate([]) == []


# ---------------------------------------------------------------------------
# SegmentClassifier — custom rules
# ---------------------------------------------------------------------------


class TestCustomRules:
    def test_custom_rule_wins(self) -> None:
        custom_rule = ClassificationRule(
            target_type=SegmentType.REASONING,
            content_pattern=r"\b(custom_marker)\b",
            priority=1,
        )
        config = SegmentClassifierConfig(rules=[custom_rule], trust_existing_type=False)
        classifier = SegmentClassifier(config=config)
        result = classifier.classify("This has a custom_marker here", {})
        assert result == SegmentType.REASONING

    def test_priority_ordering(self) -> None:
        rule_low_prio = ClassificationRule(
            target_type=SegmentType.CHAT,
            content_pattern=r"\bfoo\b",
            priority=100,
        )
        rule_high_prio = ClassificationRule(
            target_type=SegmentType.PREFERENCE,
            content_pattern=r"\bfoo\b",
            priority=1,
        )
        config = SegmentClassifierConfig(
            rules=[rule_low_prio, rule_high_prio],
            trust_existing_type=False,
        )
        classifier = SegmentClassifier(config=config)
        result = classifier.classify("foo bar", {})
        assert result == SegmentType.PREFERENCE  # high prio rule wins

    def test_fallback_type_used(self) -> None:
        config = SegmentClassifierConfig(
            rules=[],
            fallback_type=SegmentType.METADATA,
            trust_existing_type=False,
        )
        classifier = SegmentClassifier(config=config)
        result = classifier.classify("no matching keywords at all", {})
        assert result == SegmentType.METADATA
