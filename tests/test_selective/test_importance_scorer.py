"""Tests for agent_session_linker.selective.importance_scorer."""
from __future__ import annotations

import pytest

from agent_session_linker.selective.importance_scorer import (
    ImportanceScorer,
    ImportanceScorerConfig,
    ScoredSegment,
    SegmentType,
    _TYPE_PRIORS,
)


# ---------------------------------------------------------------------------
# SegmentType
# ---------------------------------------------------------------------------


class TestSegmentType:
    def test_all_types_have_priors(self) -> None:
        for st in SegmentType:
            assert st in _TYPE_PRIORS

    def test_preference_priority_highest(self) -> None:
        assert _TYPE_PRIORS[SegmentType.PREFERENCE] >= _TYPE_PRIORS[SegmentType.TASK_STATE]
        assert _TYPE_PRIORS[SegmentType.TASK_STATE] > _TYPE_PRIORS[SegmentType.REASONING]
        assert _TYPE_PRIORS[SegmentType.REASONING] > _TYPE_PRIORS[SegmentType.CHAT]

    def test_chat_has_lowest_prior(self) -> None:
        chat_prior = _TYPE_PRIORS[SegmentType.CHAT]
        for st, prior in _TYPE_PRIORS.items():
            if st not in (SegmentType.CHAT, SegmentType.UNKNOWN):
                assert prior > chat_prior


# ---------------------------------------------------------------------------
# ScoredSegment
# ---------------------------------------------------------------------------


class TestScoredSegment:
    def test_construction(self) -> None:
        seg = ScoredSegment(
            segment_id="s1",
            segment_type=SegmentType.PREFERENCE,
            importance_score=0.90,
            token_count=50,
        )
        assert seg.segment_id == "s1"
        assert seg.importance_score == pytest.approx(0.90)

    def test_is_high_importance_default_threshold(self) -> None:
        high = ScoredSegment("s1", SegmentType.PREFERENCE, 0.90, 10)
        low = ScoredSegment("s2", SegmentType.CHAT, 0.30, 10)
        assert high.is_high_importance() is True
        assert low.is_high_importance() is False

    def test_is_high_importance_custom_threshold(self) -> None:
        seg = ScoredSegment("s1", SegmentType.REASONING, 0.60, 10)
        assert seg.is_high_importance(0.60) is True
        assert seg.is_high_importance(0.61) is False

    def test_to_dict(self) -> None:
        seg = ScoredSegment(
            segment_id="s1",
            segment_type=SegmentType.TASK_STATE,
            importance_score=0.85,
            token_count=100,
            content_preview="Task: complete report",
        )
        d = seg.to_dict()
        assert d["segment_id"] == "s1"
        assert d["segment_type"] == "task_state"
        assert d["importance_score"] == pytest.approx(0.85)

    def test_frozen(self) -> None:
        seg = ScoredSegment("s1", SegmentType.CHAT, 0.30, 10)
        with pytest.raises((TypeError, AttributeError)):
            seg.importance_score = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ImportanceScorer.score_segment
# ---------------------------------------------------------------------------


class TestImportanceScorerScoreSegment:
    def test_preference_high_score(self) -> None:
        scorer = ImportanceScorer()
        scored = scorer.score_segment(
            "s1", SegmentType.PREFERENCE, "User prefers JSON", 10
        )
        assert scored.importance_score >= 0.85

    def test_task_state_high_score(self) -> None:
        scorer = ImportanceScorer()
        scored = scorer.score_segment(
            "s1", SegmentType.TASK_STATE, "Current task: write report", 20
        )
        assert scored.importance_score >= 0.80

    def test_chat_low_score(self) -> None:
        scorer = ImportanceScorer()
        scored = scorer.score_segment(
            "s1", SegmentType.CHAT, "Hello, how are you?", 10
        )
        assert scored.importance_score <= 0.50

    def test_keyword_boost_applies(self) -> None:
        scorer = ImportanceScorer()
        with_keyword = scorer.score_segment(
            "s1", SegmentType.CHAT, "Critical: this is important", 10
        )
        without_keyword = scorer.score_segment(
            "s2", SegmentType.CHAT, "just a regular message", 10
        )
        assert with_keyword.importance_score > without_keyword.importance_score

    def test_recency_boost_applies(self) -> None:
        scorer = ImportanceScorer()
        old = scorer.score_segment(
            "s1", SegmentType.CHAT, "old message", 10, recency_rank=0.0
        )
        recent = scorer.score_segment(
            "s2", SegmentType.CHAT, "recent message", 10, recency_rank=1.0
        )
        assert recent.importance_score > old.importance_score

    def test_score_clamped_to_1(self) -> None:
        scorer = ImportanceScorer()
        # preference (0.90) + keyword_boost (0.10) + recency_boost (0.05) > 1.0
        scored = scorer.score_segment(
            "s1", SegmentType.PREFERENCE, "critical preference note", 10,
            recency_rank=1.0
        )
        assert scored.importance_score <= 1.0

    def test_score_clamped_to_0(self) -> None:
        config = ImportanceScorerConfig(type_modifiers={"chat": -1.0})
        scorer = ImportanceScorer(config)
        scored = scorer.score_segment("s1", SegmentType.CHAT, "msg", 5)
        assert scored.importance_score >= 0.0

    def test_content_preview_truncated(self) -> None:
        scorer = ImportanceScorer()
        long_content = "x" * 200
        scored = scorer.score_segment("s1", SegmentType.CHAT, long_content, 50)
        assert len(scored.content_preview) == 120


# ---------------------------------------------------------------------------
# ImportanceScorer.score_segments
# ---------------------------------------------------------------------------


class TestImportanceScorerScoreSegments:
    def _make_segment(self, seg_id: str, seg_type: str, content: str, tokens: int) -> dict:
        return {
            "segment_id": seg_id,
            "segment_type": seg_type,
            "content": content,
            "token_count": tokens,
        }

    def test_empty_list(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.score_segments([]) == []

    def test_returns_same_count(self) -> None:
        scorer = ImportanceScorer()
        segs = [
            self._make_segment("s1", "preference", "content", 10),
            self._make_segment("s2", "chat", "hello", 5),
        ]
        result = scorer.score_segments(segs)
        assert len(result) == 2

    def test_order_preserved(self) -> None:
        scorer = ImportanceScorer()
        segs = [
            self._make_segment("a", "chat", "msg1", 10),
            self._make_segment("b", "preference", "prefer json", 10),
        ]
        result = scorer.score_segments(segs)
        assert result[0].segment_id == "a"
        assert result[1].segment_id == "b"

    def test_most_recent_gets_higher_recency(self) -> None:
        scorer = ImportanceScorer()
        segs = [
            self._make_segment("s1", "chat", "msg", 10),
            self._make_segment("s2", "chat", "msg", 10),
            self._make_segment("s3", "chat", "msg", 10),
        ]
        result = scorer.score_segments(segs)
        # s3 (last, recency=1.0) should have highest score
        assert result[2].importance_score >= result[0].importance_score

    def test_invalid_type_falls_back_to_unknown(self) -> None:
        scorer = ImportanceScorer()
        segs = [self._make_segment("s1", "nonexistent_type", "content", 10)]
        result = scorer.score_segments(segs)
        assert result[0].segment_type == SegmentType.UNKNOWN


# ---------------------------------------------------------------------------
# ImportanceScorer.rank_by_importance
# ---------------------------------------------------------------------------


class TestRankByImportance:
    def test_ordered_highest_first(self) -> None:
        scorer = ImportanceScorer()
        segs = [
            {"segment_id": "s1", "segment_type": "chat", "content": "msg", "token_count": 5},
            {"segment_id": "s2", "segment_type": "preference", "content": "prefer JSON", "token_count": 10},
            {"segment_id": "s3", "segment_type": "task_state", "content": "task: report", "token_count": 15},
        ]
        ranked = scorer.rank_by_importance(segs)
        scores = [s.importance_score for s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_returns_empty(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.rank_by_importance([]) == []


# ---------------------------------------------------------------------------
# ImportanceScorer.type_prior
# ---------------------------------------------------------------------------


class TestTypePrior:
    def test_returns_correct_prior(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.type_prior(SegmentType.PREFERENCE) == pytest.approx(0.90)
        assert scorer.type_prior(SegmentType.CHAT) == pytest.approx(0.30)

    def test_type_modifiers_applied(self) -> None:
        config = ImportanceScorerConfig(type_modifiers={"preference": -0.20})
        scorer = ImportanceScorer(config)
        scored = scorer.score_segment(
            "s1", SegmentType.PREFERENCE, "some preference", 10
        )
        # 0.90 - 0.20 = 0.70 (before other boosts)
        assert scored.importance_score < 0.90


# ---------------------------------------------------------------------------
# Keyword boost disabled
# ---------------------------------------------------------------------------


class TestKeywordBoostDisabled:
    def test_no_boost_when_disabled(self) -> None:
        config_on = ImportanceScorerConfig(keyword_boost_enabled=True)
        config_off = ImportanceScorerConfig(keyword_boost_enabled=False)
        scorer_on = ImportanceScorer(config_on)
        scorer_off = ImportanceScorer(config_off)
        content = "Critical: this is important"
        score_on = scorer_on.score_segment("s", SegmentType.CHAT, content, 10).importance_score
        score_off = scorer_off.score_segment("s", SegmentType.CHAT, content, 10).importance_score
        assert score_on > score_off
