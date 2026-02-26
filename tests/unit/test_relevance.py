"""Tests for RelevanceScorer and helper functions."""
from __future__ import annotations

import pytest

from agent_session_linker.context.relevance import RelevanceScorer


class TestRelevanceScorerScore:
    def test_identical_text_and_query_high_score(self) -> None:
        scorer = RelevanceScorer()
        score = scorer.score("machine learning models", "machine learning models")
        assert score > 0.0

    def test_empty_segment_returns_zero(self) -> None:
        scorer = RelevanceScorer()
        assert scorer.score("", "machine learning") == pytest.approx(0.0)

    def test_empty_query_returns_zero(self) -> None:
        scorer = RelevanceScorer()
        assert scorer.score("machine learning models", "") == pytest.approx(0.0)

    def test_no_overlap_returns_zero(self) -> None:
        scorer = RelevanceScorer()
        score = scorer.score("bananas and apples", "quantum physics theories")
        # No token overlap.
        assert score == pytest.approx(0.0)

    def test_partial_overlap_returns_positive(self) -> None:
        scorer = RelevanceScorer()
        score = scorer.score("deep learning neural networks", "deep learning")
        assert score > 0.0

    def test_smooth_idf_false(self) -> None:
        scorer = RelevanceScorer(smooth_idf=False)
        score = scorer.score("neural network training", "neural network")
        assert score >= 0.0

    def test_sublinear_tf_true(self) -> None:
        scorer = RelevanceScorer(sublinear_tf=True)
        score = scorer.score("neural network neural network neural network", "neural")
        assert score >= 0.0


class TestRelevanceScorerRank:
    def test_empty_segments_returns_empty(self) -> None:
        scorer = RelevanceScorer()
        assert scorer.rank([], "query") == []

    def test_returns_all_segments(self) -> None:
        scorer = RelevanceScorer()
        segments = ["alpha beta", "gamma delta", "epsilon zeta"]
        result = scorer.rank(segments, "alpha")
        assert len(result) == 3

    def test_most_relevant_first(self) -> None:
        scorer = RelevanceScorer()
        segments = [
            "deep learning neural networks",
            "unrelated content about bananas",
        ]
        result = scorer.rank(segments, "deep learning")
        assert "deep learning" in result[0][1]

    def test_segment_objects_with_content_attribute(self) -> None:
        scorer = RelevanceScorer()

        class Seg:
            def __init__(self, content: str) -> None:
                self.content = content

        segments = [Seg("neural network"), Seg("random forest")]
        result = scorer.rank(segments, "neural")
        assert len(result) == 2

    def test_segment_objects_without_content_attribute(self) -> None:
        scorer = RelevanceScorer()
        result = scorer.rank([42, 99], "number")
        assert len(result) == 2

    def test_sorted_descending(self) -> None:
        scorer = RelevanceScorer()
        segments = ["python programming language", "cooking recipes desserts"]
        result = scorer.rank(segments, "python")
        scores = [score for score, _ in result]
        assert scores == sorted(scores, reverse=True)


class TestRelevanceScorerScoreMany:
    def test_empty_list_returns_empty(self) -> None:
        scorer = RelevanceScorer()
        assert scorer.score_many([], "query") == []

    def test_returns_correct_length(self) -> None:
        scorer = RelevanceScorer()
        texts = ["text one", "text two", "text three"]
        result = scorer.score_many(texts, "text")
        assert len(result) == 3

    def test_scores_are_non_negative(self) -> None:
        scorer = RelevanceScorer()
        texts = ["alpha", "beta gamma", "delta epsilon zeta"]
        scores = scorer.score_many(texts, "alpha")
        assert all(s >= 0.0 for s in scores)

    def test_shared_idf_more_accurate(self) -> None:
        scorer = RelevanceScorer()
        texts = ["the quick brown fox", "the lazy dog"]
        scores = scorer.score_many(texts, "quick")
        # "the" is in both docs so gets lower IDF; unique terms score higher.
        assert scores[0] > scores[1]
