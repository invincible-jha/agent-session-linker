"""Additional branch-coverage tests for agent_session_linker.context.relevance.

Targets lines not yet covered by test_relevance.py:
  - _term_frequency with empty list (line 41-45)
  - _compute_idf with empty corpus (lines 64-70)
  - RelevanceScorer._build_idf with smooth_idf=False (line 244)
  - RelevanceScorer._apply_tf with sublinear_tf=True (line 272)
  - RelevanceScorer._tfidf_similarity edge cases (line 302)
"""
from __future__ import annotations

import math

import pytest

from agent_session_linker.context.relevance import (
    RelevanceScorer,
    _compute_idf,
    _term_frequency,
    _tokenize,
)


# ---------------------------------------------------------------------------
# _term_frequency
# ---------------------------------------------------------------------------


class TestTermFrequencyExtra:
    def test_empty_tokens_returns_empty_dict(self) -> None:
        assert _term_frequency([]) == {}

    def test_single_token_has_unit_frequency(self) -> None:
        tf = _term_frequency(["hello"])
        assert tf["hello"] == pytest.approx(1.0)

    def test_two_identical_tokens_frequency_is_one(self) -> None:
        tf = _term_frequency(["word", "word"])
        assert tf["word"] == pytest.approx(1.0)

    def test_mixed_tokens_frequencies_sum_to_one(self) -> None:
        tf = _term_frequency(["a", "b", "c"])
        assert sum(tf.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_idf
# ---------------------------------------------------------------------------


class TestComputeIdfExtra:
    def test_empty_corpus_returns_empty(self) -> None:
        assert _compute_idf([]) == {}

    def test_single_doc_single_term(self) -> None:
        idf = _compute_idf([["dog"]])
        assert "dog" in idf
        assert idf["dog"] > 0.0

    def test_term_in_all_docs_lower_idf(self) -> None:
        docs = [["common", "unique1"], ["common", "unique2"]]
        idf = _compute_idf(docs)
        # "common" is in 2 docs, unique terms in 1.
        assert idf["common"] < idf["unique1"]


# ---------------------------------------------------------------------------
# RelevanceScorer._build_idf (smooth_idf=False branch)
# ---------------------------------------------------------------------------


class TestRelevanceScorerBuildIdf:
    def test_smooth_idf_false_gives_non_negative_values(self) -> None:
        scorer = RelevanceScorer(smooth_idf=False)
        docs = [["cat", "dog"], ["dog", "fish"]]
        idf = scorer._build_idf(docs)
        # All IDF values must be >= 0 even with raw formula.
        assert all(v >= 0.0 for v in idf.values())

    def test_smooth_idf_false_empty_docs_returns_empty(self) -> None:
        scorer = RelevanceScorer(smooth_idf=False)
        assert scorer._build_idf([]) == {}

    def test_smooth_idf_true_and_false_differ(self) -> None:
        docs = [["cat"], ["dog"]]
        scorer_smooth = RelevanceScorer(smooth_idf=True)
        scorer_raw = RelevanceScorer(smooth_idf=False)
        idf_smooth = scorer_smooth._build_idf(docs)
        idf_raw = scorer_raw._build_idf(docs)
        # The values should differ for at least one term.
        assert any(
            abs(idf_smooth.get(t, 0) - idf_raw.get(t, 0)) > 1e-9
            for t in idf_smooth
        )


# ---------------------------------------------------------------------------
# RelevanceScorer._apply_tf (sublinear_tf=True branch)
# ---------------------------------------------------------------------------


class TestRelevanceScorerApplyTf:
    def test_sublinear_tf_returns_non_empty_for_tokens(self) -> None:
        scorer = RelevanceScorer(sublinear_tf=True)
        tf = scorer._apply_tf(["word", "word", "word"])
        assert "word" in tf

    def test_sublinear_tf_value_uses_log(self) -> None:
        scorer = RelevanceScorer(sublinear_tf=True)
        tf = scorer._apply_tf(["word", "word", "word"])
        expected = 1.0 + math.log(3)
        assert tf["word"] == pytest.approx(expected)

    def test_sublinear_tf_empty_tokens_returns_empty(self) -> None:
        scorer = RelevanceScorer(sublinear_tf=True)
        assert scorer._apply_tf([]) == {}

    def test_linear_tf_returns_count_normalised(self) -> None:
        scorer = RelevanceScorer(sublinear_tf=False)
        tf = scorer._apply_tf(["a", "a", "b"])
        assert tf["a"] == pytest.approx(2 / 3)
        assert tf["b"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# RelevanceScorer._tfidf_similarity edge cases
# ---------------------------------------------------------------------------


class TestRelevanceScorerTfidfSimilarity:
    def test_empty_query_returns_zero(self) -> None:
        scorer = RelevanceScorer()
        idf = {"cat": 1.0}
        assert scorer._tfidf_similarity([], ["cat"], idf) == pytest.approx(0.0)

    def test_empty_doc_returns_zero(self) -> None:
        scorer = RelevanceScorer()
        idf = {"cat": 1.0}
        assert scorer._tfidf_similarity(["cat"], [], idf) == pytest.approx(0.0)

    def test_matching_terms_positive_score(self) -> None:
        scorer = RelevanceScorer()
        idf = {"machine": 2.0, "learning": 1.5}
        score = scorer._tfidf_similarity(
            ["machine", "learning"], ["machine", "learning", "model"], idf
        )
        assert score > 0.0

    def test_non_overlapping_terms_zero_score(self) -> None:
        scorer = RelevanceScorer()
        idf = {"cat": 1.0, "dog": 1.0}
        score = scorer._tfidf_similarity(["cat"], ["banana", "apple"], idf)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# RelevanceScorer.score with smooth_idf=False
# ---------------------------------------------------------------------------


class TestRelevanceScorerScoreVariants:
    def test_smooth_false_score_non_negative(self) -> None:
        scorer = RelevanceScorer(smooth_idf=False)
        score = scorer.score("neural network training", "neural network")
        assert score >= 0.0

    def test_sublinear_true_score_non_negative(self) -> None:
        scorer = RelevanceScorer(sublinear_tf=True)
        score = scorer.score("neural network neural network", "neural")
        assert score >= 0.0

    def test_both_options_combined(self) -> None:
        scorer = RelevanceScorer(smooth_idf=False, sublinear_tf=True)
        score = scorer.score("python programming language", "python")
        assert score >= 0.0


# ---------------------------------------------------------------------------
# RelevanceScorer.score_many additional paths
# ---------------------------------------------------------------------------


class TestScoreManyExtra:
    def test_score_many_single_item(self) -> None:
        scorer = RelevanceScorer()
        result = scorer.score_many(["hello world"], "hello")
        assert len(result) == 1

    def test_score_many_with_sublinear_tf(self) -> None:
        scorer = RelevanceScorer(sublinear_tf=True)
        result = scorer.score_many(["alpha beta", "gamma"], "alpha")
        assert len(result) == 2
        assert all(s >= 0.0 for s in result)
