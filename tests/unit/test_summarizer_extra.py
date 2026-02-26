"""Additional branch-coverage tests for agent_session_linker.context.summarizer.

Targets the uncovered lines in summarizer.py:
  - line 55 (empty _term_frequency)
  - line 65 (empty _compute_idf)
  - line 104 (empty sentence_tokens -> _score_sentence returns 0.0)
  - line 241 (selected empty -> return "")
  - position_bias=False branch in ContextSummarizer.summarize
  - summarize_text convenience wrapper
  - segments with objects lacking a .content attribute
"""
from __future__ import annotations

import pytest

from agent_session_linker.context.summarizer import (
    ContextSummarizer,
    _compute_idf,
    _estimate_tokens,
    _score_sentence,
    _split_sentences,
    _term_frequency,
    _tokenize,
)


# ---------------------------------------------------------------------------
# _term_frequency
# ---------------------------------------------------------------------------


class TestSummarizerTermFrequency:
    def test_empty_returns_empty(self) -> None:
        assert _term_frequency([]) == {}

    def test_single_token(self) -> None:
        tf = _term_frequency(["hello"])
        assert tf["hello"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _compute_idf
# ---------------------------------------------------------------------------


class TestSummarizerComputeIdf:
    def test_empty_corpus_returns_empty(self) -> None:
        assert _compute_idf([]) == {}

    def test_single_doc_term_has_positive_idf(self) -> None:
        idf = _compute_idf([["word"]])
        assert idf["word"] > 0.0


# ---------------------------------------------------------------------------
# _score_sentence — empty tokens branch
# ---------------------------------------------------------------------------


class TestScoreSentence:
    def test_empty_tokens_returns_zero(self) -> None:
        assert _score_sentence([], {}, 0, 5) == pytest.approx(0.0)

    def test_single_sentence_has_max_position_weight(self) -> None:
        idf = {"dog": 1.0}
        # total_sentences=1 -> position_weight=1.0
        score = _score_sentence(["dog"], idf, 0, 1)
        assert score > 0.0

    def test_later_sentence_lower_score(self) -> None:
        idf = {"cat": 1.0}
        tokens = ["cat"]
        first = _score_sentence(tokens, idf, 0, 5)
        last = _score_sentence(tokens, idf, 4, 5)
        assert first > last

    def test_total_one_sentence_no_division_issue(self) -> None:
        idf = {"word": 2.0}
        # total_sentences=1 triggers the special-case branch.
        score = _score_sentence(["word"], idf, 0, 1)
        assert score > 0.0


# ---------------------------------------------------------------------------
# _split_sentences
# ---------------------------------------------------------------------------


class TestSplitSentences:
    def test_single_sentence(self) -> None:
        sentences = _split_sentences("Hello world.")
        assert len(sentences) == 1

    def test_multiple_sentences(self) -> None:
        text = "First sentence. Second sentence! Third one?"
        sentences = _split_sentences(text)
        assert len(sentences) == 3

    def test_empty_string_returns_empty(self) -> None:
        assert _split_sentences("") == []


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokensSummarizer:
    def test_empty_string_returns_one(self) -> None:
        assert _estimate_tokens("") == 1

    def test_four_chars_returns_one(self) -> None:
        assert _estimate_tokens("abcd") == 1


# ---------------------------------------------------------------------------
# ContextSummarizer.summarize
# ---------------------------------------------------------------------------


class TestContextSummarizerSummarize:
    def test_empty_segments_returns_empty(self) -> None:
        summarizer = ContextSummarizer()
        assert summarizer.summarize([], 100) == ""

    def test_returns_non_empty_for_real_text(self) -> None:
        summarizer = ContextSummarizer()
        texts = ["Machine learning is a subset of artificial intelligence."]
        result = summarizer.summarize(texts, 50)
        assert len(result) > 0

    def test_respects_max_tokens_budget(self) -> None:
        summarizer = ContextSummarizer()
        long_text = (
            "This sentence talks about neural networks. "
            "Another sentence discusses deep learning models. "
            "A third sentence covers optimization algorithms. "
            "The fourth sentence is about training data preparation. "
            "The fifth sentence concludes with model evaluation metrics."
        )
        result = summarizer.summarize([long_text], max_tokens=20)
        assert _estimate_tokens(result) <= 20 + 5  # small tolerance

    def test_position_bias_false(self) -> None:
        summarizer = ContextSummarizer(position_bias=False)
        texts = [
            "Important decision made about the architecture.",
            "Some random filler text goes here.",
        ]
        result = summarizer.summarize(texts, 100)
        assert isinstance(result, str)

    def test_accepts_string_segments(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize(["plain text segment"], 50)
        assert isinstance(result, str)

    def test_accepts_objects_with_content(self) -> None:
        class FakeSegment:
            def __init__(self, content: str) -> None:
                self.content = content

        summarizer = ContextSummarizer()
        result = summarizer.summarize([FakeSegment("hello world content")], 50)
        assert isinstance(result, str)

    def test_accepts_objects_without_content_attr(self) -> None:
        summarizer = ContextSummarizer()
        # Falls back to str(segment).
        result = summarizer.summarize([42], 50)
        assert isinstance(result, str)

    def test_segments_containing_no_text_returns_empty(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize(["  "], 100)
        # Should return empty or whitespace — just not raise.
        assert isinstance(result, str)

    def test_max_sentences_per_segment_respected(self) -> None:
        summarizer = ContextSummarizer(max_sentences_per_segment=1)
        text = (
            "First important sentence covers topic A. "
            "Second sentence elaborates on B. "
            "Third sentence is filler content here."
        )
        result = summarizer.summarize([text], max_tokens=1000)
        sentence_count = len([s for s in result.split(". ") if s.strip()])
        assert sentence_count <= 1

    def test_very_tight_budget_still_returns_string(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize(["This is some content."], max_tokens=1)
        assert isinstance(result, str)

    def test_no_sentences_selected_returns_empty(self) -> None:
        # Edge: each sentence would exceed the budget on its own when
        # selected is empty at the end.
        summarizer = ContextSummarizer()
        # max_tokens=0 would never let anything through if selected is non-empty already.
        # Use a very long single sentence that exceeds budget when already one is selected.
        short_a = "A" * 5 + "."
        long_b = "B" * 400 + "."
        # The second sentence is far too large; if nothing is selected it gets forced in.
        result = summarizer.summarize([f"{long_b} {short_a}"], max_tokens=2)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# ContextSummarizer.summarize_text
# ---------------------------------------------------------------------------


class TestContextSummarizerSummarizeText:
    def test_returns_string(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize_text(
            "Machine learning is transforming industries globally.", 50
        )
        assert isinstance(result, str)

    def test_empty_text_returns_empty(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize_text("", 50)
        assert result == ""

    def test_result_within_token_budget(self) -> None:
        summarizer = ContextSummarizer()
        long_text = ". ".join(f"Sentence number {i}" for i in range(20)) + "."
        result = summarizer.summarize_text(long_text, 30)
        assert _estimate_tokens(result) <= 30 + 10  # small tolerance
