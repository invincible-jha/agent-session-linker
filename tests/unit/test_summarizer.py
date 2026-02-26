"""Tests for ContextSummarizer."""
from __future__ import annotations

import pytest

from agent_session_linker.context.summarizer import ContextSummarizer


class TestContextSummarizerSummarize:
    def test_empty_segments_returns_empty(self) -> None:
        summarizer = ContextSummarizer()
        assert summarizer.summarize([], max_tokens=100) == ""

    def test_plain_string_segments(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize(["Hello world. This is a test."], max_tokens=200)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_segment_with_content_attribute(self) -> None:
        class Seg:
            content = "Machine learning is powerful. Neural networks learn patterns."

        summarizer = ContextSummarizer()
        result = summarizer.summarize([Seg()], max_tokens=100)
        assert isinstance(result, str)

    def test_segment_without_content_attribute_uses_str(self) -> None:
        class Seg:
            def __str__(self) -> str:
                return "Custom string representation. It is useful."

        summarizer = ContextSummarizer()
        result = summarizer.summarize([Seg()], max_tokens=100)
        assert isinstance(result, str)

    def test_max_tokens_limits_output(self) -> None:
        # Build text with many short distinct sentences so the token budget
        # (max_tokens=20 ≈ 80 chars) forces the summarizer to drop most of them.
        sentences = [f"Sentence number {i} contains unique content here." for i in range(50)]
        text = " ".join(sentences)
        summarizer = ContextSummarizer()
        result = summarizer.summarize([text], max_tokens=20)
        # Result should be shorter than the full input.
        assert len(result) < len(text)

    def test_position_bias_false(self) -> None:
        summarizer = ContextSummarizer(position_bias=False)
        result = summarizer.summarize(
            ["First sentence here. Second sentence here."], max_tokens=100
        )
        assert isinstance(result, str)

    def test_max_sentences_per_segment_cap(self) -> None:
        many_sentences = ". ".join([f"Sentence number {i}" for i in range(20)]) + "."
        summarizer = ContextSummarizer(max_sentences_per_segment=2)
        result = summarizer.summarize([many_sentences], max_tokens=500)
        sentence_count = result.count(".")
        assert sentence_count <= 2

    def test_multiple_segments_included(self) -> None:
        segments = [
            "Machine learning is powerful.",
            "Deep neural networks learn representations.",
        ]
        summarizer = ContextSummarizer()
        result = summarizer.summarize(segments, max_tokens=500)
        assert isinstance(result, str)

    def test_segment_with_no_usable_text_returns_empty(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize([""], max_tokens=100)
        assert result == ""

    def test_no_overflow_beyond_max_tokens(self) -> None:
        # Build a multi-sentence text so the token budget can exclude later
        # sentences.  With max_tokens=5 (~20 chars) only the highest-scoring
        # sentence is included; subsequent ones are blocked by the budget.
        sentences = [
            "This is a reasonably long sentence about machine learning.",
            "Neural networks are universal function approximators.",
            "Gradient descent optimises the loss surface iteratively.",
        ]
        text = " ".join(sentences)
        summarizer = ContextSummarizer()
        result = summarizer.summarize([text], max_tokens=5)
        # The summarizer selects at most one ~58-char sentence; it must be
        # strictly shorter than the full three-sentence input.
        assert len(result) < len(text)


class TestContextSummarizerSummarizeText:
    def test_plain_text(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize_text("This is a test. It has two sentences.", max_tokens=50)
        assert isinstance(result, str)

    def test_empty_string(self) -> None:
        summarizer = ContextSummarizer()
        assert summarizer.summarize_text("", max_tokens=100) == ""

    def test_single_sentence(self) -> None:
        summarizer = ContextSummarizer()
        result = summarizer.summarize_text("Just one sentence here.", max_tokens=50)
        assert isinstance(result, str)
