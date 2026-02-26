"""Extractive context summarization.

Compresses a list of context segments into a shorter summary string using
sentence-level TF-IDF scoring weighted by position (earlier sentences in a
segment score higher).

Classes
-------
- ContextSummarizer  — extractive summarizer with token-budget control
"""
from __future__ import annotations

import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "with", "as", "by", "from",
        "this", "that", "was", "are", "be", "been", "have", "has",
        "do", "did", "will", "would", "could", "should", "may", "can",
        "i", "you", "we", "they", "he", "she", "its", "their", "our",
        "so", "if", "then", "just", "also", "about", "there", "here",
        "up", "out", "when", "what", "which", "who", "how", "all",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words and short tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on common sentence-boundary punctuation."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if s.strip()]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: one token per ~4 characters."""
    return max(1, len(text) // 4)


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    """Compute normalised TF for a token list."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute IDF across a corpus of tokenised documents."""
    num_docs = len(documents)
    if num_docs == 0:
        return {}
    document_freq: Counter[str] = Counter()
    for doc_tokens in documents:
        document_freq.update(set(doc_tokens))
    return {
        term: math.log((1 + num_docs) / (1 + df)) + 1
        for term, df in document_freq.items()
    }


def _score_sentence(
    sentence_tokens: list[str],
    idf: dict[str, float],
    position_index: int,
    total_sentences: int,
) -> float:
    """Score one sentence by TF-IDF sum weighted by position.

    Sentences appearing earlier receive a higher positional boost because
    the first sentences of a segment tend to carry higher information density
    (topic introduction, conclusions, decisions).

    Parameters
    ----------
    sentence_tokens:
        Tokenised words in the sentence.
    idf:
        IDF scores for all terms in the corpus.
    position_index:
        Zero-based position of this sentence within its segment.
    total_sentences:
        Total sentence count in the segment (used for normalisation).

    Returns
    -------
    float
        Composite score for this sentence.
    """
    if not sentence_tokens:
        return 0.0

    tf = _term_frequency(sentence_tokens)
    tfidf_sum = sum(tf.get(term, 0.0) * idf.get(term, 0.0) for term in tf)

    # Positional weight: first sentence = 1.0, last = ~0.5 (linear decay).
    if total_sentences <= 1:
        position_weight = 1.0
    else:
        position_weight = 1.0 - 0.5 * (position_index / (total_sentences - 1))

    return tfidf_sum * position_weight


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class ContextSummarizer:
    """Compress a list of context segments into an extractive summary.

    Uses sentence-level TF-IDF scoring combined with a positional bias
    (earlier sentences score higher) to select the most informative
    sentences from the input segments.  Selected sentences are ordered
    by their original document position, not by score, so the output
    reads naturally.

    Parameters
    ----------
    max_sentences_per_segment:
        Hard cap on how many sentences may be drawn from a single segment.
        Default: 5.
    position_bias:
        When True (default), earlier sentences receive a higher weight.
    """

    def __init__(
        self,
        max_sentences_per_segment: int = 5,
        position_bias: bool = True,
    ) -> None:
        self.max_sentences_per_segment = max_sentences_per_segment
        self.position_bias = position_bias

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize(self, segments: list[object], max_tokens: int) -> str:
        """Produce an extractive summary within ``max_tokens``.

        Each element of ``segments`` may be either a
        ``ContextSegment``-like object (with a ``.content`` attribute) or
        a plain string.  Both forms are accepted so the summarizer can be
        used independently of the session model.

        Parameters
        ----------
        segments:
            Ordered sequence of context segments or plain text strings.
        max_tokens:
            Target token budget for the returned summary.  The summarizer
            selects sentences greedily until this budget is exhausted.

        Returns
        -------
        str
            The extractive summary.  An empty string is returned when
            ``segments`` is empty or contains no usable text.
        """
        if not segments:
            return ""

        # Extract text content from segments (support both str and objects).
        texts: list[str] = []
        for segment in segments:
            if isinstance(segment, str):
                texts.append(segment)
            elif hasattr(segment, "content"):
                texts.append(str(segment.content))
            else:
                texts.append(str(segment))

        # Build sentence list: (segment_index, sentence_index, raw_sentence).
        all_sentences: list[tuple[int, int, str]] = []
        segment_sentence_lists: list[list[str]] = []

        for seg_idx, text in enumerate(texts):
            sentence_list = _split_sentences(text)
            segment_sentence_lists.append(sentence_list)
            for sent_idx, sentence in enumerate(sentence_list):
                all_sentences.append((seg_idx, sent_idx, sentence))

        if not all_sentences:
            return ""

        # Tokenize all sentences for IDF computation.
        all_token_lists: list[list[str]] = [
            _tokenize(sentence) for _, _, sentence in all_sentences
        ]
        idf = _compute_idf(all_token_lists)

        # Score every sentence.
        scored: list[tuple[float, int, int, str]] = []
        for (seg_idx, sent_idx, sentence), tokens in zip(all_sentences, all_token_lists):
            total_in_seg = len(segment_sentence_lists[seg_idx])
            if self.position_bias:
                score = _score_sentence(tokens, idf, sent_idx, total_in_seg)
            else:
                tf = _term_frequency(tokens)
                score = sum(tf.get(t, 0.0) * idf.get(t, 0.0) for t in tf)
            scored.append((score, seg_idx, sent_idx, sentence))

        # Sort descending by score.
        scored.sort(key=lambda item: item[0], reverse=True)

        # Greedily select sentences within token budget, respecting per-segment cap.
        selected: list[tuple[int, int, str]] = []
        tokens_used = 0
        per_segment_counts: dict[int, int] = {}

        for score, seg_idx, sent_idx, sentence in scored:
            if tokens_used >= max_tokens:
                break
            segment_count = per_segment_counts.get(seg_idx, 0)
            if segment_count >= self.max_sentences_per_segment:
                continue
            sentence_tokens = _estimate_tokens(sentence)
            if tokens_used + sentence_tokens > max_tokens and selected:
                # Skip sentences that would overflow the budget (unless nothing selected yet).
                continue
            selected.append((seg_idx, sent_idx, sentence))
            tokens_used += sentence_tokens
            per_segment_counts[seg_idx] = segment_count + 1

        if not selected:
            return ""

        # Re-order selected sentences by original document order.
        selected.sort(key=lambda item: (item[0], item[1]))

        return " ".join(sentence for _, _, sentence in selected)

    def summarize_text(self, text: str, max_tokens: int) -> str:
        """Convenience wrapper: summarize a single plain-text string.

        Parameters
        ----------
        text:
            Raw text to summarize.
        max_tokens:
            Target token budget.

        Returns
        -------
        str
            Extractive summary.
        """
        return self.summarize([text], max_tokens)
