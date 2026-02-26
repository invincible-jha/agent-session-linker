"""TF-IDF relevance scoring without external ML dependencies.

Provides segment-level and corpus-level relevance scoring for use in
context selection and retrieval pipelines.

Classes
-------
- RelevanceScorer  — TF-IDF implementation for scoring and ranking segments
"""
from __future__ import annotations

import math
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Text helpers (module-private)
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "with", "as", "by", "from",
        "this", "that", "was", "are", "be", "been", "have", "has",
        "do", "did", "will", "would", "could", "should", "may", "can",
        "i", "you", "we", "they", "he", "she", "its", "their", "our",
        "so", "if", "then", "just", "also", "about", "there", "here",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric, remove stop words and short tokens."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    """Return length-normalised TF for a token list."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute smoothed IDF over a corpus of tokenised documents.

    Uses the formula: ``log((1 + N) / (1 + df)) + 1`` so that terms
    appearing in every document still receive a small positive weight.

    Parameters
    ----------
    documents:
        Each element is the token list for one document.

    Returns
    -------
    dict[str, float]
        IDF value per unique term.
    """
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


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class RelevanceScorer:
    """Score and rank context segments by TF-IDF similarity to a query.

    The scorer maintains an internal IDF cache that is rebuilt automatically
    whenever the corpus of documents changes.  Scores are always in [0.0, ∞)
    — they are not normalised to [0, 1] by default, though scores are
    comparable within a single call to ``rank``.

    Parameters
    ----------
    smooth_idf:
        When True (default), uses the smoothed IDF formula
        ``log((1+N)/(1+df)) + 1``.  When False, uses ``log(N/df)``.
    sublinear_tf:
        When True, applies ``1 + log(tf)`` instead of raw TF, damping the
        influence of very high-frequency terms.  Default: False.
    """

    def __init__(
        self,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
    ) -> None:
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    # ------------------------------------------------------------------
    # Primary public methods
    # ------------------------------------------------------------------

    def score(self, segment_text: str, query: str) -> float:
        """Compute TF-IDF similarity between ``segment_text`` and ``query``.

        The document corpus for IDF purposes consists of only the segment
        and the query — this is a lightweight two-document IDF.  For
        corpus-aware scoring across many segments prefer ``rank``.

        Parameters
        ----------
        segment_text:
            The candidate segment content to score.
        query:
            The reference query string.

        Returns
        -------
        float
            Similarity score >= 0.0.  Higher means more relevant.
        """
        query_tokens = _tokenize(query)
        doc_tokens = _tokenize(segment_text)

        if not query_tokens or not doc_tokens:
            return 0.0

        idf = self._build_idf([doc_tokens, query_tokens])
        return self._tfidf_similarity(query_tokens, doc_tokens, idf)

    def rank(
        self,
        segments: list[object],
        query: str,
    ) -> list[tuple[float, object]]:
        """Rank segments by TF-IDF similarity to ``query``.

        IDF is computed across all provided segments at once for accuracy.

        Parameters
        ----------
        segments:
            Sequence of segments to rank.  Each element may be a
            ``ContextSegment``-like object with a ``.content`` attribute or
            a plain string.
        query:
            The reference query string.

        Returns
        -------
        list[tuple[float, object]]
            List of ``(score, segment)`` pairs sorted by score descending.
            The original segment objects (not copies) are returned.
        """
        if not segments:
            return []

        query_tokens = _tokenize(query)

        # Extract text from each segment.
        texts: list[str] = []
        for segment in segments:
            if isinstance(segment, str):
                texts.append(segment)
            elif hasattr(segment, "content"):
                texts.append(str(segment.content))
            else:
                texts.append(str(segment))

        all_token_lists: list[list[str]] = [_tokenize(text) for text in texts]

        # Build IDF over the full corpus plus the query.
        idf = self._build_idf(all_token_lists + [query_tokens])

        scored: list[tuple[float, object]] = []
        for segment, doc_tokens in zip(segments, all_token_lists):
            similarity = self._tfidf_similarity(query_tokens, doc_tokens, idf)
            scored.append((similarity, segment))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    def score_many(
        self,
        segment_texts: list[str],
        query: str,
    ) -> list[float]:
        """Return per-segment TF-IDF scores for a list of plain text strings.

        Uses a shared IDF built from the full corpus, giving more accurate
        scores than calling ``score`` individually for each segment.

        Parameters
        ----------
        segment_texts:
            Raw text strings to score.
        query:
            The reference query string.

        Returns
        -------
        list[float]
            Scores in the same order as ``segment_texts``.
        """
        if not segment_texts:
            return []

        query_tokens = _tokenize(query)
        all_token_lists = [_tokenize(text) for text in segment_texts]
        idf = self._build_idf(all_token_lists + [query_tokens])

        return [
            self._tfidf_similarity(query_tokens, doc_tokens, idf)
            for doc_tokens in all_token_lists
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_idf(self, documents: list[list[str]]) -> dict[str, float]:
        """Build an IDF mapping from a list of tokenised documents.

        Parameters
        ----------
        documents:
            Tokenised corpus.

        Returns
        -------
        dict[str, float]
            IDF score per term.
        """
        num_docs = len(documents)
        if num_docs == 0:
            return {}

        document_freq: Counter[str] = Counter()
        for doc_tokens in documents:
            document_freq.update(set(doc_tokens))

        result: dict[str, float] = {}
        for term, df in document_freq.items():
            if self.smooth_idf:
                result[term] = math.log((1 + num_docs) / (1 + df)) + 1
            else:
                result[term] = math.log(num_docs / max(1, df))
        return result

    def _apply_tf(self, tokens: list[str]) -> dict[str, float]:
        """Compute TF, optionally with sublinear scaling.

        Parameters
        ----------
        tokens:
            Token list for a single document.

        Returns
        -------
        dict[str, float]
            TF per term.
        """
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        if self.sublinear_tf:
            return {term: 1.0 + math.log(count) for term, count in counts.items()}
        return {term: count / total for term, count in counts.items()}

    def _tfidf_similarity(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
        idf: dict[str, float],
    ) -> float:
        """Compute TF-IDF dot-product similarity.

        Parameters
        ----------
        query_tokens:
            Tokenised query.
        doc_tokens:
            Tokenised document.
        idf:
            Pre-computed IDF map.

        Returns
        -------
        float
            Similarity score >= 0.0.
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        doc_tf = self._apply_tf(doc_tokens)

        return sum(
            doc_tf.get(token, 0.0) * idf.get(token, 0.0)
            for token in set(query_tokens)
        )
