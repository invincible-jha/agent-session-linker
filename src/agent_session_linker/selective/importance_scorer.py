"""Importance scorer — assigns importance scores to session segments.

Each session segment is scored on a scale of 0.0 to 1.0 based on its type
and optional metadata signals.  The type-based prior scores are:

    preference    : 0.90  (user preferences should always be restored)
    task_state    : 0.85  (active task state is critical for continuity)
    reasoning     : 0.60  (intermediate reasoning is moderately important)
    metadata      : 0.50  (session metadata is marginally useful)
    chat          : 0.30  (raw chat history has low restoration priority)

These priors can be adjusted via a modifier dictionary, and individual
segment-level boost factors (e.g. recency, keyword matches) further tune
the final score.

The scorer deliberately avoids any ML-based embeddings or similarity
computations to keep the implementation commodity and zero-dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


# ---------------------------------------------------------------------------
# Segment type enumeration
# ---------------------------------------------------------------------------


class SegmentType(str, Enum):
    """Categorical segment types with associated importance priors."""

    PREFERENCE = "preference"
    TASK_STATE = "task_state"
    REASONING = "reasoning"
    METADATA = "metadata"
    CHAT = "chat"
    UNKNOWN = "unknown"


# Base importance prior for each segment type.
_TYPE_PRIORS: dict[SegmentType, float] = {
    SegmentType.PREFERENCE: 0.90,
    SegmentType.TASK_STATE: 0.85,
    SegmentType.REASONING: 0.60,
    SegmentType.METADATA: 0.50,
    SegmentType.CHAT: 0.30,
    SegmentType.UNKNOWN: 0.40,
}

# Keywords that boost segment importance regardless of type.
_BOOST_KEYWORDS: list[str] = [
    "critical", "important", "required", "must", "always", "never",
    "prefer", "preference", "remember", "note:", "todo:", "action:",
    "deadline", "urgent", "blocked",
]


def _clamp(value: float) -> float:
    """Clamp value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# Scored segment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredSegment:
    """A session segment paired with its importance score.

    Attributes
    ----------
    segment_id:
        Unique identifier for the session segment.
    segment_type:
        The classified type of this segment.
    importance_score:
        Computed importance score in [0.0, 1.0].
    token_count:
        Estimated token count for this segment.
    content_preview:
        First 120 characters of the segment content (for inspection).
    metadata:
        Arbitrary extra data from the original segment.
    """

    segment_id: str
    segment_type: SegmentType
    importance_score: float
    token_count: int
    content_preview: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def is_high_importance(self, threshold: float = 0.60) -> bool:
        """Return True when the importance score meets the given threshold."""
        return self.importance_score >= threshold

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict."""
        return {
            "segment_id": self.segment_id,
            "segment_type": self.segment_type.value,
            "importance_score": self.importance_score,
            "token_count": self.token_count,
            "content_preview": self.content_preview,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# ImportanceScorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportanceScorerConfig:
    """Configuration for ImportanceScorer.

    Attributes
    ----------
    type_modifiers:
        Additive modifiers applied on top of each type's prior score.
        Keys are SegmentType values; values are in [-1.0, 1.0].
    keyword_boost:
        Score boost applied when a boost keyword is found in the content.
    recency_boost:
        Maximum additional score granted to the most recent segments.
        Applied linearly: the most recent segment gets full recency_boost,
        the oldest gets 0.0.
    keyword_boost_enabled:
        Whether keyword detection is active.
    recency_boost_enabled:
        Whether recency boosting is active.
    """

    type_modifiers: dict[str, float] = field(default_factory=dict)
    keyword_boost: float = 0.10
    recency_boost: float = 0.05
    keyword_boost_enabled: bool = True
    recency_boost_enabled: bool = True


class ImportanceScorer:
    """Scores session segments by importance for selective restoration.

    Parameters
    ----------
    config:
        Scorer configuration.  Defaults to standard settings.

    Example
    -------
    >>> scorer = ImportanceScorer()
    >>> segments = [{"segment_id": "s1", "segment_type": "preference",
    ...              "content": "User prefers metric units", "token_count": 10}]
    >>> scored = scorer.score_segments(segments)
    >>> scored[0].importance_score >= 0.85
    True
    """

    def __init__(self, config: ImportanceScorerConfig | None = None) -> None:
        self._config = config if config is not None else ImportanceScorerConfig()

    def score_segment(
        self,
        segment_id: str,
        segment_type: SegmentType,
        content: str,
        token_count: int,
        *,
        recency_rank: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> ScoredSegment:
        """Compute the importance score for a single segment.

        Parameters
        ----------
        segment_id:
            Unique identifier for the segment.
        segment_type:
            The classified type of this segment.
        content:
            Full text content (used for keyword detection).
        token_count:
            Estimated token count.
        recency_rank:
            A value in [0.0, 1.0] where 1.0 means the most recent segment.
            Used to apply the recency boost.
        metadata:
            Optional extra data to carry through.

        Returns
        -------
        ScoredSegment
            Scored segment.
        """
        # Base score from type prior
        prior = _TYPE_PRIORS.get(segment_type, 0.40)

        # Apply type modifier
        modifier_key = segment_type.value
        modifier = self._config.type_modifiers.get(modifier_key, 0.0)
        score = prior + modifier

        # Keyword boost
        if self._config.keyword_boost_enabled:
            lower_content = content.lower()
            if any(kw in lower_content for kw in _BOOST_KEYWORDS):
                score += self._config.keyword_boost

        # Recency boost
        if self._config.recency_boost_enabled:
            score += recency_rank * self._config.recency_boost

        return ScoredSegment(
            segment_id=segment_id,
            segment_type=segment_type,
            importance_score=round(_clamp(score), 4),
            token_count=token_count,
            content_preview=content[:120],
            metadata=dict(metadata or {}),
        )

    def score_segments(
        self,
        segments: list[dict[str, object]],
    ) -> list[ScoredSegment]:
        """Score a list of raw segment dicts.

        Each dict must have the following keys:
        - ``segment_id`` (str)
        - ``segment_type`` (str — a SegmentType value)
        - ``content`` (str)
        - ``token_count`` (int)

        Optional keys:
        - ``metadata`` (dict)

        Segments are assumed to be in chronological order (oldest first).
        The most recent segment receives recency_rank=1.0.

        Parameters
        ----------
        segments:
            List of raw segment dicts in chronological order.

        Returns
        -------
        list[ScoredSegment]
            Scored segments in the same order as input.
        """
        if not segments:
            return []

        total = len(segments)
        scored: list[ScoredSegment] = []

        for index, seg in enumerate(segments):
            segment_id = str(seg.get("segment_id", f"seg-{index}"))
            raw_type = str(seg.get("segment_type", SegmentType.UNKNOWN.value))
            try:
                segment_type = SegmentType(raw_type)
            except ValueError:
                segment_type = SegmentType.UNKNOWN

            content = str(seg.get("content", ""))
            token_count = int(seg.get("token_count", 0))
            metadata = dict(seg.get("metadata", {}))

            # Recency rank: 0.0 for oldest, 1.0 for newest
            recency_rank = index / max(total - 1, 1)

            scored.append(
                self.score_segment(
                    segment_id=segment_id,
                    segment_type=segment_type,
                    content=content,
                    token_count=token_count,
                    recency_rank=recency_rank,
                    metadata=metadata,
                )
            )

        return scored

    def rank_by_importance(
        self,
        segments: list[dict[str, object]],
    ) -> list[ScoredSegment]:
        """Score segments and return them sorted by descending importance.

        Parameters
        ----------
        segments:
            List of raw segment dicts.

        Returns
        -------
        list[ScoredSegment]
            Scored segments sorted highest importance first.
        """
        scored = self.score_segments(segments)
        return sorted(scored, key=lambda s: s.importance_score, reverse=True)

    def type_prior(self, segment_type: SegmentType) -> float:
        """Return the base importance prior for a given segment type.

        Parameters
        ----------
        segment_type:
            The segment type to look up.

        Returns
        -------
        float
            Base importance prior in [0.0, 1.0].
        """
        return _TYPE_PRIORS.get(segment_type, 0.40)
