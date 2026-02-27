"""Selective loader — load session segments within a token budget.

SelectiveLoader takes a list of session segments, scores them by importance,
and returns only those segments that:
1. Meet or exceed the minimum importance threshold.
2. Fit within the configured token budget.

Selection strategy
------------------
1. Score all segments.
2. Sort by importance (descending).
3. Greedily select segments in importance order until the token budget is
   exhausted or all qualifying segments are loaded.
4. Return selected segments sorted back into their original chronological
   order (preserving conversation coherence).

This approach ensures the most important context is always loaded first,
and lower-priority content is included only if budget allows.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent_session_linker.selective.importance_scorer import (
    ImportanceScorer,
    ImportanceScorerConfig,
    ScoredSegment,
    SegmentType,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectiveLoaderConfig:
    """Configuration for SelectiveLoader.

    Attributes
    ----------
    token_budget:
        Maximum total token count of segments to load.
    importance_threshold:
        Minimum importance score (inclusive) for a segment to be considered.
    max_segments:
        Hard limit on the number of segments returned, regardless of budget.
    preserve_order:
        When True (default), returned segments are in their original
        chronological order even though selection is done by importance.
    always_include_types:
        Segment types that are always included if they exist, regardless of
        threshold (but still subject to token budget).
    """

    token_budget: int = 4000
    importance_threshold: float = 0.50
    max_segments: int = 100
    preserve_order: bool = True
    always_include_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Load result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadResult:
    """The result of a selective load operation.

    Attributes
    ----------
    selected_segments:
        The segments chosen for loading, in chronological order.
    total_tokens_loaded:
        Sum of token counts across all selected segments.
    total_tokens_available:
        Sum of token counts across all qualifying segments (before budget).
    segments_considered:
        Total number of segments evaluated (after type/threshold filter).
    segments_skipped:
        Number of segments that met the threshold but were dropped for budget.
    budget_used_pct:
        Fraction of the token budget consumed.
    """

    selected_segments: list[ScoredSegment]
    total_tokens_loaded: int
    total_tokens_available: int
    segments_considered: int
    segments_skipped: int
    budget_used_pct: float

    def to_dict(self) -> dict[str, object]:
        """Serialise the result summary (without full segment content)."""
        return {
            "selected_count": len(self.selected_segments),
            "total_tokens_loaded": self.total_tokens_loaded,
            "total_tokens_available": self.total_tokens_available,
            "segments_considered": self.segments_considered,
            "segments_skipped": self.segments_skipped,
            "budget_used_pct": self.budget_used_pct,
        }


# ---------------------------------------------------------------------------
# SelectiveLoader
# ---------------------------------------------------------------------------


class SelectiveLoader:
    """Load session segments selectively within a token budget.

    Parameters
    ----------
    config:
        Loader configuration.  Defaults to standard settings.
    scorer_config:
        Optional config for the underlying ImportanceScorer.

    Example
    -------
    >>> loader = SelectiveLoader()
    >>> segments = [
    ...     {"segment_id": "s1", "segment_type": "preference",
    ...      "content": "User prefers JSON output", "token_count": 15},
    ...     {"segment_id": "s2", "segment_type": "chat",
    ...      "content": "Hello, how are you?", "token_count": 10},
    ... ]
    >>> result = loader.load(segments)
    >>> len(result.selected_segments) >= 1
    True
    """

    def __init__(
        self,
        config: SelectiveLoaderConfig | None = None,
        scorer_config: ImportanceScorerConfig | None = None,
    ) -> None:
        self._config = config if config is not None else SelectiveLoaderConfig()
        self._scorer = ImportanceScorer(scorer_config)

    @property
    def config(self) -> SelectiveLoaderConfig:
        """The active loader configuration."""
        return self._config

    def load(self, segments: list[dict[str, object]]) -> LoadResult:
        """Select segments within the configured token budget.

        Parameters
        ----------
        segments:
            Raw segment dicts in chronological order.  Each dict requires:
            ``segment_id``, ``segment_type``, ``content``, ``token_count``.

        Returns
        -------
        LoadResult
            Selection result with metadata.
        """
        if not segments:
            return LoadResult(
                selected_segments=[],
                total_tokens_loaded=0,
                total_tokens_available=0,
                segments_considered=0,
                segments_skipped=0,
                budget_used_pct=0.0,
            )

        # Score all segments, preserving original index for order restoration.
        scored_with_index: list[tuple[int, ScoredSegment]] = []
        for original_index, raw in enumerate(segments):
            scored_list = self._scorer.score_segments([raw])
            scored = scored_list[0]
            scored_with_index.append((original_index, scored))

        always_include_types = {
            SegmentType(t)
            for t in self._config.always_include_types
            if t in SegmentType._value2member_map_
        }

        # Separate always-include from threshold-filtered
        always_include: list[tuple[int, ScoredSegment]] = []
        threshold_candidates: list[tuple[int, ScoredSegment]] = []

        for idx, scored in scored_with_index:
            if scored.segment_type in always_include_types:
                always_include.append((idx, scored))
            elif scored.importance_score >= self._config.importance_threshold:
                threshold_candidates.append((idx, scored))

        # Sort threshold candidates by importance descending
        threshold_candidates.sort(key=lambda t: t[1].importance_score, reverse=True)

        # Greedily fill budget
        selected_pairs: list[tuple[int, ScoredSegment]] = []
        used_tokens = 0

        # Always-include segments go first (subject to budget)
        for idx, scored in always_include:
            if (
                used_tokens + scored.token_count <= self._config.token_budget
                and len(selected_pairs) < self._config.max_segments
            ):
                selected_pairs.append((idx, scored))
                used_tokens += scored.token_count

        segments_skipped = 0
        for idx, scored in threshold_candidates:
            if len(selected_pairs) >= self._config.max_segments:
                segments_skipped += 1
                continue
            if used_tokens + scored.token_count > self._config.token_budget:
                segments_skipped += 1
                continue
            selected_pairs.append((idx, scored))
            used_tokens += scored.token_count

        # Restore chronological order
        if self._config.preserve_order:
            selected_pairs.sort(key=lambda t: t[0])

        selected = [s for _, s in selected_pairs]

        total_available = sum(
            s.token_count
            for _, s in (always_include + threshold_candidates)
        )

        budget_used_pct = round(
            used_tokens / max(self._config.token_budget, 1), 4
        )

        return LoadResult(
            selected_segments=selected,
            total_tokens_loaded=used_tokens,
            total_tokens_available=total_available,
            segments_considered=len(always_include) + len(threshold_candidates),
            segments_skipped=segments_skipped,
            budget_used_pct=budget_used_pct,
        )

    def load_scored(self, scored_segments: list[ScoredSegment]) -> LoadResult:
        """Load from pre-scored segments (skips the scoring step).

        Parameters
        ----------
        scored_segments:
            Already-scored segments in chronological order.

        Returns
        -------
        LoadResult
            Selection result.
        """
        raw = [
            {
                "segment_id": s.segment_id,
                "segment_type": s.segment_type.value,
                "content": s.content_preview,
                "token_count": s.token_count,
                "metadata": s.metadata,
            }
            for s in scored_segments
        ]
        return self.load(raw)
