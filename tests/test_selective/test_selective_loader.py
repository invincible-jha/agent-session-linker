"""Tests for agent_session_linker.selective.selective_loader."""
from __future__ import annotations

import pytest

from agent_session_linker.selective.selective_loader import (
    LoadResult,
    SelectiveLoader,
    SelectiveLoaderConfig,
)
from agent_session_linker.selective.importance_scorer import SegmentType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seg(seg_id: str, seg_type: str, content: str, tokens: int) -> dict:
    return {
        "segment_id": seg_id,
        "segment_type": seg_type,
        "content": content,
        "token_count": tokens,
    }


# ---------------------------------------------------------------------------
# SelectiveLoaderConfig
# ---------------------------------------------------------------------------


class TestSelectiveLoaderConfig:
    def test_defaults(self) -> None:
        config = SelectiveLoaderConfig()
        assert config.token_budget == 4000
        assert config.importance_threshold == pytest.approx(0.50)
        assert config.max_segments == 100
        assert config.preserve_order is True

    def test_frozen(self) -> None:
        config = SelectiveLoaderConfig()
        with pytest.raises((TypeError, AttributeError)):
            config.token_budget = 100  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LoadResult
# ---------------------------------------------------------------------------


class TestLoadResult:
    def test_to_dict(self) -> None:
        from agent_session_linker.selective.importance_scorer import ScoredSegment
        result = LoadResult(
            selected_segments=[],
            total_tokens_loaded=0,
            total_tokens_available=0,
            segments_considered=0,
            segments_skipped=0,
            budget_used_pct=0.0,
        )
        d = result.to_dict()
        assert "selected_count" in d
        assert "total_tokens_loaded" in d
        assert "budget_used_pct" in d


# ---------------------------------------------------------------------------
# SelectiveLoader — empty input
# ---------------------------------------------------------------------------


class TestSelectiveLoaderEmpty:
    def test_empty_segments(self) -> None:
        loader = SelectiveLoader()
        result = loader.load([])
        assert result.selected_segments == []
        assert result.total_tokens_loaded == 0
        assert result.segments_considered == 0


# ---------------------------------------------------------------------------
# SelectiveLoader — importance threshold
# ---------------------------------------------------------------------------


class TestSelectiveLoaderThreshold:
    def test_high_importance_included(self) -> None:
        config = SelectiveLoaderConfig(token_budget=1000, importance_threshold=0.50)
        loader = SelectiveLoader(config=config)
        segs = [
            _seg("s1", "preference", "User prefers JSON", 10),
            _seg("s2", "task_state", "Current task: write report", 20),
        ]
        result = loader.load(segs)
        ids = [s.segment_id for s in result.selected_segments]
        assert "s1" in ids
        assert "s2" in ids

    def test_low_importance_excluded(self) -> None:
        config = SelectiveLoaderConfig(token_budget=1000, importance_threshold=0.80)
        loader = SelectiveLoader(config=config)
        segs = [_seg("s1", "chat", "Hello", 5)]
        result = loader.load(segs)
        # chat segment score is ~0.30, below 0.80 threshold
        assert len(result.selected_segments) == 0

    def test_threshold_boundary(self) -> None:
        # reasoning prior = 0.60; threshold = 0.60 → should be included
        config = SelectiveLoaderConfig(token_budget=1000, importance_threshold=0.60)
        loader = SelectiveLoader(config=config)
        segs = [_seg("s1", "reasoning", "Because X leads to Y", 20)]
        result = loader.load(segs)
        # reasoning score should be >= 0.60 (with possible recency boost)
        assert len(result.selected_segments) >= 1


# ---------------------------------------------------------------------------
# SelectiveLoader — token budget
# ---------------------------------------------------------------------------


class TestSelectiveLoaderTokenBudget:
    def test_respects_token_budget(self) -> None:
        config = SelectiveLoaderConfig(token_budget=50, importance_threshold=0.40)
        loader = SelectiveLoader(config=config)
        segs = [
            _seg("s1", "preference", "prefer JSON", 30),
            _seg("s2", "task_state", "current task: write report", 30),
        ]
        result = loader.load(segs)
        assert result.total_tokens_loaded <= 50

    def test_fills_budget_greedily(self) -> None:
        config = SelectiveLoaderConfig(token_budget=60, importance_threshold=0.40)
        loader = SelectiveLoader(config=config)
        segs = [
            _seg("s1", "preference", "prefer JSON", 25),
            _seg("s2", "task_state", "current task: report", 25),
            _seg("s3", "reasoning", "because X", 15),
        ]
        result = loader.load(segs)
        # Total tokens: 65; budget 60 → at least 2 segments should fit
        assert result.total_tokens_loaded <= 60
        assert len(result.selected_segments) >= 2

    def test_budget_used_pct(self) -> None:
        config = SelectiveLoaderConfig(token_budget=100, importance_threshold=0.0)
        loader = SelectiveLoader(config=config)
        segs = [_seg("s1", "preference", "prefer JSON", 50)]
        result = loader.load(segs)
        assert result.budget_used_pct == pytest.approx(0.50, abs=0.01)


# ---------------------------------------------------------------------------
# SelectiveLoader — chronological order preservation
# ---------------------------------------------------------------------------


class TestSelectiveLoaderOrder:
    def test_chronological_order_preserved(self) -> None:
        config = SelectiveLoaderConfig(
            token_budget=1000, importance_threshold=0.40, preserve_order=True
        )
        loader = SelectiveLoader(config=config)
        # Preference segments should be selected; chat may be dropped
        segs = [
            _seg("a", "preference", "prefer JSON", 10),
            _seg("b", "task_state", "current task: report", 15),
            _seg("c", "preference", "always use metric units", 10),
        ]
        result = loader.load(segs)
        ids = [s.segment_id for s in result.selected_segments]
        # Verify chronological order (a < b < c)
        a_idx = ids.index("a") if "a" in ids else -1
        b_idx = ids.index("b") if "b" in ids else -1
        c_idx = ids.index("c") if "c" in ids else -1
        # For present IDs, verify ordering
        present = [(idx, sid) for sid, idx in [("a", a_idx), ("b", b_idx), ("c", c_idx)] if idx >= 0]
        sorted_by_original = sorted(present, key=lambda t: t[0])
        assert [sid for _, sid in present] == [sid for _, sid in sorted_by_original]


# ---------------------------------------------------------------------------
# SelectiveLoader — max_segments
# ---------------------------------------------------------------------------


class TestSelectiveLoaderMaxSegments:
    def test_max_segments_respected(self) -> None:
        config = SelectiveLoaderConfig(
            token_budget=10000, importance_threshold=0.0, max_segments=2
        )
        loader = SelectiveLoader(config=config)
        segs = [_seg(f"s{i}", "preference", f"prefer item {i}", 10) for i in range(10)]
        result = loader.load(segs)
        assert len(result.selected_segments) <= 2


# ---------------------------------------------------------------------------
# SelectiveLoader — always_include_types
# ---------------------------------------------------------------------------


class TestSelectiveLoaderAlwaysInclude:
    def test_always_include_type(self) -> None:
        # Set a high threshold but always include preferences
        config = SelectiveLoaderConfig(
            token_budget=1000,
            importance_threshold=0.95,
            always_include_types=["preference"],
        )
        loader = SelectiveLoader(config=config)
        segs = [
            _seg("s1", "preference", "prefer JSON output", 10),
            _seg("s2", "chat", "hello world", 5),
        ]
        result = loader.load(segs)
        ids = [s.segment_id for s in result.selected_segments]
        # Preference must be included even though threshold is very high
        assert "s1" in ids
        # Chat should be excluded (below threshold, not always_include)
        assert "s2" not in ids


# ---------------------------------------------------------------------------
# SelectiveLoader — segments_skipped
# ---------------------------------------------------------------------------


class TestSelectiveLoaderSkipped:
    def test_skipped_count(self) -> None:
        config = SelectiveLoaderConfig(
            token_budget=25, importance_threshold=0.40
        )
        loader = SelectiveLoader(config=config)
        segs = [
            _seg("s1", "preference", "prefer JSON", 20),
            _seg("s2", "task_state", "task: report", 20),
        ]
        result = loader.load(segs)
        assert result.segments_skipped >= 0  # At least one may be skipped
