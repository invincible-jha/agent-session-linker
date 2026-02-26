"""Unit tests for agent_session_linker.middleware.context_window.

ContextWindowManager and the module-level _estimate_tokens helper.
"""
from __future__ import annotations

import pytest

from agent_session_linker.middleware.context_window import (
    ContextWindowManager,
    _estimate_tokens,
)
from agent_session_linker.session.state import ContextSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    content: str,
    role: str = "user",
    token_count: int = 0,
) -> ContextSegment:
    """Create a minimal ContextSegment for testing."""
    from datetime import datetime, timezone
    return ContextSegment(
        role=role,
        content=content,
        token_count=token_count,
        turn_index=0,
        segment_type="conversation",
        timestamp=datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# _estimate_tokens
# ---------------------------------------------------------------------------


class TestEstimateTokens:
    def test_empty_string_returns_one(self) -> None:
        assert _estimate_tokens("") == 1

    def test_four_chars_returns_one(self) -> None:
        assert _estimate_tokens("abcd") == 1

    def test_eight_chars_returns_two(self) -> None:
        assert _estimate_tokens("abcdefgh") == 2

    def test_long_text_proportional(self) -> None:
        text = "x" * 400
        assert _estimate_tokens(text) == 100


# ---------------------------------------------------------------------------
# ContextWindowManager construction
# ---------------------------------------------------------------------------


class TestContextWindowManagerConstruction:
    def test_default_parameters(self) -> None:
        mgr = ContextWindowManager()
        assert mgr.max_tokens == 4000
        assert mgr.max_segments == 50

    def test_custom_parameters(self) -> None:
        mgr = ContextWindowManager(max_tokens=1000, max_segments=10)
        assert mgr.max_tokens == 1000
        assert mgr.max_segments == 10

    def test_max_tokens_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            ContextWindowManager(max_tokens=0)

    def test_max_segments_zero_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="max_segments"):
            ContextWindowManager(max_segments=0)

    def test_repr_contains_token_info(self) -> None:
        mgr = ContextWindowManager(max_tokens=100)
        assert "100" in repr(mgr)

    def test_initial_token_count_is_zero(self) -> None:
        mgr = ContextWindowManager()
        assert mgr.token_count() == 0

    def test_initial_len_is_zero(self) -> None:
        mgr = ContextWindowManager()
        assert len(mgr) == 0


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestContextWindowManagerAdd:
    def test_add_single_segment(self) -> None:
        mgr = ContextWindowManager()
        seg = _make_segment("hello world", token_count=5)
        mgr.add(seg)
        assert len(mgr) == 1

    def test_add_uses_explicit_token_count(self) -> None:
        mgr = ContextWindowManager()
        seg = _make_segment("text", token_count=50)
        mgr.add(seg)
        assert mgr.token_count() == 50

    def test_add_estimates_tokens_when_zero(self) -> None:
        mgr = ContextWindowManager()
        # 8 chars -> 2 tokens estimated.
        seg = _make_segment("abcdefgh", token_count=0)
        mgr.add(seg)
        assert mgr.token_count() == 2

    def test_add_evicts_oldest_when_over_token_budget(self) -> None:
        mgr = ContextWindowManager(max_tokens=20)
        for i in range(3):
            seg = _make_segment(f"segment {i}", token_count=10)
            mgr.add(seg)
        # Only the last two segments (10 + 10 = 20) should fit.
        assert len(mgr) == 2

    def test_add_evicts_when_segment_count_exceeded(self) -> None:
        mgr = ContextWindowManager(max_tokens=10000, max_segments=2)
        for i in range(4):
            seg = _make_segment(f"msg-{i}", token_count=1)
            mgr.add(seg)
        assert len(mgr) == 2

    def test_add_very_large_segment_accepted_as_lone_entry(self) -> None:
        mgr = ContextWindowManager(max_tokens=10)
        huge = _make_segment("x" * 400, token_count=9999)
        mgr.add(huge)
        assert len(mgr) == 1

    def test_add_cumulates_token_count(self) -> None:
        mgr = ContextWindowManager(max_tokens=100)
        mgr.add(_make_segment("a", token_count=10))
        mgr.add(_make_segment("b", token_count=20))
        assert mgr.token_count() == 30


# ---------------------------------------------------------------------------
# get_window
# ---------------------------------------------------------------------------


class TestContextWindowManagerGetWindow:
    def test_get_window_empty_returns_empty_string(self) -> None:
        mgr = ContextWindowManager()
        assert mgr.get_window() == ""

    def test_get_window_contains_role_label(self) -> None:
        mgr = ContextWindowManager()
        seg = _make_segment("hello", role="user", token_count=5)
        mgr.add(seg)
        assert "USER" in mgr.get_window()

    def test_get_window_contains_content(self) -> None:
        mgr = ContextWindowManager()
        seg = _make_segment("my content", token_count=5)
        mgr.add(seg)
        assert "my content" in mgr.get_window()

    def test_get_window_uses_role_separator(self) -> None:
        mgr = ContextWindowManager(role_separator=" >>> ")
        seg = _make_segment("text", role="assistant", token_count=5)
        mgr.add(seg)
        assert " >>> " in mgr.get_window()

    def test_get_window_uses_segment_separator(self) -> None:
        mgr = ContextWindowManager(max_tokens=1000, segment_separator="---SEP---")
        mgr.add(_make_segment("first", token_count=5))
        mgr.add(_make_segment("second", token_count=5))
        assert "---SEP---" in mgr.get_window()

    def test_get_window_segments_in_order(self) -> None:
        mgr = ContextWindowManager(max_tokens=1000)
        mgr.add(_make_segment("alpha", token_count=5))
        mgr.add(_make_segment("beta", token_count=5))
        window = mgr.get_window()
        assert window.index("alpha") < window.index("beta")


# ---------------------------------------------------------------------------
# get_segments
# ---------------------------------------------------------------------------


class TestContextWindowManagerGetSegments:
    def test_get_segments_returns_list(self) -> None:
        mgr = ContextWindowManager()
        mgr.add(_make_segment("hello", token_count=5))
        segments = mgr.get_segments()
        assert isinstance(segments, list)

    def test_get_segments_is_a_copy(self) -> None:
        mgr = ContextWindowManager()
        mgr.add(_make_segment("hello", token_count=5))
        segments = mgr.get_segments()
        segments.clear()
        assert len(mgr) == 1

    def test_get_segments_preserves_order(self) -> None:
        mgr = ContextWindowManager(max_tokens=1000)
        s1 = _make_segment("first", token_count=5)
        s2 = _make_segment("second", token_count=5)
        mgr.add(s1)
        mgr.add(s2)
        segments = mgr.get_segments()
        assert segments[0].content == "first"
        assert segments[1].content == "second"


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


class TestContextWindowManagerClear:
    def test_clear_empties_window(self) -> None:
        mgr = ContextWindowManager()
        mgr.add(_make_segment("hello", token_count=10))
        mgr.clear()
        assert len(mgr) == 0

    def test_clear_resets_token_count(self) -> None:
        mgr = ContextWindowManager()
        mgr.add(_make_segment("hello", token_count=50))
        mgr.clear()
        assert mgr.token_count() == 0

    def test_clear_makes_window_empty_string(self) -> None:
        mgr = ContextWindowManager()
        mgr.add(_make_segment("hello", token_count=5))
        mgr.clear()
        assert mgr.get_window() == ""
