"""Sliding context window with token-budget management.

Maintains a bounded window of the most-recent context segments and can
render them as a formatted string for injection into prompts.

Classes
-------
- ContextWindowManager  — sliding window over ContextSegment objects
"""
from __future__ import annotations

from collections import deque

from agent_session_linker.session.state import ContextSegment


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: one token per ~4 characters."""
    return max(1, len(text) // 4)


class ContextWindowManager:
    """Manage a sliding window of context segments within a token budget.

    Segments are added via ``add()``.  When the cumulative token count
    would exceed ``max_tokens``, the oldest segment(s) are evicted until
    the budget constraint is satisfied.

    Parameters
    ----------
    max_tokens:
        Maximum token count allowed in the window at any time.  Default: 4000.
    max_segments:
        Hard cap on the number of segments in the window, independent of
        the token budget.  Default: 50.
    role_separator:
        String placed between the role label and content in the rendered
        window.  Default: ``": "``.
    segment_separator:
        String placed between consecutive segments in the rendered output.
        Default: ``"\\n\\n"``.
    """

    def __init__(
        self,
        max_tokens: int = 4000,
        max_segments: int = 50,
        role_separator: str = ": ",
        segment_separator: str = "\n\n",
    ) -> None:
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens!r}.")
        if max_segments < 1:
            raise ValueError(f"max_segments must be >= 1, got {max_segments!r}.")

        self.max_tokens = max_tokens
        self.max_segments = max_segments
        self.role_separator = role_separator
        self.segment_separator = segment_separator

        self._window: deque[ContextSegment] = deque()
        self._token_total: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, segment: ContextSegment) -> None:
        """Append a segment to the window, evicting old segments if needed.

        Token count is taken from ``segment.token_count`` when non-zero,
        otherwise estimated from content length.

        Parameters
        ----------
        segment:
            The segment to append.
        """
        segment_tokens = segment.token_count or _estimate_tokens(segment.content)

        # If a single segment is larger than the entire budget, accept it
        # as a lone entry (the window holds at least one segment always).
        if segment_tokens > self.max_tokens and not self._window:
            self._window.append(segment)
            self._token_total += segment_tokens
            return

        # Evict from the front until there is room.
        while self._window and (
            self._token_total + segment_tokens > self.max_tokens
            or len(self._window) >= self.max_segments
        ):
            evicted = self._window.popleft()
            self._token_total -= evicted.token_count or _estimate_tokens(evicted.content)

        self._window.append(segment)
        self._token_total += segment_tokens

    def get_window(self) -> str:
        """Render the current window as a formatted string.

        Each segment is rendered as ``"<ROLE>: <content>"`` with segments
        separated by ``self.segment_separator``.

        Returns
        -------
        str
            Formatted context window string.  Empty string when the window
            is empty.
        """
        if not self._window:
            return ""

        parts: list[str] = []
        for segment in self._window:
            role_label = segment.role.upper()
            parts.append(f"{role_label}{self.role_separator}{segment.content}")

        return self.segment_separator.join(parts)

    def get_segments(self) -> list[ContextSegment]:
        """Return a copy of the current window as an ordered list.

        Returns
        -------
        list[ContextSegment]
            Segments from oldest to newest.
        """
        return list(self._window)

    def token_count(self) -> int:
        """Return the current cumulative token count in the window.

        Returns
        -------
        int
            Total estimated tokens for all segments in the window.
        """
        return self._token_total

    def clear(self) -> None:
        """Remove all segments from the window."""
        self._window.clear()
        self._token_total = 0

    def __len__(self) -> int:
        return len(self._window)

    def __repr__(self) -> str:
        return (
            f"ContextWindowManager("
            f"segments={len(self._window)}, "
            f"tokens={self._token_total}/{self.max_tokens})"
        )
