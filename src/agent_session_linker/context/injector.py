"""Context injection with TF-IDF relevance scoring and token budgeting.

Selects the most relevant context segments from one or more sessions and
formats them for injection into a new agent system prompt.

Classes
-------
- InjectionConfig  — configuration for the injector
- ContextInjector  — scores, selects, and formats context
"""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agent_session_linker.context.freshness import DecayCurve, FreshnessDecay
from agent_session_linker.session.state import ContextSegment, EntityReference, SessionState, TaskStatus


class InjectionConfig(BaseModel):
    """Configuration parameters for ``ContextInjector``.

    Parameters
    ----------
    token_budget:
        Maximum tokens to include in the injected context block.
        Default: 2000.
    max_segments:
        Hard cap on the number of segments included regardless of budget.
        Default: 20.
    freshness_curve:
        Decay curve to use for age-based freshness weighting.
    max_age_hours:
        Oldest segments to consider, in hours.  Segments older than this
        are excluded entirely.  Default: 168 (one week).
    decay_rate:
        Decay rate parameter for the exponential curve.  Default: 0.01.
    step_thresholds:
        Threshold pair for the step curve.  Default: (24.0, 168.0).
    relevance_weight:
        Weight applied to the TF-IDF relevance score (0.0–1.0).
    freshness_weight:
        Weight applied to the freshness score (0.0–1.0).
    type_priority_weight:
        Weight applied to the segment type priority bonus.
    type_priorities:
        Mapping of ``segment_type`` label to a priority score in [0, 1].
        Higher means more likely to be selected.
    include_summary:
        Include the session summary at the top of the injection block.
    include_active_tasks:
        Include pending/in-progress tasks in the injection block.
    include_entities:
        Include tracked entities in the injection block.
    """

    token_budget: int = 2000
    max_segments: int = 20
    freshness_curve: DecayCurve = DecayCurve.EXPONENTIAL
    max_age_hours: float = 168.0
    decay_rate: float = 0.01
    step_thresholds: tuple[float, float] = (24.0, 168.0)
    relevance_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    freshness_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    type_priority_weight: float = Field(default=0.2, ge=0.0, le=1.0)
    type_priorities: dict[str, float] = Field(
        default_factory=lambda: {
            "plan": 1.0,
            "reasoning": 0.9,
            "code": 0.85,
            "output": 0.7,
            "conversation": 0.5,
            "metadata": 0.3,
        }
    )
    include_summary: bool = True
    include_active_tasks: bool = True
    include_entities: bool = True

    model_config = {"frozen": False}


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "with", "as", "by", "from",
        "this", "that", "was", "are", "be", "been", "have", "has",
        "do", "did", "will", "would", "could", "should", "may", "can",
        "i", "you", "we", "they", "he", "she", "its", "their", "our",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, remove stop words."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _term_frequency(tokens: list[str]) -> dict[str, float]:
    """Compute raw TF for a token list (normalised by document length)."""
    if not tokens:
        return {}
    counts = Counter(tokens)
    total = len(tokens)
    return {term: count / total for term, count in counts.items()}


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute IDF scores across a corpus of tokenised documents."""
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


def _tfidf_score(query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float]) -> float:
    """Score a document against a query using TF-IDF overlap."""
    if not query_tokens or not doc_tokens:
        return 0.0
    tf = _term_frequency(doc_tokens)
    score = sum(
        tf.get(token, 0.0) * idf.get(token, 0.0)
        for token in set(query_tokens)
    )
    return score


class ContextInjector:
    """Score and inject relevant context segments into agent prompts.

    Uses a combination of TF-IDF relevance, freshness decay, and segment
    type priority to rank segments, then selects the top candidates within
    a configurable token budget.

    Parameters
    ----------
    config:
        Injection configuration.  Defaults to ``InjectionConfig()``.
    """

    def __init__(self, config: InjectionConfig | None = None) -> None:
        self.config = config or InjectionConfig()
        self._freshness = FreshnessDecay(
            curve=self.config.freshness_curve,
            max_age_hours=self.config.max_age_hours,
            decay_rate=self.config.decay_rate,
            step_thresholds=self.config.step_thresholds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, sessions: list[SessionState], query: str) -> str:
        """Select relevant context and return a formatted injection block.

        Parameters
        ----------
        sessions:
            One or more prior sessions to draw context from.  Sessions are
            processed together; segments from all sessions compete for
            the token budget.
        query:
            The current user query or task description.  Used to compute
            TF-IDF relevance scores.

        Returns
        -------
        str
            A formatted string ready for inclusion in a system prompt.
            Empty string if no relevant content was found.
        """
        if not sessions:
            return ""

        query_tokens = _tokenize(query)
        now = datetime.now(timezone.utc)

        # Collect all segments and filter by max_age.
        eligible: list[tuple[ContextSegment, SessionState]] = []
        for session in sessions:
            for segment in session.segments:
                age_hours = (now - segment.timestamp).total_seconds() / 3600.0
                if age_hours <= self.config.max_age_hours:
                    eligible.append((segment, session))

        if not eligible:
            return self._build_header(sessions, query_tokens)

        # Build IDF corpus from all eligible segment texts.
        all_doc_tokens = [_tokenize(seg.content) for seg, _ in eligible]
        idf = _compute_idf(all_doc_tokens)

        # Score each segment.
        scored: list[tuple[float, ContextSegment, SessionState]] = []
        for (segment, session), doc_tokens in zip(eligible, all_doc_tokens):
            age_hours = (now - segment.timestamp).total_seconds() / 3600.0
            freshness = self._freshness.score(age_hours)
            relevance = _tfidf_score(query_tokens, doc_tokens, idf)
            type_score = self.config.type_priorities.get(segment.segment_type, 0.5)
            combined = (
                self.config.relevance_weight * relevance
                + self.config.freshness_weight * freshness
                + self.config.type_priority_weight * type_score
            )
            scored.append((combined, segment, session))

        # Sort descending by score.
        scored.sort(key=lambda triple: triple[0], reverse=True)

        # Select within token budget.
        selected: list[tuple[ContextSegment, SessionState]] = []
        token_total = 0
        for _score, segment, session in scored:
            if len(selected) >= self.config.max_segments:
                break
            tokens = segment.token_count or len(segment.content) // 4
            if token_total + tokens > self.config.token_budget:
                continue
            selected.append((segment, session))
            token_total += tokens

        return self._format(sessions, selected, query_tokens)

    def score_segment(
        self, segment: ContextSegment, query: str, reference_segments: list[ContextSegment]
    ) -> float:
        """Return the combined relevance+freshness+type score for one segment.

        Parameters
        ----------
        segment:
            The segment to score.
        query:
            Current query text.
        reference_segments:
            All segments in the context corpus (for IDF computation).

        Returns
        -------
        float
            Combined score.
        """
        now = datetime.now(timezone.utc)
        query_tokens = _tokenize(query)
        doc_tokens = _tokenize(segment.content)
        all_doc_tokens = [_tokenize(s.content) for s in reference_segments]
        idf = _compute_idf(all_doc_tokens)
        age_hours = (now - segment.timestamp).total_seconds() / 3600.0
        freshness = self._freshness.score(age_hours)
        relevance = _tfidf_score(query_tokens, doc_tokens, idf)
        type_score = self.config.type_priorities.get(segment.segment_type, 0.5)
        return (
            self.config.relevance_weight * relevance
            + self.config.freshness_weight * freshness
            + self.config.type_priority_weight * type_score
        )

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _build_header(self, sessions: list[SessionState], query_tokens: list[str]) -> str:
        """Build the header block (summary + tasks + entities) even when
        no segments pass the age filter."""
        parts: list[str] = []
        parts.append("--- PRIOR SESSION CONTEXT ---")

        if self.config.include_summary:
            for session in sessions:
                if session.summary:
                    parts.append(f"\n[Summary from session {session.session_id[:8]}]")
                    parts.append(session.summary)

        if self.config.include_active_tasks:
            active_tasks = [
                task
                for session in sessions
                for task in session.tasks
                if task.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
            ]
            if active_tasks:
                parts.append("\n[Active Tasks]")
                for task in active_tasks:
                    status_label = task.status.value.replace("_", " ").title()
                    parts.append(f"  - [{status_label}] {task.title}")
                    if task.description:
                        parts.append(f"    {task.description}")

        if self.config.include_entities:
            all_entities: list[EntityReference] = [
                entity for session in sessions for entity in session.entities
            ]
            if all_entities and query_tokens:
                relevant_entities = self._filter_entities(all_entities, query_tokens)
                if relevant_entities:
                    parts.append("\n[Relevant Entities]")
                    for entity in relevant_entities[:10]:
                        aliases = (
                            f" (aka {', '.join(entity.aliases[:3])})"
                            if entity.aliases
                            else ""
                        )
                        parts.append(f"  - {entity.canonical_name}{aliases} [{entity.entity_type}]")

        parts.append("\n--- END PRIOR CONTEXT ---")
        return "\n".join(parts)

    def _format(
        self,
        sessions: list[SessionState],
        selected: list[tuple[ContextSegment, SessionState]],
        query_tokens: list[str],
    ) -> str:
        """Format selected segments and metadata into the injection block."""
        parts: list[str] = [self._build_header(sessions, query_tokens)]

        if selected:
            parts.append("\n[Relevant Context Segments]")
            for segment, session in selected:
                role_label = segment.role.upper()
                turn_info = f"turn={segment.turn_index}"
                session_info = f"session={session.session_id[:8]}"
                parts.append(f"\n[{role_label} | {segment.segment_type} | {turn_info} | {session_info}]")
                parts.append(segment.content)

        return "\n".join(parts)

    @staticmethod
    def _filter_entities(
        entities: list[EntityReference], query_tokens: list[str]
    ) -> list[EntityReference]:
        """Filter entities whose name or aliases overlap with ``query_tokens``."""
        query_set = set(query_tokens)
        relevant: list[EntityReference] = []
        for entity in entities:
            name_tokens = set(_tokenize(entity.canonical_name))
            alias_tokens: set[str] = set()
            for alias in entity.aliases:
                alias_tokens.update(_tokenize(alias))
            if name_tokens & query_set or alias_tokens & query_set:
                relevant.append(entity)
        return relevant
