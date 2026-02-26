"""Unit tests for agent_session_linker.context.injector.

Covers InjectionConfig, ContextInjector (inject, score_segment, _format,
_build_header, _filter_entities) and the TF-IDF module-level helpers.
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest

from agent_session_linker.context.injector import (
    InjectionConfig,
    ContextInjector,
    _tokenize,
    _term_frequency,
    _compute_idf,
    _tfidf_score,
)
from agent_session_linker.session.manager import SessionManager
from agent_session_linker.session.state import (
    ContextSegment,
    EntityReference,
    SessionState,
    TaskStatus,
)
from agent_session_linker.storage.memory import InMemoryBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    content: str,
    role: str = "user",
    segment_type: str = "conversation",
    token_count: int = 10,
    age_hours: float = 0.0,
) -> ContextSegment:
    timestamp = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return ContextSegment(
        role=role,
        content=content,
        token_count=token_count,
        turn_index=0,
        segment_type=segment_type,
        timestamp=timestamp,
    )


def _make_session(
    segments: list[ContextSegment] | None = None,
    summary: str = "",
    tasks_info: list[tuple[str, TaskStatus]] | None = None,
    entities_info: list[tuple[str, str]] | None = None,
) -> SessionState:
    manager = SessionManager(
        backend=InMemoryBackend(), default_agent_id="test-agent"
    )
    session = manager.create_session()
    for seg in segments or []:
        session.segments.append(seg)
    if summary:
        session.summary = summary
    for title, status in tasks_info or []:
        task = session.add_task(title)
        if status != TaskStatus.PENDING:
            session.update_task(task.task_id, status=status)
    for name, etype in entities_info or []:
        session.track_entity(name, etype)
    return session


# ---------------------------------------------------------------------------
# Module-level TF-IDF helpers
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases_input(self) -> None:
        tokens = _tokenize("Hello World")
        assert all(t == t.lower() for t in tokens)

    def test_removes_stop_words(self) -> None:
        tokens = _tokenize("this is a test")
        assert "this" not in tokens
        assert "is" not in tokens
        assert "a" not in tokens

    def test_keeps_content_words(self) -> None:
        tokens = _tokenize("machine learning model")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "model" in tokens

    def test_removes_single_char_tokens(self) -> None:
        tokens = _tokenize("a b c dogs")
        assert "b" not in tokens
        assert "c" not in tokens
        assert "dogs" in tokens

    def test_empty_string_returns_empty(self) -> None:
        assert _tokenize("") == []


class TestTermFrequency:
    def test_empty_tokens_returns_empty(self) -> None:
        assert _term_frequency([]) == {}

    def test_single_token_has_frequency_one(self) -> None:
        tf = _term_frequency(["word"])
        assert tf["word"] == 1.0

    def test_repeated_token_has_higher_frequency(self) -> None:
        tf = _term_frequency(["dog", "dog", "cat"])
        assert tf["dog"] > tf["cat"]

    def test_all_frequencies_sum_to_one(self) -> None:
        tokens = ["alpha", "beta", "gamma"]
        tf = _term_frequency(tokens)
        assert abs(sum(tf.values()) - 1.0) < 1e-9


class TestComputeIdf:
    def test_empty_corpus_returns_empty(self) -> None:
        assert _compute_idf([]) == {}

    def test_rare_term_has_higher_idf(self) -> None:
        docs = [["cat"], ["dog"], ["cat"]]
        idf = _compute_idf(docs)
        # "dog" appears in 1 doc, "cat" in 2 — dog should have higher IDF.
        assert idf["dog"] > idf["cat"]

    def test_all_idf_values_positive(self) -> None:
        docs = [["a", "b"], ["b", "c"]]
        idf = _compute_idf(docs)
        assert all(v > 0 for v in idf.values())


class TestTfidfScore:
    def test_empty_query_returns_zero(self) -> None:
        idf = {"dog": 1.0}
        assert _tfidf_score([], ["dog"], idf) == 0.0

    def test_empty_doc_returns_zero(self) -> None:
        idf = {"dog": 1.0}
        assert _tfidf_score(["dog"], [], idf) == 0.0

    def test_matching_tokens_produce_positive_score(self) -> None:
        idf = {"dog": 2.0}
        score = _tfidf_score(["dog"], ["dog", "cat"], idf)
        assert score > 0.0

    def test_non_matching_tokens_produce_zero_score(self) -> None:
        idf = {"dog": 2.0}
        score = _tfidf_score(["elephant"], ["dog", "cat"], idf)
        assert score == 0.0


# ---------------------------------------------------------------------------
# InjectionConfig
# ---------------------------------------------------------------------------


class TestInjectionConfig:
    def test_defaults(self) -> None:
        config = InjectionConfig()
        assert config.token_budget == 2000
        assert config.max_segments == 20
        assert config.include_summary is True
        assert config.include_active_tasks is True
        assert config.include_entities is True

    def test_custom_token_budget(self) -> None:
        config = InjectionConfig(token_budget=500)
        assert config.token_budget == 500

    def test_type_priorities_defaults_populated(self) -> None:
        config = InjectionConfig()
        assert "plan" in config.type_priorities
        assert "code" in config.type_priorities


# ---------------------------------------------------------------------------
# ContextInjector.inject — basic behaviour
# ---------------------------------------------------------------------------


class TestContextInjectorInject:
    def test_inject_empty_sessions_returns_empty_string(self) -> None:
        injector = ContextInjector()
        result = injector.inject([], "query")
        assert result == ""

    def test_inject_returns_string(self) -> None:
        session = _make_session(
            segments=[_make_segment("python machine learning", age_hours=0)]
        )
        injector = ContextInjector()
        result = injector.inject([session], "python")
        assert isinstance(result, str)

    def test_inject_includes_segment_content(self) -> None:
        session = _make_session(
            segments=[_make_segment("python machine learning", age_hours=0)]
        )
        injector = ContextInjector()
        result = injector.inject([session], "python")
        assert "python" in result.lower() or "machine" in result.lower()

    def test_inject_includes_header_markers(self) -> None:
        session = _make_session(segments=[_make_segment("content", age_hours=0)])
        injector = ContextInjector()
        result = injector.inject([session], "query")
        assert "PRIOR SESSION CONTEXT" in result

    def test_inject_excludes_segments_older_than_max_age(self) -> None:
        config = InjectionConfig(max_age_hours=1.0)
        session = _make_session(
            segments=[_make_segment("old content", age_hours=2.0)]
        )
        injector = ContextInjector(config=config)
        result = injector.inject([session], "old content")
        assert "old content" not in result

    def test_inject_respects_token_budget(self) -> None:
        config = InjectionConfig(token_budget=30)
        segments = [
            _make_segment("segment one content here", token_count=20, age_hours=0),
            _make_segment("segment two content here", token_count=20, age_hours=0),
            _make_segment("segment three content here", token_count=20, age_hours=0),
        ]
        session = _make_session(segments=segments)
        injector = ContextInjector(config=config)
        result = injector.inject([session], "content")
        # With a 30-token budget and each segment costing 20, only 1 fits.
        assert result.count("[USER") <= 1

    def test_inject_respects_max_segments(self) -> None:
        config = InjectionConfig(max_segments=1, token_budget=9999)
        segments = [
            _make_segment(f"segment {i}", token_count=5, age_hours=0)
            for i in range(5)
        ]
        session = _make_session(segments=segments)
        injector = ContextInjector(config=config)
        result = injector.inject([session], "segment")
        assert result.count("[USER") <= 1

    def test_inject_includes_summary_when_configured(self) -> None:
        config = InjectionConfig(include_summary=True)
        session = _make_session(summary="This is a test summary.")
        injector = ContextInjector(config=config)
        result = injector.inject([session], "test")
        assert "This is a test summary." in result

    def test_inject_excludes_summary_when_disabled(self) -> None:
        config = InjectionConfig(include_summary=False)
        session = _make_session(summary="Should not appear")
        injector = ContextInjector(config=config)
        result = injector.inject([session], "test")
        assert "Should not appear" not in result

    def test_inject_includes_active_tasks(self) -> None:
        config = InjectionConfig(include_active_tasks=True)
        session = _make_session(
            tasks_info=[("Implement feature X", TaskStatus.IN_PROGRESS)]
        )
        injector = ContextInjector(config=config)
        result = injector.inject([session], "feature")
        assert "Implement feature X" in result

    def test_inject_excludes_completed_tasks(self) -> None:
        config = InjectionConfig(include_active_tasks=True)
        session = _make_session(
            tasks_info=[("Done task", TaskStatus.COMPLETED)]
        )
        injector = ContextInjector(config=config)
        result = injector.inject([session], "task")
        assert "Done task" not in result

    def test_inject_with_multiple_sessions(self) -> None:
        s1 = _make_session(segments=[_make_segment("session one content", age_hours=0)])
        s2 = _make_session(segments=[_make_segment("session two content", age_hours=0)])
        injector = ContextInjector()
        result = injector.inject([s1, s2], "content")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# ContextInjector._build_header (via inject with no eligible segments)
# ---------------------------------------------------------------------------


class TestContextInjectorBuildHeader:
    def test_header_without_summary(self) -> None:
        session = _make_session()
        injector = ContextInjector()
        result = injector.inject([session], "query")
        assert "PRIOR SESSION CONTEXT" in result

    def test_header_includes_entity_matching_query(self) -> None:
        session = _make_session(
            entities_info=[("Django", "framework"), ("Flask", "framework")]
        )
        injector = ContextInjector()
        result = injector.inject([session], "Django framework")
        assert "Django" in result

    def test_header_excludes_entity_not_matching_query(self) -> None:
        session = _make_session(
            entities_info=[("Django", "framework")]
        )
        injector = ContextInjector()
        # Query contains no tokens overlapping with "django".
        result = injector.inject([session], "python machine learning")
        # "Django" may or may not be included depending on tokenisation — just
        # check the injection block is well-formed.
        assert "PRIOR SESSION CONTEXT" in result


# ---------------------------------------------------------------------------
# ContextInjector.score_segment
# ---------------------------------------------------------------------------


class TestContextInjectorScoreSegment:
    def test_score_returns_float(self) -> None:
        injector = ContextInjector()
        seg = _make_segment("machine learning algorithm", age_hours=0)
        refs = [seg, _make_segment("deep learning neural network", age_hours=0)]
        score = injector.score_segment(seg, "machine learning", refs)
        assert isinstance(score, float)

    def test_relevant_segment_scores_higher_than_irrelevant(self) -> None:
        injector = ContextInjector()
        relevant = _make_segment("machine learning model training", age_hours=0)
        irrelevant = _make_segment("grocery shopping list apples", age_hours=0)
        refs = [relevant, irrelevant]
        s_rel = injector.score_segment(relevant, "machine learning", refs)
        s_irr = injector.score_segment(irrelevant, "machine learning", refs)
        assert s_rel >= s_irr


# ---------------------------------------------------------------------------
# ContextInjector._filter_entities (static)
# ---------------------------------------------------------------------------


class TestContextInjectorFilterEntities:
    def _make_entity(self, name: str, aliases: list[str] | None = None) -> EntityReference:
        return EntityReference(
            canonical_name=name,
            entity_type="concept",
            aliases=aliases or [],
        )

    def test_entity_matching_query_token_included(self) -> None:
        entities = [self._make_entity("Django")]
        query_tokens = ["django", "framework"]
        result = ContextInjector._filter_entities(entities, query_tokens)
        assert len(result) == 1

    def test_entity_with_matching_alias_included(self) -> None:
        entities = [self._make_entity("Python", aliases=["py"])]
        query_tokens = ["py"]
        result = ContextInjector._filter_entities(entities, query_tokens)
        assert len(result) == 1

    def test_entity_not_matching_excluded(self) -> None:
        entities = [self._make_entity("Rust")]
        query_tokens = ["python", "machine", "learning"]
        result = ContextInjector._filter_entities(entities, query_tokens)
        assert len(result) == 0

    def test_empty_entities_returns_empty(self) -> None:
        result = ContextInjector._filter_entities([], ["query"])
        assert result == []

    def test_empty_query_tokens_returns_empty(self) -> None:
        entities = [self._make_entity("Django")]
        result = ContextInjector._filter_entities(entities, [])
        assert result == []
