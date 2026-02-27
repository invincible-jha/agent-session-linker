"""Selective context restoration package.

Provides tools for loading only the most important segments of a session
within a token budget, classified by segment type.

Classes
-------
ImportanceScorer     — score session segments by importance
SelectiveLoader      — load segments above importance threshold within budget
SegmentClassifier    — classify segments by type
"""
from __future__ import annotations

from agent_session_linker.selective.importance_scorer import (
    ImportanceScorer,
    ScoredSegment,
    SegmentType,
)
from agent_session_linker.selective.selective_loader import (
    LoadResult,
    SelectiveLoader,
    SelectiveLoaderConfig,
)
from agent_session_linker.selective.segment_classifier import (
    ClassificationRule,
    SegmentClassifier,
    SegmentClassifierConfig,
)

__all__ = [
    # Importance scorer
    "ImportanceScorer",
    "ScoredSegment",
    "SegmentType",
    # Selective loader
    "LoadResult",
    "SelectiveLoader",
    "SelectiveLoaderConfig",
    # Segment classifier
    "ClassificationRule",
    "SegmentClassifier",
    "SegmentClassifierConfig",
]
