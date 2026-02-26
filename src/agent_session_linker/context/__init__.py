"""Context processing subpackage.

Provides tools for scoring, selecting, and injecting relevant context
from prior sessions into new agent prompts.

Public surface
--------------
- FreshnessDecay     — configurable age-based score decay curves
- ContextInjector    — TF-IDF + freshness scoring, token-budget injection
- ContextSummarizer  — heuristic conversation compression
"""
from __future__ import annotations

from agent_session_linker.context.freshness import DecayCurve, FreshnessDecay
from agent_session_linker.context.injector import ContextInjector, InjectionConfig
from agent_session_linker.context.summarizer import ContextSummarizer

__all__ = [
    "ContextInjector",
    "ContextSummarizer",
    "DecayCurve",
    "FreshnessDecay",
    "InjectionConfig",
]
