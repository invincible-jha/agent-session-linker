"""agent-session-linker — Cross-session context persistence and resumption.

Public API
----------
The stable public surface is everything exported from this module.
Anything inside submodules not re-exported here is considered private
and may change without notice.

Example
-------
>>> import agent_session_linker
>>> agent_session_linker.__version__
'0.1.0'
"""
from __future__ import annotations

# Session core
from agent_session_linker.session.state import (
    ContextSegment,
    EntityReference,
    SessionState,
    TaskState,
    TaskStatus,
    ToolContext,
)
from agent_session_linker.session.manager import SessionManager, SessionNotFoundError
from agent_session_linker.session.serializer import SessionSerializer, SchemaVersionError

# Storage backends
from agent_session_linker.storage.base import StorageBackend
from agent_session_linker.storage.memory import InMemoryBackend
from agent_session_linker.storage.filesystem import FilesystemBackend
from agent_session_linker.storage.sqlite import SQLiteBackend

# Context processing
from agent_session_linker.context.freshness import DecayCurve, FreshnessDecay
from agent_session_linker.context.injector import ContextInjector, InjectionConfig
from agent_session_linker.context.summarizer import ContextSummarizer
from agent_session_linker.context.relevance import RelevanceScorer

# Entity extraction, tracking, and linking
from agent_session_linker.entity.extractor import Entity, EntityExtractor
from agent_session_linker.entity.tracker import EntityTracker, TrackedEntity
from agent_session_linker.entity.linker import EntityLinker

# Middleware
from agent_session_linker.middleware.session_middleware import SessionMiddleware
from agent_session_linker.middleware.context_window import ContextWindowManager
from agent_session_linker.middleware.checkpoint import CheckpointManager, CheckpointRecord

# Session linking
from agent_session_linker.linking.session_linker import LinkedSession, SessionLinker
from agent_session_linker.linking.chain import SessionChain

# Portable / Universal Session Format
from agent_session_linker.portable.usf import (
    USFVersion,
    USFMessage,
    USFEntity,
    USFTaskState,
    UniversalSession,
)
from agent_session_linker.portable.exporters import (
    SessionExporter,
    LangChainExporter,
    CrewAIExporter,
    OpenAIExporter,
)
from agent_session_linker.portable.importers import (
    SessionImporter,
    LangChainImporter,
    CrewAIImporter,
    OpenAIImporter,
)

__version__: str = "0.1.0"

__all__ = [
    "__version__",
    # Session core
    "ContextSegment",
    "EntityReference",
    "SchemaVersionError",
    "SessionManager",
    "SessionNotFoundError",
    "SessionSerializer",
    "SessionState",
    "TaskState",
    "TaskStatus",
    "ToolContext",
    # Storage
    "FilesystemBackend",
    "InMemoryBackend",
    "SQLiteBackend",
    "StorageBackend",
    # Context
    "ContextInjector",
    "ContextSummarizer",
    "DecayCurve",
    "FreshnessDecay",
    "InjectionConfig",
    "RelevanceScorer",
    # Entity
    "Entity",
    "EntityExtractor",
    "EntityLinker",
    "EntityTracker",
    "TrackedEntity",
    # Middleware
    "CheckpointManager",
    "CheckpointRecord",
    "ContextWindowManager",
    "SessionMiddleware",
    # Linking
    "LinkedSession",
    "SessionChain",
    "SessionLinker",
    # Portable / USF
    "USFVersion",
    "USFMessage",
    "USFEntity",
    "USFTaskState",
    "UniversalSession",
    "SessionExporter",
    "LangChainExporter",
    "CrewAIExporter",
    "OpenAIExporter",
    "SessionImporter",
    "LangChainImporter",
    "CrewAIImporter",
    "OpenAIImporter",
]
