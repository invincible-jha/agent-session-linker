"""Enhanced USF session portability — additional framework adapters.

Extends the base importers/exporters with enhanced versions that handle
edge cases, add metadata enrichment, and support streaming partial imports.

Classes
-------
EnhancedLangChainImporter
    LangChain importer with tool-call extraction and memory enrichment.
EnhancedCrewAIImporter
    CrewAI importer with crew metadata and agent-role mapping.
SessionPortabilityKit
    Convenience wrapper combining all importers and exporters.
"""
from __future__ import annotations

from agent_session_linker.portability.usf_enhanced import (
    EnhancedCrewAIImporter,
    EnhancedLangChainImporter,
    SessionPortabilityKit,
)

__all__ = [
    "EnhancedCrewAIImporter",
    "EnhancedLangChainImporter",
    "SessionPortabilityKit",
]
