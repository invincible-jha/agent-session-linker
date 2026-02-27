"""Universal Session Format (USF) — cross-framework session portability.

This package implements Phase 6B of the AumOS implementation plan.  It provides
a vendor-neutral interchange format (USF) for agent session data, together with
exporters and importers for the three most common agent frameworks.

Public surface
--------------
- :class:`UniversalSession`    — the canonical USF session model (Pydantic v2)
- :class:`USFMessage`          — a single conversation message
- :class:`USFEntity`           — a named entity captured in a session
- :class:`USFTaskState`        — a tracked task within a session
- :data:`USFVersion`           — current format version string

Exporters
---------
- :class:`LangChainExporter`
- :class:`CrewAIExporter`
- :class:`OpenAIExporter`

Importers
---------
- :class:`LangChainImporter`
- :class:`CrewAIImporter`
- :class:`OpenAIImporter`
"""
from __future__ import annotations

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

__all__ = [
    # USF core
    "USFVersion",
    "USFMessage",
    "USFEntity",
    "USFTaskState",
    "UniversalSession",
    # Exporters
    "SessionExporter",
    "LangChainExporter",
    "CrewAIExporter",
    "OpenAIExporter",
    # Importers
    "SessionImporter",
    "LangChainImporter",
    "CrewAIImporter",
    "OpenAIImporter",
]
