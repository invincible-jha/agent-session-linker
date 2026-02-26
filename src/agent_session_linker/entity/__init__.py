"""Entity extraction, tracking, and linking subpackage.

Provides regex-based named entity recognition (NER), cross-turn frequency
tracking, and edit-distance entity linking.

Public surface
--------------
- Entity         — dataclass for a single extracted entity
- EntityExtractor — regex NER (PERSON, ORG, EMAIL, URL, DATE, NUMBER, MONEY)
- TrackedEntity  — dataclass for a frequency-tracked entity
- EntityTracker  — aggregate and query entity statistics across turns
- EntityLinker   — link mentions to known entities via edit distance
"""
from __future__ import annotations

from agent_session_linker.entity.extractor import Entity, EntityExtractor
from agent_session_linker.entity.linker import EntityLinker
from agent_session_linker.entity.tracker import EntityTracker, TrackedEntity

__all__ = [
    "Entity",
    "EntityExtractor",
    "EntityLinker",
    "EntityTracker",
    "TrackedEntity",
]
