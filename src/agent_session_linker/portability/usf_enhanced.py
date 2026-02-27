"""Enhanced USF portability — framework adapters with richer metadata.

This module extends the base importers in ``portable.importers`` with
enhanced versions that:

- Extract tool-call data from LangChain ``additional_kwargs``
- Map CrewAI agent roles to USF roles with crew-level metadata
- Provide a :class:`SessionPortabilityKit` convenience wrapper

Design
------
Each enhanced importer delegates core parsing to the base importer then
applies enrichment passes.  No proprietary logic is added — all
enrichments are standard transformations on the imported data.

Usage
-----
::

    from agent_session_linker.portability import EnhancedLangChainImporter

    importer = EnhancedLangChainImporter(extract_tool_calls=True)
    session = importer.import_session(langchain_memory_dict)
    print(session.framework_source)    # "langchain_enhanced"
    print(len(session.task_state))     # tool calls mapped to task states
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from agent_session_linker.portable.importers import (
    CrewAIImporter,
    LangChainImporter,
    _langchain_type_to_usf_role,
    _parse_timestamp,
    _utc_now,
)
from agent_session_linker.portable.usf import (
    USFEntity,
    USFMessage,
    USFTaskState,
    USFVersion,
    UniversalSession,
)


# ---------------------------------------------------------------------------
# EnhancedLangChainImporter
# ---------------------------------------------------------------------------


class EnhancedLangChainImporter:
    """Enhanced LangChain importer with tool-call extraction.

    Extends the base :class:`~agent_session_linker.portable.importers.LangChainImporter`
    to extract tool invocations from ``additional_kwargs.function_call`` or
    ``additional_kwargs.tool_calls`` and represent them as
    :class:`~agent_session_linker.portable.usf.USFTaskState` entries.

    Parameters
    ----------
    extract_tool_calls:
        If True (default), tool-call data in ``additional_kwargs`` is
        extracted into task states.
    enrich_working_memory:
        If True (default), any ``input_variables`` key found in the root
        dict is merged into ``working_memory``.
    """

    def __init__(
        self,
        extract_tool_calls: bool = True,
        enrich_working_memory: bool = True,
    ) -> None:
        self._extract_tool_calls = extract_tool_calls
        self._enrich_working_memory = enrich_working_memory
        self._base = LangChainImporter()

    def import_session(self, data: dict[str, object]) -> UniversalSession:
        """Convert a LangChain memory dict to an enhanced :class:`UniversalSession`.

        Parameters
        ----------
        data:
            LangChain memory dict (same structure accepted by
            :class:`~agent_session_linker.portable.importers.LangChainImporter`).

        Returns
        -------
        UniversalSession
            With ``framework_source="langchain_enhanced"`` and optional
            enrichment.
        """
        # Base import
        session = self._base.import_session(data)

        # Enrich working memory with input_variables
        working_memory = dict(session.working_memory)
        if self._enrich_working_memory:
            input_vars: dict[str, object] = dict(data.get("input_variables") or {})
            working_memory.update(input_vars)

        # Extract tool calls -> task states
        task_state: list[USFTaskState] = list(session.task_state)
        if self._extract_tool_calls:
            raw_messages: list[object] = data.get("messages") or []
            for entry in raw_messages:
                if not isinstance(entry, dict):
                    continue
                additional = entry.get("additional_kwargs") or {}
                # OpenAI function_call style
                func_call = additional.get("function_call")
                if func_call and isinstance(func_call, dict):
                    task_state.append(
                        USFTaskState(
                            task_id=str(uuid4()),
                            status="completed",
                            progress=1.0,
                            result=str(func_call.get("name", "unknown_function")),
                        )
                    )
                # OpenAI tool_calls style (list)
                tool_calls = additional.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if isinstance(tc, dict):
                            task_state.append(
                                USFTaskState(
                                    task_id=tc.get("id") or str(uuid4()),
                                    status="completed",
                                    progress=1.0,
                                    result=str(
                                        (tc.get("function") or {}).get("name", "unknown_tool")
                                    ),
                                )
                            )

        return UniversalSession(
            version=session.version,
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            framework_source="langchain_enhanced",
            messages=session.messages,
            working_memory=working_memory,
            entities=session.entities,
            task_state=task_state,
            metadata={"base_framework": "langchain"},
        )


# ---------------------------------------------------------------------------
# EnhancedCrewAIImporter
# ---------------------------------------------------------------------------


class EnhancedCrewAIImporter:
    """Enhanced CrewAI importer with crew metadata and agent-role mapping.

    Extends the base :class:`~agent_session_linker.portable.importers.CrewAIImporter`
    to:

    - Map crew-level metadata (``crew_name``, ``crew_id``) into session
      metadata.
    - Normalise agent_id fields from ``task_results`` into working_memory.
    - Extract crew ``agents`` list into :class:`USFEntity` entries.

    Parameters
    ----------
    include_crew_metadata:
        If True (default), crew-level metadata is captured.
    map_agents_to_entities:
        If True (default), each agent in the crew is added as a USFEntity
        with ``entity_type="agent"``.
    """

    def __init__(
        self,
        include_crew_metadata: bool = True,
        map_agents_to_entities: bool = True,
    ) -> None:
        self._include_crew_metadata = include_crew_metadata
        self._map_agents_to_entities = map_agents_to_entities
        self._base = CrewAIImporter()

    def import_session(self, data: dict[str, object]) -> UniversalSession:
        """Convert a CrewAI context dict to an enhanced :class:`UniversalSession`.

        Parameters
        ----------
        data:
            CrewAI context dict.  The top-level may include:

            - ``"crew_name"`` (str): Name of the crew.
            - ``"crew_id"`` (str): Unique crew identifier.
            - ``"agents"`` (list of dicts): Agent definitions.
            - ``"context"`` (dict): As consumed by the base importer.
            - ``"task_results"`` (list): As consumed by the base importer.

        Returns
        -------
        UniversalSession
            With ``framework_source="crewai_enhanced"``.
        """
        session = self._base.import_session(data)

        # Crew metadata
        metadata: dict[str, object] = {"base_framework": "crewai"}
        if self._include_crew_metadata:
            if crew_name := data.get("crew_name"):
                metadata["crew_name"] = str(crew_name)
            if crew_id := data.get("crew_id"):
                metadata["crew_id"] = str(crew_id)

        # Agent entities
        entities: list[USFEntity] = list(session.entities)
        if self._map_agents_to_entities:
            agents: list[object] = data.get("agents") or []
            for agent in agents:
                if not isinstance(agent, dict):
                    continue
                agent_name = str(agent.get("name") or agent.get("role") or "unknown_agent")
                entity = USFEntity(
                    name=agent_name,
                    entity_type="agent",
                    value=str(agent.get("role") or ""),
                    confidence=1.0,
                )
                # Avoid duplicates
                existing_names = {e.name for e in entities}
                if agent_name not in existing_names:
                    entities.append(entity)

        # Enrich working_memory with agent_id from task_results
        working_memory: dict[str, object] = dict(session.working_memory)
        task_results: list[object] = data.get("task_results") or []
        agent_ids: list[str] = []
        for tr in task_results:
            if isinstance(tr, dict) and (aid := tr.get("agent_id")):
                agent_ids.append(str(aid))
        if agent_ids:
            working_memory["task_agent_ids"] = agent_ids

        return UniversalSession(
            version=session.version,
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=session.updated_at,
            framework_source="crewai_enhanced",
            messages=session.messages,
            working_memory=working_memory,
            entities=entities,
            task_state=session.task_state,
            metadata=metadata,
        )


# ---------------------------------------------------------------------------
# SessionPortabilityKit
# ---------------------------------------------------------------------------


class SessionPortabilityKit:
    """Convenience wrapper providing all importers and exporters in one place.

    Parameters
    ----------
    enhanced:
        If True (default), use enhanced importers.  Otherwise use base
        importers.

    Example
    -------
    ::

        kit = SessionPortabilityKit()
        session = kit.import_langchain(langchain_dict)
        session2 = kit.import_crewai(crewai_dict)
        json_str = session.to_json()
    """

    def __init__(self, enhanced: bool = True) -> None:
        self._enhanced = enhanced
        if enhanced:
            self._langchain_importer: object = EnhancedLangChainImporter()
            self._crewai_importer: object = EnhancedCrewAIImporter()
        else:
            self._langchain_importer = LangChainImporter()
            self._crewai_importer = CrewAIImporter()

    @property
    def is_enhanced(self) -> bool:
        """True when enhanced importers are active."""
        return self._enhanced

    def import_langchain(self, data: dict[str, object]) -> UniversalSession:
        """Import a LangChain memory dict.

        Parameters
        ----------
        data:
            LangChain memory dict.

        Returns
        -------
        UniversalSession
        """
        return self._langchain_importer.import_session(data)

    def import_crewai(self, data: dict[str, object]) -> UniversalSession:
        """Import a CrewAI context dict.

        Parameters
        ----------
        data:
            CrewAI context dict.

        Returns
        -------
        UniversalSession
        """
        return self._crewai_importer.import_session(data)

    def export_to_json(self, session: UniversalSession) -> str:
        """Export a :class:`UniversalSession` to JSON.

        Parameters
        ----------
        session:
            The session to export.

        Returns
        -------
        str
            JSON representation.
        """
        return session.to_json()

    @staticmethod
    def import_json(json_str: str) -> UniversalSession:
        """Import a USF session from a JSON string.

        Parameters
        ----------
        json_str:
            JSON string previously produced by :meth:`export_to_json`.

        Returns
        -------
        UniversalSession
        """
        return UniversalSession.from_json(json_str)
