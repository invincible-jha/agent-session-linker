"""Tests for agent_session_linker.portability.usf_enhanced."""
from __future__ import annotations

import pytest

from agent_session_linker.portable.usf import UniversalSession
from agent_session_linker.portability.usf_enhanced import (
    EnhancedCrewAIImporter,
    EnhancedLangChainImporter,
    SessionPortabilityKit,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _langchain_data(with_tool_calls: bool = False) -> dict:
    msgs = [
        {"type": "human", "content": "Deploy service X."},
        {"type": "ai", "content": "Starting deployment."},
    ]
    if with_tool_calls:
        msgs.append({
            "type": "ai",
            "content": "",
            "additional_kwargs": {
                "function_call": {"name": "run_deploy_script", "arguments": "{}"}
            },
        })
    return {
        "messages": msgs,
        "memory_variables": {"history": "short"},
        "input_variables": {"task": "deploy"},
    }


def _crewai_data(with_agents: bool = True) -> dict:
    data: dict = {
        "crew_name": "DevOps Crew",
        "crew_id": "crew-001",
        "context": {
            "session_id": "sess-xyz",
            "messages": [
                {"role": "user", "content": "Deploy to prod."},
            ],
            "working_memory": {"environment": "production"},
            "entities": [
                {"name": "prod-env", "entity_type": "environment", "value": "prod", "confidence": 1.0}
            ],
        },
        "task_results": [
            {
                "task_id": "task-1",
                "status": "completed",
                "progress": 1.0,
                "result": "Deployed",
                "agent_id": "agent-deploy",
            }
        ],
    }
    if with_agents:
        data["agents"] = [
            {"name": "deploy-agent", "role": "deployer"},
            {"name": "monitor-agent", "role": "monitor"},
        ]
    return data


# ===========================================================================
# EnhancedLangChainImporter
# ===========================================================================


class TestEnhancedLangChainImporter:
    def test_returns_universal_session(self) -> None:
        importer = EnhancedLangChainImporter()
        session = importer.import_session(_langchain_data())
        assert isinstance(session, UniversalSession)

    def test_framework_source_langchain_enhanced(self) -> None:
        importer = EnhancedLangChainImporter()
        session = importer.import_session(_langchain_data())
        assert session.framework_source == "langchain_enhanced"

    def test_messages_imported(self) -> None:
        importer = EnhancedLangChainImporter()
        session = importer.import_session(_langchain_data())
        assert len(session.messages) == 2

    def test_working_memory_from_memory_variables(self) -> None:
        importer = EnhancedLangChainImporter()
        session = importer.import_session(_langchain_data())
        assert session.working_memory.get("history") == "short"

    def test_input_variables_merged_into_working_memory(self) -> None:
        importer = EnhancedLangChainImporter(enrich_working_memory=True)
        session = importer.import_session(_langchain_data())
        assert session.working_memory.get("task") == "deploy"

    def test_input_variables_not_merged_when_disabled(self) -> None:
        importer = EnhancedLangChainImporter(enrich_working_memory=False)
        session = importer.import_session(_langchain_data())
        assert "task" not in session.working_memory

    def test_tool_call_extracted_as_task_state(self) -> None:
        importer = EnhancedLangChainImporter(extract_tool_calls=True)
        session = importer.import_session(_langchain_data(with_tool_calls=True))
        assert len(session.task_state) == 1
        assert session.task_state[0].result == "run_deploy_script"
        assert session.task_state[0].status == "completed"

    def test_tool_call_not_extracted_when_disabled(self) -> None:
        importer = EnhancedLangChainImporter(extract_tool_calls=False)
        session = importer.import_session(_langchain_data(with_tool_calls=True))
        assert len(session.task_state) == 0

    def test_metadata_includes_base_framework(self) -> None:
        importer = EnhancedLangChainImporter()
        session = importer.import_session(_langchain_data())
        assert session.metadata.get("base_framework") == "langchain"

    def test_tool_calls_list_style(self) -> None:
        data = {
            "messages": [{
                "type": "ai",
                "content": "",
                "additional_kwargs": {
                    "tool_calls": [
                        {"id": "call_1", "function": {"name": "search_web"}},
                        {"id": "call_2", "function": {"name": "write_file"}},
                    ]
                }
            }]
        }
        importer = EnhancedLangChainImporter(extract_tool_calls=True)
        session = importer.import_session(data)
        assert len(session.task_state) == 2
        results = {t.result for t in session.task_state}
        assert "search_web" in results
        assert "write_file" in results

    def test_no_messages_returns_empty(self) -> None:
        importer = EnhancedLangChainImporter()
        session = importer.import_session({})
        assert len(session.messages) == 0


# ===========================================================================
# EnhancedCrewAIImporter
# ===========================================================================


class TestEnhancedCrewAIImporter:
    def test_returns_universal_session(self) -> None:
        importer = EnhancedCrewAIImporter()
        session = importer.import_session(_crewai_data())
        assert isinstance(session, UniversalSession)

    def test_framework_source_crewai_enhanced(self) -> None:
        importer = EnhancedCrewAIImporter()
        session = importer.import_session(_crewai_data())
        assert session.framework_source == "crewai_enhanced"

    def test_session_id_preserved(self) -> None:
        importer = EnhancedCrewAIImporter()
        session = importer.import_session(_crewai_data())
        assert session.session_id == "sess-xyz"

    def test_messages_imported(self) -> None:
        importer = EnhancedCrewAIImporter()
        session = importer.import_session(_crewai_data())
        assert len(session.messages) == 1

    def test_crew_name_in_metadata(self) -> None:
        importer = EnhancedCrewAIImporter(include_crew_metadata=True)
        session = importer.import_session(_crewai_data())
        assert session.metadata.get("crew_name") == "DevOps Crew"

    def test_crew_id_in_metadata(self) -> None:
        importer = EnhancedCrewAIImporter(include_crew_metadata=True)
        session = importer.import_session(_crewai_data())
        assert session.metadata.get("crew_id") == "crew-001"

    def test_crew_metadata_excluded_when_disabled(self) -> None:
        importer = EnhancedCrewAIImporter(include_crew_metadata=False)
        session = importer.import_session(_crewai_data())
        assert "crew_name" not in session.metadata

    def test_agents_mapped_to_entities(self) -> None:
        importer = EnhancedCrewAIImporter(map_agents_to_entities=True)
        session = importer.import_session(_crewai_data(with_agents=True))
        agent_entities = [e for e in session.entities if e.entity_type == "agent"]
        assert len(agent_entities) == 2

    def test_agent_entity_names_correct(self) -> None:
        importer = EnhancedCrewAIImporter(map_agents_to_entities=True)
        session = importer.import_session(_crewai_data(with_agents=True))
        names = {e.name for e in session.entities if e.entity_type == "agent"}
        assert "deploy-agent" in names
        assert "monitor-agent" in names

    def test_agents_not_mapped_when_disabled(self) -> None:
        importer = EnhancedCrewAIImporter(map_agents_to_entities=False)
        session = importer.import_session(_crewai_data(with_agents=True))
        agent_entities = [e for e in session.entities if e.entity_type == "agent"]
        assert len(agent_entities) == 0

    def test_context_entities_preserved(self) -> None:
        importer = EnhancedCrewAIImporter(map_agents_to_entities=False)
        session = importer.import_session(_crewai_data())
        env_entities = [e for e in session.entities if e.entity_type == "environment"]
        assert len(env_entities) == 1

    def test_task_agent_ids_in_working_memory(self) -> None:
        importer = EnhancedCrewAIImporter()
        session = importer.import_session(_crewai_data())
        assert "task_agent_ids" in session.working_memory
        assert "agent-deploy" in session.working_memory["task_agent_ids"]

    def test_task_state_imported(self) -> None:
        importer = EnhancedCrewAIImporter()
        session = importer.import_session(_crewai_data())
        assert len(session.task_state) == 1
        assert session.task_state[0].status == "completed"

    def test_base_framework_in_metadata(self) -> None:
        importer = EnhancedCrewAIImporter()
        session = importer.import_session(_crewai_data())
        assert session.metadata.get("base_framework") == "crewai"


# ===========================================================================
# SessionPortabilityKit
# ===========================================================================


class TestSessionPortabilityKit:
    def test_enhanced_by_default(self) -> None:
        kit = SessionPortabilityKit()
        assert kit.is_enhanced is True

    def test_non_enhanced(self) -> None:
        kit = SessionPortabilityKit(enhanced=False)
        assert kit.is_enhanced is False

    def test_import_langchain(self) -> None:
        kit = SessionPortabilityKit()
        session = kit.import_langchain(_langchain_data())
        assert isinstance(session, UniversalSession)
        assert session.framework_source == "langchain_enhanced"

    def test_import_crewai(self) -> None:
        kit = SessionPortabilityKit()
        session = kit.import_crewai(_crewai_data())
        assert isinstance(session, UniversalSession)
        assert session.framework_source == "crewai_enhanced"

    def test_export_to_json(self) -> None:
        kit = SessionPortabilityKit()
        session = kit.import_langchain(_langchain_data())
        json_str = kit.export_to_json(session)
        assert isinstance(json_str, str)
        assert "langchain_enhanced" in json_str

    def test_import_json_roundtrip(self) -> None:
        kit = SessionPortabilityKit()
        session = kit.import_langchain(_langchain_data())
        json_str = kit.export_to_json(session)
        restored = SessionPortabilityKit.import_json(json_str)
        assert restored.session_id == session.session_id

    def test_base_import_langchain(self) -> None:
        kit = SessionPortabilityKit(enhanced=False)
        session = kit.import_langchain(_langchain_data())
        assert session.framework_source == "langchain"

    def test_base_import_crewai(self) -> None:
        kit = SessionPortabilityKit(enhanced=False)
        session = kit.import_crewai(_crewai_data())
        assert session.framework_source == "crewai"
