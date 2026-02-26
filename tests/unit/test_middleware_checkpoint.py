"""Unit tests for agent_session_linker.middleware.checkpoint.

Uses InMemoryBackend so no disk I/O is required.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agent_session_linker.middleware.checkpoint import (
    CheckpointManager,
    CheckpointRecord,
    _build_checkpoint_key,
)
from agent_session_linker.session.manager import SessionManager
from agent_session_linker.session.state import SessionState
from agent_session_linker.storage.memory import InMemoryBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend() -> InMemoryBackend:
    return InMemoryBackend()


@pytest.fixture()
def manager(backend: InMemoryBackend) -> SessionManager:
    return SessionManager(backend=backend, default_agent_id="test-agent")


@pytest.fixture()
def checkpoint_manager(
    backend: InMemoryBackend, manager: SessionManager
) -> CheckpointManager:
    return CheckpointManager(backend=backend, manager=manager)


@pytest.fixture()
def session(manager: SessionManager) -> SessionState:
    s = manager.create_session()
    s.add_segment("user", "hello", token_count=10)
    s.add_segment("assistant", "world", token_count=20)
    manager.save_session(s)
    return s


# ---------------------------------------------------------------------------
# _build_checkpoint_key helper
# ---------------------------------------------------------------------------


class TestBuildCheckpointKey:
    def test_contains_session_id(self) -> None:
        key = _build_checkpoint_key("session-abc", 0)
        assert "session-abc" in key

    def test_contains_zero_padded_sequence(self) -> None:
        key = _build_checkpoint_key("s", 3)
        assert "0003" in key

    def test_distinct_keys_for_different_sequences(self) -> None:
        k0 = _build_checkpoint_key("s", 0)
        k1 = _build_checkpoint_key("s", 1)
        assert k0 != k1


# ---------------------------------------------------------------------------
# CheckpointRecord serialisation
# ---------------------------------------------------------------------------


class TestCheckpointRecord:
    def test_to_dict_roundtrip(self) -> None:
        now = datetime.now(timezone.utc)
        record = CheckpointRecord(
            checkpoint_id="cp-id",
            session_id="sess-id",
            label="my-label",
            created_at=now,
            segment_count=3,
            token_count=100,
        )
        data = record.to_dict()
        restored = CheckpointRecord.from_dict(data)
        assert restored.checkpoint_id == record.checkpoint_id
        assert restored.session_id == record.session_id
        assert restored.label == record.label
        assert restored.segment_count == 3
        assert restored.token_count == 100

    def test_from_dict_parses_iso_timestamp(self) -> None:
        now = datetime.now(timezone.utc)
        data = {
            "checkpoint_id": "cp",
            "session_id": "s",
            "label": "lbl",
            "created_at": now.isoformat(),
            "segment_count": "2",
            "token_count": "50",
        }
        record = CheckpointRecord.from_dict(data)
        assert isinstance(record.created_at, datetime)


# ---------------------------------------------------------------------------
# CheckpointManager.create_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointManagerCreate:
    def test_returns_checkpoint_record(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session, label="v1")
        assert isinstance(record, CheckpointRecord)

    def test_record_has_correct_session_id(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        assert record.session_id == session.session_id

    def test_record_has_correct_segment_count(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        assert record.segment_count == 2

    def test_record_has_correct_token_count(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        assert record.token_count == 30

    def test_label_defaults_to_iso_timestamp(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        # ISO timestamp contains "T" and "+" or "Z".
        assert "T" in record.label or "-" in record.label

    def test_custom_label_stored(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session, label="before-refactor")
        assert record.label == "before-refactor"

    def test_checkpoint_data_persisted_in_backend(
        self,
        checkpoint_manager: CheckpointManager,
        session: SessionState,
        backend: InMemoryBackend,
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        assert backend.exists(record.checkpoint_id)

    def test_sequential_checkpoints_have_distinct_ids(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        r1 = checkpoint_manager.create_checkpoint(session, label="first")
        r2 = checkpoint_manager.create_checkpoint(session, label="second")
        assert r1.checkpoint_id != r2.checkpoint_id

    def test_max_checkpoints_evicts_oldest(
        self, backend: InMemoryBackend, manager: SessionManager, session: SessionState
    ) -> None:
        cp_manager = CheckpointManager(
            backend=backend, manager=manager, max_checkpoints_per_session=2
        )
        r1 = cp_manager.create_checkpoint(session, label="r1")
        cp_manager.create_checkpoint(session, label="r2")
        cp_manager.create_checkpoint(session, label="r3")
        # r1 should have been evicted — its key is gone.
        assert not backend.exists(r1.checkpoint_id)

    def test_index_updated_after_create(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        checkpoint_manager.create_checkpoint(session)
        records = checkpoint_manager.list_checkpoints(session.session_id)
        assert len(records) == 1


# ---------------------------------------------------------------------------
# CheckpointManager.restore_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointManagerRestore:
    def test_restore_returns_session_state(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        restored = checkpoint_manager.restore_checkpoint(record.checkpoint_id)
        assert isinstance(restored, SessionState)

    def test_restored_session_has_correct_id(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        restored = checkpoint_manager.restore_checkpoint(record.checkpoint_id)
        assert restored.session_id == session.session_id

    def test_restored_session_has_same_segments(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        restored = checkpoint_manager.restore_checkpoint(record.checkpoint_id)
        assert len(restored.segments) == 2

    def test_restore_missing_checkpoint_raises_key_error(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        with pytest.raises(KeyError, match="ghost-checkpoint"):
            checkpoint_manager.restore_checkpoint("ghost-checkpoint")


# ---------------------------------------------------------------------------
# CheckpointManager.list_checkpoints
# ---------------------------------------------------------------------------


class TestCheckpointManagerList:
    def test_list_empty_for_unknown_session(
        self, checkpoint_manager: CheckpointManager
    ) -> None:
        assert checkpoint_manager.list_checkpoints("unknown-session") == []

    def test_list_returns_records_in_creation_order(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        r1 = checkpoint_manager.create_checkpoint(session, label="first")
        r2 = checkpoint_manager.create_checkpoint(session, label="second")
        records = checkpoint_manager.list_checkpoints(session.session_id)
        assert records[0].checkpoint_id == r1.checkpoint_id
        assert records[1].checkpoint_id == r2.checkpoint_id

    def test_list_count_matches_creates(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        for i in range(3):
            checkpoint_manager.create_checkpoint(session, label=f"v{i}")
        records = checkpoint_manager.list_checkpoints(session.session_id)
        assert len(records) == 3


# ---------------------------------------------------------------------------
# CheckpointManager.delete_checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointManagerDelete:
    def test_delete_removes_from_backend(
        self,
        checkpoint_manager: CheckpointManager,
        session: SessionState,
        backend: InMemoryBackend,
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        checkpoint_manager.delete_checkpoint(record.checkpoint_id, session.session_id)
        assert not backend.exists(record.checkpoint_id)

    def test_delete_updates_index(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        record = checkpoint_manager.create_checkpoint(session)
        checkpoint_manager.delete_checkpoint(record.checkpoint_id, session.session_id)
        assert checkpoint_manager.list_checkpoints(session.session_id) == []

    def test_delete_missing_raises_key_error(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        with pytest.raises(KeyError):
            checkpoint_manager.delete_checkpoint("nonexistent", session.session_id)

    def test_delete_only_removes_target(
        self, checkpoint_manager: CheckpointManager, session: SessionState
    ) -> None:
        r1 = checkpoint_manager.create_checkpoint(session, label="keep")
        r2 = checkpoint_manager.create_checkpoint(session, label="remove")
        checkpoint_manager.delete_checkpoint(r2.checkpoint_id, session.session_id)
        remaining = checkpoint_manager.list_checkpoints(session.session_id)
        assert any(r.checkpoint_id == r1.checkpoint_id for r in remaining)
        assert not any(r.checkpoint_id == r2.checkpoint_id for r in remaining)
