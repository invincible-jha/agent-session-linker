"""Tests for agent_session_linker.branching.branch_manager."""
from __future__ import annotations

import threading

import pytest

from agent_session_linker.branching.branch_manager import (
    BranchConfig,
    BranchManager,
    SessionBranch,
)
from agent_session_linker.session.state import SessionState, TaskStatus


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_session(agent_id: str = "agent_x") -> SessionState:
    s = SessionState(agent_id=agent_id)
    s.add_segment("user", "First message")
    s.add_segment("assistant", "Response one")
    s.add_segment("user", "Second message")
    s.add_task(title="Task A")
    s.track_entity("project_x", entity_type="project")
    s.preferences["tone"] = "formal"
    s.summary = "Summary of session."
    return s


def _manager(parent_id: str = "parent-session-001") -> BranchManager:
    return BranchManager(parent_session_id=parent_id)


# ===========================================================================
# BranchConfig
# ===========================================================================


class TestBranchConfig:
    def test_defaults(self) -> None:
        cfg = BranchConfig()
        assert cfg.copy_segments is True
        assert cfg.copy_tasks is True
        assert cfg.copy_entities is True
        assert cfg.copy_preferences is True
        assert cfg.max_segments is None
        assert cfg.branch_label == ""

    def test_frozen(self) -> None:
        cfg = BranchConfig()
        with pytest.raises(Exception):
            cfg.copy_segments = False  # type: ignore[misc]

    def test_negative_max_segments_raises(self) -> None:
        with pytest.raises(ValueError, match="max_segments"):
            BranchConfig(max_segments=-1)


# ===========================================================================
# BranchManager — construction
# ===========================================================================


class TestBranchManagerConstruction:
    def test_empty_parent_id_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_session_id"):
            BranchManager(parent_session_id="")

    def test_initial_branch_count_zero(self) -> None:
        mgr = _manager()
        assert mgr.branch_count == 0

    def test_parent_session_id_stored(self) -> None:
        mgr = BranchManager(parent_session_id="sess-999")
        assert mgr.parent_session_id == "sess-999"


# ===========================================================================
# BranchManager — create_branch
# ===========================================================================


class TestCreateBranch:
    def test_creates_branch(self) -> None:
        mgr = _manager()
        session = _make_session()
        branch = mgr.create_branch(session, branch_name="variant_a")
        assert isinstance(branch, SessionBranch)
        assert "variant_a" in mgr

    def test_branch_count_increments(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "a")
        mgr.create_branch(session, "b")
        assert mgr.branch_count == 2

    def test_duplicate_name_raises(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "variant_a")
        with pytest.raises(ValueError, match="already exists"):
            mgr.create_branch(session, "variant_a")

    def test_empty_branch_name_raises(self) -> None:
        mgr = _manager()
        with pytest.raises(ValueError, match="branch_name"):
            mgr.create_branch(_make_session(), branch_name="")

    def test_branch_parent_session_id(self) -> None:
        mgr = BranchManager(parent_session_id="parent-abc")
        branch = mgr.create_branch(_make_session(), "v1")
        assert branch.parent_session_id == "parent-abc"

    def test_segments_copied(self) -> None:
        mgr = _manager()
        session = _make_session()
        branch = mgr.create_branch(session, "v1")
        assert branch.segment_count == 3

    def test_tasks_copied(self) -> None:
        mgr = _manager()
        session = _make_session()
        branch = mgr.create_branch(session, "v1")
        assert branch.task_count == 1

    def test_entities_copied(self) -> None:
        mgr = _manager()
        session = _make_session()
        branch = mgr.create_branch(session, "v1")
        assert len(branch.session.entities) == 1

    def test_preferences_copied(self) -> None:
        mgr = _manager()
        session = _make_session()
        branch = mgr.create_branch(session, "v1")
        assert branch.session.preferences.get("tone") == "formal"

    def test_deep_copy_independence(self) -> None:
        """Mutating branch segments must not affect the parent session."""
        mgr = _manager()
        session = _make_session()
        branch = mgr.create_branch(session, "v1")
        branch.add_divergent_segment("user", "Branch-only message")
        assert len(session.segments) == 3  # parent unchanged
        assert branch.segment_count == 4

    def test_max_segments_config(self) -> None:
        mgr = _manager()
        session = _make_session()
        config = BranchConfig(max_segments=2)
        branch = mgr.create_branch(session, "v1", config=config)
        assert branch.segment_count == 2

    def test_copy_tasks_false(self) -> None:
        mgr = _manager()
        session = _make_session()
        config = BranchConfig(copy_tasks=False)
        branch = mgr.create_branch(session, "v1", config=config)
        assert branch.task_count == 0

    def test_copy_entities_false(self) -> None:
        mgr = _manager()
        session = _make_session()
        config = BranchConfig(copy_entities=False)
        branch = mgr.create_branch(session, "v1", config=config)
        assert len(branch.session.entities) == 0

    def test_copy_preferences_false(self) -> None:
        mgr = _manager()
        session = _make_session()
        config = BranchConfig(copy_preferences=False)
        branch = mgr.create_branch(session, "v1", config=config)
        assert branch.session.preferences == {}

    def test_metadata_passed(self) -> None:
        mgr = _manager()
        branch = mgr.create_branch(
            _make_session(), "v1", metadata={"model": "gpt-4o"}
        )
        assert branch.metadata["model"] == "gpt-4o"


# ===========================================================================
# BranchManager — queries
# ===========================================================================


class TestBranchManagerQueries:
    def test_get_branch_returns_branch(self) -> None:
        mgr = _manager()
        mgr.create_branch(_make_session(), "v1")
        branch = mgr.get_branch("v1")
        assert branch.branch_name == "v1"

    def test_get_branch_unknown_raises(self) -> None:
        mgr = _manager()
        with pytest.raises(KeyError):
            mgr.get_branch("nonexistent")

    def test_list_branch_names_sorted(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "beta")
        mgr.create_branch(session, "alpha")
        mgr.create_branch(session, "gamma")
        assert mgr.list_branch_names() == ["alpha", "beta", "gamma"]

    def test_list_branches_returns_all(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "a")
        mgr.create_branch(session, "b")
        branches = mgr.list_branches()
        assert len(branches) == 2

    def test_delete_branch_returns_true(self) -> None:
        mgr = _manager()
        mgr.create_branch(_make_session(), "v1")
        result = mgr.delete_branch("v1")
        assert result is True
        assert "v1" not in mgr

    def test_delete_nonexistent_returns_false(self) -> None:
        mgr = _manager()
        assert mgr.delete_branch("nonexistent") is False

    def test_clear_removes_all(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "a")
        mgr.create_branch(session, "b")
        mgr.clear()
        assert mgr.branch_count == 0

    def test_len(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "a")
        mgr.create_branch(session, "b")
        assert len(mgr) == 2


# ===========================================================================
# BranchManager — comparison
# ===========================================================================


class TestBranchComparison:
    def test_compare_segment_counts(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "a")
        mgr.create_branch(session, "b", config=BranchConfig(max_segments=1))
        counts = mgr.compare_segment_counts()
        assert counts["a"] == 3
        assert counts["b"] == 1

    def test_compare_task_counts(self) -> None:
        mgr = _manager()
        session = _make_session()
        mgr.create_branch(session, "with_tasks")
        mgr.create_branch(session, "no_tasks", config=BranchConfig(copy_tasks=False))
        counts = mgr.compare_task_counts()
        assert counts["with_tasks"] == 1
        assert counts["no_tasks"] == 0


# ===========================================================================
# SessionBranch helpers
# ===========================================================================


class TestSessionBranch:
    def _make_branch(self) -> SessionBranch:
        mgr = _manager()
        session = _make_session()
        return mgr.create_branch(session, "test_branch")

    def test_summary_line_contains_name(self) -> None:
        branch = self._make_branch()
        assert "test_branch" in branch.summary_line()

    def test_add_divergent_segment_increases_count(self) -> None:
        branch = self._make_branch()
        before = branch.segment_count
        branch.add_divergent_segment("user", "Extra context in this branch")
        assert branch.segment_count == before + 1

    def test_pending_task_count(self) -> None:
        branch = self._make_branch()
        # All tasks default to PENDING
        assert branch.pending_task_count() == 1


# ===========================================================================
# Thread safety
# ===========================================================================


class TestThreadSafety:
    def test_concurrent_create_branch(self) -> None:
        mgr = _manager()
        session = _make_session()
        errors: list[Exception] = []

        def create(name: str) -> None:
            try:
                mgr.create_branch(session, name)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=create, args=(f"branch_{i}",))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert mgr.branch_count == 20
