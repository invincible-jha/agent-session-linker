"""Session branching — fork a session for A/B testing or exploration.

Design
------
A :class:`SessionBranch` is an independent copy of a parent session up to
a specific branch point.  Branches are named (e.g. ``"variant_a"``,
``"variant_b"``) and carry a ``branch_id``, ``parent_session_id``, and
the segments from the parent up to the branch point.

The :class:`BranchManager` coordinates branch creation and comparison.
It stores all branches for a parent session and provides utilities for
querying and merging.

Usage
-----
::

    from agent_session_linker.branching import BranchManager, BranchConfig

    manager = BranchManager(parent_session_id="sess-123")
    branch_a = manager.create_branch(
        source_session=session,
        branch_name="variant_a",
    )
    branch_b = manager.create_branch(
        source_session=session,
        branch_name="variant_b",
        config=BranchConfig(copy_tasks=False),
    )
    print(manager.list_branch_names())
"""
from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from agent_session_linker.session.state import SessionState, TaskStatus


# ---------------------------------------------------------------------------
# BranchConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchConfig:
    """Configuration controlling what is copied when a branch is created.

    Parameters
    ----------
    copy_segments:
        Whether to copy context segments from the parent.  Default True.
    copy_tasks:
        Whether to copy task states.  Default True.
    copy_entities:
        Whether to copy entity references.  Default True.
    copy_preferences:
        Whether to copy preferences.  Default True.
    max_segments:
        Maximum number of segments to copy.  ``None`` copies all.
    branch_label:
        Optional descriptive label for this branch.
    """

    copy_segments: bool = True
    copy_tasks: bool = True
    copy_entities: bool = True
    copy_preferences: bool = True
    max_segments: int | None = None
    branch_label: str = ""

    def __post_init__(self) -> None:
        if self.max_segments is not None and self.max_segments < 0:
            raise ValueError(
                f"max_segments must be non-negative or None, "
                f"got {self.max_segments!r}."
            )


# ---------------------------------------------------------------------------
# SessionBranch
# ---------------------------------------------------------------------------


class SessionBranch(BaseModel):
    """An independent session branch derived from a parent session.

    Parameters
    ----------
    branch_id:
        Unique identifier for this branch.
    branch_name:
        Human-readable name (e.g. ``"variant_a"``).
    parent_session_id:
        The session this branch was forked from.
    branch_label:
        Optional descriptive label.
    session:
        The forked :class:`SessionState` (deep copy of parent at fork time).
    created_at:
        UTC timestamp when the branch was created.
    metadata:
        Arbitrary branch annotations.
    """

    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    branch_id: str = Field(default_factory=lambda: str(uuid4()))
    branch_name: str
    parent_session_id: str
    branch_label: str = ""
    session: SessionState
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, object] = Field(default_factory=dict)

    @property
    def segment_count(self) -> int:
        """Number of segments in this branch."""
        return len(self.session.segments)

    @property
    def task_count(self) -> int:
        """Number of tasks in this branch."""
        return len(self.session.tasks)

    def summary_line(self) -> str:
        """Return a one-line human-readable description."""
        return (
            f"Branch[{self.branch_name}] id={self.branch_id[:8]} "
            f"parent={self.parent_session_id[:8]} "
            f"segments={self.segment_count} tasks={self.task_count}"
        )

    def add_divergent_segment(self, role: str, content: str, **kwargs: object) -> None:
        """Add a segment specific to this branch (does not affect parent).

        Parameters
        ----------
        role:
            Message role.
        content:
            Segment text content.
        **kwargs:
            Additional kwargs forwarded to :meth:`SessionState.add_segment`.
        """
        self.session.add_segment(role, content, **kwargs)

    def pending_task_count(self) -> int:
        """Return the number of PENDING tasks in this branch."""
        return sum(1 for t in self.session.tasks if t.status == TaskStatus.PENDING)


# ---------------------------------------------------------------------------
# BranchManager
# ---------------------------------------------------------------------------


class BranchManager:
    """Create and manage session branches for A/B testing.

    All public methods are thread-safe.

    Parameters
    ----------
    parent_session_id:
        The session ID all branches originate from.

    Example
    -------
    ::

        manager = BranchManager(parent_session_id="sess-abc")
        branch = manager.create_branch(source_session=session, branch_name="control")
        assert manager.branch_count == 1
    """

    def __init__(self, parent_session_id: str) -> None:
        if not parent_session_id.strip():
            raise ValueError("parent_session_id must not be empty.")
        self._parent_session_id = parent_session_id
        self._branches: dict[str, SessionBranch] = {}
        self._lock = threading.Lock()

    @property
    def parent_session_id(self) -> str:
        """The parent session ID."""
        return self._parent_session_id

    @property
    def branch_count(self) -> int:
        """Number of branches currently managed."""
        with self._lock:
            return len(self._branches)

    # ------------------------------------------------------------------
    # Branch creation
    # ------------------------------------------------------------------

    def create_branch(
        self,
        source_session: SessionState,
        branch_name: str,
        config: BranchConfig | None = None,
        metadata: dict[str, object] | None = None,
    ) -> SessionBranch:
        """Fork *source_session* into a new named branch.

        Parameters
        ----------
        source_session:
            The session to fork.  A deep copy is made.
        branch_name:
            Unique name for this branch within this manager.
        config:
            Optional :class:`BranchConfig`.  Defaults to copying everything.
        metadata:
            Optional annotations for the branch.

        Returns
        -------
        SessionBranch

        Raises
        ------
        ValueError
            If *branch_name* already exists or is empty.
        """
        if not branch_name.strip():
            raise ValueError("branch_name must not be empty.")

        cfg = config or BranchConfig()

        with self._lock:
            if branch_name in self._branches:
                raise ValueError(
                    f"Branch {branch_name!r} already exists for session "
                    f"{self._parent_session_id!r}."
                )

            # Build the forked SessionState
            forked = SessionState(
                agent_id=source_session.agent_id,
                parent_session_id=self._parent_session_id,
            )

            # Segments
            if cfg.copy_segments:
                segs = list(source_session.segments)
                if cfg.max_segments is not None:
                    segs = segs[-cfg.max_segments:]
                forked.segments = copy.deepcopy(segs)

            # Tasks
            if cfg.copy_tasks:
                forked.tasks = copy.deepcopy(list(source_session.tasks))

            # Entities
            if cfg.copy_entities:
                forked.entities = copy.deepcopy(list(source_session.entities))

            # Preferences
            if cfg.copy_preferences:
                forked.preferences = dict(source_session.preferences)

            forked.summary = source_session.summary

            branch = SessionBranch(
                branch_name=branch_name,
                parent_session_id=self._parent_session_id,
                branch_label=cfg.branch_label,
                session=forked,
                metadata=metadata or {},
            )
            self._branches[branch_name] = branch

        return branch

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_branch(self, branch_name: str) -> SessionBranch:
        """Return the branch with *branch_name*.

        Raises
        ------
        KeyError
            If the branch does not exist.
        """
        with self._lock:
            if branch_name not in self._branches:
                raise KeyError(
                    f"Branch {branch_name!r} not found for session "
                    f"{self._parent_session_id!r}."
                )
            return self._branches[branch_name]

    def list_branch_names(self) -> list[str]:
        """Return sorted list of branch names."""
        with self._lock:
            return sorted(self._branches.keys())

    def list_branches(self) -> list[SessionBranch]:
        """Return all branches sorted by name."""
        with self._lock:
            return [self._branches[name] for name in sorted(self._branches.keys())]

    def delete_branch(self, branch_name: str) -> bool:
        """Delete a branch by name.

        Returns
        -------
        bool
            True if the branch existed and was deleted.
        """
        with self._lock:
            if branch_name not in self._branches:
                return False
            del self._branches[branch_name]
            return True

    def clear(self) -> None:
        """Delete all branches."""
        with self._lock:
            self._branches.clear()

    def __contains__(self, branch_name: object) -> bool:
        with self._lock:
            return branch_name in self._branches

    def __len__(self) -> int:
        with self._lock:
            return len(self._branches)

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_segment_counts(self) -> dict[str, int]:
        """Return a mapping of branch_name -> segment count for all branches."""
        with self._lock:
            return {name: branch.segment_count for name, branch in self._branches.items()}

    def compare_task_counts(self) -> dict[str, int]:
        """Return a mapping of branch_name -> task count for all branches."""
        with self._lock:
            return {name: branch.task_count for name, branch in self._branches.items()}
