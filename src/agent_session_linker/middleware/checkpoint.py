"""Periodic session checkpointing.

Creates named snapshots of ``SessionState`` and allows restoration to any
prior checkpoint.  Checkpoints are stored in the same ``StorageBackend``
as sessions, using a distinguished key prefix.

Classes
-------
- CheckpointRecord   — dataclass describing a single checkpoint
- CheckpointManager  — create, list, and restore session checkpoints
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from agent_session_linker.session.manager import SessionManager
from agent_session_linker.session.serializer import SessionSerializer
from agent_session_linker.session.state import SessionState
from agent_session_linker.storage.base import StorageBackend

logger = logging.getLogger(__name__)

_CHECKPOINT_KEY_PREFIX = "__checkpoint__"
_INDEX_KEY_SUFFIX = "__index__"


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class CheckpointRecord:
    """Metadata about a single session checkpoint.

    Parameters
    ----------
    checkpoint_id:
        Unique identifier for this checkpoint (storage key).
    session_id:
        The session this checkpoint was taken from.
    label:
        Human-readable label assigned at creation time.
    created_at:
        UTC timestamp of when the checkpoint was created.
    segment_count:
        Number of segments in the session at snapshot time.
    token_count:
        Total tokens in the session at snapshot time.
    """

    checkpoint_id: str
    session_id: str
    label: str
    created_at: datetime
    segment_count: int
    token_count: int

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict for JSON storage."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "session_id": self.session_id,
            "label": self.label,
            "created_at": self.created_at.isoformat(),
            "segment_count": self.segment_count,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CheckpointRecord:
        """Deserialise from a dict produced by ``to_dict``."""
        return cls(
            checkpoint_id=str(data["checkpoint_id"]),
            session_id=str(data["session_id"]),
            label=str(data["label"]),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            segment_count=int(str(data["segment_count"])),
            token_count=int(str(data["token_count"])),
        )


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class CheckpointManager:
    """Create, restore, and list periodic session snapshots.

    Checkpoints are stored as independent serialised ``SessionState``
    documents in the storage backend alongside regular sessions, using a
    key pattern of ``__checkpoint__<session_id>__<n>``.  An index document
    (``__checkpoint__<session_id>__index__``) tracks all checkpoints for
    each session.

    Parameters
    ----------
    backend:
        The storage backend to write checkpoint data into.
    manager:
        Optional ``SessionManager`` used to load the live session when
        creating a checkpoint from a session ID.  If omitted the caller
        must supply the ``SessionState`` directly to ``create_checkpoint``.
    max_checkpoints_per_session:
        Maximum number of checkpoints to retain per session.  When this
        limit is reached the oldest checkpoint is deleted before creating
        the new one.  Default: 10.
    """

    def __init__(
        self,
        backend: StorageBackend,
        manager: SessionManager | None = None,
        max_checkpoints_per_session: int = 10,
    ) -> None:
        self._backend = backend
        self._manager = manager
        self._serializer = SessionSerializer(validate_checksum=False)
        self.max_checkpoints_per_session = max_checkpoints_per_session

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        session: SessionState,
        label: str = "",
    ) -> CheckpointRecord:
        """Snapshot ``session`` and store it as a named checkpoint.

        Parameters
        ----------
        session:
            The current session state to snapshot.
        label:
            Optional human-readable label (e.g. "before-refactor",
            "post-planning-step").  Defaults to an ISO timestamp.

        Returns
        -------
        CheckpointRecord
            Metadata about the newly created checkpoint.
        """
        now = datetime.now(timezone.utc)
        existing = self._load_index(session.session_id)

        # Determine next checkpoint sequence number.
        sequence = len(existing)
        checkpoint_id = _build_checkpoint_key(session.session_id, sequence)

        # Evict oldest checkpoint if at capacity.
        if len(existing) >= self.max_checkpoints_per_session and existing:
            oldest = existing[0]
            old_key = oldest.checkpoint_id
            if self._backend.exists(old_key):
                self._backend.delete(old_key)
            existing = existing[1:]

        # Serialise the session snapshot.
        raw = self._serializer.to_json(session)
        self._backend.save(checkpoint_id, raw)

        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            session_id=session.session_id,
            label=label or now.isoformat(),
            created_at=now,
            segment_count=len(session.segments),
            token_count=session.total_tokens(),
        )
        existing.append(record)
        self._save_index(session.session_id, existing)

        logger.debug(
            "CheckpointManager: created checkpoint %r for session %r",
            checkpoint_id,
            session.session_id,
        )
        return record

    def restore_checkpoint(self, checkpoint_id: str) -> SessionState:
        """Load and return the session state stored in a checkpoint.

        Parameters
        ----------
        checkpoint_id:
            The ``checkpoint_id`` from a ``CheckpointRecord``.

        Returns
        -------
        SessionState
            The session state at the time the checkpoint was taken.

        Raises
        ------
        KeyError
            If no checkpoint with ``checkpoint_id`` exists.
        """
        if not self._backend.exists(checkpoint_id):
            raise KeyError(f"Checkpoint {checkpoint_id!r} not found.")
        raw = self._backend.load(checkpoint_id)
        state = self._serializer.from_json(raw)
        logger.debug("CheckpointManager: restored checkpoint %r", checkpoint_id)
        return state

    def list_checkpoints(self, session_id: str) -> list[CheckpointRecord]:
        """Return all checkpoint records for ``session_id``, oldest first.

        Parameters
        ----------
        session_id:
            The session whose checkpoints to list.

        Returns
        -------
        list[CheckpointRecord]
            Checkpoint records in creation order (oldest first).
        """
        return self._load_index(session_id)

    def delete_checkpoint(self, checkpoint_id: str, session_id: str) -> None:
        """Remove a checkpoint and update the session index.

        Parameters
        ----------
        checkpoint_id:
            The checkpoint to delete.
        session_id:
            The owning session ID (needed to update the index).

        Raises
        ------
        KeyError
            If the checkpoint does not exist.
        """
        if not self._backend.exists(checkpoint_id):
            raise KeyError(f"Checkpoint {checkpoint_id!r} not found.")

        self._backend.delete(checkpoint_id)

        index = self._load_index(session_id)
        updated = [r for r in index if r.checkpoint_id != checkpoint_id]
        self._save_index(session_id, updated)

        logger.debug(
            "CheckpointManager: deleted checkpoint %r from session %r",
            checkpoint_id,
            session_id,
        )

    # ------------------------------------------------------------------
    # Index helpers
    # ------------------------------------------------------------------

    def _index_key(self, session_id: str) -> str:
        return f"{_CHECKPOINT_KEY_PREFIX}{session_id}{_INDEX_KEY_SUFFIX}"

    def _load_index(self, session_id: str) -> list[CheckpointRecord]:
        """Load the checkpoint index for a session."""
        index_key = self._index_key(session_id)
        if not self._backend.exists(index_key):
            return []
        raw = self._backend.load(index_key)
        records_data: list[dict[str, object]] = json.loads(raw)
        return [CheckpointRecord.from_dict(record) for record in records_data]

    def _save_index(self, session_id: str, records: list[CheckpointRecord]) -> None:
        """Persist the checkpoint index for a session."""
        index_key = self._index_key(session_id)
        data = [record.to_dict() for record in records]
        self._backend.save(index_key, json.dumps(data, default=str))


# ---------------------------------------------------------------------------
# Key building helper
# ---------------------------------------------------------------------------


def _build_checkpoint_key(session_id: str, sequence: int) -> str:
    """Build a deterministic storage key for a checkpoint.

    Parameters
    ----------
    session_id:
        Owning session ID.
    sequence:
        Monotonically increasing sequence number.

    Returns
    -------
    str
        Storage key string.
    """
    return f"{_CHECKPOINT_KEY_PREFIX}{session_id}__{sequence:04d}"
