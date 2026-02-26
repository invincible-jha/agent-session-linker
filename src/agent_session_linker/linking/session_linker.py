"""Cross-session relationship linking.

Maintains a graph of directed relationships between sessions and provides
efficient lookup of all sessions linked to a given session.

Classes
-------
- LinkedSession  — dataclass describing a directed session relationship
- SessionLinker  — link and query related sessions
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Domain model
# ---------------------------------------------------------------------------


@dataclass
class LinkedSession:
    """A directed relationship between two sessions.

    Parameters
    ----------
    source_session_id:
        The originating session in this relationship.
    target_session_id:
        The related session that was linked to.
    relationship:
        A string label describing the relationship, e.g. ``"continues"``,
        ``"references"``, ``"spawned"``, ``"merged_into"``.
    created_at:
        UTC timestamp when this link was created.
    metadata:
        Optional key-value metadata attached to this link.
    """

    source_session_id: str
    target_session_id: str
    relationship: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, str] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"LinkedSession("
            f"source={self.source_session_id!r}, "
            f"target={self.target_session_id!r}, "
            f"relationship={self.relationship!r})"
        )


# ---------------------------------------------------------------------------
# Linker
# ---------------------------------------------------------------------------


class SessionLinker:
    """Create and query relationships between sessions.

    Relationships are stored in an in-memory adjacency list.  To persist
    links across process restarts, serialise ``export_links()`` and restore
    via ``import_links()``.

    Parameters
    ----------
    allow_self_links:
        When False (default), ``link(a, b)`` raises ``ValueError`` if
        ``a == b``.
    """

    def __init__(self, allow_self_links: bool = False) -> None:
        self.allow_self_links = allow_self_links
        # Maps session_id -> list of LinkedSession (outgoing links).
        self._outgoing: dict[str, list[LinkedSession]] = {}
        # Maps session_id -> list of LinkedSession (incoming links, for reverse lookup).
        self._incoming: dict[str, list[LinkedSession]] = {}

    # ------------------------------------------------------------------
    # Link management
    # ------------------------------------------------------------------

    def link(
        self,
        source_session_id: str,
        target_session_id: str,
        relationship: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> LinkedSession:
        """Create a directed relationship from ``source`` to ``target``.

        Duplicate links (same source, target, and relationship) are silently
        ignored — the existing link is returned unchanged.

        Parameters
        ----------
        source_session_id:
            The session that initiates the relationship.
        target_session_id:
            The related session.
        relationship:
            Descriptive label for the relationship type.
        metadata:
            Optional additional key-value data to attach to the link.

        Returns
        -------
        LinkedSession
            The newly created (or existing duplicate) link.

        Raises
        ------
        ValueError
            If ``source_session_id == target_session_id`` and
            ``allow_self_links`` is False.
        """
        if source_session_id == target_session_id and not self.allow_self_links:
            raise ValueError(
                f"Self-links are not allowed. "
                f"source_session_id and target_session_id are both {source_session_id!r}."
            )

        # Check for existing duplicate.
        for existing_link in self._outgoing.get(source_session_id, []):
            if (
                existing_link.target_session_id == target_session_id
                and existing_link.relationship == relationship
            ):
                return existing_link

        linked = LinkedSession(
            source_session_id=source_session_id,
            target_session_id=target_session_id,
            relationship=relationship,
            metadata=metadata or {},
        )

        self._outgoing.setdefault(source_session_id, []).append(linked)
        self._incoming.setdefault(target_session_id, []).append(linked)
        return linked

    def unlink(
        self,
        source_session_id: str,
        target_session_id: str,
        relationship: str,
    ) -> None:
        """Remove a specific directed link.

        Parameters
        ----------
        source_session_id:
            Source of the link to remove.
        target_session_id:
            Target of the link to remove.
        relationship:
            Relationship label of the link to remove.

        Raises
        ------
        KeyError
            If no matching link exists.
        """
        outgoing = self._outgoing.get(source_session_id, [])
        before = len(outgoing)
        self._outgoing[source_session_id] = [
            link for link in outgoing
            if not (
                link.target_session_id == target_session_id
                and link.relationship == relationship
            )
        ]
        if len(self._outgoing[source_session_id]) == before:
            raise KeyError(
                f"No link from {source_session_id!r} to {target_session_id!r} "
                f"with relationship {relationship!r}."
            )

        incoming = self._incoming.get(target_session_id, [])
        self._incoming[target_session_id] = [
            link for link in incoming
            if not (
                link.source_session_id == source_session_id
                and link.relationship == relationship
            )
        ]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def get_linked(
        self,
        session_id: str,
        relationship: str | None = None,
        direction: str = "both",
    ) -> list[LinkedSession]:
        """Return all links connected to ``session_id``.

        Parameters
        ----------
        session_id:
            The session to query.
        relationship:
            When provided, filter results to only this relationship type.
        direction:
            One of ``"outgoing"``, ``"incoming"``, or ``"both"`` (default).

        Returns
        -------
        list[LinkedSession]
            Matching links sorted by ``created_at`` ascending.
        """
        links: list[LinkedSession] = []

        if direction in ("outgoing", "both"):
            links.extend(self._outgoing.get(session_id, []))
        if direction in ("incoming", "both"):
            links.extend(self._incoming.get(session_id, []))

        # De-duplicate (a link may appear in both directions when queried both ways
        # — here we deduplicate by object identity which is fine for "both" direction).
        seen_ids: set[int] = set()
        unique: list[LinkedSession] = []
        for link in links:
            if id(link) not in seen_ids:
                seen_ids.add(id(link))
                unique.append(link)

        if relationship is not None:
            unique = [link for link in unique if link.relationship == relationship]

        unique.sort(key=lambda link: link.created_at)
        return unique

    def get_related_session_ids(
        self,
        session_id: str,
        relationship: str | None = None,
    ) -> list[str]:
        """Return the IDs of all sessions linked to ``session_id``.

        Parameters
        ----------
        session_id:
            The session to query.
        relationship:
            Optional filter.

        Returns
        -------
        list[str]
            Unique session IDs (excluding ``session_id`` itself).
        """
        related: set[str] = set()
        for link in self.get_linked(session_id, relationship=relationship):
            related.add(link.source_session_id)
            related.add(link.target_session_id)
        related.discard(session_id)
        return sorted(related)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def export_links(self) -> list[dict[str, object]]:
        """Export all links as a list of dicts for persistence.

        Returns
        -------
        list[dict[str, object]]
            Serialisable representation of all links.
        """
        all_links: set[int] = set()
        result: list[dict[str, object]] = []
        for links in self._outgoing.values():
            for link in links:
                if id(link) not in all_links:
                    all_links.add(id(link))
                    result.append(
                        {
                            "source_session_id": link.source_session_id,
                            "target_session_id": link.target_session_id,
                            "relationship": link.relationship,
                            "created_at": link.created_at.isoformat(),
                            "metadata": link.metadata,
                        }
                    )
        return result

    def import_links(self, data: list[dict[str, object]]) -> None:
        """Restore links from a previously exported list.

        Existing links are preserved; only new links are added.

        Parameters
        ----------
        data:
            List of dicts as produced by ``export_links``.
        """
        for record in data:
            linked = LinkedSession(
                source_session_id=str(record["source_session_id"]),
                target_session_id=str(record["target_session_id"]),
                relationship=str(record["relationship"]),
                created_at=datetime.fromisoformat(str(record["created_at"])),
                metadata=dict(record.get("metadata", {})),  # type: ignore[arg-type]
            )
            # Check for duplicate before inserting.
            existing_outgoing = self._outgoing.get(linked.source_session_id, [])
            is_duplicate = any(
                e.target_session_id == linked.target_session_id
                and e.relationship == linked.relationship
                for e in existing_outgoing
            )
            if not is_duplicate:
                self._outgoing.setdefault(linked.source_session_id, []).append(linked)
                self._incoming.setdefault(linked.target_session_id, []).append(linked)

    def __repr__(self) -> str:
        total_links = sum(len(links) for links in self._outgoing.values())
        return f"SessionLinker(sessions={len(self._outgoing)}, links={total_links})"
