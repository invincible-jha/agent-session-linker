"""Tests for SessionLinker and LinkedSession."""
from __future__ import annotations

import pytest

from agent_session_linker.linking.session_linker import LinkedSession, SessionLinker


class TestLinkedSessionRepr:
    def test_repr_contains_source(self) -> None:
        link = LinkedSession(
            source_session_id="session-a",
            target_session_id="session-b",
            relationship="continues",
        )
        assert "session-a" in repr(link)

    def test_repr_contains_relationship(self) -> None:
        link = LinkedSession(
            source_session_id="a",
            target_session_id="b",
            relationship="references",
        )
        assert "references" in repr(link)


class TestSessionLinkerLink:
    def test_link_creates_relationship(self) -> None:
        linker = SessionLinker()
        link = linker.link("session-a", "session-b", "continues")
        assert link.source_session_id == "session-a"
        assert link.target_session_id == "session-b"
        assert link.relationship == "continues"

    def test_duplicate_link_returns_existing(self) -> None:
        linker = SessionLinker()
        link1 = linker.link("a", "b", "continues")
        link2 = linker.link("a", "b", "continues")
        assert link1 is link2

    def test_self_link_raises_by_default(self) -> None:
        linker = SessionLinker(allow_self_links=False)
        with pytest.raises(ValueError):
            linker.link("session-a", "session-a", "self")

    def test_self_link_allowed_when_flag_set(self) -> None:
        linker = SessionLinker(allow_self_links=True)
        link = linker.link("session-a", "session-a", "self")
        assert link.source_session_id == "session-a"

    def test_link_with_metadata(self) -> None:
        linker = SessionLinker()
        link = linker.link("a", "b", "spawned", metadata={"reason": "fork"})
        assert link.metadata.get("reason") == "fork"


class TestSessionLinkerUnlink:
    def test_unlink_removes_relationship(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        linker.unlink("a", "b", "continues")
        linked = linker.get_linked("a")
        assert len(linked) == 0

    def test_unlink_nonexistent_raises(self) -> None:
        linker = SessionLinker()
        with pytest.raises(KeyError):
            linker.unlink("a", "b", "nonexistent")

    def test_unlink_removes_from_incoming_too(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        linker.unlink("a", "b", "continues")
        incoming = linker.get_linked("b", direction="incoming")
        assert len(incoming) == 0


class TestSessionLinkerGetLinked:
    def _linker_with_links(self) -> SessionLinker:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        linker.link("a", "c", "references")
        linker.link("d", "a", "spawned")
        return linker

    def test_get_linked_both_directions(self) -> None:
        linker = self._linker_with_links()
        links = linker.get_linked("a")
        # a -> b, a -> c outgoing; d -> a incoming.
        assert len(links) == 3

    def test_get_linked_outgoing_only(self) -> None:
        linker = self._linker_with_links()
        links = linker.get_linked("a", direction="outgoing")
        assert all(link.source_session_id == "a" for link in links)

    def test_get_linked_incoming_only(self) -> None:
        linker = self._linker_with_links()
        links = linker.get_linked("a", direction="incoming")
        assert all(link.target_session_id == "a" for link in links)

    def test_get_linked_with_relationship_filter(self) -> None:
        linker = self._linker_with_links()
        links = linker.get_linked("a", relationship="continues")
        assert all(link.relationship == "continues" for link in links)

    def test_get_linked_no_relationships_returns_empty(self) -> None:
        linker = SessionLinker()
        assert linker.get_linked("ghost") == []

    def test_get_linked_sorted_by_created_at(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        linker.link("a", "c", "references")
        links = linker.get_linked("a")
        timestamps = [link.created_at for link in links]
        assert timestamps == sorted(timestamps)


class TestSessionLinkerGetRelatedSessionIds:
    def test_returns_related_ids(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        linker.link("c", "a", "spawned")
        related = linker.get_related_session_ids("a")
        assert "b" in related
        assert "c" in related
        assert "a" not in related  # self excluded

    def test_with_relationship_filter(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        linker.link("a", "c", "references")
        related = linker.get_related_session_ids("a", relationship="continues")
        assert "b" in related
        assert "c" not in related


class TestSessionLinkerExportImport:
    def test_export_links(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        exported = linker.export_links()
        assert len(exported) == 1
        assert exported[0]["source_session_id"] == "a"
        assert exported[0]["target_session_id"] == "b"

    def test_import_links_restores_data(self) -> None:
        original = SessionLinker()
        original.link("a", "b", "continues", metadata={"note": "x"})
        exported = original.export_links()

        restored = SessionLinker()
        restored.import_links(exported)
        links = restored.get_linked("a")
        assert len(links) == 1
        assert links[0].relationship == "continues"

    def test_import_links_idempotent(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        exported = linker.export_links()

        linker.import_links(exported)  # Should not add duplicates.
        links = linker.get_linked("a")
        assert len(links) == 1

    def test_repr(self) -> None:
        linker = SessionLinker()
        linker.link("a", "b", "continues")
        assert "SessionLinker" in repr(linker)
