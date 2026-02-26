"""Cross-turn entity linking via normalised edit distance.

Maps entity surface forms to canonical known entities using the normalised
Levenshtein edit distance.  No external NLP libraries are required.

Classes
-------
- EntityLinker  — link entity mentions to a known-entity catalogue
"""
from __future__ import annotations

from agent_session_linker.entity.extractor import Entity


# ---------------------------------------------------------------------------
# Edit distance
# ---------------------------------------------------------------------------


def _normalised_edit_distance(source: str, target: str) -> float:
    """Compute the normalised Levenshtein edit distance between two strings.

    The normalised distance is in [0.0, 1.0], where 0.0 means identical
    and 1.0 means completely different.  Normalisation is by the length of
    the longer string so the measure is symmetric.

    Parameters
    ----------
    source:
        First string.
    target:
        Second string.

    Returns
    -------
    float
        Normalised edit distance in [0.0, 1.0].
    """
    if source == target:
        return 0.0

    source_length = len(source)
    target_length = len(target)

    if source_length == 0:
        return 1.0 if target_length > 0 else 0.0
    if target_length == 0:
        return 1.0

    # Build a (source_length+1) x (target_length+1) DP matrix.
    # Use two rolling rows to limit memory to O(min(m,n)).
    if source_length < target_length:
        source, target = target, source
        source_length, target_length = target_length, source_length

    previous_row = list(range(target_length + 1))
    current_row: list[int] = [0] * (target_length + 1)

    for row_index in range(1, source_length + 1):
        current_row[0] = row_index
        for col_index in range(1, target_length + 1):
            insert_cost = current_row[col_index - 1] + 1
            delete_cost = previous_row[col_index] + 1
            replace_cost = previous_row[col_index - 1] + (
                0 if source[row_index - 1] == target[col_index - 1] else 1
            )
            current_row[col_index] = min(insert_cost, delete_cost, replace_cost)
        previous_row, current_row = current_row, previous_row

    raw_distance = previous_row[target_length]
    return raw_distance / max(source_length, target_length)


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class EntityLinker:
    """Link entity mentions to a catalogue of known entities.

    Uses normalised edit distance to find the best match from a list of
    candidate entities.  Type-based filtering is applied first so that, for
    example, a PERSON mention is never linked to an ORG entity.

    Parameters
    ----------
    similarity_threshold:
        Minimum similarity (1 - normalised_edit_distance) required to
        accept a link.  Must be in (0.0, 1.0].  Default: 0.8.
    case_sensitive:
        When False (default), comparisons are made in lowercase.
    require_same_type:
        When True (default), a mention is only linked to known entities
        of the same ``entity_type``.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.8,
        case_sensitive: bool = False,
        require_same_type: bool = True,
    ) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be in (0.0, 1.0], got {similarity_threshold!r}."
            )
        self.similarity_threshold = similarity_threshold
        self.case_sensitive = case_sensitive
        self.require_same_type = require_same_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def link(self, entity: Entity, known: list[Entity]) -> Entity | None:
        """Find the best matching known entity for a mention.

        Parameters
        ----------
        entity:
            The entity mention to link.
        known:
            Catalogue of known entities to search.

        Returns
        -------
        Entity | None
            The best-matching known entity if similarity exceeds the
            threshold, otherwise None.
        """
        if not known:
            return None

        candidates = known
        if self.require_same_type:
            candidates = [k for k in known if k.entity_type == entity.entity_type]

        if not candidates:
            return None

        mention_text = entity.text if self.case_sensitive else entity.text.lower()

        best_entity: Entity | None = None
        best_similarity: float = -1.0

        for candidate in candidates:
            candidate_text = (
                candidate.text if self.case_sensitive else candidate.text.lower()
            )
            distance = _normalised_edit_distance(mention_text, candidate_text)
            similarity = 1.0 - distance

            if similarity > best_similarity:
                best_similarity = similarity
                best_entity = candidate

        if best_similarity >= self.similarity_threshold:
            return best_entity
        return None

    def link_all(
        self,
        entities: list[Entity],
        known: list[Entity],
    ) -> list[tuple[Entity, Entity | None]]:
        """Link every entity in ``entities`` against the catalogue.

        Parameters
        ----------
        entities:
            Mentions to link.
        known:
            Known entity catalogue.

        Returns
        -------
        list[tuple[Entity, Entity | None]]
            Each pair is ``(mention, linked_entity)``.  The second element
            is None when no match met the similarity threshold.
        """
        return [(entity, self.link(entity, known)) for entity in entities]

    def similarity(self, text_a: str, text_b: str) -> float:
        """Return the similarity (1 - normalised edit distance) between two strings.

        Parameters
        ----------
        text_a:
            First string.
        text_b:
            Second string.

        Returns
        -------
        float
            Similarity score in [0.0, 1.0].
        """
        source = text_a if self.case_sensitive else text_a.lower()
        target = text_b if self.case_sensitive else text_b.lower()
        return 1.0 - _normalised_edit_distance(source, target)
