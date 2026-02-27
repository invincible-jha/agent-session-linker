"""Segment classifier — classify session segments by type.

SegmentClassifier inspects a segment's metadata and content to determine
which SegmentType it belongs to.  Classification is rule-based:

1. Metadata field rules — examine explicit fields (e.g. ``segment_type``
   field already set, ``role`` field).
2. Keyword rules — examine the content for characteristic keyword patterns.
3. Fallback — unmatched segments are classified as CHAT.

Classification rules are evaluated in priority order.  The first matching
rule wins.

This is intentionally a simple, commodity implementation with no external
dependencies.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent_session_linker.selective.importance_scorer import SegmentType


# ---------------------------------------------------------------------------
# Classification rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassificationRule:
    """A single classification rule.

    Attributes
    ----------
    target_type:
        The SegmentType to assign when this rule matches.
    field_name:
        When set, check whether the named metadata field contains
        ``field_value`` (case-insensitive substring match).
    field_value:
        The value to look for in ``field_name``.
    content_pattern:
        When set, check whether the segment content matches this regex pattern.
    priority:
        Lower values are evaluated first.
    """

    target_type: SegmentType
    field_name: str = ""
    field_value: str = ""
    content_pattern: str = ""
    priority: int = 50


# ---------------------------------------------------------------------------
# Classifier configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentClassifierConfig:
    """Configuration for SegmentClassifier.

    Attributes
    ----------
    rules:
        Ordered list of classification rules.  When empty, the classifier
        uses the built-in default rule set.
    fallback_type:
        Type to assign when no rule matches.
    trust_existing_type:
        When True and the segment already has a non-empty ``segment_type``
        field that is a valid SegmentType, use it directly without applying
        rules.
    """

    rules: list[ClassificationRule] = field(default_factory=list)
    fallback_type: SegmentType = SegmentType.CHAT
    trust_existing_type: bool = True


# ---------------------------------------------------------------------------
# Default built-in rules
# ---------------------------------------------------------------------------

_DEFAULT_RULES: list[ClassificationRule] = [
    # Metadata field rules (priority 10–20)
    ClassificationRule(
        target_type=SegmentType.PREFERENCE,
        field_name="segment_type",
        field_value="preference",
        priority=10,
    ),
    ClassificationRule(
        target_type=SegmentType.TASK_STATE,
        field_name="segment_type",
        field_value="task_state",
        priority=10,
    ),
    ClassificationRule(
        target_type=SegmentType.REASONING,
        field_name="segment_type",
        field_value="reasoning",
        priority=10,
    ),
    ClassificationRule(
        target_type=SegmentType.METADATA,
        field_name="segment_type",
        field_value="metadata",
        priority=10,
    ),
    ClassificationRule(
        target_type=SegmentType.CHAT,
        field_name="segment_type",
        field_value="chat",
        priority=10,
    ),
    ClassificationRule(
        target_type=SegmentType.CHAT,
        field_name="segment_type",
        field_value="conversation",
        priority=12,
    ),
    # Role-based rules (priority 20)
    ClassificationRule(
        target_type=SegmentType.METADATA,
        field_name="role",
        field_value="system",
        priority=20,
    ),
    # Content keyword rules (priority 30–50)
    ClassificationRule(
        target_type=SegmentType.PREFERENCE,
        content_pattern=(
            r"\b(prefer|preference|always use|always respond|never use|"
            r"style guide|my preference|user prefers?|format preference|"
            r"output format|language preference)\b"
        ),
        priority=30,
    ),
    ClassificationRule(
        target_type=SegmentType.TASK_STATE,
        content_pattern=(
            r"\b(task:|todo:|action item:|in progress:|completed:|"
            r"pending:|current task|working on|next step|status:)"
        ),
        priority=30,
    ),
    ClassificationRule(
        target_type=SegmentType.REASONING,
        content_pattern=(
            r"\b(because|therefore|reasoning:|rationale:|"
            r"i chose|i decided|my reasoning|analysis:|"
            r"considering|given that|since|thus)\b"
        ),
        priority=40,
    ),
    ClassificationRule(
        target_type=SegmentType.METADATA,
        content_pattern=(
            r"\b(session_id|agent_id|timestamp|version:|schema_version|"
            r"created_at|updated_at|trace_id|span_id|correlation_id)\b"
        ),
        priority=40,
    ),
]


# ---------------------------------------------------------------------------
# SegmentClassifier
# ---------------------------------------------------------------------------


class SegmentClassifier:
    """Classify session segments into SegmentType categories.

    Parameters
    ----------
    config:
        Classifier configuration.  Uses default built-in rules when not
        provided.

    Example
    -------
    >>> classifier = SegmentClassifier()
    >>> seg_type = classifier.classify(
    ...     content="User prefers JSON output format",
    ...     metadata={"role": "user"},
    ... )
    >>> seg_type
    <SegmentType.PREFERENCE: 'preference'>
    """

    def __init__(self, config: SegmentClassifierConfig | None = None) -> None:
        self._config = config if config is not None else SegmentClassifierConfig()
        rules = (
            self._config.rules if self._config.rules else list(_DEFAULT_RULES)
        )
        self._rules: list[ClassificationRule] = sorted(
            rules, key=lambda r: r.priority
        )
        self._compiled: dict[str, re.Pattern[str]] = {}
        for rule in self._rules:
            if rule.content_pattern and rule.content_pattern not in self._compiled:
                self._compiled[rule.content_pattern] = re.compile(
                    rule.content_pattern, re.IGNORECASE
                )

    def classify(
        self,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> SegmentType:
        """Classify a single segment into a SegmentType.

        Parameters
        ----------
        content:
            The text content of the segment.
        metadata:
            Optional segment metadata dict.  Examined for field-based rules.

        Returns
        -------
        SegmentType
            The classified segment type.
        """
        meta = metadata or {}

        # Trust explicit type if present and valid
        if self._config.trust_existing_type:
            existing = str(meta.get("segment_type", "")).strip()
            if existing:
                try:
                    return SegmentType(existing)
                except ValueError:
                    pass  # fall through to rules

        for rule in self._rules:
            # When trust_existing_type is False, skip field-based rules that
            # examine the "segment_type" field — those rules exist to honour an
            # already-assigned type, which is exactly what we are ignoring.
            if (
                not self._config.trust_existing_type
                and rule.field_name == "segment_type"
            ):
                continue
            if self._matches_rule(rule, content, meta):
                return rule.target_type

        return self._config.fallback_type

    def classify_batch(
        self,
        segments: list[dict[str, object]],
    ) -> list[SegmentType]:
        """Classify a list of raw segment dicts.

        Each dict should have a ``content`` key.  A ``metadata`` key is
        optional.

        Parameters
        ----------
        segments:
            List of raw segment dicts.

        Returns
        -------
        list[SegmentType]
            Classification for each segment in input order.
        """
        results: list[SegmentType] = []
        for seg in segments:
            content = str(seg.get("content", ""))
            metadata = dict(seg.get("metadata", {}))
            # If segment_type is a top-level key, pass it via metadata
            for key in ("segment_type", "role"):
                if key in seg and key not in metadata:
                    metadata[key] = str(seg[key])
            results.append(self.classify(content, metadata))
        return results

    def annotate(
        self,
        segments: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Classify segments and annotate each with a ``segment_type`` field.

        The original dict is not mutated; a new dict is returned for each
        segment with the ``segment_type`` key set.

        Parameters
        ----------
        segments:
            List of raw segment dicts.

        Returns
        -------
        list[dict[str, Any]]
            Copies of input dicts with ``segment_type`` filled in.
        """
        annotated: list[dict[str, object]] = []
        for seg in segments:
            content = str(seg.get("content", ""))
            metadata = dict(seg.get("metadata", {}))
            for key in ("segment_type", "role"):
                if key in seg and key not in metadata:
                    metadata[key] = str(seg[key])
            classified = self.classify(content, metadata)
            new_seg = dict(seg)
            new_seg["segment_type"] = classified.value
            annotated.append(new_seg)
        return annotated

    def _matches_rule(
        self,
        rule: ClassificationRule,
        content: str,
        metadata: dict[str, object],
    ) -> bool:
        """Evaluate whether a rule matches the given segment.

        A rule matches when at least one of its conditions is defined and
        satisfied.  If a rule has both ``field_name`` and ``content_pattern``,
        either matching is sufficient (OR semantics).

        Parameters
        ----------
        rule:
            The rule to evaluate.
        content:
            Segment content text.
        metadata:
            Segment metadata dict.

        Returns
        -------
        bool
            True when the rule matches.
        """
        has_condition = False

        if rule.field_name and rule.field_value:
            has_condition = True
            field_val = str(metadata.get(rule.field_name, "")).lower()
            if rule.field_value.lower() in field_val:
                return True

        if rule.content_pattern:
            has_condition = True
            pattern = self._compiled.get(rule.content_pattern)
            if pattern and pattern.search(content):
                return True

        return False
