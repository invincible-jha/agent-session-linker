"""Session serialization with schema versioning.

Supports JSON and YAML round-trips.  Schema version is embedded in every
serialised document so that future readers can perform migrations.

Classes
-------
- SessionSerializer  — serialize/deserialize SessionState to JSON or YAML
"""
from __future__ import annotations

import json
from typing import Literal

import yaml

from agent_session_linker.session.state import SessionState

_SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0"})


class SchemaVersionError(ValueError):
    """Raised when a serialised document uses an unsupported schema version."""

    def __init__(self, version: str) -> None:
        self.version = version
        supported = ", ".join(sorted(_SUPPORTED_SCHEMA_VERSIONS))
        super().__init__(
            f"Unsupported schema version {version!r}. "
            f"Supported versions: {supported}"
        )


class SessionSerializer:
    """Serialize and deserialize ``SessionState`` objects.

    All output documents embed a ``schema_version`` field.  On load, the
    version is checked against the set of supported versions before
    deserialization proceeds.

    Parameters
    ----------
    validate_checksum:
        When True (default), ``load_json`` and ``load_yaml`` will verify
        the embedded SHA-256 checksum and raise ``ValueError`` on mismatch.
    """

    def __init__(self, validate_checksum: bool = True) -> None:
        self.validate_checksum = validate_checksum

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------

    def to_json(self, state: SessionState, *, indent: int = 2) -> str:
        """Serialise a ``SessionState`` to a JSON string.

        A fresh checksum is computed and embedded before serialisation.

        Parameters
        ----------
        state:
            The session to serialise.
        indent:
            JSON indentation level (default 2).

        Returns
        -------
        str
            JSON-encoded document with schema_version and checksum fields.
        """
        state.compute_checksum()
        data = state.model_dump(mode="json")
        return json.dumps(data, indent=indent, default=str)

    def from_json(self, raw: str) -> SessionState:
        """Deserialize a ``SessionState`` from a JSON string.

        Parameters
        ----------
        raw:
            JSON string previously produced by ``to_json``.

        Returns
        -------
        SessionState
            The reconstructed session.

        Raises
        ------
        SchemaVersionError
            If the ``schema_version`` field is not in the supported set.
        ValueError
            If ``validate_checksum`` is True and the checksum does not match.
        json.JSONDecodeError
            If ``raw`` is not valid JSON.
        """
        data: dict[str, object] = json.loads(raw)
        return self._deserialize(data)

    # ------------------------------------------------------------------
    # YAML
    # ------------------------------------------------------------------

    def to_yaml(self, state: SessionState) -> str:
        """Serialise a ``SessionState`` to a YAML string.

        Parameters
        ----------
        state:
            The session to serialise.

        Returns
        -------
        str
            YAML-encoded document.
        """
        state.compute_checksum()
        data = state.model_dump(mode="json")
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=True)

    def from_yaml(self, raw: str) -> SessionState:
        """Deserialize a ``SessionState`` from a YAML string.

        Parameters
        ----------
        raw:
            YAML string previously produced by ``to_yaml``.

        Returns
        -------
        SessionState
            The reconstructed session.

        Raises
        ------
        SchemaVersionError
            If the ``schema_version`` field is not in the supported set.
        ValueError
            If ``validate_checksum`` is True and the checksum does not match.
        """
        data: dict[str, object] = yaml.safe_load(raw)
        return self._deserialize(data)

    # ------------------------------------------------------------------
    # Format dispatch
    # ------------------------------------------------------------------

    def serialize(
        self, state: SessionState, format: Literal["json", "yaml"] = "json"
    ) -> str:
        """Serialize using the named format.

        Parameters
        ----------
        state:
            Session to serialize.
        format:
            Either ``"json"`` (default) or ``"yaml"``.

        Returns
        -------
        str
            Serialized document string.
        """
        if format == "yaml":
            return self.to_yaml(state)
        return self.to_json(state)

    def deserialize(
        self, raw: str, format: Literal["json", "yaml"] = "json"
    ) -> SessionState:
        """Deserialize using the named format.

        Parameters
        ----------
        raw:
            Serialized document string.
        format:
            Either ``"json"`` (default) or ``"yaml"``.

        Returns
        -------
        SessionState
            Reconstructed session.
        """
        if format == "yaml":
            return self.from_yaml(raw)
        return self.from_json(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deserialize(self, data: dict[str, object]) -> SessionState:
        """Common deserialization path for both JSON and YAML.

        Parameters
        ----------
        data:
            Dictionary decoded from the serialized document.

        Returns
        -------
        SessionState
            Validated and (optionally checksum-verified) session state.
        """
        version = str(data.get("schema_version", ""))
        if version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionError(version)

        state = SessionState.model_validate(data)

        if self.validate_checksum and state.checksum:
            stored = state.checksum
            computed = state.compute_checksum()
            if stored != computed:
                raise ValueError(
                    f"Checksum mismatch for session {state.session_id!r}: "
                    f"stored={stored!r} computed={computed!r}"
                )

        return state
