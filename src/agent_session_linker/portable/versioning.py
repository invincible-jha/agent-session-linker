"""USF schema versioning and migration utilities.

This module tracks the current USF schema version, validates whether an
incoming payload's version is supported, and provides a registration-based
migration system for evolving the schema over time.

Classes
-------
SchemaVersion
    Static class holding the current and supported version strings.
SchemaMigrator
    Registers and applies migration functions between schema versions.
"""
from __future__ import annotations

from typing import Callable


# ---------------------------------------------------------------------------
# SchemaVersion
# ---------------------------------------------------------------------------


class SchemaVersion:
    """USF schema version registry.

    Attributes
    ----------
    CURRENT:
        The schema version produced by the current library build.
    SUPPORTED:
        The set of schema versions this library can read.
    """

    CURRENT: str = "1.0"
    SUPPORTED: set[str] = {"1.0"}

    @staticmethod
    def is_supported(version: str) -> bool:
        """Return ``True`` when *version* can be read by this library.

        Parameters
        ----------
        version:
            A schema version string such as ``"1.0"``.

        Returns
        -------
        bool
            ``True`` if *version* is in :attr:`SUPPORTED`.
        """
        return version in SchemaVersion.SUPPORTED


# ---------------------------------------------------------------------------
# SchemaMigrator
# ---------------------------------------------------------------------------


class SchemaMigrator:
    """Registration-based schema migration engine for USF payloads.

    Migrations are keyed by ``(from_version, to_version)`` pairs.  When
    :meth:`migrate` is called with a payload whose version differs from
    *target_version*, the registered migration function is applied.

    Example
    -------
    .. code-block:: python

        migrator = SchemaMigrator()
        migrator.register_migration("1.0", "2.0", lambda d: {**d, "new_field": None})
        updated = migrator.migrate(payload_v1, target_version="2.0")
    """

    def __init__(self) -> None:
        self._migrations: dict[
            tuple[str, str], Callable[[dict[str, object]], dict[str, object]]
        ] = {}

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migrate_fn: Callable[[dict[str, object]], dict[str, object]],
    ) -> None:
        """Register a migration function for a specific version pair.

        Parameters
        ----------
        from_version:
            The source schema version string.
        to_version:
            The target schema version string.
        migrate_fn:
            A callable that accepts a ``dict`` payload at *from_version* and
            returns a new ``dict`` payload at *to_version*.
        """
        self._migrations[(from_version, to_version)] = migrate_fn

    def migrate(
        self,
        data: dict[str, object],
        target_version: str | None = None,
    ) -> dict[str, object]:
        """Apply the registered migration to *data*, if required.

        Parameters
        ----------
        data:
            A payload dict that should contain a ``"version"`` key.
        target_version:
            The desired output version.  Defaults to :attr:`SchemaVersion.CURRENT`.

        Returns
        -------
        dict[str, object]
            The migrated payload (or *data* unchanged if already at target).

        Raises
        ------
        ValueError
            If no migration is registered for the required ``(from, to)`` path.
        """
        target = target_version if target_version is not None else SchemaVersion.CURRENT
        current = str(data.get("version", SchemaVersion.CURRENT))
        if current == target:
            return data
        key = (current, target)
        if key not in self._migrations:
            raise ValueError(
                f"No migration path registered from version {current!r} to {target!r}"
            )
        return self._migrations[key](data)

    def detect_version(self, data: dict[str, object]) -> str:
        """Return the schema version declared in *data*.

        Falls back to :attr:`SchemaVersion.CURRENT` when the ``"version"``
        key is absent.

        Parameters
        ----------
        data:
            A payload dict.

        Returns
        -------
        str
            The schema version string.
        """
        return str(data.get("version", SchemaVersion.CURRENT))
