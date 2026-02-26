"""Freshness decay functions for context scoring.

Provides three decay curves that map segment age (in hours) to a score
multiplier in [0.0, 1.0].

Classes
-------
- DecayCurve    — enum of supported curve names
- FreshnessDecay — compute freshness scores with configurable curves
"""
from __future__ import annotations

import math
from enum import Enum


class DecayCurve(str, Enum):
    """Names for the available freshness decay curves."""

    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    STEP = "step"


class FreshnessDecay:
    """Compute freshness multipliers for context segments.

    The freshness score modulates how much weight is given to context
    based on its age.  Three curves are available:

    - ``linear``      — ``1 - age / max_age`` — drops to 0 at ``max_age``
    - ``exponential`` — ``exp(-decay_rate * age)`` — never reaches 0
    - ``step``        — 1.0 if young, 0.5 if middle-aged, 0.1 if old

    All curves return values in [0.0, 1.0].

    Parameters
    ----------
    curve:
        The decay curve to use.  Defaults to ``DecayCurve.EXPONENTIAL``.
    max_age_hours:
        For ``linear``: the age at which the score reaches 0.
        For ``step``: the first threshold where score drops to 0.5.
        Default: 168 (one week).
    decay_rate:
        For ``exponential``: controls how quickly the score falls.
        Higher values decay faster.  Default: 0.01.
    step_thresholds:
        For ``step``: a two-element tuple ``(t1, t2)`` in hours.
        Score is 1.0 when ``age < t1``, 0.5 when ``t1 <= age < t2``,
        and 0.1 otherwise.  Defaults to ``(24, 168)``.
    """

    def __init__(
        self,
        curve: DecayCurve = DecayCurve.EXPONENTIAL,
        max_age_hours: float = 168.0,
        decay_rate: float = 0.01,
        step_thresholds: tuple[float, float] = (24.0, 168.0),
    ) -> None:
        self.curve = curve
        self.max_age_hours = max_age_hours
        self.decay_rate = decay_rate
        self.step_thresholds = step_thresholds

    # ------------------------------------------------------------------
    # Public scoring method
    # ------------------------------------------------------------------

    def score(self, age_hours: float) -> float:
        """Return the freshness score for a segment of age ``age_hours``.

        Parameters
        ----------
        age_hours:
            The age of the segment in hours (must be >= 0).

        Returns
        -------
        float
            Freshness score in [0.0, 1.0].  A score of 1.0 means fully
            fresh; 0.0 means completely stale.
        """
        age = max(0.0, age_hours)

        if self.curve is DecayCurve.LINEAR:
            return self._linear(age)
        if self.curve is DecayCurve.EXPONENTIAL:
            return self._exponential(age)
        if self.curve is DecayCurve.STEP:
            return self._step(age)

        # Fallback — should not be reachable with a valid DecayCurve.
        return 1.0

    # ------------------------------------------------------------------
    # Curve implementations
    # ------------------------------------------------------------------

    def _linear(self, age: float) -> float:
        """Linear decay: ``max(0, 1 - age / max_age_hours)``."""
        if self.max_age_hours <= 0:
            return 0.0
        return max(0.0, 1.0 - age / self.max_age_hours)

    def _exponential(self, age: float) -> float:
        """Exponential decay: ``exp(-decay_rate * age)``."""
        return math.exp(-self.decay_rate * age)

    def _step(self, age: float) -> float:
        """Step decay: 1.0 / 0.5 / 0.1 based on threshold bands."""
        t1, t2 = self.step_thresholds
        if age < t1:
            return 1.0
        if age < t2:
            return 0.5
        return 0.1

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def score_many(self, ages_hours: list[float]) -> list[float]:
        """Return freshness scores for a list of ages.

        Parameters
        ----------
        ages_hours:
            List of segment ages in hours.

        Returns
        -------
        list[float]
            Corresponding freshness scores.
        """
        return [self.score(age) for age in ages_hours]

    def __repr__(self) -> str:
        return (
            f"FreshnessDecay(curve={self.curve.value!r}, "
            f"max_age_hours={self.max_age_hours}, "
            f"decay_rate={self.decay_rate})"
        )
