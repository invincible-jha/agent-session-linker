"""Tests for FreshnessDecay and DecayCurve."""
from __future__ import annotations

import math

import pytest

from agent_session_linker.context.freshness import DecayCurve, FreshnessDecay


class TestDecayCurveEnum:
    def test_enum_values(self) -> None:
        assert DecayCurve.LINEAR.value == "linear"
        assert DecayCurve.EXPONENTIAL.value == "exponential"
        assert DecayCurve.STEP.value == "step"


class TestFreshnessDecayLinear:
    def _linear(self, max_age: float = 100.0) -> FreshnessDecay:
        return FreshnessDecay(curve=DecayCurve.LINEAR, max_age_hours=max_age)

    def test_age_zero_returns_one(self) -> None:
        assert self._linear().score(0.0) == pytest.approx(1.0)

    def test_age_at_max_returns_zero(self) -> None:
        assert self._linear(100.0).score(100.0) == pytest.approx(0.0)

    def test_age_beyond_max_clamped_to_zero(self) -> None:
        assert self._linear(100.0).score(200.0) == pytest.approx(0.0)

    def test_midpoint_returns_half(self) -> None:
        assert self._linear(100.0).score(50.0) == pytest.approx(0.5)

    def test_negative_age_treated_as_zero(self) -> None:
        assert self._linear().score(-10.0) == pytest.approx(1.0)

    def test_max_age_zero_returns_zero(self) -> None:
        fd = FreshnessDecay(curve=DecayCurve.LINEAR, max_age_hours=0.0)
        assert fd.score(0.0) == 0.0


class TestFreshnessDecayExponential:
    def _exp(self, rate: float = 0.01) -> FreshnessDecay:
        return FreshnessDecay(curve=DecayCurve.EXPONENTIAL, decay_rate=rate)

    def test_age_zero_returns_one(self) -> None:
        assert self._exp().score(0.0) == pytest.approx(1.0)

    def test_positive_age_returns_less_than_one(self) -> None:
        assert self._exp().score(10.0) < 1.0

    def test_higher_rate_decays_faster(self) -> None:
        slow = self._exp(rate=0.01).score(100.0)
        fast = self._exp(rate=0.1).score(100.0)
        assert fast < slow

    def test_result_never_zero(self) -> None:
        assert self._exp().score(10000.0) > 0.0

    def test_matches_math_formula(self) -> None:
        rate = 0.05
        age = 30.0
        expected = math.exp(-rate * age)
        fd = FreshnessDecay(curve=DecayCurve.EXPONENTIAL, decay_rate=rate)
        assert fd.score(age) == pytest.approx(expected)


class TestFreshnessDecayStep:
    def _step(
        self, t1: float = 24.0, t2: float = 168.0
    ) -> FreshnessDecay:
        return FreshnessDecay(
            curve=DecayCurve.STEP, step_thresholds=(t1, t2)
        )

    def test_young_segment_returns_one(self) -> None:
        assert self._step().score(0.0) == pytest.approx(1.0)

    def test_age_below_t1_returns_one(self) -> None:
        assert self._step(t1=24.0).score(23.9) == pytest.approx(1.0)

    def test_age_at_t1_returns_half(self) -> None:
        fd = self._step(t1=24.0, t2=168.0)
        assert fd.score(24.0) == pytest.approx(0.5)

    def test_age_between_t1_and_t2_returns_half(self) -> None:
        assert self._step(t1=24.0, t2=168.0).score(100.0) == pytest.approx(0.5)

    def test_age_at_t2_returns_point_one(self) -> None:
        assert self._step(t1=24.0, t2=168.0).score(168.0) == pytest.approx(0.1)

    def test_age_beyond_t2_returns_point_one(self) -> None:
        assert self._step(t1=24.0, t2=168.0).score(9999.0) == pytest.approx(0.1)


class TestFreshnessDecayScoreMany:
    def test_empty_list_returns_empty(self) -> None:
        fd = FreshnessDecay()
        assert fd.score_many([]) == []

    def test_returns_correct_length(self) -> None:
        fd = FreshnessDecay()
        result = fd.score_many([0.0, 10.0, 100.0])
        assert len(result) == 3

    def test_scores_in_descending_order_for_increasing_ages(self) -> None:
        fd = FreshnessDecay(curve=DecayCurve.LINEAR)
        scores = fd.score_many([0.0, 50.0, 100.0])
        assert scores[0] > scores[1] > scores[2]


class TestFreshnessDecayRepr:
    def test_repr_contains_curve_name(self) -> None:
        fd = FreshnessDecay(curve=DecayCurve.LINEAR)
        assert "linear" in repr(fd)

    def test_repr_contains_max_age(self) -> None:
        fd = FreshnessDecay(max_age_hours=72.0)
        assert "72.0" in repr(fd)
