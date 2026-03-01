"""Structural tests for agent-session-linker benchmarks.

Verifies that each benchmark function is callable and returns a dict
with the expected required keys.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "benchmarks"))

_REQUIRED_KEYS = {"operation", "ops_per_second", "avg_latency_ms"}


def test_bench_session_throughput_returns_expected_keys() -> None:
    """bench_session_save_load_throughput returns a dict with required keys."""
    from bench_session_throughput import run_benchmark

    result = run_benchmark()
    assert isinstance(result, dict)
    for key in _REQUIRED_KEYS:
        assert key in result, f"Missing key: {key!r}"


def test_bench_session_throughput_ops_per_second_positive() -> None:
    """ops_per_second must be a positive float."""
    from bench_session_throughput import run_benchmark

    result = run_benchmark()
    assert float(result["ops_per_second"]) > 0.0  # type: ignore[arg-type]


def test_bench_session_latency_returns_expected_keys() -> None:
    """bench_session_linking_latency returns a dict with required keys."""
    from bench_session_latency import run_benchmark

    result = run_benchmark()
    assert isinstance(result, dict)
    for key in _REQUIRED_KEYS:
        assert key in result, f"Missing key: {key!r}"


def test_bench_session_latency_p99_present() -> None:
    """p99_latency_ms key must be present and non-negative."""
    from bench_session_latency import run_benchmark

    result = run_benchmark()
    assert "p99_latency_ms" in result
    assert float(result["p99_latency_ms"]) >= 0.0  # type: ignore[arg-type]


def test_bench_memory_usage_returns_expected_keys() -> None:
    """bench_session_memory_usage returns a dict with required keys."""
    from bench_memory_usage import run_benchmark

    result = run_benchmark()
    assert isinstance(result, dict)
    assert "operation" in result
    assert "peak_memory_kb" in result
    assert float(result["peak_memory_kb"]) >= 0.0  # type: ignore[arg-type]
