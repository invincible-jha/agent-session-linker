"""Benchmark: Session linking latency — per-link p50/p95/p99.

Measures the per-call latency of SessionLinker.link() as the graph grows,
capturing the distribution of link creation times.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_session_linker.linking.session_linker import SessionLinker

_WARMUP: int = 200
_ITERATIONS: int = 5_000


def bench_session_linking_latency() -> dict[str, object]:
    """Benchmark SessionLinker.link() per-call latency.

    Returns
    -------
    dict with keys: operation, iterations, total_seconds, ops_per_second,
    avg_latency_ms, p99_latency_ms, memory_peak_mb.
    """
    linker = SessionLinker()

    # Warmup phase — populate the graph.
    for i in range(_WARMUP):
        source_id = f"session-warmup-{i}"
        target_id = f"session-warmup-{i + 1}"
        linker.link(source_id, target_id, "continues")

    latencies_ms: list[float] = []
    for i in range(_ITERATIONS):
        source_id = f"session-bench-{i}"
        target_id = f"session-bench-{i + 1}"
        t0 = time.perf_counter()
        linker.link(source_id, target_id, "references")
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    total = sum(latencies_ms) / 1000

    result: dict[str, object] = {
        "operation": "session_linking_latency",
        "iterations": _ITERATIONS,
        "total_seconds": round(total, 4),
        "ops_per_second": round(_ITERATIONS / total, 1),
        "avg_latency_ms": round(sum(latencies_ms) / n, 4),
        "p99_latency_ms": round(sorted_lats[min(int(n * 0.99), n - 1)], 4),
        "memory_peak_mb": 0.0,
    }
    print(
        f"[bench_session_latency] {result['operation']}: "
        f"p99={result['p99_latency_ms']:.4f}ms  "
        f"mean={result['avg_latency_ms']:.4f}ms"
    )
    return result


def run_benchmark() -> dict[str, object]:
    """Entry point returning the benchmark result dict."""
    return bench_session_linking_latency()


if __name__ == "__main__":
    result = run_benchmark()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "latency_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Results saved to {output_path}")
