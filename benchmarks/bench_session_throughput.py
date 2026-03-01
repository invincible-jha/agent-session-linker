"""Benchmark: Session save/load throughput — operations per second.

Measures how many session save+load round-trips can be completed per second
using the in-memory backend.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_session_linker.session.manager import SessionManager
from agent_session_linker.storage.memory import InMemoryBackend

_ITERATIONS: int = 5_000


def bench_session_save_load_throughput() -> dict[str, object]:
    """Benchmark SessionManager save+load round-trip throughput.

    Returns
    -------
    dict with keys: operation, iterations, total_seconds, ops_per_second,
    avg_latency_ms, p99_latency_ms, memory_peak_mb.
    """
    backend = InMemoryBackend()
    manager = SessionManager(backend=backend, default_agent_id="bench-agent")

    # Pre-create sessions so we can load them in the loop.
    session = manager.create_session(agent_id="bench-agent")
    session_id = manager.save_session(session)

    latencies_ms: list[float] = []
    for _ in range(_ITERATIONS):
        t0 = time.perf_counter()
        new_session = manager.create_session(agent_id="bench-agent")
        sid = manager.save_session(new_session)
        manager.load_session(sid)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

    total = sum(latencies_ms) / 1000
    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)

    result: dict[str, object] = {
        "operation": "session_save_load_throughput",
        "iterations": _ITERATIONS,
        "total_seconds": round(total, 4),
        "ops_per_second": round(_ITERATIONS / total, 1),
        "avg_latency_ms": round(sum(latencies_ms) / n, 4),
        "p99_latency_ms": round(sorted_lats[min(int(n * 0.99), n - 1)], 4),
        "memory_peak_mb": 0.0,
    }
    print(
        f"[bench_session_throughput] {result['operation']}: "
        f"{result['ops_per_second']:,.0f} ops/sec  "
        f"avg {result['avg_latency_ms']:.4f} ms"
    )
    return result


def run_benchmark() -> dict[str, object]:
    """Entry point returning the benchmark result dict."""
    return bench_session_save_load_throughput()


if __name__ == "__main__":
    result = run_benchmark()
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "throughput_baseline.json"
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"Results saved to {output_path}")
