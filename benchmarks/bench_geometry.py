"""Benchmark the Geometry & Visibility Engine against reference implementations.

Answers the brief's "computing limitations" point with numbers:

    loop       - Python loop per link (typical prototype)
    broadcast  - NumPy broadcasting with a (T, S, G, 3) intermediate
    engine     - dot-product / BLAS formulation used by E2EPS

Usage:
    python benchmarks/bench_geometry.py
"""

from __future__ import annotations

import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from e2eps.config import load_scenario  # noqa: E402
from e2eps.geometry import GeometryEngine, GroundPoints  # noqa: E402
from e2eps.geometry.reference import geometry_broadcast, geometry_loop  # noqa: E402
from e2eps.orbit import build_propagator  # noqa: E402
from e2eps.timegrid import iter_chunks, make_time_grid  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def measure(fn, repeat=3):
    """Best wall time (no tracing) and peak traced memory (separate run)."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return best, peak / 1e6


def main():
    sc = load_scenario(ROOT / "scenarios" / "leo_ku_europe.yaml")
    prop = build_propagator(sc.constellation, sc.epoch)
    ground = GroundPoints.from_sites(sc.user_terminals.all_sites(),
                                     sc.user_terminals.min_elevation_deg)
    engine = GeometryEngine(ground)
    engine32 = GeometryEngine(ground, dtype=np.float32)

    # ---- 1. per-method comparison on one chunk ---------------------------
    T_small = 2
    sat_small = prop.positions_ecef(np.arange(T_small) * 30.0)
    sat_chunk = prop.positions_ecef(np.arange(sc.simulation.chunk_steps) * 30.0)
    n_small = sat_small.shape[0] * sat_small.shape[1] * ground.size
    n_chunk = sat_chunk.shape[0] * sat_chunk.shape[1] * ground.size

    rows = []
    t, m = measure(lambda: geometry_loop(sat_small, ground), repeat=1)
    m = float("nan")  # tracing a pure-Python loop is not meaningful
    rows.append(("loop (Python)", n_small, t, m))
    for name, fn in (("broadcast (NumPy 4-D)", lambda: geometry_broadcast(sat_chunk, ground)),
                     ("engine float64", lambda: engine.compute(sat_chunk)),
                     ("engine float32 out", lambda: engine32.compute(sat_chunk))):
        t, m = measure(fn)
        rows.append((name, n_chunk, t, m))

    base_ns = rows[0][2] / rows[0][1] * 1e9
    print(f"\nScenario: {sc.name}  S={prop.num_sats}  G={ground.size}\n")
    print("| Method | Links | Time [s] | ns / link | Speed-up | Peak mem [MB] |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, n, t, m in rows:
        ns = t / n * 1e9
        mem = "-" if m != m else f"{m:.0f}"
        print(f"| {name} | {n:,} | {t:.3f} | {ns:.1f} | {base_ns / ns:,.0f}x | {mem} |")

    # ---- 2. full reference run (propagation + geometry), chunked --------
    t_grid = make_time_grid(sc.simulation)
    total_links = t_grid.size * prop.num_sats * ground.size
    t0 = time.perf_counter()
    for _, t_chunk in iter_chunks(t_grid, sc.simulation.chunk_steps):
        engine.compute(prop.positions_ecef(t_chunk))
    full = time.perf_counter() - t0
    loop_est = total_links * base_ns / 1e9

    print(f"\nFull scenario: {t_grid.size} steps x {prop.num_sats} sats x {ground.size} terminals"
          f" = {total_links:,} links")
    print(f"  engine (chunked):        {full:8.1f} s")
    print(f"  Python loop (estimated): {loop_est:8.0f} s  (~{loop_est / 60:.0f} min)")
    print(f"\nRun at {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}, NumPy {np.__version__}")


if __name__ == "__main__":
    main()
