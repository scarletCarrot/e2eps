"""Simulation time grid and chunking."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .config.schema import SimulationConfig


def make_time_grid(sim: SimulationConfig) -> np.ndarray:
    """Seconds since epoch for every step: (T,)."""
    return np.arange(sim.num_steps, dtype=float) * sim.step_s


def iter_chunks(t_s: np.ndarray, chunk_steps: int) -> Iterator[tuple[slice, np.ndarray]]:
    """Yield (slice, times) per chunk so callers can write results back in place."""
    for start in range(0, t_s.size, chunk_steps):
        sl = slice(start, min(start + chunk_steps, t_s.size))
        yield sl, t_s[sl]
