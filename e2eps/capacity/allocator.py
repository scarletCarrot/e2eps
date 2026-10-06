"""Capacity & Resource Allocator.

Baseline policy (A8, A14, A15):
  1. A terminal can use a satellite only if the link closes (Module B) and the
     satellite sees at least one gateway (bent-pipe payload).
  2. Each terminal connects to the highest-elevation usable satellite.
  3. A satellite's channel is shared equally (time-shared) by its terminals:
     throughput = link_rate / users_on_that_satellite.

The policy is isolated here so a load-aware or operator-specific scheduler
can replace it without touching geometry or RF.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import GeometryResult
from ..rf import LinkResult


@dataclass
class AllocationResult:
    serving_sat: np.ndarray            # (T, G) int32, -1 = no service
    users_per_sat: np.ndarray          # (T, S) int32
    throughput_bps: np.ndarray         # (T, G) float32, 0 when not served
    serving_link_rate_bps: np.ndarray  # (T, G) float32
    serving_modcod: np.ndarray         # (T, G) int16
    serving_elevation_deg: np.ndarray  # (T, G) float32, NaN when not served


def gather_serving(arr: np.ndarray, serving: np.ndarray, fill=np.nan) -> np.ndarray:
    """Pick arr[t, serving[t, g], g] for every (t, g): (T, S, G) -> (T, G)."""
    idx = np.clip(serving, 0, None)[:, None, :]
    out = np.take_along_axis(arr, idx, axis=1)[:, 0, :]
    return np.where(serving >= 0, out, fill)


class CapacityAllocator:
    def allocate(self, geo: GeometryResult, link: LinkResult,
                 sat_has_gateway: np.ndarray) -> AllocationResult:
        T, S, G = geo.shape
        usable = link.available & sat_has_gateway[:, :, None]

        score = np.where(usable, geo.elevation_deg, -np.inf)
        serving = np.argmax(score, axis=1).astype(np.int32)               # (T, G)
        served = np.isfinite(np.take_along_axis(score, serving[:, None, :], axis=1)[:, 0, :])
        serving[~served] = -1

        users = np.zeros((T, S), dtype=np.int32)
        t_idx, g_idx = np.nonzero(served)
        np.add.at(users, (t_idx, serving[t_idx, g_idx]), 1)

        rate = gather_serving(link.link_rate_bps, serving, fill=0.0).astype(np.float32)
        n_share = np.where(served, users[np.arange(T)[:, None], np.clip(serving, 0, None)], 1)
        throughput = (rate / n_share).astype(np.float32)

        return AllocationResult(
            serving_sat=serving,
            users_per_sat=users,
            throughput_bps=throughput,
            serving_link_rate_bps=rate,
            serving_modcod=gather_serving(link.modcod_idx, serving, fill=-1).astype(np.int16),
            serving_elevation_deg=gather_serving(geo.elevation_deg, serving).astype(np.float32),
        )
