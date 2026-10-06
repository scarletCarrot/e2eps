"""Latency Model (A17).

Physics-level latency for a bent-pipe path:

    one-way = (|terminal - satellite| + |satellite - gateway|) / c
              + processing + terrestrial backhaul

The gateway is the closest one visible from the serving satellite. Queuing
delay is out of scope - it belongs to the packet-level network emulator.
"""

from __future__ import annotations

import numpy as np

from ..capacity import gather_serving
from ..config.schema import LatencyConfig
from ..constants import C_LIGHT
from ..geometry import GeometryResult


class LatencyModel:
    def __init__(self, cfg: LatencyConfig):
        self.fixed_ms = cfg.processing_ms + cfg.terrestrial_ms

    @staticmethod
    def feeder_range(geo_gw: GeometryResult) -> np.ndarray:
        """Shortest satellite-gateway range per satellite: (T, S), inf if no gateway."""
        rng = np.where(geo_gw.visible, geo_gw.slant_range_m, np.inf)
        return rng.min(axis=2)

    def one_way_ms(self, geo_ut: GeometryResult, serving: np.ndarray,
                   feeder_range_m: np.ndarray) -> np.ndarray:
        """One-way latency per terminal: (T, G), NaN when not served."""
        user_leg = gather_serving(geo_ut.slant_range_m, serving).astype(np.float64)
        T = serving.shape[0]
        feeder_leg = feeder_range_m[np.arange(T)[:, None], np.clip(serving, 0, None)]
        feeder_leg = np.where(serving >= 0, feeder_leg, np.nan)
        return (user_leg + feeder_leg) / C_LIGHT * 1e3 + self.fixed_ms
