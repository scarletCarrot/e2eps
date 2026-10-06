"""Simulation Orchestrator.

Runs the per-chunk pipeline and keeps only compact (T, G) / (T, S) outputs,
so memory depends on chunk size, not on the number of time steps:

    propagate -> geometry (terminals, gateways) -> link budget
              -> allocation -> latency -> store
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .capacity import CapacityAllocator
from .config.schema import Scenario
from .geometry import GeometryEngine, GroundPoints
from .latency import LatencyModel
from .orbit import build_propagator
from .rf import LinkBudgetEngine
from .timegrid import iter_chunks, make_time_grid

log = logging.getLogger("e2eps")


@dataclass
class SimulationResult:
    scenario: Scenario
    t_s: np.ndarray                    # (T,)
    terminals: GroundPoints
    num_sats: int
    serving_sat: np.ndarray            # (T, G) int32
    throughput_bps: np.ndarray         # (T, G) float32
    link_rate_bps: np.ndarray          # (T, G) float32
    modcod_idx: np.ndarray             # (T, G) int16
    elevation_deg: np.ndarray          # (T, G) float32
    latency_ms: np.ndarray             # (T, G) float32
    visible_sats: np.ndarray           # (T, G) int16
    users_per_sat: np.ndarray          # (T, S) int32
    timings_s: dict = field(default_factory=dict)

    @property
    def served(self) -> np.ndarray:
        return self.serving_sat >= 0


def run_simulation(sc: Scenario) -> SimulationResult:
    t_grid = make_time_grid(sc.simulation)
    prop = build_propagator(sc.constellation, sc.epoch)
    terminals = GroundPoints.from_sites(sc.user_terminals.all_sites(),
                                        sc.user_terminals.min_elevation_deg)
    gateways = GroundPoints.from_sites(sc.gateways.sites, sc.gateways.min_elevation_deg)

    geo_ut_engine = GeometryEngine(terminals, dtype=np.float32)
    geo_gw_engine = GeometryEngine(gateways, dtype=np.float32)
    link_engine = LinkBudgetEngine(sc.rf.downlink, sc.user_terminals.g_over_t_dbk)
    allocator = CapacityAllocator()
    latency = LatencyModel(sc.latency)

    T, S, G = t_grid.size, prop.num_sats, terminals.size
    out = dict(
        serving_sat=np.full((T, G), -1, np.int32),
        throughput_bps=np.zeros((T, G), np.float32),
        link_rate_bps=np.zeros((T, G), np.float32),
        modcod_idx=np.full((T, G), -1, np.int16),
        elevation_deg=np.full((T, G), np.nan, np.float32),
        latency_ms=np.full((T, G), np.nan, np.float32),
        visible_sats=np.zeros((T, G), np.int16),
        users_per_sat=np.zeros((T, S), np.int32),
    )
    timings = dict.fromkeys(["orbit", "geometry", "link_budget", "allocation", "latency"], 0.0)

    log.info("scenario %s: %d steps x %d sats x %d terminals (%d gateways)",
             sc.name, T, S, G, gateways.size)

    for sl, t_chunk in iter_chunks(t_grid, sc.simulation.chunk_steps):
        t0 = time.perf_counter()
        sat = prop.positions_ecef(t_chunk)
        t1 = time.perf_counter()
        geo_ut = geo_ut_engine.compute(sat)
        geo_gw = geo_gw_engine.compute(sat)
        t2 = time.perf_counter()
        link = link_engine.compute(geo_ut)
        t3 = time.perf_counter()
        feeder = latency.feeder_range(geo_gw)
        alloc = allocator.allocate(geo_ut, link, np.isfinite(feeder))
        t4 = time.perf_counter()
        lat = latency.one_way_ms(geo_ut, alloc.serving_sat, feeder)
        t5 = time.perf_counter()

        out["serving_sat"][sl] = alloc.serving_sat
        out["throughput_bps"][sl] = alloc.throughput_bps
        out["link_rate_bps"][sl] = alloc.serving_link_rate_bps
        out["modcod_idx"][sl] = alloc.serving_modcod
        out["elevation_deg"][sl] = alloc.serving_elevation_deg
        out["latency_ms"][sl] = lat
        out["visible_sats"][sl] = geo_ut.visible_count()
        out["users_per_sat"][sl] = alloc.users_per_sat

        for k, dt in zip(timings, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
            timings[k] += dt

    timings["total"] = sum(timings.values())
    log.info("done in %.1f s (%s)", timings["total"],
             ", ".join(f"{k} {v:.1f}s" for k, v in timings.items() if k != "total"))
    return SimulationResult(scenario=sc, t_s=t_grid, terminals=terminals, num_sats=S,
                            timings_s=timings, **out)
