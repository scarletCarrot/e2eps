"""KPI Aggregator.

Reduces the (T, G) time series into the numbers engineers actually compare:

Per terminal
    availability_pct      % of time steps with a closing link (A16)
    throughput_mean_mbps  time-average, outages counted as 0
    throughput_p5/p50/p95 percentiles while served (P5 = "worst typical")
    latency_mean/p95_ms   one-way, while served
    visible_sats_mean     coverage depth
    handovers_per_hour    serving-satellite changes

System
    availability, throughput, latency percentiles over all terminals and time,
    active satellites and satellite load.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..simulation import SimulationResult


def _pct(a: np.ndarray, q: float, axis=None):
    """nanpercentile that returns NaN (no warning) for all-NaN columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanpercentile(a, q, axis=axis)


def _nanmean(a: np.ndarray, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def terminal_kpis(r: SimulationResult) -> list[dict]:
    served = r.served
    thr = r.throughput_bps / 1e6
    thr_served = np.where(served, thr, np.nan)
    hours = (r.t_s[-1] - r.t_s[0]) / 3600.0 or 1.0
    ho = ((r.serving_sat[1:] != r.serving_sat[:-1])
          & (r.serving_sat[1:] >= 0) & (r.serving_sat[:-1] >= 0)).sum(axis=0)

    cols = dict(
        availability_pct=100.0 * served.mean(axis=0),
        throughput_mean_mbps=thr.mean(axis=0),
        throughput_p5_mbps=_pct(thr_served, 5, axis=0),
        throughput_p50_mbps=_pct(thr_served, 50, axis=0),
        throughput_p95_mbps=_pct(thr_served, 95, axis=0),
        latency_mean_ms=_nanmean(r.latency_ms, axis=0),
        latency_p95_ms=_pct(r.latency_ms, 95, axis=0),
        visible_sats_mean=r.visible_sats.mean(axis=0),
        handovers_per_hour=ho / hours,
    )
    rows = []
    for g, tid in enumerate(r.terminals.ids):
        row = {"terminal": tid,
               "lat_deg": float(r.terminals.lat_deg[g]),
               "lon_deg": float(r.terminals.lon_deg[g])}
        row.update({k: (None if np.isnan(v[g]) else round(float(v[g]), 3)) for k, v in cols.items()})
        rows.append(row)
    return rows


def system_kpis(r: SimulationResult) -> dict:
    served = r.served
    thr = r.throughput_bps[served] / 1e6
    lat = r.latency_ms[served]
    active = (r.users_per_sat > 0).sum(axis=1)
    load = r.users_per_sat[r.users_per_sat > 0]
    total_tp = r.throughput_bps.sum(axis=1) / 1e9

    def f(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 3)

    return {
        "scenario": r.scenario.name,
        "time_steps": int(r.t_s.size),
        "satellites": int(r.num_sats),
        "terminals": int(r.terminals.size),
        "availability_pct": f(100.0 * served.mean()),
        "terminals_below_99pct_availability": int((served.mean(axis=0) < 0.99).sum()),
        "throughput_mean_mbps": f(thr.mean()) if thr.size else None,
        "throughput_p5_mbps": f(np.percentile(thr, 5)) if thr.size else None,
        "throughput_p50_mbps": f(np.percentile(thr, 50)) if thr.size else None,
        "throughput_p95_mbps": f(np.percentile(thr, 95)) if thr.size else None,
        "latency_mean_ms": f(lat.mean()) if lat.size else None,
        "latency_p95_ms": f(np.percentile(lat, 95)) if lat.size else None,
        "total_delivered_gbps_mean": f(total_tp.mean()),
        "active_satellites_mean": f(active.mean()),
        "users_per_active_satellite_mean": f(load.mean()) if load.size else None,
        "runtime_s": {k: round(v, 2) for k, v in r.timings_s.items()},
    }


def system_timeseries(r: SimulationResult) -> dict[str, np.ndarray]:
    served = r.served
    return {
        "t_s": r.t_s,
        "availability_pct": 100.0 * served.mean(axis=1),
        "total_delivered_gbps": r.throughput_bps.sum(axis=1) / 1e9,
        "throughput_mean_mbps": r.throughput_bps.mean(axis=1) / 1e6,
        "latency_mean_ms": _nanmean(r.latency_ms, axis=1),
        "active_satellites": (r.users_per_sat > 0).sum(axis=1),
    }
