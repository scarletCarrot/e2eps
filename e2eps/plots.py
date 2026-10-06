"""Result plots (PNG) for a quick visual check of a run."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .kpi import system_timeseries, terminal_kpis
from .rf import ModcodTable
from .simulation import SimulationResult


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_coverage_map(r: SimulationResult, path: Path) -> Path:
    rows = terminal_kpis(r)
    lat = np.array([x["lat_deg"] for x in rows])
    lon = np.array([x["lon_deg"] for x in rows])
    p50 = np.array([np.nan if x["throughput_p50_mbps"] is None else x["throughput_p50_mbps"]
                    for x in rows])
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sc = ax.scatter(lon, lat, c=p50, s=120, marker="s", cmap="viridis", edgecolors="none")
    fig.colorbar(sc, ax=ax, label="Median user throughput [Mbps]")
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(f"{r.scenario.name}: median throughput per terminal")
    ax.set_aspect(1.3)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_system_timeseries(r: SimulationResult, path: Path) -> Path:
    ts = system_timeseries(r)
    hours = ts["t_s"] / 3600.0
    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax1.plot(hours, ts["total_delivered_gbps"], lw=1.5, label="Delivered capacity")
    ax1.set_xlabel("Time [h]")
    ax1.set_ylabel("Delivered capacity [Gbps]")
    ax2 = ax1.twinx()
    ax2.plot(hours, ts["active_satellites"], color="tab:orange", lw=1, alpha=0.8,
             label="Active satellites")
    ax2.set_ylabel("Active satellites")
    ax1.set_title(f"{r.scenario.name}: system capacity over time")
    ax1.grid(alpha=0.3)
    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88))
    return _save(fig, path)


def plot_throughput_cdf(r: SimulationResult, path: Path) -> Path:
    thr = np.sort(r.throughput_bps[r.served] / 1e6)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if thr.size:
        ax.plot(thr, np.linspace(0, 1, thr.size), lw=2)
        for q in (5, 50, 95):
            v = np.percentile(thr, q)
            ax.axvline(v, ls="--", lw=0.8, color="grey")
            ax.text(v, 0.03 + q / 120, f" P{q} = {v:.0f}", fontsize=9)
    ax.set_xlabel("User throughput [Mbps]")
    ax.set_ylabel("CDF (all terminals, all time steps)")
    ax.set_title("User throughput distribution")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_modcod_usage(r: SimulationResult, path: Path) -> Path:
    mc = ModcodTable.dvb_s2()
    idx = r.modcod_idx[r.served]
    counts = np.bincount(idx, minlength=len(mc.names)) / max(idx.size, 1) * 100
    used = counts > 0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.array(mc.names)[used], counts[used])
    ax.set_ylabel("Share of served samples [%]")
    ax.set_title("MODCOD usage (adaptive coding and modulation)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def plot_terminal_timeline(r: SimulationResult, path: Path, terminal: int = 0) -> Path:
    hours = r.t_s / 3600.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(hours, r.throughput_bps[:, terminal] / 1e6, lw=1)
    a1.set_ylabel("Throughput [Mbps]")
    a1.set_title(f"Terminal '{r.terminals.ids[terminal]}': throughput and serving geometry")
    a1.grid(alpha=0.3)
    a2.plot(hours, r.elevation_deg[:, terminal], lw=1, color="tab:green")
    ho = np.flatnonzero(np.diff(r.serving_sat[:, terminal]) != 0) + 1
    a2.plot(hours[ho], r.elevation_deg[ho, terminal], "k|", ms=5, alpha=0.5, label="handover")
    a2.set_ylabel("Serving elevation [deg]")
    a2.set_xlabel("Time [h]")
    a2.legend(loc="lower right")
    a2.grid(alpha=0.3)
    return _save(fig, path)


def write_plots(r: SimulationResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return {
        "coverage_map": plot_coverage_map(r, out / "coverage_map.png"),
        "system_timeseries": plot_system_timeseries(r, out / "system_timeseries.png"),
        "throughput_cdf": plot_throughput_cdf(r, out / "throughput_cdf.png"),
        "modcod_usage": plot_modcod_usage(r, out / "modcod_usage.png"),
        "terminal_timeline": plot_terminal_timeline(r, out / "terminal_timeline.png"),
    }
