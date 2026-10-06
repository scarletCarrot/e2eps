"""Command-line interface.

    e2eps run scenarios/leo_ku_europe.yaml --out output/
    e2eps run scenarios/leo_ku_europe.yaml --rain-rate 25 --out output/rain
    e2eps link-budget scenarios/leo_ku_europe.yaml --elevation 25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

from . import __version__
from .config import ScenarioError, load_scenario
from .constants import R_EARTH_EQ


def _with_overrides(sc, args):
    dl = sc.rf.downlink
    if args.rain_rate is not None:
        dl = dl.model_copy(update={"rain_rate_mm_h": args.rain_rate})
    sim = sc.simulation
    if args.duration is not None:
        sim = sim.model_copy(update={"duration_s": args.duration})
    return sc.model_copy(update={"rf": sc.rf.model_copy(update={"downlink": dl}), "simulation": sim})


def cmd_run(args) -> int:
    from .report import write_outputs
    from .simulation import run_simulation

    sc = _with_overrides(load_scenario(args.scenario), args)
    result = run_simulation(sc)
    files = write_outputs(result, args.out)
    if not args.no_plots:
        try:
            from .plots import write_plots
            files.update(write_plots(result, args.out))
        except ImportError:
            pass

    kpi = json.loads(files["system_kpis"].read_text())
    print("\nSystem KPIs")
    for k in ("availability_pct", "throughput_p5_mbps", "throughput_p50_mbps",
              "throughput_p95_mbps", "latency_mean_ms", "latency_p95_ms",
              "total_delivered_gbps_mean", "active_satellites_mean"):
        print(f"  {k:28s} {kpi[k]}")
    print(f"\nOutputs written to {args.out}:")
    for p in files.values():
        print(f"  {p}")
    return 0


def cmd_link_budget(args) -> int:
    from .rf import LinkBudgetEngine

    sc = _with_overrides(load_scenario(args.scenario), args)
    h = sc.constellation.altitude_km * 1e3
    el = np.radians(args.elevation)
    re = R_EARTH_EQ
    eta = np.arcsin(re / (re + h) * np.cos(el))
    rng = np.sqrt((re + h) ** 2 - (re * np.cos(el)) ** 2) - re * np.sin(el)
    eng = LinkBudgetEngine(sc.rf.downlink, sc.user_terminals.g_over_t_dbk)
    b = eng.breakdown(args.elevation, rng, np.degrees(eta))
    mc_idx, eff = eng.modcods.select(np.array([b.esn0_db - sc.rf.downlink.implementation_margin_db]))

    print(f"\nDownlink budget - elevation {args.elevation:.1f} deg, "
          f"range {rng / 1e3:.0f} km, off-nadir {np.degrees(eta):.1f} deg")
    rows = [("Satellite EIRP", b.eirp_dbw, "dBW"), ("Scan loss", -b.scan_loss_db, "dB"),
            ("Free-space loss", -b.fspl_db, "dB"), ("Gaseous loss", -b.gas_db, "dB"),
            ("Rain loss", -b.rain_db, "dB"), ("Misc losses", -b.misc_db, "dB"),
            ("Terminal G/T", b.g_over_t_dbk, "dB/K"), ("Boltzmann", 228.6, "dBW/K/Hz"),
            ("C/N0", b.cn0_dbhz, "dBHz"), ("Es/N0", b.esn0_db, "dB")]
    for name, v, unit in rows:
        print(f"  {name:18s} {v:9.2f} {unit}")
    name = eng.modcods.names[mc_idx[0]] if mc_idx[0] >= 0 else "OUTAGE"
    print(f"  {'MODCOD':18s} {name}  ({eff[0] * eng.symbol_rate / 1e6:.0f} Mbps full channel)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="e2eps", description="E2EPS - End-to-End Performance Simulator")
    p.add_argument("--version", action="version", version=f"e2eps {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("scenario", help="scenario YAML file")
        sp.add_argument("--rain-rate", type=float, help="override rain rate [mm/h]")
        sp.add_argument("--duration", type=float, help="override duration [s]")

    r = sub.add_parser("run", help="run a full simulation")
    common(r)
    r.add_argument("--out", default="output", help="output directory")
    r.add_argument("--no-plots", action="store_true")
    r.set_defaults(func=cmd_run)

    lb = sub.add_parser("link-budget", help="print a single-link budget")
    common(lb)
    lb.add_argument("--elevation", type=float, default=45.0, help="elevation angle [deg]")
    lb.set_defaults(func=cmd_link_budget)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    try:
        return args.func(args)
    except ScenarioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
