"""Write simulation outputs to disk (CSV + JSON)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .kpi import system_kpis, system_timeseries, terminal_kpis
from .simulation import SimulationResult


def write_outputs(r: SimulationResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {}

    sys_kpi = system_kpis(r)
    files["system_kpis"] = out / "system_kpis.json"
    files["system_kpis"].write_text(json.dumps(sys_kpi, indent=2), encoding="utf-8")

    rows = terminal_kpis(r)
    files["terminal_kpis"] = out / "terminal_kpis.csv"
    with files["terminal_kpis"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ts = system_timeseries(r)
    files["timeseries"] = out / "system_timeseries.csv"
    with files["timeseries"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(ts))
        for i in range(r.t_s.size):
            w.writerow([round(float(ts[k][i]), 4) for k in ts])

    return files
