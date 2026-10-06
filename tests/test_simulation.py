import json
from pathlib import Path

import numpy as np

from e2eps.cli import main
from e2eps.config import load_scenario
from e2eps.kpi import system_kpis, terminal_kpis
from e2eps.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "scenarios" / "small_test.yaml"


def test_run_simulation_shapes_and_sanity():
    r = run_simulation(load_scenario(SMALL))
    T, G = r.t_s.size, r.terminals.size
    assert r.serving_sat.shape == (T, G) and r.users_per_sat.shape == (T, r.num_sats)
    served = r.served
    # Conservation: users counted on satellites == served terminals
    assert (r.users_per_sat.sum(axis=1) == served.sum(axis=1)).all()
    assert (r.throughput_bps[~served] == 0).all()
    assert (r.throughput_bps[served] > 0).all()
    assert np.isnan(r.latency_ms[~served]).all()
    # One-way bent-pipe latency: fixed 15 ms + a few ms propagation
    lat = r.latency_ms[served]
    assert (lat > 15.0).all() and (lat < 30.0).all()


def test_kpis_consistent():
    r = run_simulation(load_scenario(SMALL))
    rows = terminal_kpis(r)
    sk = system_kpis(r)
    assert [row["terminal"] for row in rows] == ["madrid", "london"]
    for row in rows:
        assert 0 <= row["availability_pct"] <= 100
    assert 0 <= sk["availability_pct"] <= 100
    assert sk["satellites"] == 60


def test_cli_run_writes_outputs(tmp_path, capsys):
    rc = main(["run", str(SMALL), "--out", str(tmp_path), "--no-plots"])
    assert rc == 0
    for name in ("system_kpis.json", "terminal_kpis.csv", "system_timeseries.csv"):
        assert (tmp_path / name).exists()
    kpi = json.loads((tmp_path / "system_kpis.json").read_text())
    assert kpi["scenario"] == "small-test"
    assert "System KPIs" in capsys.readouterr().out


def test_cli_rain_override_reduces_throughput(tmp_path):
    main(["run", str(SMALL), "--out", str(tmp_path / "clear"), "--no-plots"])
    main(["run", str(SMALL), "--out", str(tmp_path / "rain"), "--no-plots", "--rain-rate", "50"])
    clear = json.loads((tmp_path / "clear" / "system_kpis.json").read_text())
    rain = json.loads((tmp_path / "rain" / "system_kpis.json").read_text())
    assert rain["throughput_p50_mbps"] < clear["throughput_p50_mbps"]


def test_cli_link_budget(capsys):
    assert main(["link-budget", str(SMALL), "--elevation", "25"]) == 0
    out = capsys.readouterr().out
    assert "Es/N0" in out and "MODCOD" in out


def test_cli_bad_scenario(tmp_path, capsys):
    assert main(["run", str(tmp_path / "missing.yaml")]) == 2
