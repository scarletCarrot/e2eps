from pathlib import Path

import pytest

from e2eps.config import ScenarioError, load_scenario
from e2eps.config.schema import TerminalGrid, WalkerDeltaConfig

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "scenarios"


def test_reference_scenario_loads():
    sc = load_scenario(SCENARIOS / "leo_ku_europe.yaml")
    assert sc.constellation.num_sats == 528
    assert sc.simulation.num_steps == 721
    assert sc.epoch.tzinfo is not None
    # 5 named cities + 13 x 18 grid points
    assert len(sc.user_terminals.all_sites()) == 5 + 13 * 18


def test_small_scenario_uses_defaults():
    sc = load_scenario(SCENARIOS / "small_test.yaml")
    assert sc.user_terminals.min_elevation_deg == 25
    assert sc.rf.downlink.implementation_margin_db == 1.0
    assert sc.latency.processing_ms == 5.0


def test_grid_expansion():
    g = TerminalGrid(lat_min_deg=0, lat_max_deg=4, lon_min_deg=10, lon_max_deg=12, step_deg=2)
    assert len(g.sites()) == 3 * 2


def test_invalid_phasing_rejected():
    with pytest.raises(ValueError):
        WalkerDeltaConfig(altitude_km=550, inclination_deg=53, num_planes=4,
                          sats_per_plane=4, phasing=4)


def test_invalid_altitude_rejected():
    with pytest.raises(ValueError):
        WalkerDeltaConfig(altitude_km=36000, inclination_deg=0, num_planes=1, sats_per_plane=1)


def test_unknown_field_rejected(tmp_path):
    bad = (SCENARIOS / "small_test.yaml").read_text() + "\nunexpected_key: 1\n"
    p = tmp_path / "bad.yaml"
    p.write_text(bad)
    with pytest.raises(ScenarioError):
        load_scenario(p)


def test_missing_file():
    with pytest.raises(ScenarioError):
        load_scenario("does_not_exist.yaml")
