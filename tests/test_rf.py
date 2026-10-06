from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from e2eps.config import load_scenario
from e2eps.config.schema import DownlinkConfig
from e2eps.geometry import GeometryEngine, GeometryResult, GroundPoints
from e2eps.orbit import build_propagator
from e2eps.rf import LinkBudgetEngine, ModcodTable
from e2eps.rf.propagation import free_space_loss_db, gaseous_loss_db, rain_loss_db

ROOT = Path(__file__).resolve().parents[1]
DL = DownlinkConfig(frequency_ghz=11.7, bandwidth_mhz=240, sat_eirp_dbw=36.0)


# ------------------------------------------------------------------ losses
def test_free_space_loss_reference_value():
    # 550 km at 11.7 GHz: 92.45 + 20log10(550) + 20log10(11.7) = 168.6 dB
    assert np.isclose(free_space_loss_db(550e3, 11.7e9), 168.62, atol=0.02)


def test_free_space_loss_6db_per_doubling():
    assert np.isclose(free_space_loss_db(2e6, 12e9) - free_space_loss_db(1e6, 12e9), 6.02, atol=0.01)


def test_gaseous_loss_cosecant():
    assert np.isclose(gaseous_loss_db(90.0, 0.3), 0.3)
    assert np.isclose(gaseous_loss_db(30.0, 0.3), 0.6)


def test_rain_loss_behaviour():
    assert np.all(rain_loss_db(np.array([30.0]), 11.7, 0.0, 4.0) == 0.0)
    a_light = rain_loss_db(45.0, 11.7, 5.0, 4.0)
    a_heavy = rain_loss_db(45.0, 11.7, 30.0, 4.0)
    a_low = rain_loss_db(25.0, 11.7, 30.0, 4.0)
    assert 0 < a_light < a_heavy < a_low
    assert 1.0 < a_heavy < 10.0          # plausible Ku-band heavy-rain fade


# ------------------------------------------------------------------ modcod
def test_modcod_selection_boundaries():
    mc = ModcodTable.dvb_s2()
    idx, eff = mc.select(np.array([-5.0, mc.min_esn0_db, 1.0, 30.0]))
    assert idx[0] == -1 and eff[0] == 0.0
    assert idx[1] == 0
    assert mc.names[idx[2]] == "QPSK 1/2"
    assert np.isclose(eff[3], 4.453)


def test_modcod_efficiency_monotonic():
    assert np.all(np.diff(ModcodTable.dvb_s2().efficiency) >= 0)


# ------------------------------------------------------------- link budget
def test_nadir_esn0_hand_calculation():
    eng = LinkBudgetEngine(DL, g_over_t_dbk=12.0)
    b = eng.breakdown(elevation_deg=90.0, slant_range_m=550e3, off_nadir_deg=0.0)
    # 36 - 168.62 - 0.3 - 1.0 + 12 + 228.6 - 10log10(240e6/1.1)
    expected = 36 - 168.62 - 0.3 - 1.0 + 12 + 228.6 - 10 * np.log10(240e6 / 1.1)
    assert np.isclose(b.esn0_db, expected, atol=0.03)
    assert b.scan_loss_db == pytest.approx(0.0, abs=1e-9)


def test_edge_of_coverage_is_worse_than_nadir():
    eng = LinkBudgetEngine(DL, g_over_t_dbk=12.0)
    nadir = eng.breakdown(90.0, 550e3, 0.0).esn0_db
    edge = eng.breakdown(25.0, 1.12e6, 56.0).esn0_db
    assert nadir - edge > 8.0     # path loss + scan loss + atmosphere


def _geo(el, rng, nad, vis):
    a = lambda x, d=float: np.asarray(x, dtype=d).reshape(1, -1, 1)
    return GeometryResult(a(el), a(rng), a(nad), a(vis, bool))


def test_compute_masks_invisible_and_out_of_scan():
    eng = LinkBudgetEngine(DL, g_over_t_dbk=12.0)
    res = eng.compute(_geo([90, 25, 50, 30], [550e3, 1.12e6, 7e5, 9e5], [0, 56, 70, 50],
                           [True, True, True, False]))
    assert res.available[0, :, 0].tolist() == [True, True, False, False]
    assert res.link_rate_bps[0, 2, 0] == 0 and np.isnan(res.esn0_db[0, 3, 0])
    assert res.link_rate_bps[0, 0, 0] > res.link_rate_bps[0, 1, 0] > 0


def test_rain_reduces_rate():
    clear = LinkBudgetEngine(DL, 12.0)
    rain = LinkBudgetEngine(DL.model_copy(update={"rain_rate_mm_h": 40.0}), 12.0)
    g = _geo([30], [1.0e6], [50], [True])
    assert rain.compute(g).link_rate_bps[0, 0, 0] < clear.compute(g).link_rate_bps[0, 0, 0]


def test_reference_scenario_end_to_end():
    sc = load_scenario(ROOT / "scenarios" / "small_test.yaml")
    prop = build_propagator(sc.constellation, sc.epoch)
    ground = GroundPoints.from_sites(sc.user_terminals.all_sites(), sc.user_terminals.min_elevation_deg)
    geo = GeometryEngine(ground).compute(prop.positions_ecef(np.arange(0, 3600, 60.0)))
    link = LinkBudgetEngine(sc.rf.downlink, sc.user_terminals.g_over_t_dbk).compute(geo)
    # Every visible link within scan range closes at least the lowest MODCOD in clear sky
    in_scan = geo.visible & (geo.off_nadir_deg <= sc.rf.downlink.max_scan_deg)
    assert np.array_equal(link.available, in_scan)
    rates = link.link_rate_bps[link.available]
    assert 300e6 < rates.max() < 1.0e9
