from datetime import datetime, timezone

import numpy as np
import pytest

from e2eps.config.schema import Site, WalkerDeltaConfig
from e2eps.constants import R_EARTH_EQ
from e2eps.geometry import GeometryEngine, GroundPoints
from e2eps.orbit import WalkerDeltaPropagator

R = R_EARTH_EQ
H = 550e3


def equator_point(min_el=0.0):
    return GroundPoints.from_sites([Site(id="eq", lat_deg=0.0, lon_deg=0.0)], min_el)


def run(sats, ground, **kw):
    sat = np.asarray(sats, dtype=float).reshape(1, -1, 3)
    return GeometryEngine(ground, **kw).compute(sat)


def test_overhead_satellite():
    res = run([[R + H, 0, 0]], equator_point())
    assert np.isclose(res.elevation_deg[0, 0, 0], 90.0)
    assert np.isclose(res.slant_range_m[0, 0, 0], H)
    assert np.isclose(res.off_nadir_deg[0, 0, 0], 0.0, atol=1e-6)
    assert res.visible[0, 0, 0]


def test_horizon_and_45_degrees():
    res = run([[R, 1e6, 0], [R + 1e6, 1e6, 0]], equator_point())
    assert np.isclose(res.elevation_deg[0, 0, 0], 0.0, atol=1e-9)
    assert np.isclose(res.elevation_deg[0, 1, 0], 45.0)


def test_satellite_behind_earth_not_visible():
    res = run([[-(R + H), 0, 0]], equator_point())
    assert res.elevation_deg[0, 0, 0] < 0
    assert not res.visible[0, 0, 0]


@pytest.mark.parametrize("el_deg", [10.0, 25.0, 40.0, 70.0])
def test_off_nadir_matches_spherical_law_of_sines(el_deg):
    # Place a satellite at altitude H seen from the equator point at elevation el.
    el = np.radians(el_deg)
    eta = np.arcsin(R / (R + H) * np.cos(el))        # off-nadir at satellite
    central = np.pi / 2 - el - eta                   # Earth central angle
    sat = [(R + H) * np.cos(central), (R + H) * np.sin(central), 0.0]
    res = run([sat], equator_point())
    assert np.isclose(res.elevation_deg[0, 0, 0], el_deg, atol=1e-6)
    assert np.isclose(res.off_nadir_deg[0, 0, 0], np.degrees(eta), atol=1e-6)


def test_min_elevation_mask_and_scan_limit():
    el = np.radians(20.0)
    eta = np.arcsin(R / (R + H) * np.cos(el))
    c = np.pi / 2 - el - eta
    sat = [[(R + H) * np.cos(c), (R + H) * np.sin(c), 0.0]]
    assert run(sat, equator_point(10.0)).visible[0, 0, 0]
    assert not run(sat, equator_point(25.0)).visible[0, 0, 0]
    assert not run(sat, equator_point(10.0), max_off_nadir_deg=50.0).visible[0, 0, 0]


def test_constellation_shapes_and_coverage():
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prop = WalkerDeltaPropagator(
        WalkerDeltaConfig(altitude_km=550, inclination_deg=53, num_planes=24, sats_per_plane=22),
        epoch)
    sites = [Site(id="madrid", lat_deg=40.4, lon_deg=-3.7), Site(id="oslo", lat_deg=59.9, lon_deg=10.7)]
    ground = GroundPoints.from_sites(sites, 25.0)
    res = GeometryEngine(ground).compute(prop.positions_ecef(np.arange(0, 3600, 60.0)))
    assert res.shape == (60, 528, 2)
    counts = res.visible_count()
    # A 528-sat shell at 53 deg keeps mid-latitude Europe covered
    assert counts.min() >= 1
    assert (res.slant_range_m[res.visible] < 1.3e6).all()


# ---------------------------------------------------------------- optimisation
from e2eps.geometry.reference import geometry_broadcast, geometry_loop  # noqa: E402


def _random_case(seed=0, T=3, S=12, G=7):
    rng = np.random.default_rng(seed)
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    prop = WalkerDeltaPropagator(
        WalkerDeltaConfig(altitude_km=550, inclination_deg=53, num_planes=3, sats_per_plane=4),
        epoch)
    sat = prop.positions_ecef(rng.uniform(0, 6000, T))
    sites = [Site(id=f"p{i}", lat_deg=float(rng.uniform(-60, 60)),
                  lon_deg=float(rng.uniform(-180, 180)), alt_m=float(rng.uniform(0, 2000)))
             for i in range(G)]
    return sat, GroundPoints.from_sites(sites, 25.0)


def test_engine_matches_reference_implementations():
    sat, ground = _random_case()
    res = GeometryEngine(ground).compute(sat)
    for el, rng_, nad in (geometry_loop(sat, ground), geometry_broadcast(sat, ground)):
        assert np.allclose(res.elevation_deg, el, atol=1e-6)
        assert np.allclose(res.slant_range_m, rng_, atol=1e-3)     # millimetres
        assert np.allclose(res.off_nadir_deg, nad, atol=1e-6)


def test_float32_output_close_to_float64():
    sat, ground = _random_case(seed=1)
    r64 = GeometryEngine(ground).compute(sat)
    r32 = GeometryEngine(ground, dtype=np.float32).compute(sat)
    assert r32.slant_range_m.dtype == np.float32
    assert np.allclose(r32.slant_range_m, r64.slant_range_m, rtol=1e-6)
    assert np.allclose(r32.elevation_deg, r64.elevation_deg, atol=1e-4)
    assert np.array_equal(r32.visible, r64.visible)
