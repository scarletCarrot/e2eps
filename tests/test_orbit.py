from datetime import datetime, timezone

import numpy as np
import pytest

from e2eps.config.schema import WalkerDeltaConfig
from e2eps.constants import R_EARTH_EQ
from e2eps.orbit import WalkerDeltaPropagator, build_propagator

EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def make(j2=True, **kw):
    cfg = dict(altitude_km=550, inclination_deg=53, num_planes=6, sats_per_plane=10,
               phasing=1, include_j2=j2)
    cfg.update(kw)
    return WalkerDeltaPropagator(WalkerDeltaConfig(**cfg), EPOCH)


def test_shape_and_constant_radius():
    p = make()
    t = np.arange(0, 6000, 60.0)
    r = p.positions_ecef(t)
    assert r.shape == (t.size, 60, 3)
    assert np.allclose(np.linalg.norm(r, axis=-1), R_EARTH_EQ + 550e3, rtol=1e-12)


def test_max_latitude_matches_inclination():
    p = make()
    r = p.positions_eci(np.arange(0, p.period_s, 10.0))
    max_z = np.abs(r[..., 2]).max() / p.a
    assert np.isclose(max_z, np.sin(np.deg2rad(53)), atol=1e-4)


def test_two_body_returns_after_one_period():
    p = make(j2=False)
    r0 = p.positions_eci(np.array([0.0]))
    r1 = p.positions_eci(np.array([p.period_s]))
    assert np.allclose(r0, r1, atol=1e-3)


def test_period_about_95_minutes():
    assert 94 * 60 < make(j2=False).period_s < 97 * 60


def test_j2_raan_drift_magnitude():
    # 550 km / 53 deg drifts westward by roughly 4.5 deg/day
    drift_deg_day = np.rad2deg(make().raan_rate) * 86400
    assert -4.7 < drift_deg_day < -4.3


def test_walker_plane_spacing():
    p = make()
    raans = np.unique(np.round(np.rad2deg(p.raan0), 6))
    assert np.allclose(np.diff(raans), 60.0)


def test_satellites_are_distinct():
    r = make().positions_eci(np.array([0.0]))[0]
    d = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=-1)
    assert (d[~np.eye(60, dtype=bool)] > 1e3).all()


def test_factory_returns_walker():
    cfg = WalkerDeltaConfig(altitude_km=550, inclination_deg=53, num_planes=2, sats_per_plane=2)
    assert isinstance(build_propagator(cfg, EPOCH), WalkerDeltaPropagator)


@pytest.mark.parametrize("inc", [0.0, 97.6])
def test_other_inclinations(inc):
    p = make(inclination_deg=inc)
    r = p.positions_ecef(np.array([0.0, 600.0]))
    assert np.isfinite(r).all()
