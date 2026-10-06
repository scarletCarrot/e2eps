from datetime import datetime, timezone

import numpy as np

from e2eps.constants import R_EARTH_EQ, WGS84_E2
from e2eps.frames import eci_to_ecef, geodetic_to_ecef, gmst_rad, julian_date


def test_julian_date_j2000():
    assert julian_date(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)) == 2451545.0


def test_gmst_at_j2000():
    g = np.rad2deg(gmst_rad(datetime(2000, 1, 1, 12, tzinfo=timezone.utc)))
    assert np.isclose(g, 280.46061837, atol=1e-6)


def test_geodetic_equator_and_pole():
    eq = geodetic_to_ecef(0.0, 0.0)
    assert np.allclose(eq, [R_EARTH_EQ, 0.0, 0.0])
    pole = geodetic_to_ecef(90.0, 0.0)
    b = R_EARTH_EQ * np.sqrt(1 - WGS84_E2)
    assert np.isclose(pole[2], b, atol=1e-3)


def test_geodetic_vectorised_shape():
    out = geodetic_to_ecef(np.array([0.0, 10.0, 20.0]), np.array([0.0, 5.0, 10.0]))
    assert out.shape == (3, 3)


def test_eci_to_ecef_rotation_sense():
    # A point on the ECI x-axis appears west of Greenwich after Earth rotates east.
    r = np.array([[[1.0, 0.0, 0.0]]])
    out = eci_to_ecef(r, np.array([np.pi / 2]))
    assert np.allclose(out[0, 0], [0.0, -1.0, 0.0], atol=1e-12)
