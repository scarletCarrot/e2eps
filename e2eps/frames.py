"""Reference frames and time.

ECI here is a simplified inertial frame aligned with the equator and the
vernal equinox at epoch. Earth rotation is modelled with GMST only - no
precession, nutation or polar motion (A4). The resulting error is metres,
which is irrelevant at link-budget level.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .constants import OMEGA_EARTH, R_EARTH_EQ, WGS84_E2

_J2000_JD = 2451545.0


def julian_date(t: datetime) -> float:
    """Julian date of a UTC datetime."""
    t = t.astimezone(timezone.utc)
    unix = t.timestamp()
    return unix / 86400.0 + 2440587.5


def gmst_rad(t: datetime) -> float:
    """Greenwich Mean Sidereal Time [rad] (IAU 1982 expression)."""
    jd = julian_date(t)
    d = jd - _J2000_JD
    tc = d / 36525.0
    deg = 280.46061837 + 360.98564736629 * d + 0.000387933 * tc**2 - tc**3 / 38710000.0
    return np.deg2rad(deg % 360.0)


def earth_rotation_angle(epoch: datetime, t_s: np.ndarray) -> np.ndarray:
    """Earth rotation angle theta(t) [rad] for seconds since epoch."""
    return gmst_rad(epoch) + OMEGA_EARTH * np.asarray(t_s, dtype=float)


def eci_to_ecef(r_eci: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate ECI positions into the Earth-fixed frame.

    r_eci: (T, S, 3), theta: (T,) -> (T, S, 3)
    """
    c = np.cos(theta)[:, None]
    s = np.sin(theta)[:, None]
    x, y, z = r_eci[..., 0], r_eci[..., 1], r_eci[..., 2]
    return np.stack((c * x + s * y, -s * x + c * y, z), axis=-1)


def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0.0) -> np.ndarray:
    """WGS84 geodetic coordinates to ECEF [m]. Inputs broadcast; output (..., 3)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    alt = np.asarray(alt_m, dtype=float)
    sin_lat = np.sin(lat)
    n = R_EARTH_EQ / np.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x = (n + alt) * np.cos(lat) * np.cos(lon)
    y = (n + alt) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - WGS84_E2) + alt) * sin_lat
    return np.stack(np.broadcast_arrays(x, y, z), axis=-1)
