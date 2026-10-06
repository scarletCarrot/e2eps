"""Propagation loss models (A10).

All functions are vectorised over NumPy arrays of any shape.
"""

from __future__ import annotations

import numpy as np

from ..constants import C_LIGHT

_MIN_EL_DEG = 5.0  # cosecant laws are invalid near the horizon


def free_space_loss_db(range_m, freq_hz):
    """Free-space path loss: 20 log10(4 pi d f / c)."""
    return 20.0 * np.log10(4.0 * np.pi * np.asarray(range_m) * freq_hz / C_LIGHT)


def gaseous_loss_db(elevation_deg, zenith_loss_db):
    """Oxygen + water vapour, cosecant scaling of the zenith value."""
    el = np.radians(np.maximum(elevation_deg, _MIN_EL_DEG))
    return zenith_loss_db / np.sin(el)


# ITU-R P.838-3 coefficients: f [GHz], kH, alphaH, kV, alphaV
_P838 = np.array([
    [10.0, 0.01217, 1.2571, 0.01129, 1.2156],
    [12.0, 0.02386, 1.1825, 0.02455, 1.1216],
    [15.0, 0.04481, 1.1233, 0.05008, 1.0440],
    [20.0, 0.09164, 1.0568, 0.09611, 0.9847],
    [25.0, 0.15710, 0.9991, 0.15330, 0.9491],
    [30.0, 0.24030, 0.9485, 0.22910, 0.9129],
])


def rain_coefficients(freq_ghz: float) -> tuple[float, float]:
    """(k, alpha) for circular polarisation, log-interpolated in frequency."""
    f = np.clip(freq_ghz, _P838[0, 0], _P838[-1, 0])
    lf = np.log10(_P838[:, 0])
    kh = 10 ** np.interp(np.log10(f), lf, np.log10(_P838[:, 1]))
    kv = 10 ** np.interp(np.log10(f), lf, np.log10(_P838[:, 3]))
    ah = np.interp(np.log10(f), lf, _P838[:, 2])
    av = np.interp(np.log10(f), lf, _P838[:, 4])
    k = (kh + kv) / 2.0
    alpha = (kh * ah + kv * av) / (2.0 * k)
    return float(k), float(alpha)


def rain_loss_db(elevation_deg, freq_ghz, rain_rate_mm_h, rain_height_km, station_alt_km=0.0):
    """Simplified slant-path rain attenuation (ITU-R P.838 + P.618-style reduction).

    gamma = k R^alpha            specific attenuation [dB/km]
    Ls    = (hR - hs) / sin(el)  slant path below the rain height [km]
    r     = 1 / (1 + Lg / L0)    path reduction, L0 = 35 exp(-0.015 R)
    A     = gamma * Ls * r
    """
    if rain_rate_mm_h <= 0:
        return np.zeros_like(np.asarray(elevation_deg, dtype=float))
    k, alpha = rain_coefficients(freq_ghz)
    gamma = k * rain_rate_mm_h**alpha
    el = np.radians(np.maximum(elevation_deg, _MIN_EL_DEG))
    ls = max(rain_height_km - station_alt_km, 0.0) / np.sin(el)
    lg = ls * np.cos(el)
    l0 = 35.0 * np.exp(-0.015 * min(rain_rate_mm_h, 100.0))
    return gamma * ls / (1.0 + lg / l0)
