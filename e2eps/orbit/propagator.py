"""Orbit propagation.

Design: every propagator answers one question - "where are all satellites at
these times?" - and returns an array (T, S, 3) in ECEF metres. Downstream
modules depend only on that contract, so this analytic Walker-Delta model can
be replaced by SGP4 (TLEs) or operational ephemeris without touching them.

Model (A1-A4): circular orbits, two-body motion plus J2 secular drift of the
right ascension of the ascending node (RAAN) and argument of latitude. J2 is
the dominant perturbation in LEO over hours to days; drag and higher harmonics
are ignored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

import numpy as np

from ..config.schema import WalkerDeltaConfig
from ..constants import J2, MU_EARTH, R_EARTH_EQ
from ..frames import earth_rotation_angle, eci_to_ecef


class Propagator(Protocol):
    num_sats: int

    def positions_ecef(self, t_s: np.ndarray) -> np.ndarray:
        """Satellite positions (T, S, 3) [m] for seconds since epoch (T,)."""
        ...


class WalkerDeltaPropagator:
    """Analytic propagator for a Walker-Delta shell i:T/P/F."""

    def __init__(self, cfg: WalkerDeltaConfig, epoch: datetime):
        self.cfg = cfg
        self.epoch = epoch
        self.num_sats = cfg.num_sats

        self.a = R_EARTH_EQ + cfg.altitude_km * 1e3          # semi-major axis [m]
        self.inc = np.deg2rad(cfg.inclination_deg)
        self.n = np.sqrt(MU_EARTH / self.a**3)                # mean motion [rad/s]

        if cfg.include_j2:
            k = 1.5 * J2 * (R_EARTH_EQ / self.a) ** 2
            cos_i = np.cos(self.inc)
            self.raan_rate = -k * self.n * cos_i                       # [rad/s]
            self.u_rate = self.n * (1.0 + k * (4.0 * cos_i**2 - 1.0))  # [rad/s]
        else:
            self.raan_rate = 0.0
            self.u_rate = self.n

        # Walker-Delta initial elements, satellite index s = plane * spp + slot
        p_idx, k_idx = np.divmod(np.arange(cfg.num_sats), cfg.sats_per_plane)
        self.plane = p_idx
        self.raan0 = np.deg2rad(cfg.raan_spread_deg) * p_idx / cfg.num_planes
        self.u0 = (2 * np.pi * k_idx / cfg.sats_per_plane
                   + 2 * np.pi * cfg.phasing * p_idx / cfg.num_sats)

    @property
    def period_s(self) -> float:
        return 2 * np.pi / self.u_rate

    def positions_eci(self, t_s: np.ndarray) -> np.ndarray:
        t = np.asarray(t_s, dtype=float)[:, None]          # (T, 1)
        raan = self.raan0[None, :] + self.raan_rate * t    # (T, S)
        u = self.u0[None, :] + self.u_rate * t             # (T, S)

        cos_r, sin_r = np.cos(raan), np.sin(raan)
        cos_u, sin_u = np.cos(u), np.sin(u)
        cos_i, sin_i = np.cos(self.inc), np.sin(self.inc)

        x = cos_r * cos_u - sin_r * sin_u * cos_i
        y = sin_r * cos_u + cos_r * sin_u * cos_i
        z = sin_u * sin_i
        return self.a * np.stack((x, y, z), axis=-1)

    def positions_ecef(self, t_s: np.ndarray) -> np.ndarray:
        theta = earth_rotation_angle(self.epoch, t_s)
        return eci_to_ecef(self.positions_eci(t_s), theta)


def build_propagator(cfg: WalkerDeltaConfig, epoch: datetime) -> Propagator:
    """Factory - the single place to add SGP4 / ephemeris-file propagators."""
    if cfg.type == "walker_delta":
        return WalkerDeltaPropagator(cfg, epoch)
    raise ValueError(f"unsupported constellation type: {cfg.type}")
