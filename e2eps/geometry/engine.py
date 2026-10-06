"""Module A - Geometry & Visibility Engine.

Turns satellite and ground positions into the geometric quantities every
downstream model needs:

    elevation   (T, S, G) [deg]  visibility, atmospheric path, rain loss
    slant range (T, S, G) [m]    free-space path loss, propagation delay
    off-nadir   (T, S, G) [deg]  satellite antenna scan loss
    visible     (T, S, G) bool   which links can physically exist

T = time steps in the chunk, S = satellites, G = ground points.

Performance
-----------
The naive approach builds the ground-to-satellite vector d = sat - ground for
every link, i.e. a (T, S, G, 3) array. Here every quantity is rewritten in
terms of dot products, so it reduces to two matrix multiplications that NumPy
hands to BLAS:

    |d|^2       = |sat|^2 - 2 sat.g + |g|^2
    d.up        = sat.up - g.up
    sat.d       = |sat|^2 - sat.g

No 4-D intermediate is created and work arrays are updated in place, so
memory and runtime both drop several-fold (see benchmarks/results.md). Maths is done in float64; outputs can be stored
as float32 to halve memory again.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ground import GroundPoints


@dataclass
class GeometryResult:
    elevation_deg: np.ndarray
    slant_range_m: np.ndarray
    off_nadir_deg: np.ndarray
    visible: np.ndarray

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.visible.shape

    def visible_count(self) -> np.ndarray:
        """Number of satellites in view per ground point: (T, G)."""
        return self.visible.sum(axis=1)


class GeometryEngine:
    """Vectorised line-of-sight geometry between satellites and ground points.

    Parameters
    ----------
    ground : GroundPoints
        Terminals or gateways, prepared once per run.
    max_off_nadir_deg : float, optional
        Satellite antenna steering limit. Links beyond it are not visible.
    dtype : numpy dtype
        Storage type of the output angle and range arrays (float64 or float32).
    """

    def __init__(self, ground: GroundPoints, max_off_nadir_deg: float | None = None,
                 dtype=np.float64):
        self.ground = ground
        self.max_off_nadir_deg = max_off_nadir_deg
        self.dtype = np.dtype(dtype)

        # Ground-only terms, computed once
        self._g_t = np.ascontiguousarray(ground.ecef.T)               # (3, G)
        self._up_t = np.ascontiguousarray(ground.up.T)                # (3, G)
        self._g_sq = np.einsum("gk,gk->g", ground.ecef, ground.ecef)  # (G,)
        self._g_up = np.einsum("gk,gk->g", ground.ecef, ground.up)    # (G,)
        self._sin_min_el = np.sin(np.radians(ground.min_elevation_deg))

    def compute(self, sat_ecef: np.ndarray) -> GeometryResult:
        """sat_ecef: (T, S, 3) [m] -> GeometryResult with (T, S, G) arrays."""
        T, S, _ = sat_ecef.shape
        sat = np.ascontiguousarray(sat_ecef, dtype=np.float64).reshape(T * S, 3)
        s_sq = np.einsum("nk,nk->n", sat, sat)[:, None]                # (TS, 1)

        # Three (TS, G) float64 work arrays, updated in place to cap peak memory.
        a = sat @ self._g_t                                            # sat.g   (BLAS)
        c = sat @ self._up_t                                           # sat.up  (BLAS)

        rng = np.multiply(a, -2.0)                                     # |d|^2 -> |d|
        rng += s_sq
        rng += self._g_sq
        np.maximum(rng, 0.0, out=rng)
        np.sqrt(rng, out=rng)

        c -= self._g_up                                                # d.up -> sin(el)
        c /= rng
        np.clip(c, -1.0, 1.0, out=c)
        visible = c >= self._sin_min_el

        np.negative(a, out=a)                                          # sat.d -> cos(nadir)
        a += s_sq
        a /= np.sqrt(s_sq)
        a /= rng
        np.clip(a, -1.0, 1.0, out=a)

        np.degrees(np.arccos(a, out=a), out=a)                         # off-nadir [deg]
        np.degrees(np.arcsin(c, out=c), out=c)                         # elevation [deg]
        if self.max_off_nadir_deg is not None:
            visible &= a <= self.max_off_nadir_deg

        shape = (T, S, self.ground.size)
        return GeometryResult(
            elevation_deg=c.astype(self.dtype, copy=False).reshape(shape),
            slant_range_m=rng.astype(self.dtype, copy=False).reshape(shape),
            off_nadir_deg=a.astype(self.dtype, copy=False).reshape(shape),
            visible=visible.reshape(shape),
        )
