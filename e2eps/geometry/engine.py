"""Module A - Geometry & Visibility Engine.

Turns satellite and ground positions into the geometric quantities every
downstream model needs:

    elevation   (T, S, G) [deg]  visibility, atmospheric path, rain loss
    slant range (T, S, G) [m]    free-space path loss, propagation delay
    off-nadir   (T, S, G) [deg]  satellite antenna scan loss
    visible     (T, S, G) bool   which links can physically exist

T = time steps in the chunk, S = satellites, G = ground points.
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
    """

    def __init__(self, ground: GroundPoints, max_off_nadir_deg: float | None = None):
        self.ground = ground
        self.max_off_nadir_deg = max_off_nadir_deg

    def compute(self, sat_ecef: np.ndarray) -> GeometryResult:
        """sat_ecef: (T, S, 3) [m] -> GeometryResult with (T, S, G) arrays."""
        g = self.ground

        # Ground-to-satellite vectors, broadcast to (T, S, G, 3)
        d = sat_ecef[:, :, None, :] - g.ecef[None, None, :, :]
        rng = np.linalg.norm(d, axis=-1)

        sin_el = np.einsum("tsgk,gk->tsg", d, g.up) / rng
        elevation = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))

        # Off-nadir: angle at the satellite between nadir and the ground point
        r_sat = np.linalg.norm(sat_ecef, axis=-1)[:, :, None]
        cos_nadir = np.einsum("tsk,tsgk->tsg", sat_ecef, d) / (r_sat * rng)
        off_nadir = np.degrees(np.arccos(np.clip(cos_nadir, -1.0, 1.0)))

        visible = elevation >= g.min_elevation_deg[None, None, :]
        if self.max_off_nadir_deg is not None:
            visible &= off_nadir <= self.max_off_nadir_deg

        return GeometryResult(elevation, rng, off_nadir, visible)
