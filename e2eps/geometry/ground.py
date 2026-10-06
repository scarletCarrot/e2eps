"""Ground points (user terminals or gateways) prepared for geometry.

Everything that depends only on the ground location (ECEF position, local
"up" vector, elevation mask) is computed once per run and reused for every
time chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import Site
from ..frames import geodetic_to_ecef


@dataclass(frozen=True)
class GroundPoints:
    ids: tuple[str, ...]
    lat_deg: np.ndarray        # (G,)
    lon_deg: np.ndarray        # (G,)
    ecef: np.ndarray           # (G, 3) [m]
    up: np.ndarray             # (G, 3) unit geodetic normal
    min_elevation_deg: np.ndarray  # (G,)

    @property
    def size(self) -> int:
        return len(self.ids)

    @classmethod
    def from_sites(cls, sites: list[Site], min_elevation_deg: float | np.ndarray) -> "GroundPoints":
        lat = np.array([s.lat_deg for s in sites], dtype=float)
        lon = np.array([s.lon_deg for s in sites], dtype=float)
        alt = np.array([s.alt_m for s in sites], dtype=float)
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        up = np.stack((np.cos(lat_r) * np.cos(lon_r),
                       np.cos(lat_r) * np.sin(lon_r),
                       np.sin(lat_r)), axis=-1)
        min_el = np.broadcast_to(np.asarray(min_elevation_deg, dtype=float), lat.shape).copy()
        return cls(
            ids=tuple(s.id for s in sites),
            lat_deg=lat,
            lon_deg=lon,
            ecef=geodetic_to_ecef(lat, lon, alt),
            up=up,
            min_elevation_deg=min_el,
        )
