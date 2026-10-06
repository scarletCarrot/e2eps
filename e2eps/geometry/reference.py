"""Reference implementations of the geometry maths.

Kept for validation and benchmarking only - the engine never calls these.

- geometry_loop:      one link at a time in Python, like a typical prototype
- geometry_broadcast: straightforward NumPy broadcasting with a (T, S, G, 3)
                      intermediate array (first vectorised version)
"""

from __future__ import annotations

import math

import numpy as np

from .ground import GroundPoints


def geometry_loop(sat_ecef: np.ndarray, ground: GroundPoints):
    T, S, _ = sat_ecef.shape
    G = ground.size
    el = np.empty((T, S, G))
    rng = np.empty((T, S, G))
    nad = np.empty((T, S, G))
    for t in range(T):
        for s in range(S):
            sx, sy, sz = sat_ecef[t, s]
            r_sat = math.sqrt(sx * sx + sy * sy + sz * sz)
            for g in range(G):
                gx, gy, gz = ground.ecef[g]
                ux, uy, uz = ground.up[g]
                dx, dy, dz = sx - gx, sy - gy, sz - gz
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                sin_el = (dx * ux + dy * uy + dz * uz) / d
                cos_n = (sx * dx + sy * dy + sz * dz) / (r_sat * d)
                el[t, s, g] = math.degrees(math.asin(max(-1.0, min(1.0, sin_el))))
                nad[t, s, g] = math.degrees(math.acos(max(-1.0, min(1.0, cos_n))))
                rng[t, s, g] = d
    return el, rng, nad


def geometry_broadcast(sat_ecef: np.ndarray, ground: GroundPoints):
    d = sat_ecef[:, :, None, :] - ground.ecef[None, None, :, :]
    rng = np.linalg.norm(d, axis=-1)
    sin_el = np.einsum("tsgk,gk->tsg", d, ground.up) / rng
    el = np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))
    r_sat = np.linalg.norm(sat_ecef, axis=-1)[:, :, None]
    cos_n = np.einsum("tsk,tsgk->tsg", sat_ecef, d) / (r_sat * rng)
    nad = np.degrees(np.arccos(np.clip(cos_n, -1.0, 1.0)))
    return el, rng, nad
