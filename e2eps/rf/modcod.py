"""MODCOD table: Es/N0 threshold -> spectral efficiency (A13).

DVB-S2 MODCODs (EN 302 307-1, normal FECFRAME, no pilots), ideal AWGN
thresholds at quasi-error-free operation. This is a subset of DVB-S2X; the
higher-order DVB-S2X MODCODs can be appended without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# name, required Es/N0 [dB], spectral efficiency [information bit / symbol]
_DVB_S2 = [
    ("QPSK 1/4", -2.35, 0.490),
    ("QPSK 1/3", -1.24, 0.656),
    ("QPSK 2/5", -0.30, 0.789),
    ("QPSK 1/2", 1.00, 0.989),
    ("QPSK 3/5", 2.23, 1.188),
    ("QPSK 2/3", 3.10, 1.322),
    ("QPSK 3/4", 4.03, 1.487),
    ("QPSK 4/5", 4.68, 1.587),
    ("QPSK 5/6", 5.18, 1.655),
    ("8PSK 3/5", 5.50, 1.779),
    ("8PSK 2/3", 6.62, 1.980),
    ("8PSK 3/4", 7.91, 2.228),
    ("16APSK 2/3", 8.97, 2.637),
    ("8PSK 5/6", 9.35, 2.479),
    ("16APSK 3/4", 10.21, 2.966),
    ("16APSK 4/5", 11.03, 3.165),
    ("16APSK 5/6", 11.61, 3.300),
    ("32APSK 3/4", 12.73, 3.703),
    ("16APSK 8/9", 12.89, 3.523),
    ("32APSK 4/5", 13.64, 3.951),
    ("32APSK 5/6", 14.28, 4.119),
    ("32APSK 8/9", 15.69, 4.397),
    ("32APSK 9/10", 16.05, 4.453),
]


@dataclass(frozen=True)
class ModcodTable:
    names: tuple[str, ...]
    esn0_db: np.ndarray      # ascending thresholds
    efficiency: np.ndarray   # bit/symbol, best achievable at or below each threshold

    @classmethod
    def dvb_s2(cls) -> "ModcodTable":
        rows = sorted(_DVB_S2, key=lambda r: r[1])
        names = tuple(r[0] for r in rows)
        thr = np.array([r[1] for r in rows])
        eff = np.array([r[2] for r in rows])
        # An ACM system never picks a MODCOD less efficient than one it can already
        # close (e.g. 8PSK 5/6 vs 16APSK 2/3), so keep the running maximum.
        return cls(names, thr, np.maximum.accumulate(eff))

    @property
    def min_esn0_db(self) -> float:
        return float(self.esn0_db[0])

    def select(self, esn0_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best MODCOD per link. Returns (index, efficiency); index -1 = outage."""
        idx = np.searchsorted(self.esn0_db, esn0_db, side="right") - 1
        eff = np.where(idx >= 0, self.efficiency[np.clip(idx, 0, None)], 0.0)
        return idx.astype(np.int16), eff
