"""Module B - RF Link Budget Engine (user downlink, A9-A13).

Converts geometry into link quality and achievable rate for every visible
satellite-terminal pair:

    EIRP(theta) = EIRP_boresight - scan_loss(off-nadir)
    C/N0        = EIRP(theta) - FSPL - L_gas - L_rain - L_misc + G/T - k
    Es/N0       = C/N0 - 10 log10(Rs),     Rs = B / (1 + rolloff)
    MODCOD      = best entry with threshold <= Es/N0 - implementation margin
    link rate   = efficiency * Rs          (if the terminal had the whole channel)

Only visible links are evaluated (typically a few % of all pairs), then
scattered back into (T, S, G) arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config.schema import DownlinkConfig
from ..constants import K_BOLTZMANN_DB
from ..geometry import GeometryResult
from .modcod import ModcodTable
from .propagation import free_space_loss_db, gaseous_loss_db, rain_loss_db


@dataclass
class LinkResult:
    esn0_db: np.ndarray          # (T, S, G) float32, NaN where no link
    modcod_idx: np.ndarray       # (T, S, G) int16, -1 = no link / outage
    spectral_eff: np.ndarray     # (T, S, G) float32 [bit/symbol]
    link_rate_bps: np.ndarray    # (T, S, G) float32 [bit/s]
    available: np.ndarray        # (T, S, G) bool: visible and Es/N0 closes a MODCOD


@dataclass(frozen=True)
class LinkBreakdown:
    """Per-term budget for a single link - for reports and sanity checks."""
    eirp_dbw: float
    scan_loss_db: float
    fspl_db: float
    gas_db: float
    rain_db: float
    misc_db: float
    g_over_t_dbk: float
    cn0_dbhz: float
    esn0_db: float


class LinkBudgetEngine:
    def __init__(self, cfg: DownlinkConfig, g_over_t_dbk: float,
                 modcods: ModcodTable | None = None):
        self.cfg = cfg
        self.g_over_t = g_over_t_dbk
        self.modcods = modcods or ModcodTable.dvb_s2()
        self.freq_hz = cfg.frequency_ghz * 1e9
        self.symbol_rate = cfg.bandwidth_mhz * 1e6 / (1.0 + cfg.rolloff)
        self._rs_db = 10.0 * np.log10(self.symbol_rate)

    # ---------------------------------------------------------------- terms
    def scan_loss_db(self, off_nadir_deg):
        cos_t = np.cos(np.radians(np.minimum(off_nadir_deg, 89.0)))
        return -10.0 * self.cfg.scan_loss_exponent * np.log10(cos_t)

    def esn0_db(self, elevation_deg, slant_range_m, off_nadir_deg):
        c = self.cfg
        cn0 = (c.sat_eirp_dbw
               - self.scan_loss_db(off_nadir_deg)
               - free_space_loss_db(slant_range_m, self.freq_hz)
               - gaseous_loss_db(elevation_deg, c.gaseous_loss_zenith_db)
               - rain_loss_db(elevation_deg, c.frequency_ghz, c.rain_rate_mm_h, c.rain_height_km)
               - c.misc_losses_db
               + self.g_over_t
               - K_BOLTZMANN_DB)
        return cn0 - self._rs_db

    def breakdown(self, elevation_deg: float, slant_range_m: float,
                  off_nadir_deg: float) -> LinkBreakdown:
        c = self.cfg
        terms = dict(
            eirp_dbw=c.sat_eirp_dbw,
            scan_loss_db=float(self.scan_loss_db(off_nadir_deg)),
            fspl_db=float(free_space_loss_db(slant_range_m, self.freq_hz)),
            gas_db=float(gaseous_loss_db(elevation_deg, c.gaseous_loss_zenith_db)),
            rain_db=float(rain_loss_db(elevation_deg, c.frequency_ghz,
                                       c.rain_rate_mm_h, c.rain_height_km)),
            misc_db=c.misc_losses_db,
            g_over_t_dbk=self.g_over_t,
        )
        cn0 = (terms["eirp_dbw"] - terms["scan_loss_db"] - terms["fspl_db"] - terms["gas_db"]
               - terms["rain_db"] - terms["misc_db"] + terms["g_over_t_dbk"] - K_BOLTZMANN_DB)
        return LinkBreakdown(**terms, cn0_dbhz=cn0, esn0_db=cn0 - self._rs_db)

    # ----------------------------------------------------------------- main
    def compute(self, geo: GeometryResult) -> LinkResult:
        shape = geo.shape
        mask = geo.visible & (geo.off_nadir_deg <= self.cfg.max_scan_deg)
        idx = np.flatnonzero(mask)

        el = geo.elevation_deg.reshape(-1)[idx].astype(np.float64)
        rng = geo.slant_range_m.reshape(-1)[idx].astype(np.float64)
        nad = geo.off_nadir_deg.reshape(-1)[idx].astype(np.float64)

        esn0 = self.esn0_db(el, rng, nad)
        mc_idx, eff = self.modcods.select(esn0 - self.cfg.implementation_margin_db)

        esn0_full = np.full(mask.size, np.nan, dtype=np.float32)
        mc_full = np.full(mask.size, -1, dtype=np.int16)
        eff_full = np.zeros(mask.size, dtype=np.float32)
        esn0_full[idx] = esn0
        mc_full[idx] = mc_idx
        eff_full[idx] = eff

        return LinkResult(
            esn0_db=esn0_full.reshape(shape),
            modcod_idx=mc_full.reshape(shape),
            spectral_eff=eff_full.reshape(shape),
            link_rate_bps=(eff_full * self.symbol_rate).astype(np.float32).reshape(shape),
            available=(mc_full >= 0).reshape(shape),
        )
