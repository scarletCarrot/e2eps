"""Scenario schema.

A scenario fully describes one simulation run: time window, constellation,
user terminals, gateways and RF parameters. Every physical parameter lives
here so engineers change scenarios, not code (see docs/assumptions.md).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- simulation
class SimulationConfig(_Model):
    duration_s: float = Field(gt=0, description="Simulated time window [s]")
    step_s: float = Field(gt=0, le=600, description="Time step [s] (A18)")
    chunk_steps: int = Field(default=40, gt=0, description="Time steps per processing chunk (A19)")

    @property
    def num_steps(self) -> int:
        return int(self.duration_s // self.step_s) + 1


# -------------------------------------------------------------- constellation
class WalkerDeltaConfig(_Model):
    """Walker-Delta shell i:T/P/F (A1-A3)."""

    type: Literal["walker_delta"] = "walker_delta"
    altitude_km: float = Field(ge=300, le=2000, description="Circular orbit altitude [km] (LEO)")
    inclination_deg: float = Field(ge=0, le=180)
    num_planes: int = Field(gt=0, description="P")
    sats_per_plane: int = Field(gt=0, description="T / P")
    phasing: int = Field(default=1, ge=0, description="F, relative phasing between planes")
    raan_spread_deg: float = Field(default=360.0, gt=0, le=360)
    include_j2: bool = Field(default=True, description="Apply J2 secular drift (A2)")

    @property
    def num_sats(self) -> int:
        return self.num_planes * self.sats_per_plane

    @model_validator(mode="after")
    def _check_phasing(self) -> "WalkerDeltaConfig":
        if self.phasing >= self.num_planes:
            raise ValueError("phasing F must be in [0, num_planes - 1]")
        return self


# ------------------------------------------------------------- ground points
class Site(_Model):
    id: str
    lat_deg: float = Field(ge=-90, le=90)
    lon_deg: float = Field(ge=-180, le=180)
    alt_m: float = 0.0


class TerminalGrid(_Model):
    """Regular lat/lon grid of terminals, used for coverage-style studies."""

    lat_min_deg: float = Field(ge=-90, le=90)
    lat_max_deg: float = Field(ge=-90, le=90)
    lon_min_deg: float = Field(ge=-180, le=180)
    lon_max_deg: float = Field(ge=-180, le=180)
    step_deg: float = Field(gt=0, le=30)

    @model_validator(mode="after")
    def _check_bounds(self) -> "TerminalGrid":
        if self.lat_min_deg > self.lat_max_deg or self.lon_min_deg > self.lon_max_deg:
            raise ValueError("grid min bound must be <= max bound")
        return self

    def sites(self) -> list[Site]:
        lats = np.arange(self.lat_min_deg, self.lat_max_deg + 1e-9, self.step_deg)
        lons = np.arange(self.lon_min_deg, self.lon_max_deg + 1e-9, self.step_deg)
        return [
            Site(id=f"grid_{lat:+.1f}_{lon:+.1f}", lat_deg=float(lat), lon_deg=float(lon))
            for lat in lats
            for lon in lons
        ]


class UserTerminalsConfig(_Model):
    min_elevation_deg: float = Field(default=25.0, ge=0, lt=90, description="(A7)")
    g_over_t_dbk: float = Field(default=12.0, description="Terminal G/T [dB/K]")
    sites: list[Site] = Field(default_factory=list)
    grid: TerminalGrid | None = None

    @model_validator(mode="after")
    def _check_any(self) -> "UserTerminalsConfig":
        if not self.sites and self.grid is None:
            raise ValueError("define at least one terminal site or a terminal grid")
        return self

    def all_sites(self) -> list[Site]:
        """Named sites first, then grid points. Order defines terminal index."""
        out = list(self.sites)
        if self.grid is not None:
            out.extend(self.grid.sites())
        ids = [s.id for s in out]
        if len(ids) != len(set(ids)):
            raise ValueError("terminal ids must be unique")
        return out


class GatewaysConfig(_Model):
    min_elevation_deg: float = Field(default=10.0, ge=0, lt=90, description="(A7)")
    sites: list[Site] = Field(min_length=1)


# ------------------------------------------------------------------------ RF
class DownlinkConfig(_Model):
    """User downlink, satellite -> terminal (A9-A13)."""

    frequency_ghz: float = Field(gt=0, le=100)
    bandwidth_mhz: float = Field(gt=0, description="User-link channel bandwidth per satellite")
    rolloff: float = Field(default=0.1, ge=0, le=0.5, description="Pulse-shaping roll-off")
    sat_eirp_dbw: float = Field(description="Satellite EIRP at boresight (nadir) [dBW]")
    scan_loss_exponent: float = Field(default=1.2, ge=0,
                                      description="Array scan loss ~ cos(theta)^n (A11)")
    max_scan_deg: float = Field(default=60.0, gt=0, le=90)
    gaseous_loss_zenith_db: float = Field(default=0.3, ge=0)
    rain_rate_mm_h: float = Field(default=0.0, ge=0,
                                  description="Rain rate applied to all terminals (0 = clear sky)")
    rain_height_km: float = Field(default=4.0, gt=0)
    misc_losses_db: float = Field(default=1.0, ge=0, description="Pointing, polarisation, etc.")
    implementation_margin_db: float = Field(default=1.0, ge=0)


class RFConfig(_Model):
    downlink: DownlinkConfig


# ------------------------------------------------------------------- latency
class LatencyConfig(_Model):
    processing_ms: float = Field(default=5.0, ge=0, description="Payload + gateway processing")
    terrestrial_ms: float = Field(default=10.0, ge=0, description="Gateway to PoP backhaul")


# ------------------------------------------------------------------ scenario
class Scenario(_Model):
    name: str
    description: str = ""
    epoch: datetime = Field(description="Simulation start (UTC)")
    simulation: SimulationConfig
    constellation: WalkerDeltaConfig
    user_terminals: UserTerminalsConfig
    gateways: GatewaysConfig
    rf: RFConfig
    latency: LatencyConfig = LatencyConfig()

    @field_validator("epoch")
    @classmethod
    def _utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)
