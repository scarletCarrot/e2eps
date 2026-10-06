"""Scenario configuration: schema and loader."""

from .loader import ScenarioError, load_scenario
from .schema import (
    DownlinkConfig,
    GatewaysConfig,
    LatencyConfig,
    RFConfig,
    Scenario,
    SimulationConfig,
    Site,
    TerminalGrid,
    UserTerminalsConfig,
    WalkerDeltaConfig,
)

__all__ = [
    "DownlinkConfig", "GatewaysConfig", "LatencyConfig", "RFConfig", "Scenario",
    "ScenarioError", "SimulationConfig", "Site", "TerminalGrid", "UserTerminalsConfig",
    "WalkerDeltaConfig", "load_scenario",
]
