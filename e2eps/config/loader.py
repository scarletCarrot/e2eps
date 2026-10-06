"""Load and validate scenario files."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from .schema import Scenario


class ScenarioError(ValueError):
    """Raised when a scenario file cannot be read or is invalid."""


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    if not path.exists():
        raise ScenarioError(f"scenario file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ScenarioError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ScenarioError(f"scenario root must be a mapping: {path}")
    try:
        return Scenario.model_validate(raw)
    except ValidationError as exc:
        raise ScenarioError(f"invalid scenario {path}:\n{exc}") from exc
