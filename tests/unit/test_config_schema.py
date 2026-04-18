"""Tests for ``boa_forecaster.config_schema.BoaConfig``.

Covers three things:

1. Round-trip: ``config.example.yaml`` loads without error.
2. Default coverage: nested section defaults kick in when omitted.
3. Strict-mode forward-compat: a ``StrictBoaConfig`` subclass that flips
   ``extra="forbid"`` rejects typo'd keys with ``ValidationError``.  This
   exercises the code path that v3.0 (Z4 roadmap) will default to.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ConfigDict, ValidationError

from boa_forecaster.config_schema import (
    BoaConfig,
    DataConfig,
    FeaturesConfig,
    ModelsConfig,
    OptimizationConfig,
    StandardizationConfig,
)

EXAMPLE_YAML = Path(__file__).resolve().parents[2] / "config.example.yaml"


# ── 1. Round-trip ─────────────────────────────────────────────────────────────


def test_load_config_example_yaml() -> None:
    """The shipped example config should parse cleanly."""
    cfg = BoaConfig.load(EXAMPLE_YAML)

    assert isinstance(cfg.data, DataConfig)
    assert cfg.data.input_path == "data/input/sales.xlsx"
    assert cfg.data.date_format == "%Y%m"

    assert isinstance(cfg.standardization, StandardizationConfig)
    assert cfg.standardization.sigma_threshold == pytest.approx(2.5)

    assert isinstance(cfg.optimization, OptimizationConfig)
    assert cfg.optimization.n_calls == 50

    assert isinstance(cfg.models, ModelsConfig)
    assert cfg.models.active == "sarima"
    assert cfg.models.sarima is not None
    assert cfg.models.sarima.enabled is True

    assert isinstance(cfg.features, FeaturesConfig)
    assert cfg.features.lag_periods == [1, 2, 3, 6, 12]
    assert cfg.features.include_calendar is True


def test_metrics_components_parse() -> None:
    """The weighted-objective list should load as ``MetricComponent`` entries."""
    cfg = BoaConfig.load(EXAMPLE_YAML)
    names = [c.metric for c in cfg.metrics.components]
    weights = [c.weight for c in cfg.metrics.components]
    assert names == ["smape", "rmsle"]
    assert sum(weights) == pytest.approx(1.0)


# ── 2. Defaults when sections are omitted ─────────────────────────────────────


def test_empty_config_uses_defaults(tmp_path: Path) -> None:
    """A minimal YAML file should validate and fall back to defaults."""
    cfg_path = tmp_path / "empty.yaml"
    cfg_path.write_text("models:\n  active: sarima\n", encoding="utf-8")

    cfg = BoaConfig.load(cfg_path)

    assert cfg.models.active == "sarima"
    assert cfg.forecast.n_periods == 12
    assert cfg.logging.level == "INFO"
    assert cfg.standardization.window == 6


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        BoaConfig.load(tmp_path / "does-not-exist.yaml")


# ── 3. Strict-mode forward-compat (v3.0 Z4) ───────────────────────────────────


class StrictBoaConfig(BoaConfig):
    """Subclass that flips ``extra="forbid"`` to simulate v3.0 strict mode."""

    model_config = ConfigDict(extra="forbid")


def test_strict_mode_rejects_typoed_key(tmp_path: Path) -> None:
    """Typos like ``n_trails`` at the root must raise once strict is on."""
    bad = {
        "models": {"active": "sarima"},
        "n_trails": 50,  # noqa: S105 — deliberate typo
    }
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    with pytest.raises(ValidationError):
        StrictBoaConfig.model_validate(yaml.safe_load(cfg_path.read_text()))


def test_default_mode_accepts_typoed_key_today(tmp_path: Path) -> None:
    """Until v3.0 flips strict on, ``BoaConfig`` silently accepts extras."""
    bad = {
        "models": {"active": "sarima"},
        "n_trails": 50,  # noqa: S105 — deliberate typo, tolerated in v2.x
    }
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml.safe_dump(bad), encoding="utf-8")

    cfg = BoaConfig.load(cfg_path)
    assert cfg.models.active == "sarima"
