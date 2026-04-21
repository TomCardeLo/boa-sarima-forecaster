"""Tests for ``boa_forecaster.config_schema.BoaConfig``.

Covers five things:

1. Round-trip: ``config.example.yaml`` loads without error.
2. Default coverage: nested section defaults kick in when omitted.
3. Strict-mode forward-compat: a ``StrictBoaConfig`` subclass that flips
   ``extra="forbid"`` rejects typo'd keys with ``ValidationError``.  This
   exercises the code path that v3.0 (Z4 roadmap) will default to.
4. Literal / Field validators added in H4.
5. ``BoaConfig.from_dict`` classmethod + ``--strict`` plumbing.
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
    ForecastConfig,
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


# ── 4. Literal / Field validators (H4) ────────────────────────────────────────


class TestStandardizationLiteralMethod:
    def test_valid_sigma(self) -> None:
        cfg = StandardizationConfig(method="sigma")
        assert cfg.method == "sigma"

    def test_valid_iqr(self) -> None:
        cfg = StandardizationConfig(method="iqr")
        assert cfg.method == "iqr"

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValidationError):
            StandardizationConfig(method="gaussian")


class TestStandardizationThresholdField:
    def test_threshold_zero_fails(self) -> None:
        with pytest.raises(ValidationError):
            StandardizationConfig(threshold=0)

    def test_threshold_negative_fails(self) -> None:
        with pytest.raises(ValidationError):
            StandardizationConfig(threshold=-1.0)

    def test_threshold_ten_passes(self) -> None:
        cfg = StandardizationConfig(threshold=10)
        assert cfg.threshold == pytest.approx(10)

    def test_threshold_eleven_fails(self) -> None:
        with pytest.raises(ValidationError):
            StandardizationConfig(threshold=11)

    def test_threshold_positive_passes(self) -> None:
        cfg = StandardizationConfig(threshold=2.5)
        assert cfg.threshold == pytest.approx(2.5)


class TestDataConfigFreqLiteral:
    @pytest.mark.parametrize("freq", ["MS", "M", "W", "D", "h", "H"])
    def test_valid_freq(self, freq: str) -> None:
        cfg = DataConfig(input_path="foo.xlsx", freq=freq)
        assert cfg.freq == freq

    def test_invalid_freq_raises(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(input_path="foo.xlsx", freq="monthly")


class TestForecastNPeriodsField:
    def test_zero_fails(self) -> None:
        with pytest.raises(ValidationError):
            ForecastConfig(n_periods=0)

    def test_one_passes(self) -> None:
        cfg = ForecastConfig(n_periods=1)
        assert cfg.n_periods == 1

    def test_twelve_passes(self) -> None:
        cfg = ForecastConfig(n_periods=12)
        assert cfg.n_periods == 12


# ── 5. BoaConfig.from_dict and --strict plumbing (H4) ─────────────────────────


class TestFromDict:
    def test_happy_path(self) -> None:
        data = {"models": {"active": "sarima"}, "forecast": {"n_periods": 6}}
        cfg = BoaConfig.from_dict(data)
        assert cfg.models.active == "sarima"
        assert cfg.forecast.n_periods == 6

    def test_strict_false_accepts_typo(self) -> None:
        data = {"models": {"active": "sarima"}, "typo_key": 99}
        cfg = BoaConfig.from_dict(data, strict=False)
        assert cfg.models.active == "sarima"

    def test_strict_true_rejects_typo(self) -> None:
        data = {"models": {"active": "sarima"}, "typo_key": 99}
        with pytest.raises(ValidationError):
            BoaConfig.from_dict(data, strict=True)

    def test_strict_true_accepts_clean_config(self) -> None:
        data = {"models": {"active": "sarima"}}
        cfg = BoaConfig.from_dict(data, strict=True)
        assert cfg.models.active == "sarima"


class TestFromDictStrictNestedTypo:
    """Strict mode should also catch typos in nested sub-models."""

    def test_nested_typo_raises_under_strict(self) -> None:
        data = {
            "models": {"active": "sarima"},
            "forecast": {"n_periods": 12, "n_perods": 6},  # typo in nested
        }
        with pytest.raises(ValidationError):
            BoaConfig.from_dict(data, strict=True)

    def test_nested_typo_allowed_without_strict(self) -> None:
        data = {
            "models": {"active": "sarima"},
            "forecast": {"n_periods": 12, "n_perods": 6},
        }
        cfg = BoaConfig.from_dict(data, strict=False)
        assert cfg.forecast.n_periods == 12


# ── 6. Canary: config.example.yaml round-trips under strict=True ──────────────


def test_example_yaml_round_trips_strict() -> None:
    """config.example.yaml must load cleanly with strict=True."""
    with EXAMPLE_YAML.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = BoaConfig.from_dict(raw, strict=True)
    assert isinstance(cfg.data, DataConfig)
    assert cfg.models.active == "sarima"


# ── 7. CLI --strict plumbing ──────────────────────────────────────────────────


def test_cli_strict_flag_plumbing_via_from_dict() -> None:
    """``BoaConfig.from_dict`` with strict=True raises on a typo'd top-level key.

    This tests the plumbing that the CLI --strict flag exercises, without
    spawning a subprocess (no data file needed).
    """
    dirty = {"models": {"active": "sarima"}, "n_trails": 1}
    # strict=False → passes (back-compat default, mirrors CLI without --strict)
    cfg = BoaConfig.from_dict(dirty, strict=False)
    assert cfg.models.active == "sarima"

    # strict=True → fails (mirrors CLI --strict)
    with pytest.raises(ValidationError):
        BoaConfig.from_dict(dirty, strict=True)


# ── 8. Strict cascades into deep nesting (PR #24 review HIGH #1) ──────────────


class TestStrictCascadesIntoNested:
    """``strict=True`` must reject typos at every nesting depth, not just top.

    Regression for PR #24 code review: ``_StrictModelsConfig`` inherited the
    ``sarima: Optional[ModelEntry]`` annotation unchanged, so Pydantic kept
    validating the ``sarima`` sub-dict against the lenient ``ModelEntry``
    (``extra="allow"``).  Same bug for ``_StrictMetricsConfig.components``.
    """

    def test_models_sarima_nested_typo_raises(self) -> None:
        data = {
            "models": {
                "active": "sarima",
                "sarima": {"enabled": True, "typo_nested": 123},
            }
        }
        with pytest.raises(ValidationError):
            BoaConfig.from_dict(data, strict=True)

    def test_models_random_forest_nested_typo_raises(self) -> None:
        data = {
            "models": {
                "active": "random_forest",
                "random_forest": {"enabled": True, "typo_rf": 1},
            }
        }
        with pytest.raises(ValidationError):
            BoaConfig.from_dict(data, strict=True)

    def test_models_ensemble_nested_typo_raises(self) -> None:
        data = {
            "models": {
                "active": "ensemble",
                "ensemble": {"enabled": True, "typo_ens": 1},
            }
        }
        with pytest.raises(ValidationError):
            BoaConfig.from_dict(data, strict=True)

    def test_metrics_component_nested_typo_raises(self) -> None:
        data = {
            "metrics": {
                "components": [
                    {"metric": "smape", "weight": 0.7, "typo_in_component": 1}
                ]
            }
        }
        with pytest.raises(ValidationError):
            BoaConfig.from_dict(data, strict=True)


# ── 9. DataConfig.freq accepts any pandas alias (PR #24 review HIGH #2) ───────


class TestDataConfigFreqValidator:
    """``DataConfig.freq`` should accept anything ``to_offset`` accepts.

    The original ``Literal["MS", "M", "W", "D", "h", "H"]`` rejected common
    valid aliases like ``"15min"``, ``"Q"``, ``"B"``.  Replace with a
    field_validator that delegates to ``pd.tseries.frequencies.to_offset``.
    """

    @pytest.mark.parametrize(
        "freq",
        ["MS", "M", "W", "D", "h", "H", "B", "Q", "QS", "15min", "2h", "7D"],
    )
    def test_pandas_alias_accepted(self, freq: str) -> None:
        cfg = DataConfig(input_path="foo.xlsx", freq=freq)
        assert cfg.freq == freq

    def test_unknown_alias_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(input_path="foo.xlsx", freq="monthly")

    def test_empty_string_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(input_path="foo.xlsx", freq="")


# ── 10. sigma_threshold ↔ threshold aliasing (PR #24 review MEDIUM) ───────────


class TestStandardizationThresholdAlias:
    """``sigma_threshold`` and ``threshold`` must be the same field.

    Having two fields with the same default silently ignored whichever value
    a caller set on the "wrong" name.  AliasChoices lets both names populate
    a single canonical field.
    """

    def test_sigma_threshold_populates_threshold(self) -> None:
        cfg = StandardizationConfig(sigma_threshold=3.5)
        assert cfg.threshold == pytest.approx(3.5)

    def test_threshold_populates_threshold(self) -> None:
        cfg = StandardizationConfig(threshold=3.5)
        assert cfg.threshold == pytest.approx(3.5)

    def test_legacy_attribute_still_readable(self) -> None:
        """``cfg.sigma_threshold`` must still resolve for v2.x back-compat."""
        cfg = StandardizationConfig(threshold=3.5)
        assert cfg.sigma_threshold == pytest.approx(3.5)


# ── 11. ForecastConfig.n_periods upper bound (PR #24 review LOW) ──────────────


class TestForecastNPeriodsUpperBound:
    """Reject absurd forecast horizons that would OOM on load."""

    def test_ten_thousand_passes(self) -> None:
        cfg = ForecastConfig(n_periods=10_000)
        assert cfg.n_periods == 10_000

    def test_ten_thousand_one_fails(self) -> None:
        with pytest.raises(ValidationError):
            ForecastConfig(n_periods=10_001)
