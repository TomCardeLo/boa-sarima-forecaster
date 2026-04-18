"""Declarative Pydantic schema for the boa-forecaster YAML configuration.

``BoaConfig`` validates ``config.yaml`` (or ``config.example.yaml``) against a
typed schema.  Sub-models mirror the top-level YAML sections so callers can
access configuration with attribute syntax instead of raw-dict lookups.

Example
-------
>>> from boa_forecaster.config_schema import BoaConfig
>>> cfg = BoaConfig.load("config.example.yaml")
>>> cfg.data.input_path
'data/input/sales.xlsx'
>>> cfg.models.active
'sarima'

Strict-mode (TODO v3.0)
-----------------------
Today ``BoaConfig`` accepts unknown keys (``extra="allow"``) so that legacy
configs keep loading during the v2.x series.  In v3.0 this flips to
``extra="forbid"`` per the Z4 roadmap item so typos like ``n_trails`` raise
``ValidationError`` instead of silently being ignored.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

# TODO (v3.0, Z4): change ``extra="allow"`` → ``extra="forbid"`` across every
# sub-model below so typo'd keys raise ``ValidationError`` at load time.
_ALLOW_EXTRA: ConfigDict = ConfigDict(extra="allow")


# ── Sub-configs ───────────────────────────────────────────────────────────────


class DataConfig(BaseModel):
    """Input-data ingestion settings (matches the ``data:`` YAML block)."""

    model_config = _ALLOW_EXTRA

    input_path: str
    sheet_name: str = "Data"
    skip_rows: int = 2
    date_format: str = "%Y%m"
    end_date: Optional[str] = None
    freq: str = "MS"


class StandardizationConfig(BaseModel):
    """Outlier-clipping settings (matches the ``standardization:`` block)."""

    model_config = _ALLOW_EXTRA

    window: int = 6
    sigma_threshold: float = 2.5


class OptimizationConfig(BaseModel):
    """Legacy SARIMA search-space & TPE budget (matches ``optimization:``).

    The v2 model registry supersedes this block via
    ``models.<name>.search_space``; it is kept because ``config.example.yaml``
    still documents it for v1.x compatibility.
    """

    model_config = _ALLOW_EXTRA

    p_range: list[int] = Field(default_factory=lambda: [0, 3])
    d_range: list[int] = Field(default_factory=lambda: [0, 2])
    q_range: list[int] = Field(default_factory=lambda: [0, 3])
    P_range: list[int] = Field(default_factory=lambda: [0, 2])
    D_range: list[int] = Field(default_factory=lambda: [0, 1])
    Q_range: list[int] = Field(default_factory=lambda: [0, 2])
    n_calls: int = 50
    n_jobs: int = 1


class MetricComponent(BaseModel):
    """One ``{metric, weight}`` entry in the metrics composition."""

    model_config = _ALLOW_EXTRA

    metric: str
    weight: float


class MetricsConfig(BaseModel):
    """Weighted-objective composition (matches the ``metrics:`` block)."""

    model_config = _ALLOW_EXTRA

    components: list[MetricComponent] = Field(
        default_factory=lambda: [
            MetricComponent(metric="smape", weight=0.7),
            MetricComponent(metric="rmsle", weight=0.3),
        ]
    )


class ForecastConfig(BaseModel):
    """Forecast horizon + confidence interval (matches the ``forecast:`` block)."""

    model_config = _ALLOW_EXTRA

    n_periods: int = 12
    alpha: float = 0.05


class OutputConfig(BaseModel):
    """Output path + run identifier (matches the ``output:`` block)."""

    model_config = _ALLOW_EXTRA

    output_path: str = "data/output/"
    run_id: str = "RUN-DEFAULT"


class LoggingConfig(BaseModel):
    """Log verbosity (matches the ``logging:`` block)."""

    model_config = _ALLOW_EXTRA

    level: str = "INFO"


class ModelEntry(BaseModel):
    """Generic per-model entry.

    Every entry carries at least ``enabled`` and an opaque ``search_space``
    dict; model-specific keys (``seasonal_period``, ``forecast_horizon``,
    ``constraints``, ``warm_starts``, …) live under ``extra`` so the schema
    does not have to enumerate every model's parameters.
    """

    model_config = _ALLOW_EXTRA

    enabled: bool = True
    search_space: dict[str, Any] = Field(default_factory=dict)
    warm_starts: list[dict[str, Any]] = Field(default_factory=list)


class ModelsConfig(BaseModel):
    """Registry-driven model block (matches the ``models:`` YAML block).

    ``active`` selects which registered ``ModelSpec`` the CLI pipeline builds.
    ``sarima`` is defined explicitly to enforce the required fields; other
    models flow through ``extra="allow"``.
    """

    model_config = _ALLOW_EXTRA

    active: str = "sarima"
    sarima: Optional[ModelEntry] = None
    random_forest: Optional[ModelEntry] = None
    xgboost: Optional[ModelEntry] = None
    lightgbm: Optional[ModelEntry] = None
    ensemble: Optional[ModelEntry] = None


class FeaturesConfig(BaseModel):
    """Feature-engineering knobs for ML models (matches the ``features:`` block)."""

    model_config = _ALLOW_EXTRA

    lag_periods: list[int] = Field(default_factory=lambda: [1, 2, 3, 6, 12])
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 6, 12])
    include_calendar: bool = True
    include_trend: bool = True
    include_expanding: bool = False


# ── Top-level schema ──────────────────────────────────────────────────────────


class BoaConfig(BaseModel):
    """Validated top-level boa-forecaster configuration.

    Every nested section is optional so minimal configs (e.g. a stand-alone
    ``models:`` block in a test fixture) still load.  Use ``BoaConfig.load``
    as the single entry point from YAML on disk.
    """

    model_config = _ALLOW_EXTRA

    data: Optional[DataConfig] = None
    standardization: StandardizationConfig = Field(
        default_factory=StandardizationConfig
    )
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    # Legacy v1 block preserved for backwards compatibility; superseded by
    # ``models.<name>`` in v2.
    model: Optional[dict[str, Any]] = None

    # ── Loading ───────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Union[str, Path]) -> BoaConfig:
        """Read *path* as YAML and validate it against this schema.

        Args:
            path: Filesystem path to a ``config.yaml`` file.

        Returns:
            Validated ``BoaConfig`` instance.

        Raises:
            FileNotFoundError: *path* does not exist.
            pydantic.ValidationError: Schema validation fails.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)
