"""
boa_forecaster: Bayesian TPE optimisation framework for time series forecasting.

Phase 1 public API — core infrastructure complete.
"""

__version__ = "2.0.0-dev"

# ── Metrics (Phase 0) ─────────────────────────────────────────────────────────
# ── Feature engineering (Phase 1) ────────────────────────────────────────────
# ── Benchmarks & comparison (Phase 5) ────────────────────────────────────────
from boa_forecaster.benchmarks import (
    auto_arima_nixtla,
    ets_model,
    run_benchmark_comparison,
    run_model_comparison,
    seasonal_naive,
    summary_table,
)
from boa_forecaster.features import FeatureConfig, FeatureEngineer
from boa_forecaster.metrics import (
    METRIC_REGISTRY,
    build_combined_metric,
    combined_metric,
    mae,
    mape,
    rmse,
    rmsle,
    smape,
)

# ── Model registry & types (Phase 1) ─────────────────────────────────────────
from boa_forecaster.models import (
    MODEL_REGISTRY,
    LightGBMSpec,  # None if lightgbm not installed
    XGBoostSpec,  # None if xgboost not installed
    get_model_spec,
    register_model,
)
from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ModelSpec,
    OptimizationResult,
    SearchSpaceParam,
)
from boa_forecaster.models.random_forest import RandomForestSpec
from boa_forecaster.models.sarima import SARIMASpec, forecast_arima, pred_arima

# ── Optimisation engine (Phase 1) ────────────────────────────────────────────
from boa_forecaster.optimizer import optimize_arima, optimize_model

# ── Preprocessing & standardisation (Phase 0) ────────────────────────────────
from boa_forecaster.preprocessor import clean_zeros, fill_blanks
from boa_forecaster.standardization import clip_outliers, weighted_moving_stats

# ── Validation (Phase 0) ──────────────────────────────────────────────────────
from boa_forecaster.validation import validate_by_group, walk_forward_validation

# data_loader is available as boa_forecaster.data_loader.load_data
# (not re-exported at top level since it requires an Excel file path)

__all__ = [
    # metrics
    "smape",
    "rmsle",
    "mae",
    "rmse",
    "mape",
    "combined_metric",
    "build_combined_metric",
    "METRIC_REGISTRY",
    # validation
    "walk_forward_validation",
    "validate_by_group",
    # preprocessor
    "fill_blanks",
    "clean_zeros",
    # standardization
    "clip_outliers",
    "weighted_moving_stats",
    # feature engineering
    "FeatureConfig",
    "FeatureEngineer",
    # model registry
    "ModelSpec",
    "OptimizationResult",
    "IntParam",
    "FloatParam",
    "CategoricalParam",
    "SearchSpaceParam",
    "SARIMASpec",
    "RandomForestSpec",
    "XGBoostSpec",
    "LightGBMSpec",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_spec",
    # SARIMA helpers
    "pred_arima",
    "forecast_arima",
    # optimisation
    "optimize_model",
    "optimize_arima",
    # benchmarks & comparison
    "seasonal_naive",
    "ets_model",
    "auto_arima_nixtla",
    "run_model_comparison",
    "run_benchmark_comparison",
    "summary_table",
]
