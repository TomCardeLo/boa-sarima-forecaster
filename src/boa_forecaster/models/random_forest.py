"""RandomForestSpec — scikit-learn Random Forest plugin for boa-forecaster v2.0.

Implements the ``ModelSpec`` protocol (via ``BaseMLSpec``) for recursive
multi-step forecasting with ``RandomForestRegressor``.  Feature engineering
is delegated to ``FeatureEngineer``; forecasting predicts one step at a time,
appends the prediction to the series, and repeats for the full horizon.

Temporal integrity
------------------
See ``BaseMLSpec`` — a fresh ``FeatureEngineer`` per CV fold prevents any
test-set information from leaking into the features.
"""

from __future__ import annotations

import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from boa_forecaster.models._ml_base import BaseMLSpec
from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    IntParam,
    SearchSpaceParam,
)


class RandomForestSpec(BaseMLSpec):
    """``ModelSpec`` implementation for scikit-learn ``RandomForestRegressor``.

    Hyperparameter optimisation via Optuna TPE; forecasting via recursive
    multi-step strategy.

    Args:
        feature_config: Feature engineering settings.  Defaults to
            ``FeatureConfig()`` (lags 1/2/3/6/12, rolling 3/6/12, calendar,
            trend).
        forecast_horizon: Number of future steps to predict.  Default 12
            (one year of monthly data).

    Raises:
        ImportError: If scikit-learn is not installed.
    """

    name: str = "random_forest"

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _check_availability(self) -> None:
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for RandomForestSpec. "
                "Install it with: pip install scikit-learn>=1.3"
            )

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """Hyperparameter search space for ``RandomForestRegressor``."""
        return {
            "n_estimators": IntParam(50, 500, log=True),
            "max_depth": IntParam(2, 20),
            "min_samples_split": FloatParam(0.01, 0.3, log=True),
            "min_samples_leaf": IntParam(1, 20),
            "max_features": CategoricalParam(["sqrt", "log2", 0.5, 0.8, 1.0]),
        }

    @property
    def warm_starts(self) -> list[dict]:
        """Two sensible starting configurations for warm-starting Optuna."""
        return [
            {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 0.1,
                "min_samples_leaf": 1,
                "max_features": "sqrt",
            },
            {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 0.05,
                "min_samples_leaf": 3,
                "max_features": "log2",
            },
        ]

    def _fit_final(self, X: pd.DataFrame, y: pd.Series, params: dict):
        rf = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=42,
            n_jobs=1,
        )
        rf.fit(X, y)
        return rf
