"""XGBoostSpec — XGBoost plugin for boa-forecaster v2.0.

Implements the ``ModelSpec`` protocol (via ``BaseMLSpec``) for recursive
multi-step forecasting with ``XGBRegressor``.  Feature engineering is delegated
to ``FeatureEngineer``; forecasting uses the same recursive one-step-at-a-time
strategy as ``RandomForestSpec``.

Early stopping
--------------
Inside each cross-validation fold the training window is further split into
an inner train and an inner validation set (20 % of fold, min 6 months).
The inner validation set is passed as ``eval_set`` to ``XGBRegressor.fit`` so
that early stopping terminates training before overfitting.  The
``FeatureEngineer`` is **always** fitted on the full fold training window, so
no test-set information leaks into lag or rolling features.
"""

from __future__ import annotations

import pandas as pd

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from boa_forecaster.features import FeatureConfig
from boa_forecaster.models._ml_base import BaseMLSpec
from boa_forecaster.models._utils import split_for_early_stopping
from boa_forecaster.models.base import FloatParam, IntParam, SearchSpaceParam


class XGBoostSpec(BaseMLSpec):
    """``ModelSpec`` implementation for ``xgboost.XGBRegressor``.

    Hyperparameter optimisation via Optuna TPE; forecasting via recursive
    multi-step strategy with early stopping inside each CV fold.

    Args:
        feature_config: Feature engineering settings.  Defaults to
            ``FeatureConfig()`` (lags 1/2/3/6/12, rolling 3/6/12, calendar,
            trend).
        forecast_horizon: Number of future steps to predict.  Default 12
            (one year of monthly data).
        early_stopping_rounds: XGBoost early stopping patience.  Default 20.

    Raises:
        ImportError: If xgboost is not installed.
    """

    name: str = "xgboost"
    uses_early_stopping: bool = True

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        forecast_horizon: int = 12,
        early_stopping_rounds: int = 20,
    ) -> None:
        super().__init__(
            feature_config=feature_config, forecast_horizon=forecast_horizon
        )
        self.early_stopping_rounds: int = early_stopping_rounds

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _check_availability(self) -> None:
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for XGBoostSpec. "
                "Install it with: pip install xgboost>=1.7"
            )

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """Hyperparameter search space for ``XGBRegressor``."""
        return {
            "n_estimators": IntParam(50, 1000, log=True),
            "max_depth": IntParam(2, 10),
            "learning_rate": FloatParam(0.005, 0.3, log=True),
            "subsample": FloatParam(0.5, 1.0),
            "colsample_bytree": FloatParam(0.5, 1.0),
            "min_child_weight": IntParam(1, 20),
            "reg_alpha": FloatParam(1e-8, 10.0, log=True),
            "reg_lambda": FloatParam(1e-8, 10.0, log=True),
            "gamma": FloatParam(0.0, 5.0),
        }

    @property
    def warm_starts(self) -> list[dict]:
        """Two sensible starting configurations for warm-starting Optuna.

        Note: ``reg_alpha`` uses ``1e-8`` (log-scale floor) instead of ``0.0``
        to remain within the log-scale search bounds.
        """
        return [
            {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "reg_alpha": 1e-8,
                "reg_lambda": 1.0,
                "gamma": 0.0,
            },
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_weight": 5,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "gamma": 0.1,
            },
        ]

    def _fit_fold(self, X: pd.DataFrame, y: pd.Series, params: dict):
        X_tr, y_tr, X_val, y_val = split_for_early_stopping(X, y)
        model = xgb.XGBRegressor(
            **params,
            early_stopping_rounds=self.early_stopping_rounds,
            eval_metric="rmse",
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def _fit_final(self, X: pd.DataFrame, y: pd.Series, params: dict):
        model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
        model.fit(X, y)
        return model
