"""LightGBMSpec — LightGBM plugin for boa-forecaster v2.0.

Implements the ``ModelSpec`` protocol (via ``BaseMLSpec``) for recursive
multi-step forecasting with ``LGBMRegressor``.  Feature engineering is delegated
to ``FeatureEngineer``; forecasting uses the same recursive one-step-at-a-time
strategy as ``XGBoostSpec`` and ``RandomForestSpec``.

Key LightGBM difference vs XGBoost
------------------------------------
LightGBM grows trees leaf-wise using ``num_leaves`` as its primary complexity
control rather than ``max_depth``.  When ``max_depth > 0``, ``num_leaves`` is
soft-constrained to ``≤ 2^max_depth - 1`` inside ``suggest_params`` to keep
the search space feasible.

Early stopping
--------------
Inside each cross-validation fold the training window is further split into
an inner train and an inner validation set (20 % of fold, min 6 months).
The inner validation set is passed via ``eval_set`` with LightGBM's
``early_stopping`` callback.  The ``FeatureEngineer`` is **always** fitted on
the full fold training window so no test-set information leaks.
"""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from boa_forecaster.features import FeatureConfig
from boa_forecaster.models._ml_base import BaseMLSpec
from boa_forecaster.models._utils import split_for_early_stopping
from boa_forecaster.models.base import FloatParam, IntParam, SearchSpaceParam


class LightGBMSpec(BaseMLSpec):
    """``ModelSpec`` implementation for ``lightgbm.LGBMRegressor``.

    Hyperparameter optimisation via Optuna TPE; forecasting via recursive
    multi-step strategy with early stopping inside each CV fold.

    Args:
        feature_config: Feature engineering settings.  Defaults to
            ``FeatureConfig()`` (lags 1/2/3/6/12, rolling 3/6/12, calendar,
            trend).
        forecast_horizon: Number of future steps to predict.  Default 12
            (one year of monthly data).
        early_stopping_rounds: LightGBM early stopping patience.  Default 20.

    Raises:
        ImportError: If lightgbm is not installed.
    """

    name: str = "lightgbm"

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
        if not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is required for LightGBMSpec. "
                "Install it with: pip install lightgbm>=3.3"
            )

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """Hyperparameter search space for ``LGBMRegressor``.

        ``num_leaves`` is LightGBM's primary complexity control (leaf-wise
        growth), replacing ``max_depth`` as the principal tree size knob.
        ``max_depth=-1`` means unlimited depth; positive values apply a hard
        depth cap on top of the ``num_leaves`` limit.
        """
        return {
            "n_estimators": IntParam(50, 1000, log=True),
            "num_leaves": IntParam(8, 256, log=True),
            "max_depth": IntParam(-1, 15),
            "learning_rate": FloatParam(0.005, 0.3, log=True),
            "subsample": FloatParam(0.5, 1.0),
            "colsample_bytree": FloatParam(0.5, 1.0),
            "min_child_samples": IntParam(5, 100),
            "reg_alpha": FloatParam(1e-8, 10.0, log=True),
            "reg_lambda": FloatParam(1e-8, 10.0, log=True),
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
                "num_leaves": 31,
                "max_depth": -1,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_alpha": 1e-8,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 300,
                "num_leaves": 63,
                "max_depth": 7,
                "learning_rate": 0.02,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
                "min_child_samples": 10,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            },
        ]

    def suggest_params(self, trial) -> dict:
        """Sample a point from ``search_space`` via Optuna trial suggestions.

        Applies a soft constraint: when ``max_depth > 0``, ``num_leaves`` is
        clipped to ``2^max_depth - 1`` to keep the tree within depth bounds.
        This avoids wasting Optuna trials on structurally impossible configs.
        """
        params = super().suggest_params(trial)
        max_depth = params["max_depth"]
        if max_depth > 0:
            max_leaves = max(1, (2**max_depth) - 1)
            params["num_leaves"] = min(params["num_leaves"], max_leaves)
        return params

    def _fit_fold(self, X: pd.DataFrame, y: pd.Series, params: dict):
        X_tr, y_tr, X_val, y_val = split_for_early_stopping(X, y)
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        callbacks: list[Callable[..., Any]] = [
            lgb.early_stopping(self.early_stopping_rounds, verbose=False),
            lgb.log_evaluation(-1),
        ]
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
        return model

    def _fit_final(self, X: pd.DataFrame, y: pd.Series, params: dict):
        model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
        model.fit(X, y)
        return model
