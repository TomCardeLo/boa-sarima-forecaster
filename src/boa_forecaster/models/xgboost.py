"""XGBoostSpec — XGBoost plugin for boa-forecaster v2.0.

Implements the ``ModelSpec`` protocol for recursive multi-step forecasting
with ``XGBRegressor``.  Feature engineering is delegated to ``FeatureEngineer``;
forecasting uses the same recursive one-step-at-a-time strategy as
``RandomForestSpec``.

Early stopping
--------------
Inside each cross-validation fold the training window is further split into
an inner train and an inner validation set (20 % of fold, min 6 months).
The inner validation set is passed as ``eval_set`` to ``XGBRegressor.fit`` so
that early stopping terminates training before overfitting.  The ``FeatureEngineer``
is **always** fitted on the full fold training window, so no test-set information
leaks into lag or rolling features.

Temporal integrity
------------------
A fresh ``FeatureEngineer`` instance is created per cross-validation fold so
that rolling/expanding stats are fitted *only* on the training window of that
fold.  No test-set information leaks into the features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.features import FeatureConfig, FeatureEngineer
from boa_forecaster.models.base import FloatParam, IntParam, suggest_from_space

_MIN_TRAIN_SIZE: int = 24
_N_CV_FOLDS: int = 3
_VAL_FRACTION: float = 0.2  # fraction of training fold used as early-stopping val set
_MIN_VAL_SIZE: int = 6  # minimum months for the inner validation split


class XGBoostSpec:
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
    needs_features: bool = True

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        forecast_horizon: int = 12,
        early_stopping_rounds: int = 20,
    ) -> None:
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for XGBoostSpec. "
                "Install it with: pip install xgboost>=1.7"
            )
        self.feature_config: FeatureConfig = feature_config or FeatureConfig()
        self.forecast_horizon: int = forecast_horizon
        self.early_stopping_rounds: int = early_stopping_rounds

    # ── ModelSpec protocol ────────────────────────────────────────────────────

    @property
    def search_space(self) -> dict:
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

    def suggest_params(self, trial) -> dict:
        """Sample a point from ``search_space`` via Optuna trial suggestions."""
        return suggest_from_space(trial, self.search_space)

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn,
        feature_config: FeatureConfig | None = None,
    ) -> float:
        """Evaluate *params* via 3-fold expanding-window CV with early stopping.

        Each fold:
        1. Fits a ``FeatureEngineer`` on the full training window.
        2. Splits the resulting feature matrix into inner train / inner val
           (last ``_VAL_FRACTION`` of rows, min ``_MIN_VAL_SIZE``).
        3. Trains ``XGBRegressor`` with early stopping on the inner val set.
        4. Forecasts recursively over the test window and scores against
           *metric_fn*.

        Args:
            series: Monthly time series with ``DatetimeIndex``.
            params: Hyperparameter dict sampled from ``search_space``.
            metric_fn: ``metric_fn(y_true, y_pred) -> float`` objective.
            feature_config: Override for this evaluation only.

        Returns:
            Mean metric across valid folds, or ``OPTIMIZER_PENALTY`` on failure.
        """
        config = feature_config or self.feature_config
        n = len(series)
        horizon = self.forecast_horizon
        scores: list[float] = []

        for k in range(_N_CV_FOLDS):
            train_end = n - horizon * (k + 1)
            if train_end < _MIN_TRAIN_SIZE:
                break

            train = series.iloc[:train_end]
            test = series.iloc[train_end : train_end + horizon]
            if len(test) < horizon:
                break

            try:
                fe = FeatureEngineer(config)
                X_all, y_all = fe.fit_transform(train)

                # Inner val split for early stopping — no leakage into test fold
                val_size = max(_MIN_VAL_SIZE, int(len(X_all) * _VAL_FRACTION))
                X_tr = X_all.iloc[:-val_size]
                y_tr = y_all.iloc[:-val_size]
                X_val = X_all.iloc[-val_size:]
                y_val = y_all.iloc[-val_size:]

                model = xgb.XGBRegressor(
                    **params,
                    early_stopping_rounds=self.early_stopping_rounds,
                    eval_metric="rmse",
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

                y_pred = self._recursive_forecast(model, fe, train, len(test))
                scores.append(metric_fn(test.values, y_pred.values))
            except Exception:
                return OPTIMIZER_PENALTY

        if not scores:
            return OPTIMIZER_PENALTY

        return float(np.mean(scores))

    def build_forecaster(
        self,
        params: dict,
        feature_config: FeatureConfig | None = None,
    ):
        """Return a ``forecaster(train) -> pd.Series`` closure.

        The returned callable fits an ``XGBRegressor`` on *train* using the
        optimised *params* (no early stopping — ``n_estimators`` is already
        determined by Optuna) and produces recursive multi-step forecasts of
        length ``forecast_horizon``.

        Args:
            params: Optimal hyperparameters from ``optimize_model``.
            feature_config: Override feature engineering settings.

        Returns:
            Callable ``forecaster(train: pd.Series) -> pd.Series``.
        """
        config = feature_config or self.feature_config
        horizon = self.forecast_horizon

        def forecaster(train: pd.Series) -> pd.Series:
            fe = FeatureEngineer(config)
            X_train, y_train = fe.fit_transform(train)
            model = xgb.XGBRegressor(**params, random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            return self._recursive_forecast(model, fe, train, horizon)

        return forecaster

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _recursive_forecast(
        self,
        model,
        fe: FeatureEngineer,
        train: pd.Series,
        horizon: int,
    ) -> pd.Series:
        """Predict *horizon* steps ahead using recursive (one-step-at-a-time) forecasting.

        At each step the latest prediction is appended to the series so that lag
        and rolling features for the next step can be computed causally.

        Args:
            model: Fitted ``XGBRegressor``.
            fe: ``FeatureEngineer`` already fitted on *train* via ``fit_transform``.
            train: Training series (used to seed the extended series).
            horizon: Number of future steps to predict.

        Returns:
            ``pd.Series`` of length *horizon* with a ``DatetimeIndex`` starting
            one month after the last training date (``freq="MS"``).
        """
        extended = train.copy()
        preds: list[float] = []

        for _ in range(horizon):
            X_all = fe.transform(extended)
            y_pred = float(model.predict(X_all.iloc[[-1]])[0])

            next_date = extended.index[-1] + pd.DateOffset(months=1)
            new_point = pd.Series([y_pred], index=[next_date])
            extended = pd.concat([extended, new_point])
            preds.append(y_pred)

        future_index = pd.date_range(
            start=train.index[-1] + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )
        return pd.Series(preds, index=future_index, name="forecast")
