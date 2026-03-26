"""RandomForestSpec — scikit-learn Random Forest plugin for boa-forecaster v2.0.

Implements the ``ModelSpec`` protocol for recursive multi-step forecasting
with ``RandomForestRegressor``.  Feature engineering is delegated to
``FeatureEngineer``; the recursive forecasting strategy predicts one step at a
time, appends the prediction to the series, and repeats for the full horizon.

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
    from sklearn.ensemble import RandomForestRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.features import FeatureConfig, FeatureEngineer
from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    IntParam,
    suggest_from_space,
)

_MIN_TRAIN_SIZE: int = 24
_N_CV_FOLDS: int = 3


class RandomForestSpec:
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
    needs_features: bool = True

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        forecast_horizon: int = 12,
    ) -> None:
        if not HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for RandomForestSpec. "
                "Install it with: pip install scikit-learn>=1.3"
            )
        self.feature_config: FeatureConfig = feature_config or FeatureConfig()
        self.forecast_horizon: int = forecast_horizon

    # ── ModelSpec protocol ────────────────────────────────────────────────────

    @property
    def search_space(self) -> dict:
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
        """Evaluate *params* via 3-fold expanding-window CV.

        Each fold fits a fresh ``FeatureEngineer`` on the training window only,
        trains a ``RandomForestRegressor``, then forecasts recursively over the
        test window.

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
                X_train, y_train = fe.fit_transform(train)
                rf = RandomForestRegressor(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_split=params["min_samples_split"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_features=params["max_features"],
                    random_state=42,
                    n_jobs=1,
                )
                rf.fit(X_train, y_train)
                y_pred = self._recursive_forecast(rf, fe, train, len(test))
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

        The returned callable fits a ``RandomForestRegressor`` on *train* and
        produces recursive multi-step forecasts of length ``forecast_horizon``.

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
            rf = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=1,
            )
            rf.fit(X_train, y_train)
            return self._recursive_forecast(rf, fe, train, horizon)

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
            model: Fitted ``RandomForestRegressor``.
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
