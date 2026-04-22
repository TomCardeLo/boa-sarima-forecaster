"""QuantileMLSpec — probabilistic forecasts via LGBM/XGB quantile regression.

Fits one booster per quantile (sharing hyperparameters, differing only in the
objective's alpha).  Primary API: ``build_quantile_forecaster`` returns a
callable producing a ``QuantileForecast``.  For ModelSpec Protocol conformance
(and to slot into ``run_model_comparison``/``build_ensemble``),
``build_forecaster`` returns a callable producing only the median as a
pd.Series.

Design
------
- ``evaluate`` fits ONLY the median booster (q=0.5) and scores it against
  metric_fn.  Tail boosters are fit lazily in ``build_quantile_forecaster``
  — avoids N× cost per Optuna trial.
- ``build_quantile_forecaster`` post-processes via isotonic sort across
  quantiles per forecast step to guarantee lower ≤ median ≤ upper (prevents
  quantile crossing that independent boosters can produce).
- Reuses BaseMLSpec feature pipeline (lags, rolling, calendar) unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from boa_forecaster.features import FeatureConfig, FeatureEngineer
from boa_forecaster.models._ml_base import BaseMLSpec
from boa_forecaster.models._utils import recursive_forecast, split_for_early_stopping
from boa_forecaster.models.base import FloatParam, IntParam, SearchSpaceParam


@dataclass(frozen=True)
class QuantileForecast:
    """Container for a quantile forecast produced by QuantileMLSpec.

    Attributes
    ----------
    median : pd.Series
        Forecast for q=0.5.
    lower : pd.Series
        Forecast for the smallest configured quantile (e.g. 0.1).
    upper : pd.Series
        Forecast for the largest configured quantile (e.g. 0.9).
    quantiles : dict
        All configured quantiles mapped to their forecast series, including
        median/lower/upper.  Post-sorted so lower ≤ median ≤ upper for every
        step.
    """

    median: pd.Series
    lower: pd.Series
    upper: pd.Series
    quantiles: dict[float, pd.Series]


class QuantileMLSpec(BaseMLSpec):
    """``ModelSpec`` implementation for probabilistic quantile forecasting.

    Fits one gradient-booster per quantile (sharing hyperparameters) so that
    prediction intervals can be extracted alongside the median forecast.
    Monotonicity (lower ≤ median ≤ upper) is enforced via per-step isotonic
    sort after all boosters are fit.

    Args:
        base: Gradient-boosting backend — ``"lightgbm"`` (default) or
            ``"xgboost"``.  LightGBM uses ``objective="quantile"``; XGBoost
            uses ``objective="reg:quantileerror"``.
        quantiles: Sorted tuple of quantiles to forecast.  Must contain
            ``0.5`` (used as the median forecast) and at least 3 values.
            All values must be in ``(0, 1)`` and unique.  Defaults to
            ``(0.1, 0.5, 0.9)``.
        feature_config: Feature engineering settings.  Defaults to the
            standard ``BaseMLSpec`` lag set (lags 1/2/3/6/12 + horizon).
        forecast_horizon: Number of future steps to predict.  Default 12.
        early_stopping_rounds: Patience for early stopping in CV folds.
            Default 20.

    Raises:
        ValueError: If ``base`` is not ``"lightgbm"`` or ``"xgboost"``.
        ValueError: If ``quantiles`` fails any of the validation rules.
        ImportError: If the selected backend is not installed.
    """

    name: str = "quantile_ml"
    uses_early_stopping: bool = True

    def __init__(
        self,
        base: Literal["lightgbm", "xgboost"] = "lightgbm",
        quantiles: tuple = (0.1, 0.5, 0.9),
        feature_config: FeatureConfig | None = None,
        forecast_horizon: int = 12,
        early_stopping_rounds: int = 20,
    ) -> None:
        # Validate base FIRST (before super().__init__ calls _check_availability)
        if base not in ("lightgbm", "xgboost"):
            raise ValueError(f"base must be 'lightgbm' or 'xgboost'; got {base!r}")
        self.base = base  # must be set before _check_availability runs
        super().__init__(
            feature_config=feature_config, forecast_horizon=forecast_horizon
        )
        # Validate quantiles
        q_sorted = tuple(sorted(float(q) for q in quantiles))
        if len(q_sorted) < 3:
            raise ValueError(
                f"need at least 3 quantiles (lower, median, upper); got {len(q_sorted)}"
            )
        if not all(0.0 < q < 1.0 for q in q_sorted):
            raise ValueError(f"all quantiles must be in (0, 1); got {q_sorted}")
        if 0.5 not in q_sorted:
            raise ValueError(
                f"quantiles must include 0.5 (used as the median forecast); got {q_sorted}"
            )
        if len(set(q_sorted)) != len(q_sorted):
            raise ValueError(f"quantiles must be unique; got {q_sorted}")
        self.quantiles: tuple = q_sorted
        self.early_stopping_rounds: int = int(early_stopping_rounds)

    # ── Subclass hooks ────────────────────────────────────────────────────────

    def _check_availability(self) -> None:
        if self.base == "lightgbm" and not HAS_LIGHTGBM:
            raise ImportError(
                "lightgbm is required for QuantileMLSpec(base='lightgbm'). "
                "Install it with: pip install 'sarima-bayes[ml]'"
            )
        if self.base == "xgboost" and not HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for QuantileMLSpec(base='xgboost'). "
                "Install it with: pip install 'sarima-bayes[ml]'"
            )

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """Delegate to the selected backend's search space.

        LightGBM uses ``num_leaves`` + ``min_child_samples`` (no ``gamma``).
        XGBoost uses ``gamma`` + ``min_child_weight`` (no ``num_leaves``).
        """
        if self.base == "lightgbm":
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
        return {  # xgboost
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
        """Two warm-start configurations — identical to the point-forecast specs."""
        if self.base == "lightgbm":
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
        return [  # xgboost
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
        """Sample a point from ``search_space`` via Optuna trial suggestions.

        For LightGBM, clips ``num_leaves`` to ``2^max_depth - 1`` when
        ``max_depth > 0`` (same constraint as ``LightGBMSpec``).
        """
        params = super().suggest_params(trial)
        if self.base == "lightgbm":
            max_depth = params["max_depth"]
            if max_depth > 0:
                max_leaves = max(1, (2**max_depth) - 1)
                params["num_leaves"] = min(params["num_leaves"], max_leaves)
        return params

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _quantile_fit_kwargs(self, quantile: float) -> dict:
        """Return backend-specific kwargs that select quantile regression."""
        if self.base == "lightgbm":
            return {"objective": "quantile", "alpha": float(quantile)}
        return {"objective": "reg:quantileerror", "quantile_alpha": float(quantile)}

    def _fit_fold(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """Fit the MEDIAN booster with early stopping (used by BaseMLSpec.evaluate)."""
        merged = {**params, **self._quantile_fit_kwargs(0.5)}
        X_tr, y_tr, X_val, y_val = split_for_early_stopping(X, y)
        if self.base == "lightgbm":
            callbacks: list[Callable[..., Any]] = [
                lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(-1),
            ]
            model = lgb.LGBMRegressor(**merged, random_state=42, verbose=-1)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
            return model
        model = xgb.XGBRegressor(
            **merged,
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return model

    def _fit_final(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """Fit the MEDIAN booster on the full training window (no early stopping).

        Used by the inherited ``build_forecaster`` for the point-forecast API.
        """
        return self._fit_one_quantile(
            X, y, params, quantile=0.5, use_early_stopping=False
        )

    def _fit_one_quantile(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        params: dict,
        *,
        quantile: float,
        use_early_stopping: bool,
    ):
        """Fit a single booster for the given quantile."""
        merged = {**params, **self._quantile_fit_kwargs(quantile)}
        if self.base == "lightgbm":
            if use_early_stopping:
                X_tr, y_tr, X_val, y_val = split_for_early_stopping(X, y)
                callbacks: list[Callable[..., Any]] = [
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(-1),
                ]
                model = lgb.LGBMRegressor(**merged, random_state=42, verbose=-1)
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)
                return model
            model = lgb.LGBMRegressor(**merged, random_state=42, verbose=-1)
            model.fit(X, y)
            return model
        # xgboost
        if use_early_stopping:
            X_tr, y_tr, X_val, y_val = split_for_early_stopping(X, y)
            model = xgb.XGBRegressor(
                **merged,
                early_stopping_rounds=self.early_stopping_rounds,
                random_state=42,
                verbosity=0,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return model
        model = xgb.XGBRegressor(**merged, random_state=42, verbosity=0)
        model.fit(X, y)
        return model

    # build_forecaster is INHERITED from BaseMLSpec — returns median pd.Series.
    # No override needed; _fit_final already fits the median booster.

    def build_quantile_forecaster(
        self,
        params: dict,
        feature_config: FeatureConfig | None = None,
    ) -> Callable[[pd.Series], QuantileForecast]:
        """Return a forecaster that produces ALL configured quantiles.

        At forecast time, fits one booster per quantile (sharing
        hyperparameters) and post-sorts predictions across quantiles per
        step to guarantee lower ≤ median ≤ upper (monotonicity via isotonic
        sort — prevents quantile crossing that independent boosters produce).

        Args:
            params: Optimal hyperparameters (typically from ``optimize_model``).
            feature_config: Override feature engineering settings.

        Returns:
            Callable ``(train: pd.Series) -> QuantileForecast``.
        """
        config = feature_config or self.feature_config
        horizon = self.forecast_horizon
        quantiles = self.quantiles
        fit_one = self._fit_one_quantile

        def forecaster(train: pd.Series) -> QuantileForecast:
            fe = FeatureEngineer(config)
            X_train, y_train = fe.fit_transform(train)
            preds: dict[float, pd.Series] = {}
            for q in quantiles:
                model = fit_one(
                    X_train, y_train, params, quantile=q, use_early_stopping=False
                )
                preds[q] = recursive_forecast(model, fe, train, horizon)

            # Post-sort: rearrangement (Chernozhukov et al. 2010) across quantiles
            # per forecast step.  np.sort along axis=1 ensures lower ≤ median ≤ upper
            # at every step by reassigning crossed values to their value-order quantile
            # slot.  Quantiles are pre-sorted ascending in __init__, so the i-th sorted
            # column is by construction the empirical i-th quantile label.
            matrix = np.column_stack([preds[q].to_numpy() for q in quantiles])
            matrix_sorted = np.sort(matrix, axis=1)
            sorted_preds = {
                q: pd.Series(matrix_sorted[:, i], index=preds[q].index, name=f"q{q}")
                for i, q in enumerate(quantiles)
            }

            return QuantileForecast(
                median=sorted_preds[0.5],
                lower=sorted_preds[quantiles[0]],
                upper=sorted_preds[quantiles[-1]],
                quantiles=sorted_preds,
            )

        return forecaster
