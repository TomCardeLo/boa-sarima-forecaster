"""Shared base class for tree-based ML ``ModelSpec`` implementations.

``BaseMLSpec`` factors out the cross-validation loop, the ``build_forecaster``
closure, and the default ``suggest_params`` implementation that were previously
duplicated across ``RandomForestSpec``, ``XGBoostSpec``, and ``LightGBMSpec``.

Subclasses only override the parts that differ:

``_check_availability``
    Verify the optional dependency is installed.  Default is a no-op.
``search_space`` / ``warm_starts``
    Per-model hyperparameter space and warm starts.
``_fit_final(X, y, params) -> model``
    Build and fit the regressor.  Used by ``build_forecaster`` (no early
    stopping) and as the default for CV folds.
``_fit_fold(X, y, params) -> model``
    Optional override used by CV folds.  Default falls back to
    ``_fit_final``; XGBoost/LightGBM override it to apply early stopping
    against an inner-validation split.
``suggest_params(trial) -> dict``
    Default calls ``suggest_from_space``.  LightGBM overrides it to clip
    ``num_leaves`` against ``max_depth``.

Temporal integrity
------------------
A fresh ``FeatureEngineer`` instance is created per CV fold so that
rolling/expanding stats are fitted *only* on that fold's training window.
No test-set information leaks into the features.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import optuna
import pandas as pd

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.features import FeatureConfig, FeatureEngineer
from boa_forecaster.models._utils import (
    MIN_TRAIN_SIZE,
    N_CV_FOLDS,
    recursive_forecast,
)
from boa_forecaster.models.base import SearchSpaceParam, suggest_from_space

logger = logging.getLogger(__name__)


class BaseMLSpec:
    """Abstract base for tree-based ML ``ModelSpec`` implementations.

    Subclasses must define ``name`` (class attribute), ``search_space``,
    ``warm_starts``, and ``_fit_final``.  They may override
    ``_check_availability``, ``_fit_fold``, and ``suggest_params``.
    """

    needs_features: bool = True

    def __init__(
        self,
        feature_config: FeatureConfig | None = None,
        forecast_horizon: int = 12,
    ) -> None:
        self._check_availability()
        self.forecast_horizon: int = forecast_horizon
        if feature_config is None:
            # Auto-inject ``forecast_horizon`` into the default lag set so the
            # model sees a lag_h feature — the single most informative predictor
            # for a h-step-ahead recursive forecast (v2.3 E2).  An explicitly
            # passed ``feature_config`` is the user's choice and is never
            # rewritten, even if ``forecast_horizon`` is absent from its lags.
            lags = [1, 2, 3, 6, 12]
            if forecast_horizon not in lags:
                lags.append(forecast_horizon)
            feature_config = FeatureConfig(lag_periods=sorted(lags))
        self.feature_config: FeatureConfig = feature_config

    # ── Overridable hooks ─────────────────────────────────────────────────────

    def _check_availability(self) -> None:
        """Raise ``ImportError`` if the backing library is missing.

        Default is a no-op.  Subclasses with optional dependencies (XGBoost,
        LightGBM) override this to guard instantiation.
        """
        return

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        raise NotImplementedError

    @property
    def warm_starts(self) -> list[dict]:
        raise NotImplementedError

    def _fit_final(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """Build and fit the regressor on the full training window.

        Used by ``build_forecaster`` and as the default for CV folds.
        Subclasses must implement this.
        """
        raise NotImplementedError

    def _fit_fold(self, X: pd.DataFrame, y: pd.Series, params: dict):
        """Build and fit the regressor inside a single CV fold.

        Default delegates to ``_fit_final`` — appropriate for models without
        early stopping (e.g. Random Forest).  Subclasses that benefit from an
        inner-validation split for early stopping (XGBoost, LightGBM) override
        this.
        """
        return self._fit_final(X, y, params)

    def suggest_params(self, trial) -> dict:
        """Sample a point from ``search_space`` via Optuna trial suggestions."""
        return suggest_from_space(trial, self.search_space)

    # ── Shared CV + forecaster implementations ───────────────────────────────

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn,
        feature_config: FeatureConfig | None = None,
        feature_cache: pd.DataFrame | None = None,
        trial: optuna.Trial | None = None,
    ) -> float:
        """Evaluate *params* via expanding-window CV.

        Each fold fits a fresh ``FeatureEngineer`` on the fold's training
        window only, trains the regressor via ``_fit_fold``, then forecasts
        recursively over the test window.

        Args:
            series: Monthly time series with ``DatetimeIndex``.
            params: Hyperparameter dict sampled from ``search_space``.
            metric_fn: ``metric_fn(y_true, y_pred) -> float`` objective.
            feature_config: Override for this evaluation only.
            feature_cache: Optional deterministic-feature cache produced by
                ``features._compute_deterministic_features`` over a superset
                of ``series.index``.  The optimiser builds this once per
                study and threads it through every trial so each fold
                avoids recomputing calendar / trend columns (v2.2 P1).
            trial: Active Optuna trial.  When supplied, the running-mean
                score is reported after each completed fold via
                ``trial.report`` and ``optuna.TrialPruned`` is raised if the
                pruner decides the trial is hopeless (v2.3 E4).  ``None``
                when the spec is invoked directly (e.g. from tests).

        Returns:
            Mean metric across valid folds, or ``OPTIMIZER_PENALTY`` on failure.

        Raises:
            optuna.TrialPruned: Only propagates when *trial* is given and the
                pruner flags the trial after an intermediate report.
        """
        config = feature_config or self.feature_config
        n = len(series)
        horizon = self.forecast_horizon
        scores: list[float] = []

        for k in range(N_CV_FOLDS):
            train_end = n - horizon * (k + 1)
            if train_end < MIN_TRAIN_SIZE:
                break

            train = series.iloc[:train_end]
            test = series.iloc[train_end : train_end + horizon]
            if len(test) < horizon:
                break

            try:
                fe = FeatureEngineer(config)
                # The cache (if any) spans the full optimiser series index, so
                # slicing inside _build_features handles both the fit call and
                # the growing extended-series slices used by recursive_forecast
                # via the cache attribute stored on ``fe``.
                X_train, y_train = fe.fit_transform(train, feature_cache=feature_cache)
                model = self._fit_fold(X_train, y_train, params)
                y_pred = recursive_forecast(model, fe, train, len(test))
                scores.append(metric_fn(test.values, y_pred.values))
            except optuna.TrialPruned:
                # Pruning is expected flow-control, not a crash — must escape
                # the broad ``except Exception`` below so Optuna sees it.
                raise
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "%s.evaluate fold %d failed: %s",
                    type(self).__name__,
                    k + 1,
                    exc,
                )
                return OPTIMIZER_PENALTY

            if trial is not None:
                # Report the running mean after each successful fold — TPE's
                # MedianPruner compares this against sibling trials at the
                # same step to decide whether to abort early.
                trial.report(float(np.mean(scores)), step=k)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        if not scores:
            return OPTIMIZER_PENALTY

        return float(np.mean(scores))

    def build_forecaster(
        self,
        params: dict,
        feature_config: FeatureConfig | None = None,
    ) -> Callable[[pd.Series], pd.Series]:
        """Return a ``forecaster(train) -> pd.Series`` closure.

        The closure fits the regressor on the full training window (no early
        stopping) via ``_fit_final`` and produces recursive multi-step forecasts
        of length ``forecast_horizon``.

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
            model = self._fit_final(X_train, y_train, params)
            return recursive_forecast(model, fe, train, horizon)

        return forecaster
