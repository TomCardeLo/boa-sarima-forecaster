"""Generic Bayesian TPE optimisation engine for boa-forecaster.

``optimize_model`` accepts any ``ModelSpec`` and runs an Optuna TPE study,
returning an ``OptimizationResult``.  The legacy ``optimize_arima`` function
is preserved as a deprecated wrapper that produces the same ``(dict, float)``
tuple as sarima_bayes v1.x.

Design decisions (same as v1)
------------------------------
* **TPE sampler** with ``multivariate=True`` captures parameter correlations.
* **Warm starts** are enqueued before TPE's surrogate model takes over.
* **Soft failure** — if the whole study crashes, fall back to the first
  warm-start params and ``OPTIMIZER_PENALTY`` rather than propagating the
  exception to the caller.
* **Reproducibility** via ``seed=42`` default.
"""

from __future__ import annotations

import logging
import os
import warnings as _warnings
from typing import TYPE_CHECKING

import optuna
from optuna.samplers import TPESampler

from boa_forecaster.config import (
    DEFAULT_D_RANGE,
    DEFAULT_D_SEASONAL_RANGE,
    DEFAULT_METRIC_COMPONENTS,
    DEFAULT_P_RANGE,
    DEFAULT_P_SEASONAL_RANGE,
    DEFAULT_Q_RANGE,
    DEFAULT_Q_SEASONAL_RANGE,
    DEFAULT_SEASONAL_PERIOD,
    OPTIMIZER_PENALTY,
)
from boa_forecaster.metrics import build_combined_metric
from boa_forecaster.models.base import OptimizationResult

if TYPE_CHECKING:
    import pandas as pd

    from boa_forecaster.features import FeatureConfig
    from boa_forecaster.models.base import ModelSpec

# Suppress Optuna's internal progress logs and experimental warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
_warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

logger = logging.getLogger(__name__)


def optimize_model(
    series: pd.Series,
    model_spec: ModelSpec,
    n_calls: int = 50,
    n_jobs: int = 1,
    metric_components: list[dict] | None = None,
    feature_config: FeatureConfig | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> OptimizationResult:
    """Run a TPE Bayesian search over a ``ModelSpec``'s search space.

    Args:
        series: Training time series with ``DatetimeIndex``.
        model_spec: Any object satisfying the ``ModelSpec`` protocol.
        n_calls: Total Optuna trials (includes warm starts).
        n_jobs: Parallel workers.  ``<= 0`` → all available CPUs.
        metric_components: List of ``{"metric": str, "weight": float}``
            dicts.  Defaults to ``0.7·sMAPE + 0.3·RMSLE``.
        feature_config: Feature configuration passed to the model's
            ``evaluate`` and ``build_forecaster`` methods.  Ignored when
            ``model_spec.needs_features is False``.
        seed: TPE sampler seed for reproducibility.
        verbose: If ``True``, Optuna logs trial progress to stdout.

    Returns:
        ``OptimizationResult`` with ``best_params``, ``best_score``,
        ``n_trials``, and ``model_name``.
    """
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    components = (
        metric_components
        if metric_components is not None
        else DEFAULT_METRIC_COMPONENTS
    )
    metric_fn = build_combined_metric(components)

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(trial: optuna.Trial) -> float:
        params = model_spec.suggest_params(trial)
        return model_spec.evaluate(series, params, metric_fn, feature_config)

    sampler = TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    for warm_start in model_spec.warm_starts:
        study.enqueue_trial(warm_start)

    try:
        study.optimize(objective, n_trials=n_calls, n_jobs=n_jobs)
    except Exception as exc:
        logger.error("Optimisation study failed: %s", exc)
        fallback = model_spec.warm_starts[0] if model_spec.warm_starts else {}
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        return OptimizationResult(
            best_params=fallback,
            best_score=OPTIMIZER_PENALTY,
            n_trials=0,
            model_name=model_spec.name,
        )

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        "Best %s params: %s  |  Score: %.4f",
        model_spec.name,
        study.best_params,
        study.best_value,
    )

    return OptimizationResult(
        best_params=study.best_params,
        best_score=study.best_value,
        n_trials=len(study.trials),
        model_name=model_spec.name,
    )


def optimize_arima(
    series,
    p_range: tuple[int, int] = DEFAULT_P_RANGE,
    d_range: tuple[int, int] = DEFAULT_D_RANGE,
    q_range: tuple[int, int] = DEFAULT_Q_RANGE,
    P_range: tuple[int, int] = DEFAULT_P_SEASONAL_RANGE,
    D_range: tuple[int, int] = DEFAULT_D_SEASONAL_RANGE,
    Q_range: tuple[int, int] = DEFAULT_Q_SEASONAL_RANGE,
    m: int = DEFAULT_SEASONAL_PERIOD,
    n_calls: int = 50,
    n_jobs: int = 1,
    metric_components: list[dict] | None = None,
) -> tuple[dict, float]:
    """Search for optimal SARIMA orders using Optuna TPE.

    .. deprecated::
        Use ``optimize_model(series, SARIMASpec(...))`` instead.
        This function will be removed in v3.0.

    Maintained for full backwards compatibility with sarima_bayes v1.x:

    - Accepts a raw ``np.ndarray`` (converted internally to ``pd.Series``).
    - Returns ``(best_params, best_value)`` where ``best_params`` includes
      the ``"m"`` key matching the v1 contract.

    Args:
        series: 1-D array or ``pd.Series`` of historical observations.
        p_range: Inclusive ``(min, max)`` for AR order *p*.
        d_range: Inclusive ``(min, max)`` for differencing order *d*.
        q_range: Inclusive ``(min, max)`` for MA order *q*.
        P_range: Inclusive ``(min, max)`` for seasonal AR order *P*.
        D_range: Inclusive ``(min, max)`` for seasonal differencing *D*.
        Q_range: Inclusive ``(min, max)`` for seasonal MA order *Q*.
        m: Fixed seasonal period (not optimised).
        n_calls: Total Optuna trials.
        n_jobs: Parallel workers.
        metric_components: Custom metric composition.

    Returns:
        Tuple ``(best_params, best_value)`` where ``best_params`` has keys
        ``{"p", "d", "q", "P", "D", "Q", "m"}``.
    """
    import warnings

    import numpy as np
    import pandas as pd

    warnings.warn(
        "optimize_arima() is deprecated. Use optimize_model() with SARIMASpec instead. "
        "'optimize_arima' will be removed in v3.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    # Convert numpy arrays to pd.Series so SARIMASpec.evaluate works uniformly
    if not isinstance(series, pd.Series):
        arr = np.asarray(series)
        idx = pd.date_range("2020-01-01", periods=len(arr), freq="MS")
        series = pd.Series(arr, index=idx)

    from boa_forecaster.models.sarima import SARIMASpec

    spec = SARIMASpec(
        p_range=p_range,
        d_range=d_range,
        q_range=q_range,
        P_range=P_range,
        D_range=D_range,
        Q_range=Q_range,
        m=m,
    )
    result = optimize_model(
        series,
        spec,
        n_calls=n_calls,
        n_jobs=n_jobs,
        metric_components=metric_components,
    )

    # Add "m" to match the v1 best_params contract
    best_params = dict(result.best_params)
    best_params["m"] = m
    return best_params, result.best_score
