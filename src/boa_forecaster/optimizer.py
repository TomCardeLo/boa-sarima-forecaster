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

import numpy as np
import optuna
import pandas as pd
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
    from boa_forecaster.features import FeatureConfig
    from boa_forecaster.models.base import ModelSpec

# Suppress Optuna's internal progress logs and experimental warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
_warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

logger = logging.getLogger(__name__)

_MIN_SERIES_LENGTH: int = 24


def _compute_bias_from_last_fold(
    series: pd.Series,
    model_spec: ModelSpec,
    best_params: dict,
    feature_config: FeatureConfig | None,
) -> np.ndarray | None:
    """Compute seasonal bias factors from the last CV fold's residuals.

    Uses the same expanding-window math as ``walk_forward_validation`` with
    ``n_folds=3, test_size=model_spec.forecast_horizon (default 12),
    min_train_size=24``.  The final (third) fold's train/test split is
    replicated here so no extra imports are needed and the logic stays
    co-located with the call site.

    Returns ``None`` on any failure so callers always get a clean result.
    """
    from boa_forecaster.postprocess import compute_seasonal_bias

    n_folds = 3
    test_size = int(getattr(model_spec, "forecast_horizon", 12))
    min_train_size = 24
    required = min_train_size + n_folds * test_size

    if test_size < 2:
        logger.warning(
            "Bias computation skipped for %s: forecast_horizon=%d is too short "
            "to compute a meaningful per-period median.",
            model_spec.name,
            test_size,
        )
        return None

    if len(series) < required:
        return None

    try:
        forecaster = model_spec.build_forecaster(best_params, feature_config)
        # Last fold indices (zero-based fold index = n_folds - 1)
        last_fold = n_folds - 1
        train_end = min_train_size + last_fold * test_size
        test_end = train_end + test_size

        train = series.iloc[:train_end]
        test = series.iloc[train_end:test_end]

        predictions = forecaster(train)

        if len(predictions) < test_size:
            logger.warning(
                "Bias computation skipped for %s: forecaster returned %d predictions "
                "but test_size=%d — refusing to silently trim.",
                model_spec.name,
                len(predictions),
                test_size,
            )
            return None

        y_true = test.values[: len(predictions)]
        y_pred = predictions.values[: len(y_true)]

        # Re-slice the test index to match trimmed length for DatetimeIndex alignment
        test_index = test.index[: len(y_true)]
        y_true_s = pd.Series(y_true, index=test_index)
        y_pred_s = pd.Series(y_pred, index=test_index)

        return compute_seasonal_bias(y_true_s, y_pred_s, periods=12, start_period=1)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Bias computation failed for %s: %s — bias_correction will be None.",
            model_spec.name,
            exc,
            exc_info=True,
        )
        return None


def _validate_series(series: pd.Series, min_length: int = _MIN_SERIES_LENGTH) -> None:
    """Validate *series* before running optimisation.

    Raises:
        TypeError: If *series* is not a ``pd.Series``.
        ValueError: If the index is not a ``DatetimeIndex``, the series is
            too short, or the series contains NaN or Inf values.
    """
    if not isinstance(series, pd.Series):
        raise TypeError(f"series must be a pd.Series, got {type(series).__name__!r}.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError(
            "series must have a DatetimeIndex. "
            "Use pd.date_range() to build an appropriate index."
        )
    if len(series) < min_length:
        raise ValueError(
            f"series has {len(series)} observations but at least "
            f"{min_length} are required for meaningful optimisation."
        )
    if series.isna().any():
        raise ValueError(
            "series contains NaN values. "
            "Handle missing values before calling optimize_model()."
        )
    # ``np.isinf`` on the underlying ndarray is ~10-20× faster than
    # ``series.isin([np.inf, -np.inf])`` on large floats because it avoids
    # building a hashable set and two element-wise equality passes.
    if np.isinf(series.to_numpy()).any():
        raise ValueError(
            "series contains Inf values. "
            "Remove or replace infinite values before calling optimize_model()."
        )


_MIN_BIAS_SERIES_LENGTH: int = 36  # need at least 3 full periods for bias


def optimize_model(
    series: pd.Series,
    model_spec: ModelSpec,
    n_calls: int = 50,
    n_jobs: int = 1,
    metric_components: list[dict] | None = None,
    feature_config: FeatureConfig | None = None,
    seed: int = 42,
    verbose: bool = False,
    apply_bias_correction: bool = False,
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
        apply_bias_correction: When ``True``, compute per-period multiplicative
            bias factors from the final CV fold's residuals after the study
            completes, and attach them to ``OptimizationResult.bias_correction``.
            Default ``False`` preserves backward compatibility.

    Returns:
        ``OptimizationResult`` with ``best_params``, ``best_score``,
        ``n_trials``, and ``model_name``.  When ``apply_bias_correction=True``
        and the series is long enough, ``result.bias_correction`` contains a
        ``np.ndarray`` of shape ``(12,)`` with the per-month factors.
    """
    _validate_series(series)

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

    # Deterministic-feature cache (v2.2 P1): calendar + trend columns are pure
    # functions of the DatetimeIndex.  Pre-compute them once for the full
    # series and thread the slice through every trial / fold so the CV loop
    # stops recomputing identical sin/cos/year_norm/trend_idx arrays.
    feature_cache: pd.DataFrame | None = None
    if getattr(model_spec, "needs_features", False) and isinstance(
        series.index, pd.DatetimeIndex
    ):
        from boa_forecaster.features import (
            FeatureConfig,
            _compute_deterministic_features,
        )

        cache_config = (
            feature_config
            or getattr(model_spec, "feature_config", None)
            or FeatureConfig()
        )
        if cache_config.include_calendar or cache_config.include_trend:
            feature_cache = _compute_deterministic_features(series.index, cache_config)

    def objective(trial: optuna.Trial) -> float:
        params = model_spec.suggest_params(trial)
        if feature_cache is not None:
            return model_spec.evaluate(
                series,
                params,
                metric_fn,
                feature_config,
                feature_cache=feature_cache,
                trial=trial,
            )
        return model_spec.evaluate(
            series, params, metric_fn, feature_config, trial=trial
        )

    sampler = TPESampler(seed=seed, multivariate=True)
    # MedianPruner (v2.3 E4): abort trials whose intermediate fold scores
    # are worse than the median of completed trials at the same step.
    # ``n_startup_trials=5`` lets TPE gather a baseline before pruning
    # engages; ``n_warmup_steps=1`` ensures each trial completes at least
    # one full fold before becoming pruneable.  The study crashes on
    # anything else — ``TrialPruned`` is caught by Optuna internally, so
    # the outer ``try/except Exception`` below never sees it.
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    for warm_start in model_spec.warm_starts:
        study.enqueue_trial(warm_start)

    try:
        study.optimize(objective, n_trials=n_calls, n_jobs=n_jobs)
    except Exception as exc:
        # Soft-failure: surface the crash via ``is_fallback=True`` + a WARNING
        # log (with traceback) so callers can distinguish a genuine optimum
        # from a default warm-start returned after a study crash.
        logger.warning(
            "Optimisation study for %s failed: %s — returning fallback result.",
            model_spec.name,
            exc,
            exc_info=True,
        )
        fallback = model_spec.warm_starts[0] if model_spec.warm_starts else {}
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        return OptimizationResult(
            best_params=fallback,
            best_score=OPTIMIZER_PENALTY,
            n_trials=0,
            model_name=model_spec.name,
            is_fallback=True,
        )

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(
        "Best %s params: %s  |  Score: %.4f",
        model_spec.name,
        study.best_params,
        study.best_value,
    )

    bias: np.ndarray | None = None
    if apply_bias_correction:
        _n_folds = 3
        _test_size = int(getattr(model_spec, "forecast_horizon", 12))
        _min_train = 24
        _required = _min_train + _n_folds * _test_size
        if len(series) < _required:
            logger.warning(
                "apply_bias_correction=True for %s but series has %d observations "
                "(need %d = %d min_train + %d folds × %d test_size); "
                "bias_correction will be None.",
                model_spec.name,
                len(series),
                _required,
                _min_train,
                _n_folds,
                _test_size,
            )
        else:
            bias = _compute_bias_from_last_fold(
                series, model_spec, study.best_params, feature_config
            )

    return OptimizationResult(
        best_params=study.best_params,
        best_score=study.best_value,
        n_trials=len(study.trials),
        model_name=model_spec.name,
        is_fallback=False,
        bias_correction=bias,
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
