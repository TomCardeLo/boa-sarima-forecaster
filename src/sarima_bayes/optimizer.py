"""Bayesian Optimisation for ARIMA hyper-parameter search.

This module uses Optuna's Tree-structured Parzen Estimator (TPE) to search
the integer space of ``ARIMA(p, d, q)`` orders, minimising the combined
cost function (0.7 · sMAPE + 0.3 · RMSLE) computed on in-sample predictions.

Design decisions
----------------
* **TPE sampler** with ``multivariate=True`` captures correlations between
  the *p* and *q* parameters, improving sample efficiency over independent
  univariate TPE.
* **Warm start** injects two sensible initial trials (``ARIMA(1,1,1)`` and
  ``AR(1)``) before the probabilistic surrogate model takes over, reducing
  wasted evaluations early in the search.
* **Soft-constraint penalty** (``1e6``) is returned on model failure instead
  of raising an exception, so the TPE sampler learns to avoid infeasible
  regions without crashing the study.
* **Reproducibility** is guaranteed via ``seed=42`` in the sampler.
"""

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sarima_bayes.metrics import combined_metric

# Suppress Optuna's internal progress logs and experimental warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings as _warnings
_warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

logger = logging.getLogger(__name__)

# Penalty score returned when a trial raises an exception.
# Using a large but finite value keeps the TPE estimator numerically stable
# (float('inf') can cause issues in some Optuna internals).
_PENALTY: float = 1e6


def _evaluate_arima(
    series: np.ndarray,
    p: int,
    d: int,
    q: int,
) -> Tuple[int, int, int, float, Optional[str]]:
    """Fit ``ARIMA(p, d, q)`` and compute the in-sample combined metric.

    This function is the atomic evaluation unit called by the Optuna objective.
    On any exception it returns :data:`_PENALTY` so the optimiser can continue
    exploring other parameter regions without crashing.

    Args:
        series: 1-D NumPy array of historical observations (monthly demand).
        p: Autoregressive order.
        d: Integration order.
        q: Moving-average order.

    Returns:
        Tuple ``(p, d, q, score, error_message)`` where ``error_message``
        is ``None`` on success or a string describing the exception.
    """
    try:
        # enforce_stationarity / enforce_invertibility = False allows the
        # optimiser to test parameters that may violate strict stationarity
        # or invertibility conditions; the cost function penalises bad fits.
        model = SARIMAX(
            series,
            order=(p, d, q),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        model_fit = model.fit(disp=False)

        # In-sample predictions: assess goodness-of-fit on historical data
        pred = model_fit.predict(start=0, end=len(series) - 1)
        score = combined_metric(series, pred)

        return (p, d, q, score, None)

    except Exception as exc:
        # Soft constraint: guide TPE away from this region without crashing
        return (p, d, q, _PENALTY, str(exc))


def optimize_arima(
    series: np.ndarray,
    p_range: Tuple[int, int] = (0, 6),
    d_range: Tuple[int, int] = (0, 2),
    q_range: Tuple[int, int] = (0, 6),
    n_calls: int = 50,
    n_jobs: int = 1,
) -> Tuple[Dict[str, int], float]:
    """Search for optimal ``ARIMA(p, d, q)`` orders using Optuna TPE.

    Minimises :func:`~sarima_bayes.metrics.combined_metric`
    (0.7 · sMAPE + 0.3 · RMSLE) computed on in-sample predictions.

    The study is pre-seeded with two warm-start trials:

    - ``ARIMA(1, 1, 1)`` — a robust baseline for trending monthly demand.
    - ``AR(1)``          — covers stationary series with short-memory
      autocorrelation.

    These warm starts accelerate early convergence and are evaluated before
    the probabilistic surrogate model begins guiding suggestions.

    Args:
        series: 1-D NumPy array of historical observations (monthly demand).
            Should contain at least ``max(p_range) + d_range[1] + 5`` points
            to allow meaningful differencing.
        p_range: ``(min, max)`` inclusive integer range for *p*.
            Defaults to ``(0, 6)``.
        d_range: ``(min, max)`` inclusive integer range for *d*.
            Defaults to ``(0, 2)``.
        q_range: ``(min, max)`` inclusive integer range for *q*.
            Defaults to ``(0, 6)``.
        n_calls: Total number of Optuna trials. Defaults to ``50``.
        n_jobs: Number of parallel Optuna workers.  Use ``-1`` to
            auto-detect CPU cores.  Defaults to ``1``.

    Returns:
        Tuple ``(best_params, best_value)`` where:

        - ``best_params`` – dict with keys ``"p"``, ``"d"``, ``"q"``
          mapping to the optimal integer orders.
        - ``best_value``  – the minimum combined metric score achieved.

    Note:
        Returns ``{"p": 1, "d": 1, "q": 1}`` and ``1e6`` as a safe fallback
        if the entire study fails.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> series = rng.normal(100, 10, 48)
        >>> params, score = optimize_arima(series, n_calls=20)
        >>> set(params.keys()) == {"p", "d", "q"}
        True
    """
    # Resolve automatic CPU count
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: suggest (p, d, q) and return the combined cost."""
        p = trial.suggest_int("p", p_range[0], p_range[1])
        d = trial.suggest_int("d", d_range[0], d_range[1])
        q = trial.suggest_int("q", q_range[0], q_range[1])

        _, _, _, score, _ = _evaluate_arima(series, p, d, q)
        return score

    # ── Sampler ────────────────────────────────────────────────────────────────
    # multivariate=True: TPE captures (p, q) correlations for better proposals.
    # seed=42: ensures reproducible parameter suggestions across runs.
    sampler = TPESampler(seed=42, multivariate=True)

    # In-memory storage maximises throughput (no disk I/O overhead per trial)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # ── Warm Start ─────────────────────────────────────────────────────────────
    # Enqueue known-good starting points before TPE's probabilistic model kicks
    # in.  This prevents wasting early trials on trivial (0,0,0) combinations.
    study.enqueue_trial({"p": 1, "d": 1, "q": 1})  # robust trending baseline
    study.enqueue_trial({"p": 1, "d": 0, "q": 0})  # simple AR(1) for stationary series

    # ── Optimisation ───────────────────────────────────────────────────────────
    try:
        study.optimize(objective, n_trials=n_calls, n_jobs=n_jobs)
    except Exception as exc:
        logger.error("Optimisation study failed: %s", exc)
        return {"p": 1, "d": 1, "q": 1}, _PENALTY

    best_params = study.best_params   # {"p": int, "d": int, "q": int}
    best_value = study.best_value     # float

    logger.info(
        "Best ARIMA params: %s  |  Score: %.4f",
        best_params,
        best_value,
    )

    return best_params, best_value
