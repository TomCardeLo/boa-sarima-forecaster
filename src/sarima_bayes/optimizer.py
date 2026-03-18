"""Bayesian Optimisation for SARIMA hyper-parameter search.

This module uses Optuna's Tree-structured Parzen Estimator (TPE) to search
the integer space of ``SARIMA(p, d, q)(P, D, Q, m)`` orders, minimising the
combined cost function (0.7 · sMAPE + 0.3 · RMSLE) computed on in-sample
predictions.

Design decisions
----------------
* **TPE sampler** with ``multivariate=True`` captures correlations between
  the *p*, *q*, *P*, and *Q* parameters, improving sample efficiency over
  independent univariate TPE.
* **Warm start** injects two sensible initial trials (``SARIMA(1,1,1)(1,1,1,m)``
  and ``AR(1)``) before the probabilistic surrogate model takes over, reducing
  wasted evaluations early in the search.
* **Soft-constraint penalty** (:data:`OPTIMIZER_PENALTY`) is returned on model
  failure instead of raising an exception, so the TPE sampler learns to avoid
  infeasible regions without crashing the study.
* **Complexity constraints** prune degenerate models before fitting:
  ``(p+q) ≤ 4`` and ``(P+Q) ≤ 3``.
* **Reproducibility** is guaranteed via ``seed=42`` in the sampler.
* **Seasonal period ``m``** is a fixed config parameter — it is NOT optimised.
  Monthly data uses ``m=12`` (annual seasonality cycle).
"""

import logging
import os
import warnings as _warnings
from typing import Optional

import numpy as np
import optuna
from optuna.samplers import TPESampler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sarima_bayes.config import (
    DEFAULT_D_RANGE,
    DEFAULT_D_SEASONAL_RANGE,
    DEFAULT_P_RANGE,
    DEFAULT_P_SEASONAL_RANGE,
    DEFAULT_Q_RANGE,
    DEFAULT_Q_SEASONAL_RANGE,
    DEFAULT_SEASONAL_PERIOD,
)
from sarima_bayes.metrics import combined_metric

# Suppress Optuna's internal progress logs and experimental warnings
optuna.logging.set_verbosity(optuna.logging.WARNING)
_warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

logger = logging.getLogger(__name__)

# Penalty score returned when a trial raises an exception or violates a
# complexity constraint.  Using a large but finite value keeps the TPE
# estimator numerically stable (float('inf') can cause issues in some
# Optuna internals).
OPTIMIZER_PENALTY: float = 1e6


def _evaluate_arima(
    series: np.ndarray,
    p: int,
    d: int,
    q: int,
    P: int = 0,
    D: int = 0,
    Q: int = 0,
    m: int = DEFAULT_SEASONAL_PERIOD,
) -> tuple[int, int, int, float, Optional[str]]:
    """Fit ``SARIMA(p,d,q)(P,D,Q,m)`` and compute the in-sample combined metric.

    This function is the atomic evaluation unit called by the Optuna objective.
    On any exception it returns :data:`OPTIMIZER_PENALTY` so the optimiser can
    continue exploring other parameter regions without crashing.

    Args:
        series: 1-D NumPy array of historical observations (monthly demand).
        p: Autoregressive order.
        d: Integration order.
        q: Moving-average order.
        P: Seasonal autoregressive order.
        D: Seasonal differencing order.
        Q: Seasonal moving-average order.
        m: Seasonal period (fixed, not optimised).  Defaults to
            :data:`~sarima_bayes.config.DEFAULT_SEASONAL_PERIOD`.

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
            seasonal_order=(P, D, Q, m),
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
        return (p, d, q, OPTIMIZER_PENALTY, str(exc))


def optimize_arima(
    series: np.ndarray,
    p_range: tuple[int, int] = DEFAULT_P_RANGE,
    d_range: tuple[int, int] = DEFAULT_D_RANGE,
    q_range: tuple[int, int] = DEFAULT_Q_RANGE,
    P_range: tuple[int, int] = DEFAULT_P_SEASONAL_RANGE,
    D_range: tuple[int, int] = DEFAULT_D_SEASONAL_RANGE,
    Q_range: tuple[int, int] = DEFAULT_Q_SEASONAL_RANGE,
    m: int = DEFAULT_SEASONAL_PERIOD,
    n_calls: int = 50,
    n_jobs: int = 1,
) -> tuple[dict[str, int], float]:
    """Search for optimal ``SARIMA(p,d,q)(P,D,Q,m)`` orders using Optuna TPE.

    Minimises :func:`~sarima_bayes.metrics.combined_metric`
    (0.7 · sMAPE + 0.3 · RMSLE) computed on in-sample predictions.

    The study is pre-seeded with two warm-start trials:

    - ``SARIMA(1,1,1)(1,1,1,m)`` — a robust baseline for trending seasonal demand.
    - ``ARIMA(1,0,0)``            — covers stationary series with short-memory
      autocorrelation (no seasonal component).

    Complexity constraints applied inside the objective (before fitting):

    - ``(p + q) > 4``  → returns :data:`OPTIMIZER_PENALTY` (avoids over-parameterised
      non-seasonal component).
    - ``(P + Q) > 3``  → returns :data:`OPTIMIZER_PENALTY` (avoids over-parameterised
      seasonal component).

    These constraints prune unpromising regions cheaply without spending a full
    model-fitting trial on them.

    .. note::
        ``m`` is a **fixed config parameter** — it is NOT part of the search
        space.  For monthly demand planning, ``m=12`` captures annual seasonality.
        Varying ``m`` would conflate model selection with data-frequency assumptions.

    Args:
        series: 1-D NumPy array of historical observations (monthly demand).
            Should contain at least ``max(p_range) + d_range[1] + m * D_range[1] + 5``
            points to allow meaningful differencing.
        p_range: ``(min, max)`` inclusive integer range for *p*.
            Defaults to ``(0, 3)``.
        d_range: ``(min, max)`` inclusive integer range for *d*.
            Defaults to ``(0, 2)``.
        q_range: ``(min, max)`` inclusive integer range for *q*.
            Defaults to ``(0, 3)``.
        P_range: ``(min, max)`` inclusive integer range for seasonal AR order *P*.
            Defaults to ``(0, 2)``.
        D_range: ``(min, max)`` inclusive integer range for seasonal differencing *D*.
            Defaults to ``(0, 1)``.
        Q_range: ``(min, max)`` inclusive integer range for seasonal MA order *Q*.
            Defaults to ``(0, 2)``.
        m: Fixed seasonal period (not optimised).  Defaults to
            :data:`~sarima_bayes.config.DEFAULT_SEASONAL_PERIOD` (12).
        n_calls: Total number of Optuna trials.  Defaults to ``50``.
        n_jobs: Number of parallel Optuna workers.  Use ``-1`` to
            auto-detect CPU cores.  Defaults to ``1``.

    Returns:
        Tuple ``(best_params, best_value)`` where:

        - ``best_params`` – dict with keys ``"p"``, ``"d"``, ``"q"``,
          ``"P"``, ``"D"``, ``"Q"``, ``"m"`` mapping to the optimal orders.
        - ``best_value``  – the minimum combined metric score achieved.

    Note:
        Returns ``{"p":1,"d":1,"q":1,"P":1,"D":1,"Q":1,"m":m}`` and
        :data:`OPTIMIZER_PENALTY` as a safe fallback if the entire study fails.

    # BREAKING CHANGE: seasonal parameters now included in returned dict.
    # All call sites must unpack P, D, Q, m in addition to p, d, q.

    Example:
        >>> import numpy as np
        >>> rng = np.random.default_rng(0)
        >>> series = rng.normal(100, 10, 48)
        >>> params, score = optimize_arima(series, n_calls=5)
        >>> set(params.keys()) == {"p", "d", "q", "P", "D", "Q", "m"}
        True
    """
    # Resolve automatic CPU count
    if n_jobs is None or n_jobs < 1:
        n_jobs = os.cpu_count() or 1

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: suggest (p,d,q,P,D,Q) and return combined cost."""
        p = trial.suggest_int("p", p_range[0], p_range[1])
        d = trial.suggest_int("d", d_range[0], d_range[1])
        q = trial.suggest_int("q", q_range[0], q_range[1])
        P = trial.suggest_int("P", P_range[0], P_range[1])
        D = trial.suggest_int("D", D_range[0], D_range[1])
        Q = trial.suggest_int("Q", Q_range[0], Q_range[1])

        # Avoid over-parameterized models
        if (p + q) > 4:
            return OPTIMIZER_PENALTY

        if (P + Q) > 3:
            return OPTIMIZER_PENALTY

        _, _, _, score, _ = _evaluate_arima(series, p, d, q, P, D, Q, m)
        return score

    # ── Sampler ────────────────────────────────────────────────────────────────
    # multivariate=True: TPE captures (p, q, P, Q) correlations for better proposals.
    # seed=42: ensures reproducible parameter suggestions across runs.
    sampler = TPESampler(seed=42, multivariate=True)

    # In-memory storage maximises throughput (no disk I/O overhead per trial)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # ── Warm Start ─────────────────────────────────────────────────────────────
    # Enqueue known-good starting points before TPE's probabilistic model kicks
    # in.  This prevents wasting early trials on trivial (0,0,0)(0,0,0) combos.
    study.enqueue_trial({"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1})  # seasonal baseline
    study.enqueue_trial({"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0})  # simple AR(1)

    # ── Optimisation ───────────────────────────────────────────────────────────
    try:
        study.optimize(objective, n_trials=n_calls, n_jobs=n_jobs)
    except Exception as exc:
        logger.error("Optimisation study failed: %s", exc)
        return {"p": 1, "d": 1, "q": 1, "P": 1, "D": 1, "Q": 1, "m": m}, OPTIMIZER_PENALTY

    raw = study.best_params  # {"p": int, "d": int, "q": int, "P": int, "D": int, "Q": int}
    best_params = {
        "p": raw["p"],
        "d": raw["d"],
        "q": raw["q"],
        "P": raw["P"],
        "D": raw["D"],
        "Q": raw["Q"],
        "m": m,
    }
    best_value = study.best_value

    logger.info(
        "Best SARIMA params: %s  |  Score: %.4f",
        best_params,
        best_value,
    )

    return best_params, best_value
