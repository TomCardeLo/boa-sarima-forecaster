# Extending `boa-forecaster` with a New Model

`boa-forecaster` is built around a single pluggable interface: any class
that satisfies the `ModelSpec` Protocol can be optimised by the generic
TPE engine and validated by the walk-forward CV harness — without
touching the optimiser internals.

This guide walks through implementing a brand-new spec from scratch.
We use **Prophet** ([facebook/prophet](https://github.com/facebook/prophet))
as a worked example because it sits in the same family as SARIMA
(consumes a raw time series, no tabular feature engineering) but has a
completely different parameter space, so it exercises every part of the
contract.

> **TL;DR.** Decide whether your model needs `FeatureEngineer`.  If
> *yes* and it is a tree-based regressor, inherit from `BaseMLSpec` and
> override two methods.  If *no* (or it is not a tree), implement the
> `ModelSpec` Protocol directly — six members, ~80 lines.

---

## 1. Pick your base

| Model shape | Inherit from | Override |
|-------------|--------------|----------|
| Statistical / state-space (SARIMA, ETS, Prophet, Theta…) | nothing — implement Protocol directly | all six members |
| Tree-based ML (RF, XGB, LGBM, CatBoost…) | `BaseMLSpec` | `_fit_final`, `search_space`, `warm_starts`, optionally `_check_availability` and `_fit_fold` |
| Neural / probabilistic (DeepAR, NHITS…) | nothing — implement Protocol directly | all six members |

Prophet is in the first row, so we implement the Protocol directly.

## 2. The `ModelSpec` Protocol — six members

From `src/boa_forecaster/models/base.py`:

```python
@runtime_checkable
class ModelSpec(Protocol):
    name: str
    needs_features: bool

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]: ...
    @property
    def warm_starts(self) -> list[dict]: ...

    def suggest_params(self, trial: optuna.Trial) -> dict: ...
    def evaluate(self, series, params, metric_fn, feature_config=None) -> float: ...
    def build_forecaster(self, params, feature_config=None): ...
```

Two contracts that must hold for every spec:

- **`evaluate` never raises.**  Per-trial failures (numerical blow-ups,
  fit errors, anything) must return `OPTIMIZER_PENALTY` so the TPE
  study can keep sampling.  See [ADR-002](adr/ADR-002-optimizer-soft-failure.md).
- **`build_forecaster` returns a closure** of shape
  `(train: pd.Series) -> pd.Series` whose output index covers the next
  `forecast_horizon` periods after `train.index[-1]`.  The walk-forward
  harness relies on this exact shape.

---

## 3. Worked example — `ProphetSpec`

```python
# src/boa_forecaster/models/prophet.py
"""Prophet plugin for the boa-forecaster TPE engine."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import optuna
import pandas as pd

try:
    from prophet import Prophet  # type: ignore[import-not-found]
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    SearchSpaceParam,
    suggest_from_space,
)

logger = logging.getLogger(__name__)


class ProphetSpec:
    """ModelSpec implementation for Facebook's Prophet."""

    name: str = "prophet"
    needs_features: bool = False  # Prophet ingests the raw series

    def __init__(self, forecast_horizon: int = 12) -> None:
        if not HAS_PROPHET:
            raise ImportError(
                "prophet is required for ProphetSpec. "
                "Install with: pip install prophet>=1.1"
            )
        self.forecast_horizon = forecast_horizon

    # ── Search space ─────────────────────────────────────────────────────────

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        return {
            "changepoint_prior_scale": FloatParam(0.001, 0.5, log=True),
            "seasonality_prior_scale": FloatParam(0.01, 10.0, log=True),
            "seasonality_mode": CategoricalParam(["additive", "multiplicative"]),
        }

    @property
    def warm_starts(self) -> list[dict]:
        # Prophet defaults — known-good baseline before TPE explores.
        return [
            {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "seasonality_mode": "additive",
            },
        ]

    # ── Trial sampling ───────────────────────────────────────────────────────

    def suggest_params(self, trial: optuna.Trial) -> dict:
        return suggest_from_space(trial, self.search_space)

    # ── Evaluation (never raises) ────────────────────────────────────────────

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn,
        feature_config=None,  # ignored — needs_features is False
    ) -> float:
        try:
            df = pd.DataFrame({"ds": series.index, "y": series.to_numpy()})
            model = Prophet(**params)
            model.fit(df)
            in_sample = model.predict(df[["ds"]])["yhat"].to_numpy()
            return float(metric_fn(series.to_numpy(), in_sample))
        except Exception as exc:  # noqa: BLE001 — soft-failure contract
            logger.debug("ProphetSpec.evaluate failed: %s", exc)
            return OPTIMIZER_PENALTY

    # ── Forecaster closure ───────────────────────────────────────────────────

    def build_forecaster(
        self, params: dict, feature_config=None
    ) -> Callable[[pd.Series], pd.Series]:
        horizon = self.forecast_horizon

        def forecaster(train: pd.Series) -> pd.Series:
            df = pd.DataFrame({"ds": train.index, "y": train.to_numpy()})
            model = Prophet(**params)
            model.fit(df)
            freq = pd.infer_freq(train.index) or "MS"
            future = model.make_future_dataframe(periods=horizon, freq=freq)
            yhat = model.predict(future)["yhat"].to_numpy()[-horizon:]
            future_index = pd.date_range(
                start=train.index[-1], periods=horizon + 1, freq=freq
            )[1:]
            return pd.Series(yhat, index=future_index, name=train.name)

        return forecaster
```

That's the entire spec.  Use it the same way as any built-in model:

```python
from boa_forecaster.optimizer import optimize_model
from boa_forecaster.models.prophet import ProphetSpec

result = optimize_model(series, ProphetSpec(forecast_horizon=12), n_calls=50)
print(result.best_params, result.best_score, result.is_fallback)
```

---

## 4. Constraints — the `OPTIMIZER_PENALTY` pattern

When some hyper-parameter combinations are *invalid* but cheap to
detect, return `OPTIMIZER_PENALTY` *before* fitting so TPE learns to
avoid them.  `SARIMASpec` does this for non-stationary order sums:

```python
# models/sarima.py
if (p + q) > self.MAX_NON_SEASONAL_ORDER or (P + Q) > self.MAX_SEASONAL_ORDER:
    return OPTIMIZER_PENALTY
```

For Prophet you might block `multiplicative` mode on series that hit
zero (Prophet fails on non-positive `y` in that mode):

```python
if params["seasonality_mode"] == "multiplicative" and (series <= 0).any():
    return OPTIMIZER_PENALTY
```

---

## 5. ML branch — inherit `BaseMLSpec` instead

For tree-based regressors you do **not** re-implement `evaluate` or
`build_forecaster` — `BaseMLSpec` provides the CV loop, recursive
forecaster, and feature-engineering plumbing.  Override only:

```python
# src/boa_forecaster/models/catboost.py
from catboost import CatBoostRegressor

from boa_forecaster.models._ml_base import BaseMLSpec
from boa_forecaster.models.base import IntParam, FloatParam, SearchSpaceParam


class CatBoostSpec(BaseMLSpec):
    name: str = "catboost"

    def _check_availability(self) -> None:
        # raise ImportError if catboost is missing — see RandomForestSpec
        ...

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        return {
            "iterations": IntParam(100, 1000, log=True),
            "depth": IntParam(3, 10),
            "learning_rate": FloatParam(0.01, 0.3, log=True),
        }

    @property
    def warm_starts(self) -> list[dict]:
        return [{"iterations": 500, "depth": 6, "learning_rate": 0.1}]

    def _fit_final(self, X, y, params):
        model = CatBoostRegressor(**params, verbose=0, random_seed=42)
        model.fit(X, y)
        return model

    # Optional: override _fit_fold(X, y, params) to enable early stopping
    # against an inner-validation split (see XGBoostSpec / LightGBMSpec).
```

`BaseMLSpec` handles:

- Expanding-window CV with `N_CV_FOLDS = 3`
- A fresh `FeatureEngineer` per fold (no leakage — see
  [features.py](../src/boa_forecaster/features.py))
- Recursive multi-step forecasting via `recursive_forecast`
- Soft-failure inside each fold (`OPTIMIZER_PENALTY` on exception)

---

## 6. Test checklist

A minimal test file follows the pattern from `tests/unit/test_base.py`
and `tests/unit/test_sarima_constraints.py`:

```python
# tests/unit/test_prophet_spec.py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("prophet")

from boa_forecaster.models.base import ModelSpec
from boa_forecaster.models.prophet import ProphetSpec


def _series(n: int = 60) -> pd.Series:
    idx = pd.date_range("2019-01-01", periods=n, freq="MS")
    t = np.arange(n)
    return pd.Series(50 + 0.3 * t + 5 * np.sin(2 * np.pi * t / 12), index=idx)


class TestProphetSpec:
    def test_satisfies_protocol(self):
        assert isinstance(ProphetSpec(), ModelSpec)

    def test_search_space_keys(self):
        assert set(ProphetSpec().search_space) == {
            "changepoint_prior_scale",
            "seasonality_prior_scale",
            "seasonality_mode",
        }

    def test_warm_starts_are_valid_param_dicts(self):
        spec = ProphetSpec()
        for ws in spec.warm_starts:
            assert set(ws).issubset(spec.search_space.keys())

    def test_evaluate_never_raises_on_pathological_input(self):
        spec = ProphetSpec()
        bad = pd.Series([np.nan] * 60, index=pd.date_range("2019-01-01", periods=60, freq="MS"))
        params = spec.warm_starts[0]
        out = spec.evaluate(bad, params, metric_fn=lambda a, b: 1.0)
        assert isinstance(out, float)  # OPTIMIZER_PENALTY, not an exception

    def test_build_forecaster_returns_horizon_steps(self):
        spec = ProphetSpec(forecast_horizon=6)
        forecaster = spec.build_forecaster(spec.warm_starts[0])
        out = forecaster(_series(60))
        assert isinstance(out, pd.Series)
        assert len(out) == 6
        assert out.index[0] > _series(60).index[-1]
```

Run with `pytest tests/unit/test_prophet_spec.py -v`.  Add the marker
`@pytest.mark.requires_prophet` if you also register it in
`pyproject.toml`.

---

## 7. Optional — register in `MODEL_REGISTRY`

The registry powers config-driven model selection (`config.yaml →
models.active`).  Registration is one line:

```python
# src/boa_forecaster/models/__init__.py
from boa_forecaster.models.prophet import ProphetSpec

MODEL_REGISTRY["prophet"] = ProphetSpec
```

After this, `config.yaml` users can select your model declaratively:

```yaml
models:
  active: prophet
  prophet:
    forecast_horizon: 12
    search_space: {}      # falls back to spec defaults
    warm_starts: []
```

Skip this if your spec lives in an external package — `optimize_model`
accepts any `ModelSpec` instance; the registry is purely a convenience
for config-driven workflows.

---

## 8. Common pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `OptimizationResult.is_fallback is True` for every series | `evaluate` raised → `study.optimize` crashed | Wrap `evaluate` body in `try/except` and return `OPTIMIZER_PENALTY` (see ADR-002) |
| Walk-forward CV reports zero folds | `build_forecaster` closure returns wrong index | Ensure the returned `pd.Series` has `forecast_horizon` rows starting *after* `train.index[-1]` |
| TPE never explores beyond warm starts | Search-space bounds collapse to a single point | Check `IntParam(low, high)` is `low < high` and that `log=True` is paired with `low > 0` |
| Mypy flags Protocol mismatch | `evaluate` signature differs from Protocol | Match the exact parameter order: `(series, params, metric_fn, feature_config=None)` |
| `ImportError` only at runtime | `try/except ImportError` not paired with a `_check_availability` raise | Mirror the `HAS_SKLEARN` pattern in `RandomForestSpec` |

---

## 9. Where to read next

- [`models/base.py`](../src/boa_forecaster/models/base.py) — Protocol definition + `OptimizationResult`
- [`models/_ml_base.py`](../src/boa_forecaster/models/_ml_base.py) — shared CV loop for tree models
- [`models/sarima.py`](../src/boa_forecaster/models/sarima.py) — full statistical-model reference
- [`models/random_forest.py`](../src/boa_forecaster/models/random_forest.py) — minimal `BaseMLSpec` subclass
- [ADR-001](adr/ADR-001-modelspec-protocol.md) — why `Protocol` not `ABC`
- [ADR-002](adr/ADR-002-optimizer-soft-failure.md) — the soft-failure contract
- [ADR-003](adr/ADR-003-combined-metric-weights.md) — what `metric_fn` minimises
