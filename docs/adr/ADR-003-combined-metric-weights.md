# ADR-003 — Combined objective `0.7·sMAPE + 0.3·RMSLE`

- **Status:** Accepted
- **Date:** 2026-03-17
- **Deciders:** core maintainers
- **Related code:** `src/boa_forecaster/metrics.py`, `src/boa_forecaster/config.py`

## Context

The TPE optimiser minimises a single scalar objective.  Demand-forecasting
data at `boa-forecaster`'s scale has two awkward properties that make any
single-metric choice suboptimal:

- **Intermittent demand.**  Many SKUs have strings of zero months followed
  by bursts.  Classical MAPE is unbounded and asymmetric around zero, which
  makes its gradient unusable as a TPE objective.
- **Multi-magnitude spans.**  A national-level series sells thousands of
  units per month while a regional SKU sells single digits.  RMSE and MAE
  are scale-dependent, so aggregating them across a heterogeneous portfolio
  lets high-volume series dominate the objective.

We needed an objective that is:

1. Bounded and symmetric (for stable TPE sampling).
2. Meaningful across multiple orders of magnitude.
3. Comparable across SKUs so cross-validation scores are averageable.
4. Overridable by users with different business contexts (revenue, price).

## Decision

Default objective: **`combined = 0.7 · sMAPE + 0.3 · RMSLE`**.

The weights are exposed in three layers of increasing flexibility:

```python
# 1. Fixed default — used when nothing is specified.
from boa_forecaster.metrics import combined_metric
combined_metric(y_true, y_pred)  # 0.7·sMAPE + 0.3·RMSLE

# 2. Parameterised call — override weights inline.
combined_metric(y_true, y_pred, w_smape=0.5, w_rmsle=0.5)

# 3. Arbitrary mix via the factory — any subset of METRIC_REGISTRY.
from boa_forecaster.metrics import build_combined_metric
fn = build_combined_metric([
    {"metric": "mae",  "weight": 0.6},
    {"metric": "rmse", "weight": 0.4},
])
```

In `config.yaml`, the `metrics` section composes the objective
declaratively — no code changes required for a new metric mix.

### Why these numbers?

- **sMAPE dominates (70 %).**  sMAPE is the most interpretable percentage
  error for a business audience and is bounded in `[0, 200]`, which means
  the aggregated cross-validation score across 10⁴ SKUs cannot be
  dominated by one pathological series.
- **RMSLE as the secondary (30 %).**  RMSLE penalises under-prediction
  more harshly than over-prediction (important for inventory planning
  — stockouts cost more than overage) and remains well-behaved on
  multi-magnitude data thanks to the log transform.  Its absolute
  magnitude is smaller than sMAPE's, which is why it receives a lower
  weight despite its operational importance.
- **Ratio chosen empirically.**  During v1.0 we swept weights on the
  original BOA portfolio (synthetic + real series) and 0.7/0.3 gave
  the best walk-forward RMSE while preserving sensitivity to
  low-volume SKUs.  Ratios like 0.5/0.5 were dominated by low-volume
  noise; 0.9/0.1 underweighted inventory risk.

## Consequences

### Positive

- **Stable across SKUs.**  Percentage-based + log-based composition means
  the objective does not explode when a series swings from 2 to 2 000
  units month-over-month.
- **Interpretable default.**  "Combined metric" in a dashboard maps
  cleanly to "mostly sMAPE, with a log-scale stability bonus" — easier
  to defend to stakeholders than a raw MSE number.
- **Configurable at every layer.**  Three escape hatches
  (`w_smape`/`w_rmsle`, `build_combined_metric`, `config.yaml`) cover
  every use case we have seen without breaking the default contract.

### Negative

- **Not universally optimal.**  Revenue series with multiplicative drift
  may prefer `log-MAE`; count data may prefer a Poisson deviance.  The
  factory lets users opt out, but the *default* choice still biases new
  users towards demand forecasting.
- **Weight choice is empirical, not theoretical.**  A future recalibration
  on a wider corpus may shift the ratio.  Because the weights are a
  default — not a hard-coded constant — future shifts are a
  non-breaking config change, not a library release.

### Neutral

- Both components are non-negative, symmetric in their arguments, and
  zero at perfect prediction.  Property tests in
  `tests/unit/test_metrics_property.py` pin these invariants so any
  future metric added to `METRIC_REGISTRY` must preserve them.

## Alternatives considered

1. **Pure sMAPE.**  Rejected: no inventory-risk signal, and a perfect
   (trivially-zero) baseline wins against any non-trivial model on
   zero-demand series.
2. **Pure MAE / RMSE.**  Rejected: scale-dependent; dominated by
   high-volume SKUs when aggregated.
3. **MAPE.**  Rejected: unbounded for small `y_true`, asymmetric,
   numerically unstable on intermittent demand.
4. **Mean-quantile-loss / pinball loss.**  Rejected for v1 — requires
   quantile regression from every spec, which SARIMAX and scikit-learn
   trees do not expose uniformly.  Candidate for a future ADR if we
   add probabilistic forecasting support.
5. **Learned meta-objective (weight as hyper-parameter).**  Rejected:
   makes cross-SKU comparison ambiguous and explodes the search space
   without a clear gain.

## References

- `src/boa_forecaster/metrics.py` — `smape`, `rmsle`, `combined_metric`, `build_combined_metric`
- `src/boa_forecaster/config.py` — `DEFAULT_METRIC_COMPONENTS`
- `tests/unit/test_metrics_property.py` — invariant coverage
- v1.3.0 release — made metric composition configurable via `config.yaml`
