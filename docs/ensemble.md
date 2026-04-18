# Ensembles

`EnsembleSpec` is a post-optimisation composition that weight-averages the
forecasts of multiple already-optimised `ModelSpec` members.

## Usage

```python
from boa_forecaster import SARIMASpec, RandomForestSpec, build_ensemble

spec, params = build_ensemble(
    series,
    specs=[SARIMASpec(), RandomForestSpec()],
    weighting="inverse_cv_loss",  # default
    n_calls=30,
)
forecaster = spec.build_forecaster(params)
forecast = forecaster(series)
```

`build_ensemble` runs Bayesian optimisation for each member, records the
`best_score`, constructs the `EnsembleSpec`, and returns both the spec and
the per-member params dict that `build_forecaster` expects.

## Weighting strategies

| Strategy            | Semantics                                                          |
|---------------------|--------------------------------------------------------------------|
| `"equal"`           | 1/N per member.                                                    |
| `"inverse_cv_loss"` | Weight ∝ 1 / `best_score`; lower-loss members get more weight.     |
| `list[float]`       | Explicit weights, normalised to sum to 1.                          |

`"inverse_cv_loss"` falls back to `"equal"` when any recorded score is
non-positive or non-finite (logged at `WARNING`).

## When **not** to ensemble

- **Identical families.** Two SARIMA specs differing only by a hyperparameter
  give you the same bias-variance profile — use `optimize_model` with a wider
  search space instead.
- **One dominant model.** If one member's walk-forward score is an order of
  magnitude better than the others, `inverse_cv_loss` will assign near-100%
  weight to it anyway; the extra members just add fit/forecast time.
- **Short series (< 2·horizon).** Ensembles need enough data for every member
  to estimate non-degenerate parameters; on very short series, the ensemble
  often underperforms the best single member.
- **High-stakes point forecasts where variance matters.** Averaging reduces
  variance but also masks which model is actually responsible for a given
  prediction — bad when you need to explain forecasts to stakeholders.

## ``evaluate`` raises ``NotImplementedError``

An ensemble has no TPE-tunable hyperparameters: its parameters are fully
determined by the members' optimisation results.  Calling
`EnsembleSpec.evaluate` raises `NotImplementedError` by design.  Use
`build_ensemble` — plugging an `EnsembleSpec` directly into `optimize_model`
is not supported.
