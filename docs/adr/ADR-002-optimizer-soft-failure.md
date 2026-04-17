# ADR-002 — Soft-failure in `optimize_model` (`is_fallback`)

- **Status:** Accepted
- **Date:** 2026-04-12
- **Deciders:** core maintainers
- **Related code:** `src/boa_forecaster/optimizer.py`, `src/boa_forecaster/models/base.py`

## Context

`optimize_model` runs an Optuna TPE study over a `ModelSpec`.  Two
categories of failure arise in practice:

1. **Per-trial failure.**  A single `(p, d, q)` combination fails to
   converge in `SARIMAX.fit`, or a single ML hyper-parameter set yields
   numerical overflow.  These are expected and handled inside each
   spec's `evaluate()` method, which returns `OPTIMIZER_PENALTY` so the
   study can continue sampling elsewhere in the space.

2. **Study-level failure.**  The *entire* study raises — e.g. an
   out-of-memory error on a pathological series, a bug in a user's
   custom `metric_fn`, an Optuna internal error, or a
   `KeyboardInterrupt` during a long batch.  Before v1.6.0, these were
   either silently swallowed (returning a fallback indistinguishable
   from a real optimum) or allowed to propagate (aborting the entire
   batch over an otherwise-healthy portfolio of SKUs).

`boa-forecaster` is frequently run over 10⁴+ SKU/country series in a
single batch.  A hard-fail on one series would waste hours of compute;
a silent fallback would mislead reviewers into trusting a result that
never came from real search.

## Decision

`optimize_model` catches any exception escaping `study.optimize`, logs
it at **`WARNING`** with the full traceback, and returns an
`OptimizationResult` with a new **`is_fallback: bool`** flag set to
`True`.  On the success path `is_fallback` is explicitly `False`.

```python
# optimizer.py
try:
    study.optimize(objective, n_trials=n_calls, n_jobs=n_jobs)
except Exception as exc:
    logger.warning(
        "Optimisation study for %s failed: %s — returning fallback result.",
        model_spec.name, exc, exc_info=True,
    )
    fallback = model_spec.warm_starts[0] if model_spec.warm_starts else {}
    return OptimizationResult(
        best_params=fallback,
        best_score=OPTIMIZER_PENALTY,
        n_trials=0,
        model_name=model_spec.name,
        is_fallback=True,
    )

return OptimizationResult(
    best_params=study.best_params,
    best_score=study.best_value,
    n_trials=len(study.trials),
    model_name=model_spec.name,
    is_fallback=False,
)
```

The first warm-start (or `{}` when none are provided) is chosen as the
fallback `best_params` because it is guaranteed to be a syntactically
valid param dict for that spec — downstream code that calls
`spec.build_forecaster(result.best_params)` will not crash on a
default-initialised result.

A spec's `evaluate()` contract remains unchanged: **never raise, return
`OPTIMIZER_PENALTY` on per-trial failure.**  This ADR governs only the
study-level wrapper.

## Consequences

### Positive

- **Batch robustness.**  A single bad series no longer aborts a
  multi-hour batch.  The flag propagates to `validate_by_group`, which
  can emit a row tagged `is_fallback=True` instead of skipping the SKU
  entirely.
- **Observability.**  `WARNING` (not `info` or `debug`) ensures the
  crash appears in default log configurations; `exc_info=True` preserves
  the traceback.  Silent fallbacks are no longer possible.
- **Typed downstream branching.**  Callers who care about the
  distinction write `if result.is_fallback: …` instead of comparing
  `best_score` against `OPTIMIZER_PENALTY` (which is brittle if a real
  trial also returns the penalty).
- **Backward compatible.**  `is_fallback` defaults to `False` on the
  dataclass, and joblib-serialised pre-v1.6.0 `OptimizationResult`
  instances load without error.

### Negative

- **Silent by default for non-checking callers.**  A caller that ignores
  `is_fallback` and pushes `best_params` straight into production will
  deploy a warm-start as if it were the optimum.  Mitigated by the
  `WARNING` log and by the public-API documentation, but cannot be
  fully prevented without breaking the "always return something"
  ergonomic we want.
- **Doesn't cover `KeyboardInterrupt`.**  We catch `Exception`, not
  `BaseException`.  A Ctrl-C still aborts cleanly — deliberate.

### Neutral

- `OPTIMIZER_PENALTY` is a real number (not NaN), so a result containing
  the penalty still serialises cleanly and compares meaningfully with
  `<` and `>`.  Callers that sort results should sort ascending on
  `best_score` and use `is_fallback` as a secondary disqualifier.

## Alternatives considered

1. **Re-raise the exception.**  Rejected: hostile to batch pipelines
   that handle hundreds of series per minute.
2. **Silent swallow (v2.0 initial behaviour).**  Rejected: indistinguishable
   from success.  Caused a real incident where a stakeholder-facing
   report flagged "best params = defaults" as a meaningful recommendation.
3. **Return `None`.**  Rejected: forces every caller site to branch on
   `None` and complicates the `OptimizationResult` return type.  Also
   loses the `model_name` and any useful context.
4. **Raise a specific `OptimizerError` subclass.**  Rejected: shifts the
   same ergonomic burden onto every caller; also harder to compose in
   `joblib.Parallel` map-reduce patterns where results are collected.

## References

- `src/boa_forecaster/optimizer.py`
- `src/boa_forecaster/models/base.py` (dataclass definition)
- `tests/unit/test_optimizer_generic.py` — soft-failure coverage
- Plan item C3 (v1.6.0) — introduction of `is_fallback`
