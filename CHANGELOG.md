# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.4.0] — 2026-04-22

Track H release — "Probabilistic, Regulatory & Deep-Learning Horizons".
Bundles three new model families (Prophet, quantile gradient boosters,
LSTM), finishes the Pydantic config rollout, and lands the
2026-04-20 feedback from the CAR PM2.5 hourly pipeline (regulatory
metric presets, SARIMA hourly tuning, ensemble safety, environmental
preprocessing defaults, post-training seasonal bias correction).
No breaking API changes — additive and behaviour-tightening only.
The `sarima_bayes` compatibility shim and v1.x deprecation warnings
are unchanged from v2.3.0.

### Added — Model families (H1, H2, H3)

- **`ProphetSpec`** (`models/prophet.py`, H1) — Meta's Prophet as a
  first-class `ModelSpec` for interpretable trend + seasonality +
  holidays decomposition.  Search space covers
  `changepoint_prior_scale` (log-uniform 0.001–0.5),
  `seasonality_prior_scale` and `holidays_prior_scale` (0.01–10), and
  `seasonality_mode` (additive / multiplicative).  Behind the
  `prophet` optional extra; degrades to a `_MissingExtra` sentinel
  when not installed.  CI installs it on Python 3.11 only (wheels
  fragile on 3.12+).
- **`QuantileMLSpec`** (`models/quantile.py`, H2) — probabilistic
  forecasts via LightGBM `objective=quantile` or XGBoost
  `reg:quantileerror`.  Fits **one booster per quantile**; returns a
  `QuantileForecast(median, lower, upper)` dataclass on top of the
  shared `BaseMLSpec` feature pipeline.  Sets
  `uses_early_stopping = True` so the H9a ensemble warning fires when
  mixed with full-fold members.  `OptimizationResult` gains an
  optional `quantile_forecasts` field — additive, no break for
  point-forecast consumers.
- **New `metrics_probabilistic.py`** (H2) — `pinball_loss(y_true,
  y_pred, quantile)` and `interval_coverage(y_true, lower, upper)`.
  `pinball_loss` is registered in `METRIC_REGISTRY` so it can be
  selected as the Optuna objective from YAML.  Both are re-exported
  from the package root.
- **`LSTMSpec`** (`models/lstm.py`, H3) — PyTorch LSTM baseline
  exposing `hidden_size`, `num_layers`, `dropout`, `learning_rate`,
  `n_epochs`, `batch_size`, `window_size`.  Train-only normalisation
  (80% slice) avoids val-loss leakage; patience-5 early stopping with
  best-state restore.  CPU by default; `device="auto"` opts into
  CUDA.  Behind the `deep` optional extra (deliberately **not** in
  `[all]` — torch is heavy).  New `test-deep-extras` CI job, Python
  3.11 only, step-level `continue-on-error` tolerated.

### Added — Regulatory metrics & presets (H7-core, H7-presets)

- **`hit_rate_weighted(y_true, y_pred, edges, weights=None)`**
  (`metrics.py`, H7-core) — bucket-accuracy with per-bucket weights so
  misses in high-stakes buckets count more.  `weights=None` is
  uniform and reduces to `hit_rate`.  Registered in
  `METRIC_REGISTRY`.
- **`f1_by_bucket(y_true, y_pred, edges, labels=None) -> dict[str,
  float]`** (`metrics.py`, H7-core) — per-category F1; uses sklearn
  if installed, vectorised numpy fallback otherwise.  No new required
  dependency.  Registered in `METRIC_REGISTRY`.
- **New `presets/` package + `presets/air_quality.py`** (H7-presets)
  — first preset pack, scoped to air-quality.  Ships
  `ICA_EDGES_PM25_CO2017`, `ICA_EDGES_PM25_USAQI`, `ICA_LABELS_6`,
  `ICA_WEIGHTS_HEALTH`, plus `hit_rate_ica(...)` and
  `hit_rate_ica_weighted(...)` thin wrappers around the core
  bucketed metrics.  Imported explicitly from
  `boa_forecaster.presets.air_quality` — **not** re-exported from
  the top-level package, keeping core lean.

### Added — Hourly SARIMA & high-volatility WMA (H8, H9b)

- **`SARIMASpec.for_frequency(freq)` classmethod** (`models/sarima.py`,
  H8) — frequency-aware `seasonal_period` defaults: `MS`/`M` → 12,
  `W` → 52, `D` → 7, `h`/`H` → tuneable
  `CategoricalParam([24, 168])` (daily vs. weekly seasonality).
  Mirrors `FeatureConfig.for_frequency` (v2.3, G1).  `optimize_model`
  honours the tuneable form when present.
- **`WMA_THRESHOLD_HIGH_VOLATILITY = 3.5` named constant**
  (`standardization.py`, H9b) — opt-in alternative to the library
  default `2.5σ` for peaky series (PM2.5 pollution episodes,
  electricity demand spikes, retail stockout-recovery, financial
  fat-tail returns, IoT sensor bursts).  The library-wide default
  stays `2.5` — no behaviour change for existing callers.

### Added — Post-training bias correction (H5)

- **New `postprocess.py`** — `compute_seasonal_bias(y_true, y_pred,
  periods=12, start_period=1, clip_range=(0.5, 2.0))` and
  `apply_seasonal_bias(forecast, bias, start_period=1)` implement the
  per-calendar-period multiplicative bias correction validated in the
  CAR PM2.5 production pipeline.  Median-of-residual-ratios
  (robust to outliers), clipped to `[0.5, 2.0]` to avoid blow-ups.
  Generalisable beyond monthly via the `periods` parameter.  When
  inputs carry a `DatetimeIndex` and `periods=12`, alignment is
  always by calendar month (`bias[0]` = January) regardless of
  `start_period`.
- **`optimize_model(..., apply_bias_correction=False)`** — opt-in
  kwarg.  When `True`, bias is computed on the final CV fold's
  residuals and attached to `OptimizationResult.bias_correction`.
  Series shorter than 36 observations skip computation with a
  `WARNING` log naming the model and required length.
- **CLI `--bias-correction` flag** on `boa-forecaster run` — applies
  the stored factors during forecast and echoes the active factor
  array.

### Added — Pydantic finishing touches (H4)

- **`BoaConfig.from_dict(...)` classmethod** — programmatic
  construction for tests and library callers without round-tripping
  through YAML.
- **`Literal` validators on sub-models** —
  `StandardizationConfig.method` ∈ `{"sigma", "iqr"}`,
  `DataConfig.freq` ∈ `{"MS", "M", "W", "D", "h", "H"}` (aligns with
  G1 from v2.3).  `StandardizationConfig.threshold ∈ (0, 10]`,
  `ForecastConfig.n_periods ≥ 1`.

### Changed

- **CLI gains `--strict` flag** on `run`, `compare`, `validate` (H4)
  — flips Pydantic `extra="allow"` → `extra="forbid"` at load time.
  Default stays `allow` for v2.x back-compat.  Use `--strict` in CI
  to catch typo'd or stale config keys.
- **SARIMA `seasonal_period` is tuneable on hourly data** (H8) —
  previously a fixed constant of 12 across all frequencies.  When the
  spec is built via `SARIMASpec.for_frequency("h")` the search space
  now contains `seasonal_period` as a `CategoricalParam([24, 168])`.

### Warnings

- **`EnsembleSpec` `inverse_cv_loss` warns when mixing early-stopping
  members** (H9a) — gradient boosters with early stopping evaluate
  on an inner validation split, while SARIMA / Prophet / RF use the
  full fold.  Their CV losses are not directly comparable, so
  `inverse_cv_loss` weighting is mathematically biased toward the
  early-stopping members.  The ensemble now emits a `UserWarning`
  flagging the offending member class names and recommending
  `strategy="equal"` or explicit weights.  New `uses_early_stopping`
  attribute on `ModelSpec` Protocol; defaults to `False`, set to
  `True` on `XGBoostSpec`, `LightGBMSpec`, and `QuantileMLSpec`.

### Fixed

- **H5 code-review follow-ups** — `apply_seasonal_bias` round-trip
  now correct for `start_period != 1` (previously the DatetimeIndex
  branch and the position branch disagreed).
  `_compute_bias_from_last_fold` reads `test_size` from
  `model_spec.forecast_horizon` instead of hardcoding 12, so
  bias-correction is wired correctly for non-12 horizons.  Short
  series now emit a `WARNING` instead of silently no-op'ing.

### Contributors

Tracks H1, H2, H3, H4, H7-core, H7-presets, H8, H9a, H9b executed in
parallel by Sonnet implementer subagents (Wave A) under Opus
orchestration; Track H5 ran serially in Phase 2 after the
`OptimizationResult` shape stabilised post-H2.  Code reviews by Opus
`code-reviewer` subagent; release gate run on main thread.

Special thanks to **Daniel Méndez** and the **CAR / Cundinamarca
PM2.5 hourly pipeline team** (Bogotá + Cundinamarca, 34 monitoring
stations, 2016–2026) for the 2026-04-20 feedback (`tasks/feedback_aire.md`)
that drove H5 (`sesgo_mensual_para_ajuste.csv` pattern), H7
(weighted bucket metrics + ICA presets), H8 (hourly SARIMA), and
H9 (ensemble safety + WMA high-volatility constant).

[2.4.0]: https://github.com/TomCardeLo/boa-forecaster/compare/v2.3.0...v2.4.0

---

## [2.3.0] — 2026-04-20

Correctness & ecosystem release bundling Tracks E/F/G of the post-v2.2
plan.  Four silent-correctness bugs fixed, quality-hardening touch-ups
on the validation/metric/preprocessor surface, and small ecosystem
primitives surfaced by a real-world consumer.  No breaking API
changes — additive and behaviour-tightening only.

### Fixed — Correctness (Track E)

- **`EnsembleSpec.needs_features` now reflects its members** (E3) —
  previously a hardcoded class attribute `False`, which lied whenever
  the ensemble contained an ML member (e.g. `RandomForestSpec`) whose
  `needs_features` is `True`.  Any downstream code that gated feature
  engineering on the protocol attribute would silently skip it.  The
  attribute is now a `@property` returning
  `any(m.needs_features for m in self.members)`.
- **`BaseMLSpec` auto-injects `forecast_horizon` into default
  `FeatureConfig.lag_periods`** (E2) — setting
  `RandomForestSpec(forecast_horizon=24)` used to leave the lag list
  at `[1, 2, 3, 6, 12]`, so the model never saw `lag_24` (the single
  most informative feature for that horizon).  Explicitly-passed
  `feature_config` is **not** overridden.
- **Optuna `MedianPruner` now wired into `optimize_model`** (E4) —
  bad-hyperparameter trials were previously run to completion across
  all CV folds before being discarded.  `MedianPruner(n_startup_trials=5,
  n_warmup_steps=1)` is now attached to the study, and the `trial`
  handle is threaded through `evaluate` so each completed fold calls
  `trial.report(...)` and `trial.should_prune()`.  `optuna.TrialPruned`
  is re-raised past the fallback branch.  Expected TPE wall-time
  reduction: 20–40% with no quality loss.

### Changed — Performance (Track E)

- **`build_ensemble` parallelised** (E1) — member optimisation now
  dispatches through `joblib.Parallel(backend="loky")` with a new
  `n_jobs` kwarg (default `1` for backwards compatibility; `-1` uses
  all cores).  Same seed yields identical `member_scores` and
  `params_per_member` between `n_jobs=1` and `n_jobs=2`.  Expected
  wall-time reduction for a 4-member ensemble on 4 cores: ~75%.

### Added — Quality hardening (Track F)

- **`hit_rate(y_true, y_pred, edges)` metric** (F3) — bucket-accuracy
  metric for regulatory reporting (air-quality AQI bands, demand
  buckets, inventory tiers).  Registered in the metric registry and
  exported from the package root.  Combine into an objective via
  `build_combined_metric([{"metric": "hit_rate", "weight": 1.0,
  "edges": [...]}])`.
- **`flag_intermittent(df, group_cols, value_col, threshold=0.7)`
  helper** (F2) — returns a boolean mask of groups whose zero-ratio
  meets/exceeds `threshold`.  Complements `clean_zeros`, which only
  filters *flat-zero* groups; intermittent series like
  `[0,0,0,5,0,0,0,3,0,0]` pass `clean_zeros` but degrade SARIMA/GBM
  accuracy.  This is a flag, not a filter — the caller decides
  whether to route them to Croston/SBA, drop them, or keep them.
  NaN values are treated as zero.  Exported from the package root.

### Changed — Quality hardening (Track F)

- **`walk_forward_validation` accepts `n_folds >= 1`** (F1) —
  previously hard-floored at `n_folds >= 3`.  `n_folds < 3` carries a
  docstring warning about high variance but is now allowed.  Default
  remains `3`.  `validate_by_group` forwards the relaxed floor.
- **`walk_forward_validation(..., forecast_horizon=None)`** (F2) —
  new kwarg; if `test_size` is unspecified, the fold window defaults
  to `forecast_horizon` (falling back to `12` if both are `None`).
  Existing callers passing explicit `test_size=` are unaffected.
- **`combined_metric` now delegates to `build_combined_metric`** (F4) —
  the 0.7·sMAPE + 0.3·RMSLE composition was previously re-implemented
  manually outside the registry, so `register_metric` had no effect on
  it.  Both paths now share one code path, and `build_combined_metric`
  passes component-specific kwargs (e.g. `edges` for `hit_rate`)
  through via `inspect.signature` filtering.

### Added — Ecosystem (Track G)

- **`FeatureConfig.for_frequency(freq, **overrides)` classmethod** (G1) —
  returns a frequency-appropriate `FeatureConfig`:
  - `"MS"`/`"M"` — monthly defaults (current behaviour preserved).
  - `"W"` — weekly lags `[1, 2, 4, 13, 26, 52]`, weekly rollings.
  - `"D"` — daily lags `[1, 7, 14, 28]`.
  - `"h"`/`"H"` — hourly lags `[1, 24, 48, 168]`.

  Overrides are merged on top of the frequency defaults; unknown
  frequency aliases and override keys raise `ValueError`.  Default
  `FeatureConfig()` stays monthly to avoid breaking changes.
- **Docs: "Weighting caveats" section in `EnsembleSpec` docstring**
  (G3) — documents that `inverse_cv_loss` weighting is **biased** when
  members use different CV semantics (e.g. XGB with early-stopping
  over an inner validation split vs. RF trained on the full fold).
  Not a bug, but a usage trap flagged by a production consumer.

### Contributors

Tracks E/F/G executed in parallel by three Opus agents; plan,
review, and release orchestration in main thread.

---

## [2.2.0] — 2026-04-20

Feature release bundling Tracks A/B/C/D of the post-v2.1.0 plan:
release hygiene, coverage lift, performance, and extensibility
(CLI + config schema + ensembles).  No breaking API changes —
additive only.  The `sarima_bayes` compatibility shim still emits
`DeprecationWarning` and remains importable; internal duplicate
module files were removed in Track A but the public shim surface
is preserved until v3.0.

### Added — Extensibility (Track D)

- **Click CLI** (`boa_forecaster.cli`) — `run`, `compare`, and
  `validate` subcommands invokable via `python -m boa_forecaster`
  or the `boa-forecaster` console entry point.  Loads `config.yaml`
  through the new Pydantic schema and dispatches to the existing
  pipeline / `run_model_comparison` / `walk_forward_validation` APIs.
  See [`docs/cli.md`](docs/cli.md).
- **Pydantic v2 config schema** (`config_schema.py`) — strongly-typed
  validation of `config.yaml` at load time; catches typos, missing
  keys, and out-of-range values before the pipeline runs.  Replaces
  ad-hoc dict parsing throughout the CLI layer.
- **`EnsembleSpec`** (`models/ensemble.py`) — weighted or stacked
  ensemble over any subset of registered `ModelSpec`s; registered
  in `MODEL_REGISTRY` and exported from the package root.  See
  [`docs/ensemble.md`](docs/ensemble.md).
- **Public API** — `EnsembleSpec` and the CLI module re-exported
  from `boa_forecaster.__init__`.

### Added — Performance (Track C)

- **Deterministic feature cache** — `FeatureEngineer.build_features()`
  split into `build_deterministic()` (calendar + trend — window-
  independent, computed **once per series**) and
  `build_window_dependent()` (lag + rolling + expanding — per fold).
  `BaseMLSpec.build_forecaster()` accepts an optional
  `feature_cache: dict[str, pd.DataFrame]` keyed by `series.name`,
  reused across walk-forward CV folds.  ~30% speedup on a 60-month
  series with 10 folds.
- **Parallel walk-forward CV** — `walk_forward_validation(..., n_jobs: int = 1)`
  fans out fold evaluation via `joblib.Parallel(backend="loky")`.
  `n_jobs=1` (default) preserves the pre-existing sequential
  behaviour; `n_jobs=-1` uses all cores.
- **`pytest-benchmark` regression suite** (`tests/perf/`) —
  micro-benchmarks for `weighted_moving_stats_series`,
  `fill_blanks`, and `optimize_model` on SARIMA.  Weekly CI job
  compares against a committed baseline JSON.

### Added — Performance (Track A, #9)

- **`weighted_moving_stats_batch`** (`standardization.py`) —
  vectorised API for clipping multiple series in one call;
  re-exported from the package root.  Accompanying tests in
  `tests/unit/test_standardization.py`.

### Added — Tests (Track B)

- **`tests/unit/test_data_loader_errors.py`** — edge-case coverage
  for bad headers, missing columns, and malformed dates; raises
  `data_loader.py` to **100%**.
- **`tests/unit/test_benchmarks_v2.py`** — coverage for the v2
  `run_model_comparison` path; raises `benchmarks.py` to **95%**.
- **`tests/unit/test_validation.py` expansion** — additional walk-
  forward scenarios (including `n_jobs=2`); raises `validation.py`
  to **98%**.
- **`tests/unit/test_optional_deps.py`** (Track A) — asserts ML
  specs (`xgboost`, `lightgbm`) degrade cleanly when optional
  extras are not installed.
- **`tests/unit/test_cli.py`**, **`test_config_schema.py`**,
  **`test_ensemble.py`** (Track D) — full coverage for the new CLI,
  schema, and ensemble surfaces.

### Added — CI & tooling

- **Security scan step** (`.github/workflows/ci.yml`) — runs on
  push / PR (Track A).
- **Weekly performance regression job** — runs `pytest tests/perf/`
  with baseline comparison (Track C).

### Changed — Performance

- **`optimizer._validate_series` inf-check** — replaced
  `series.isin([np.inf, -np.inf]).any()` with
  `np.isinf(series.to_numpy()).any()`.  Short-circuits on the first
  `inf` instead of scanning the full series, ~10–20× faster on large
  float arrays.

### Fixed

- **mypy errors on Python 3.11 CI** — narrowing issues in
  `features.py`, `models/base.py`, and `models/sarima.py` flagged by
  the stricter 3.11 type-checker.

[2.2.0]: https://github.com/TomCardeLo/boa-forecaster/compare/v2.1.0...v2.2.0

---

## [2.1.0] — 2026-04-17

Feature release on the v2.x line.  Ships the full Phase A–E improvement
plan (perf, tests, code quality, CI, docs) on top of the v2.0.0 framework
foundation.  No breaking API changes since v2.0.0 — additions and
deprecations only.

> **Migration note.**  `import sarima_bayes` continues to work via a
> compatibility shim that re-exports the entire `boa_forecaster` API
> and emits a `DeprecationWarning` on import.  `pred_arima`,
> `forecast_arima`, and `optimize_arima` also keep working but warn —
> they will be removed in v3.0.

### Added — Multi-model framework

- **Pluggable `ModelSpec` `Protocol`** lets the same generic
  `optimize_model(series, spec)` engine drive SARIMA, Random Forest,
  XGBoost, LightGBM, and any user-defined spec.  See
  [ADR-001](docs/adr/ADR-001-modelspec-protocol.md).
  - `models/base.py` — `ModelSpec` Protocol, `OptimizationResult`,
    parameter descriptors (`IntParam`, `FloatParam`, `CategoricalParam`),
    `suggest_from_space` helper.
  - `models/sarima.py` — `SARIMASpec` (statsmodels SARIMAX).
  - `models/random_forest.py` — `RandomForestSpec` (scikit-learn).
  - `models/xgboost.py` — `XGBoostSpec` (optional `xgboost` extra).
  - `models/lightgbm.py` — `LightGBMSpec` (optional `lightgbm` extra).
  - `models/__init__.py` — `MODEL_REGISTRY` for config-driven selection.
- **`BaseMLSpec`** (`src/boa_forecaster/models/_ml_base.py`) — shared
  abstract base for tree-based ML specs.  Factors out the CV loop,
  recursive forecaster, and default `suggest_params`; subclasses
  override only `_fit_final`, `search_space`, and `warm_starts`.
  Removes ~329 lines of duplication across `RandomForestSpec`,
  `XGBoostSpec`, and `LightGBMSpec`.
- **Feature engineering for tabular ML** (`features.py`) —
  `FeatureConfig` + `FeatureEngineer`.  Generates lag, rolling,
  calendar, trend, and (optional) expanding features with shift-based
  temporal integrity (no look-ahead).
- **Walk-forward `validate_by_group`** — runs walk-forward CV per
  `(Country, SKU)` group with the v2 spec interface.
- **Multi-model `run_model_comparison`** — head-to-head comparison
  across any subset of registered specs; legacy
  `run_benchmark_comparison` retained as an alias.
- **Public API consolidation** — `optimize_model`, `optimize_arima`,
  `forecast_arima`, `pred_arima`, all metrics, validation helpers,
  benchmark utilities, and every `ModelSpec` class re-exported from
  `boa_forecaster.__init__`.

### Added — Reliability & observability

- **`OptimizationResult.is_fallback: bool`** — distinguishes a genuine
  optimum from a warm-start returned after a study-level crash;
  defaults to `False` for backward compatibility.  Crash now logs at
  `WARNING` with `exc_info=True` instead of being silently swallowed.
  See [ADR-002](docs/adr/ADR-002-optimizer-soft-failure.md).
- **Thread-safe `METRIC_REGISTRY`** — `register_metric` wraps
  registration in a `threading.Lock`.
- **`SARIMASpec.MAX_NON_SEASONAL_ORDER` / `MAX_SEASONAL_ORDER` named
  constants** replacing magic `4` / `3` thresholds.  Constraint
  violations now return `OPTIMIZER_PENALTY` so TPE learns to avoid them.

### Added — Tests

- **SARIMA constraint enforcement tests**
  (`tests/unit/test_sarima_constraints.py`).
- **Feature-leakage regression tests** (`tests/unit/test_features.py`)
  — assert no `t ≥ now` value appears in the feature row at position
  `t`.
- **Benchmark silent-failure tests** (`tests/unit/test_benchmarks.py`,
  `test_benchmarks_v2.py`) — mock ETS / AutoARIMA exceptions and
  assert seasonal-naive fallback.
- **Full-pipeline integration test**
  (`tests/integration/test_full_pipeline.py`) — exercises
  `data_loader → clean_zeros → fill_blanks → weighted_moving_stats_series
  → optimize_model → build_forecaster → walk_forward_validation` on
  both SARIMA and Random Forest branches, with a real Excel fixture
  including a zero-demand SKU and an injected outlier.
- **Property-based metric tests**
  (`tests/unit/test_metrics_property.py`) — 19 Hypothesis tests
  covering sMAPE / RMSLE / MAE / RMSE invariants (bounds, symmetry,
  identity, Jensen), `combined_metric` linearity, and
  `build_combined_metric` weight monotonicity.  Uses `deadline=None`
  for Windows CI stability.
- **Optimizer stress test** (`tests/unit/test_optimizer_stress.py`)
  — `@pytest.mark.slow`, runs `optimize_model` on a 500-point monthly
  series with 5 trials and asserts completion under a 30 s budget.

### Added — CI & tooling

- **`mypy` static type checking** — new CI step on the Python 3.11
  matrix entry runs `python -m mypy src/boa_forecaster`.
  `pyproject.toml` pins `python_version = "3.9"` so 3.9-incompatible
  typing is caught (e.g. PEP 604 `X | Y` in value position).
- **Weekly slow-test job** (`test-slow` in `.github/workflows/ci.yml`)
  — Mondays 06:00 UTC, Python 3.11 with `[dev,ml]` extras, runs
  `pytest -m slow --durations=10` with a 20-min timeout.  Existing
  push/PR jobs gated to those triggers only.
- **Coverage threshold `--cov-fail-under=80`** enforced in CI for both
  `test-core-only` and `test-ml-extras` jobs.
- **`hypothesis>=6.0`** added to `[dev]` extras.

### Added — Documentation

- **Architecture Decision Records** (`docs/adr/`) — ADR-001
  (`ModelSpec` as `Protocol`, not `ABC`), ADR-002 (optimizer
  soft-failure contract), ADR-003 (default
  `0.7·sMAPE + 0.3·RMSLE` objective).
- **Extension guide** (`docs/extending_models.md`) — end-to-end
  walkthrough for adding a new `ModelSpec`, with a worked Prophet
  example, the tree-model `BaseMLSpec` shortcut, a test checklist,
  and a pitfalls table.
- **Type-annotation completeness pass** across `models/base.py`,
  `validation.py`, `features.py`, `data_loader.py`.
- **Documented rationale for decaying weights `[0.3, 0.2, 0.1]`** in
  `standardization.py`.

### Changed — Performance

- **`weighted_moving_stats` vectorised** — new
  `weighted_moving_stats_series(sales_data, window_size, threshold)`
  helper uses `np.lib.stride_tricks.sliding_window_view` to compute
  rolling means and standard deviations in O(n) instead of
  O(n · window).  Mathematically identical output; **18–130× faster**
  depending on series length.  The legacy row-by-row entry point
  remains available for callers that genuinely need it.
- **`fill_blanks` vectorised** — replaced the cross-join + merge with
  a `pd.MultiIndex.from_product([dates, groups])` + `reindex`
  pipeline.  ~1.2–1.5× faster, lower peak memory.  **Behaviour
  change:** duplicate `(date, group)` rows in the input are now
  summed before reindexing (previously they were silently duplicated
  in the output).  Pipelines that run `clean_zeros` first are
  unaffected, since `clean_zeros` removes duplicates upstream.
- **`recursive_forecast` pre-allocates** the extended series instead
  of growing it via `pd.concat` in a loop.  5–20× speedup on long
  horizons.
- **`optimizer._validate_series` early-exits** via
  `series.isin([np.inf, -np.inf]).any()` instead of materialising a
  full boolean array via `np.isinf`.

### Changed — Defaults

- **CI matrix expanded** — Python 3.9 / 3.10 / 3.11 for core,
  Python 3.11 with `[dev,ml]` for ML-extras job.
- **Optimizer crash visibility** — `optimize_model` upgraded from a
  silent fallback to a `WARNING`-logged fallback with the
  `is_fallback=True` flag.  See ADR-002.

### Deprecated

- **`pred_arima` and `forecast_arima`** in
  `boa_forecaster.models.sarima` emit `DeprecationWarning` on call.
  Will be removed in v3.0.  Use `SARIMASpec` with `optimize_model`
  instead.
- **`optimize_arima`** — emits `DeprecationWarning`; use
  `optimize_model(series, SARIMASpec(...))`.
- **`sarima_bayes` package** — emits `DeprecationWarning` on import;
  re-exports everything from `boa_forecaster`.

### Fixed

- Six latent type-checker issues surfaced by the new mypy CI step:
  `_n_train` narrowing in `features.py` (×2), callbacks list type and
  `Callable` / `Any` imports in `lightgbm.py`, `weight_sum` float
  cast and empty-`ndarray` annotation in `standardization.py`, and a
  `date_values` `ndarray` annotation in `preprocessor.py`.

[2.1.0]: https://github.com/TomCardeLo/boa-forecaster/compare/v2.0.0...v2.1.0

---

## [1.4.0] — 2026-03-24

### Added

- **Optional `Country` and `SKU` columns** — the loader, preprocessor,
  optimiser, and validator now accept flat single-series workbooks
  without group columns.  `group_cols=None` falls back to a single
  ungrouped pipeline.

### Removed

- **`merge_representatives` helper** — superseded by direct group
  selection in `validate_by_group`.

[1.4.0]: https://github.com/TomCardeLo/boa-sarima-forecaster/compare/v1.3.0...v1.4.0

---

## [1.3.0] — 2026-03-24

### Added

- **Configurable metric composition** — `build_combined_metric(components)`
  factory accepts any list of `{"metric": str, "weight": float}` dicts
  drawn from `METRIC_REGISTRY` (`smape`, `rmsle`, `mae`, `rmse`,
  `mape`).  The optimisation objective can now be tuned per call or per
  config without touching source code.
- **`metrics` section in `config.example.yaml`** — declarative
  composition consumed by `optimize_model` / `optimize_arima`.
- **`mae`, `rmse`, `mape`** — newly exposed in `metrics.py` alongside
  the original `smape` / `rmsle` / `combined_metric`.

### Changed

- **`optimize_arima` / `optimize_model`** — both now accept a
  `metric_components` keyword that overrides the default
  `0.7·sMAPE + 0.3·RMSLE` mix.
- **`README.md`** — documents the metric registry and configuration
  examples for revenue and price use cases.

[1.3.0]: https://github.com/TomCardeLo/boa-sarima-forecaster/compare/v1.2.0...v1.3.0

---

## [1.2.0] — 2026-03-23

### Added

- **Configurable time-series frequency** — the pipeline now supports any pandas
  DateOffset alias, not only monthly (`"MS"`).  Pass `freq` to control the sampling
  rate and `m` to set the matching seasonal period:
  - `pred_arima`, `forecast_arima`, `forecast_arima_with_group` — new `freq: str = "MS"` parameter
  - `validate_by_group` — new `freq: str = "MS"` parameter (inserted before `**kwargs`)
  - `ets_model` — new `m: int = 12` parameter
  - `auto_arima_nixtla` — new `m: int = 12` and `freq: str = "MS"` parameters
  - `run_benchmark_comparison` — new `m: int = 12` and `freq: str = "MS"` parameters;
    both forwarded to all three baseline models
- **`_freq_to_period_alias` helper** in `preprocessor.py` — maps pandas DateOffset
  aliases (e.g. `"MS"`, `"W"`, `"D"`, `"H"`) to Period aliases required by
  `pd.Series.dt.to_period()`.
- **`data.freq` key** in `config.example.yaml` — documents supported aliases and the
  relationship between `freq` and `model.sarima.seasonal_period`.

### Changed

- `preprocessor.fill_blanks` — date normalisation is now frequency-aware; previously
  locked to `"M"` (monthly), it now derives the Period alias from `freq` via
  `_freq_to_period_alias`.
- `config.example.yaml` — `model.sarima.seasonal_period` comment updated to show the
  recommended `m` value for each supported frequency.
- `validation.validate_by_group` docstring — `date_col` description no longer says
  "monthly dates"; `freq` parameter documented.

### Backward Compatibility

All new parameters default to `freq="MS"` / `m=12`, matching the previous hardcoded
values.  Existing call sites require **no changes**.

[1.2.0]: https://github.com/TomCardeLo/boa-sarima-forecaster/compare/v1.1.0...v1.2.0

---

## [1.1.0] — 2026-03-23

### Added

- **Configurable outlier-clipping threshold** — `clip_outliers` and
  `weighted_moving_stats` now accept a `threshold` parameter (default `2.5`).
  Previously the σ multiplier was hard-coded to `2.5`; it can now be set per-call
  or globally via `config.yaml` under `standardization.threshold`.

### Changed

- `config.example.yaml` — added `standardization.threshold: 2.5` key so users can
  tune sensitivity without touching source code.
- `docs/methodology.md` — updated standardisation section to document the new parameter.

### Fixed

- Renamed internal parameter `sigma_threshold` → `threshold` in `clip_outliers` to
  match the public API expected by the test suite.
- Resolved ruff lint errors (`UP` and `E` rules) and applied black auto-formatting
  to `config.py` that were blocking CI on the feature branch.

[1.1.0]: https://github.com/TomCardeLo/boa-sarima-forecaster/compare/v1.0.0...v1.1.0

---

## [1.0.0] — 2026-03-17

### Added

- **SARIMA + Bayesian Optimisation pipeline** — end-to-end demand forecasting using
  Optuna TPE to search ARIMA orders `(p, d, q)` and seasonal orders `(P, D, Q, m)`.
- **Walk-forward (expanding-window) cross-validation** — prevents look-ahead bias by
  evaluating each fold on true out-of-sample periods.
- **Benchmark comparison** — walk-forward results compared against Seasonal Naïve,
  ETS (Holt-Winters), and AutoARIMA (statsforecast) baselines.
- **Weighted moving-average outlier standardisation** — clips demand observations to
  ±1σ of their neighbourhood; both raw and adjusted series are modelled and the better
  one is selected automatically.
- **sMAPE and RMSLE metrics** — combined cost function `0.7 × sMAPE + 0.3 × RMSLE`
  used as the Optuna objective; both metrics available individually via `sarima_bayes.metrics`.
- **Demo notebook** (`notebooks/demo.ipynb`) — end-to-end walkthrough using synthetic
  data; no real data required.
- **pytest test suite** (`tests/`) — unit and integration tests with coverage reporting.
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — runs linting (ruff, black) and
  the full test suite on every push and pull request.
- **Full type hints and Google-style docstrings** — all 19 public functions across
  `src/sarima_bayes/` annotated with Python 3.10+ `X | Y` union syntax, Args, Returns,
  Raises, and Example sections.
- **`config.yaml` / `config.example.yaml`** — YAML-driven configuration for data paths,
  optimisation budget, forecast horizon, and output location.
- **`docs/methodology.md`** — detailed technical description of the five-stage pipeline.
- **Forecast plot** (`docs/img/forecast_example.png`) — example output image showing
  training history, last-24-months actuals, point forecast, and 80%/95% CI bands;
  generated reproducibly via `scripts/generate_plots.py`.

[1.0.0]: https://github.com/TomCardeLo/boa-sarima-forecaster/releases/tag/v1.0.0
