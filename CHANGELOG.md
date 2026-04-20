# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
