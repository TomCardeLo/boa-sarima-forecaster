# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
