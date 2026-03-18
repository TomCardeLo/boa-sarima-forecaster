# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[1.0.0]: https://github.com/USERNAME/REPO_NAME/releases/tag/v1.0.0
