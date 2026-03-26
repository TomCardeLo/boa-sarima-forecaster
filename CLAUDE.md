# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"
pip install pytest-cov   # not included in [dev] extras, must be installed separately

# Install with ML model extras (RandomForest, XGBoost, LightGBM)
pip install -e ".[dev,ml]"

# Install with notebook dependencies
pip install -e ".[dev,notebooks]"

# Run all tests with coverage
pytest tests/ -v --tb=short --cov=src/boa_forecaster --cov-report=term-missing

# Run a single test file
pytest tests/unit/test_metrics.py -v

# Run a single test class or function
pytest tests/unit/test_standardization.py::TestClipOutliers -v

# Lint
python -m ruff check .

# Format check / auto-format
python -m black --check .
python -m black .
```

If `ruff` or `black` are not on PATH, always use `python -m ruff` / `python -m black`.

Ruff selects rules `E, F, W, I, UP` and ignores `E501` (line length in code is not enforced by ruff). Black still enforces the 88-char limit for formatting — long inline comments will be wrapped into parenthesised expressions.

## Architecture

**`src/boa_forecaster/`** — primary package (v2.0). `sarima_bayes` is a deprecated
compatibility shim that re-exports everything from `boa_forecaster` and emits a
`DeprecationWarning` on import.

Pipeline stages:

1. **`data_loader.py`** — reads Excel input, parses YYYYMM dates, returns a clean DataFrame
2. **`preprocessor.py`** — fills calendar gaps with zeros (`fill_blanks`), removes zero-demand groups (`clean_zeros`)
3. **`standardization.py`** — two clipping strategies: `weighted_moving_stats` (local neighbourhood smoother, row-by-row) and `clip_outliers` (global sigma or IQR). Both raw and clipped series are passed to the optimiser, which picks the lower combined metric.
4. **`features.py`** — `FeatureConfig` dataclass and `FeatureEngineer`; builds lag, rolling-window, calendar, trend, and expanding features from a raw series for ML models
5. **`models/`** — model registry and specs:
   - `base.py` — `ModelSpec` Protocol + `OptimizationResult`, `IntParam`, `FloatParam`, `CategoricalParam`
   - `sarima.py` — `SARIMASpec`: SARIMAX via statsmodels
   - `random_forest.py` — `RandomForestSpec`: scikit-learn RandomForestRegressor
   - `xgboost.py` — `XGBoostSpec`: XGBoost (requires `xgboost` extra)
   - `lightgbm.py` — `LightGBMSpec`: LightGBM (requires `lightgbm` extra)
6. **`optimizer.py`** — `optimize_model(series, model_spec, n_trials)` runs Optuna TPE over any `ModelSpec`; `optimize_arima()` is the v1.x-compatible alias (emits `DeprecationWarning`)
7. **`validation.py`** — expanding-window walk-forward CV (`walk_forward_validation`); `validate_by_group` runs it per SKU/country group
8. **`benchmarks.py`** — Seasonal Naïve, ETS, AutoARIMA baselines + `run_model_comparison` for multi-model head-to-head; `run_benchmark_comparison` is the v1.x alias

**`metrics.py`** defines the objective: `combined_metric = 0.7·sMAPE + 0.3·RMSLE` (configurable via `build_combined_metric`).

**`config.py`** holds global defaults; all values are overridable via `config.yaml` (git-ignored — copy from `config.example.yaml`).

**Public API** is re-exported from `__init__.py`: `optimize_model`, `optimize_arima`, `forecast_arima`, `pred_arima`, metric functions, validation helpers, benchmark utilities, and all `ModelSpec` classes.

## API notes

- `clip_outliers(series, method="sigma", threshold=2.5)` — the threshold parameter is named **`threshold`**, not `sigma_threshold`. `method` accepts `"sigma"` or `"iqr"`.
- `weighted_moving_stats(row_index, sales_data, window_size=3, threshold=2.5)` — takes a plain list, not a Series; call it row-by-row inside a loop over one SKU.
- The `m` (seasonal period) is **fixed** at 12 for monthly data and is not part of the Optuna search space. Set it via `SARIMASpec(seasonal_period=12)`.
- `optimize_model(series, model_spec, n_trials=50)` — v2.0 generic entry point. `model_spec` is any `ModelSpec` instance (e.g. `SARIMASpec()`, `RandomForestSpec()`). Returns `OptimizationResult`.

## Configuration

Copy `config.example.yaml` → `config.yaml` before running locally. Key sections:
- `data`: input path, sheet name, skip_rows, date format
- `optimization`: p/d/q ranges, n_calls, n_jobs (legacy SARIMA; superseded by `models.sarima.search_space` in v2.0)
- `standardization`: method (`sigma` or `iqr`), threshold (default 2.5)
- `forecast`: n_periods, confidence alpha
- `output`: output path, run_id
- `models`: active model name + per-model `search_space`, `constraints`, `warm_starts`
- `features`: `lag_periods`, `rolling_windows`, `include_calendar`, `include_trend`, `include_expanding` (consumed by ML models)

## Tests

Fixtures are in `tests/conftest.py` (`synthetic_series`, `raw_series_with_outliers`, `long_series`). The `series_with_outlier` fixture is defined locally inside `tests/unit/test_standardization.py` (not in conftest). Tests are unit-only except for `tests/integration/`. CI runs Python 3.9, 3.10, 3.11 (`test-core-only`) and Python 3.11 with ML extras (`test-ml-extras`). ML test files (`test_xgboost.py`, `test_lightgbm.py`) use `pytest.importorskip` at module level and are auto-skipped when the extras are not installed. Markers `requires_sklearn`, `requires_xgboost`, `requires_lightgbm`, `slow`, and `integration` are registered in `pyproject.toml`.
