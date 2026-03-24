# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"
pip install pytest-cov   # not included in [dev] extras, must be installed separately

# Install with notebook dependencies
pip install -e ".[dev,notebooks]"

# Run all tests with coverage
pytest tests/ -v --tb=short --cov=src/sarima_bayes --cov-report=term-missing

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

**`src/sarima_bayes/`** — src-layout package. The pipeline flows through these stages:

1. **`data_loader.py`** — reads Excel input, parses YYYYMM dates, returns a clean DataFrame
2. **`preprocessor.py`** — fills calendar gaps with zeros (`fill_blanks`), removes zero-demand groups (`clean_zeros`)
3. **`standardization.py`** — two clipping strategies: `weighted_moving_stats` (local neighbourhood smoother, called row-by-row) and `clip_outliers` (global sigma or IQR). Both raw and clipped series are passed to the optimiser, which picks the lower combined metric.
4. **`optimizer.py`** — Optuna TPE Bayesian search over (p,d,q,P,D,Q); warm-started with ARIMA(1,1,1) and AR(1); soft constraints penalise complexity (p+q≤4, P+Q≤3); returns best params
5. **`model.py`** — fits SARIMAX via statsmodels (`pred_arima`), then `forecast_arima` wraps it for a full 12-month forecast with 95% CI
6. **`validation.py`** — expanding-window walk-forward CV (`walk_forward_validation`); `validate_by_group` runs it per SKU/country group
7. **`benchmarks.py`** — Seasonal Naïve, ETS (Holt-Winters), AutoARIMA (statsforecast) baselines; `run_benchmark_comparison` and `summary_table` compare all models

**`metrics.py`** defines the objective: `combined_metric = 0.7·sMAPE + 0.3·RMSLE`.

**`config.py`** holds global defaults; all values are overridable via `config.yaml` (git-ignored — copy from `config.example.yaml`).

**Public API** is re-exported from `__init__.py`: `optimize_arima`, `forecast_arima`, `pred_arima`, metric functions, validation helpers, and benchmark utilities.

## API notes

- `clip_outliers(series, method="sigma", threshold=2.5)` — the threshold parameter is named **`threshold`**, not `sigma_threshold`. `method` accepts `"sigma"` or `"iqr"`.
- `weighted_moving_stats(row_index, sales_data, window_size=3, threshold=2.5)` — takes a plain list, not a Series; call it row-by-row inside a loop over one SKU.
- The `m` (seasonal period) is **fixed** at 12 for monthly data and is not part of the Optuna search space.

## Configuration

Copy `config.example.yaml` → `config.yaml` before running locally. Key sections:
- `data`: input path, sheet name, skip_rows, date format
- `optimization`: p/d/q ranges, n_calls, n_jobs
- `standardization`: method (`sigma` or `iqr`), threshold (default 2.5)
- `forecast`: n_periods, confidence alpha
- `output`: output path, run_id

## Tests

Fixtures are in `tests/conftest.py` (`synthetic_series`, `raw_series_with_outliers`, `long_series`). The `series_with_outlier` fixture is defined locally inside `tests/unit/test_standardization.py` (not in conftest). Tests are unit-only — no integration tests hit real Excel files. CI runs Python 3.9, 3.10, and 3.11.
