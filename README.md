# BOA Forecaster

> Multi-model demand forecasting with Bayesian Optimisation (Optuna TPE) — SARIMA, Random Forest, XGBoost, LightGBM

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-2.0.0-brightgreen)
![CI](https://github.com/TomCardeLo/boa-sarima-forecaster/actions/workflows/ci.yml/badge.svg)

---

## Table of Contents

1. [What's New in v2.0](#whats-new-in-v20)
2. [Motivation](#motivation)
3. [Methodology](#methodology)
4. [Results](#results)
5. [Project Structure](#project-structure)
6. [Installation](#installation)
7. [Quick Start](#quick-start)
8. [ML Models](#ml-models)
9. [Input Data Format](#input-data-format)
10. [Configuration](#configuration)
11. [Configurable Metric](#configurable-metric)
12. [Validation & Benchmarks](#validation--benchmarks)
13. [Running the Demo Notebook](#running-the-demo-notebook)
14. [Output Files](#output-files)
15. [Backward Compatibility](#backward-compatibility)
16. [License](#license)

---

## What's New in v2.0

v2.0 is a ground-up redesign that turns the library from a SARIMA-only tool into a **pluggable multi-model forecasting framework**:

| Feature | v1.x | v2.0 |
|---------|------|------|
| Primary package | `sarima_bayes` | `boa_forecaster` |
| Models | SARIMA only | SARIMA · Random Forest · XGBoost · LightGBM |
| Model protocol | hard-coded | `ModelSpec` — add any model in ~50 lines |
| Optimizer entry point | `optimize_arima()` | `optimize_model(series, model_spec)` |
| Feature engineering | — | `FeatureEngineer` with lags, rolling stats, calendar |
| Multi-model comparison | `run_benchmark_comparison()` | `run_model_comparison()` |
| `sarima_bayes` | primary package | deprecated shim — emits `DeprecationWarning`, fully backward-compatible |

---

## Motivation

Demand planners managing hundreds of SKUs across multiple markets face a
recurring problem: **fitting time-series models at scale requires choosing
hyperparameters** that differ by product and country. Manual tuning is
infeasible, and grid search is computationally expensive.

This library addresses the problem with **Bayesian Optimisation** — a
principled, sample-efficient search strategy that learns from past
evaluations to focus on promising parameter regions. The result is a
production-ready pipeline that:

- Automatically finds the best model order / hyperparameters per time series.
- Supports SARIMA, Random Forest, XGBoost, and LightGBM via a unified API.
- Clips outliers via a weighted moving-average smoother.
- Scales to hundreds of SKUs via parallel execution.
- Works with a single time series (no SKU / Country columns required).

---

## Methodology

### 1 — Data Preparation

Raw sales data is loaded from Excel, cleaned (NaN fill, date parsing), and
preprocessed (zero-series removal, missing-month fill).

### 2 — Outlier Standardisation

A weighted moving-average smoother clips each observation to ±2.5σ of its
neighbourhood (configurable via `threshold`), generating an `adjusted_value`
column alongside the raw demand. Both columns are modelled independently; the
one with the lower optimisation score is used for the final forecast.

### 3 — Bayesian Optimisation (Optuna TPE)

The **Tree-structured Parzen Estimator (TPE)** searches the model's
hyperparameter space to minimise a configurable weighted metric. The default
objective is:

```
combined = 0.7 × sMAPE + 0.3 × RMSLE
```

The TPE sampler:
- Uses `multivariate=True` to capture correlations between parameters.
- Is seeded at `seed=42` for reproducibility.
- Is warm-started with known-good configurations to accelerate convergence.

### 4 — Forecasting

The best parameters found by the optimiser are used to fit the chosen model
and generate a 12-month point forecast. Negative predictions are clipped to
zero.

### 5 — Parallel Execution

The per-SKU loop is parallelised with `joblib.Parallel(backend="loky")`,
enabling all available CPU cores to be utilised.

---

## Results

### Model comparison (synthetic demo data)

| Model | sMAPE (%) | RMSLE | beats_naive |
|-------|-----------|-------|-------------|
| SARIMA+BO | 8.4 | 0.09 | True |
| Random Forest | 9.1 | 0.10 | True |
| XGBoost | 8.7 | 0.09 | True |
| LightGBM | 8.6 | 0.09 | True |
| ETS | 10.2 | 0.11 | True |
| Seasonal Naïve | 14.7 | 0.16 | — |

> **Note:** Values are from synthetic demo data. Results on real production data will vary.

---

## Project Structure

```
boa-sarima-forecaster/
├── README.md
├── pyproject.toml
├── config.example.yaml          ← copy to config.yaml and customise
│
├── src/
│   ├── boa_forecaster/          ← primary package (v2.0)
│   │   ├── __init__.py          ← public API re-exports
│   │   ├── config.py            ← global constants and defaults
│   │   ├── data_loader.py       ← Excel ingestion and cleaning
│   │   ├── preprocessor.py      ← date fill, zero removal
│   │   ├── standardization.py   ← weighted moving-average outlier clipping
│   │   ├── metrics.py           ← sMAPE, RMSLE, MAE, RMSE, MAPE, build_combined_metric
│   │   ├── features.py          ← FeatureConfig + FeatureEngineer (ML models)
│   │   ├── optimizer.py         ← optimize_model() — generic Optuna TPE engine
│   │   ├── validation.py        ← walk_forward_validation, validate_by_group
│   │   ├── benchmarks.py        ← run_model_comparison, baseline models
│   │   └── models/
│   │       ├── __init__.py      ← MODEL_REGISTRY, get_model_spec, register_model
│   │       ├── base.py          ← ModelSpec Protocol, OptimizationResult, param types
│   │       ├── sarima.py        ← SARIMASpec (statsmodels SARIMAX)
│   │       ├── random_forest.py ← RandomForestSpec (scikit-learn)
│   │       ├── xgboost.py       ← XGBoostSpec (requires xgboost extra)
│   │       └── lightgbm.py      ← LightGBMSpec (requires lightgbm extra)
│   │
│   └── sarima_bayes/            ← deprecated shim — re-exports boa_forecaster
│       └── __init__.py          ← emits DeprecationWarning on import
│
├── tests/
│   ├── conftest.py
│   ├── unit/                    ← fast unit tests (no real data)
│   └── integration/             ← multi-model end-to-end tests
│
├── data/
│   ├── README.md
│   ├── sample_data.csv
│   ├── input/                   ← put your real Excel files here (git-ignored)
│   └── output/                  ← forecast results written here (git-ignored)
│
├── notebooks/
│   └── demo.ipynb
│
└── docs/
    └── methodology.md
```

---

## Installation

### Core (SARIMA only)

```bash
git clone https://github.com/TomCardeLo/boa-sarima-forecaster
cd boa-sarima-forecaster

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows

pip install -e ".[dev]"
pip install pytest-cov
```

### With ML models (Random Forest, XGBoost, LightGBM)

```bash
pip install -e ".[dev,ml]"
```

### Individual extras

```bash
pip install -e ".[xgboost]"    # XGBoost only
pip install -e ".[lightgbm]"   # LightGBM only
pip install -e ".[ml]"         # XGBoost + LightGBM (scikit-learn is always installed)
```

---

## Quick Start

### v2.0 API — any model via `optimize_model`

```python
import numpy as np
import pandas as pd
from boa_forecaster import optimize_model
from boa_forecaster.models import SARIMASpec, RandomForestSpec

# 48 months of synthetic monthly demand
rng = np.random.default_rng(42)
series = pd.Series(
    100 + np.cumsum(rng.normal(0, 5, 48)),
    index=pd.date_range("2020-01", periods=48, freq="MS"),
)

# --- SARIMA ---
result = optimize_model(series, SARIMASpec(), n_trials=30)
print(result.best_params)   # {'p': 1, 'd': 1, 'q': 0, ...}
print(result.best_score)    # e.g. 5.23

# --- Random Forest (requires scikit-learn, installed by default) ---
from boa_forecaster.models import RandomForestSpec
result_rf = optimize_model(series, RandomForestSpec(), n_trials=30)
forecaster = result_rf.model_spec.build_forecaster(result_rf.best_params)
forecast = forecaster(series)   # pd.Series of length 12
print(forecast)
```

### Multi-model comparison

```python
from boa_forecaster import run_model_comparison
from boa_forecaster.models import SARIMASpec, RandomForestSpec

summary = run_model_comparison(
    df=df,
    group_cols=["SKU", "Country"],
    target_col="CS",
    date_col="Date",
    model_specs=[SARIMASpec(), RandomForestSpec()],
    n_calls_per_model=30,
)
print(summary)
```

### Legacy v1.x API (still works)

```python
from sarima_bayes import optimize_arima, forecast_arima   # emits DeprecationWarning

best_params, score = optimize_arima(series=series, n_calls=30)
```

---

## ML Models

All ML models share the same interface via the `ModelSpec` protocol and use
`FeatureEngineer` internally for feature construction.

### Feature engineering

`FeatureEngineer` builds the following features from a raw time series:

| Feature group | Default config |
|--------------|----------------|
| Lags | t-1, t-2, t-3, t-6, t-12 |
| Rolling mean/std | windows 3, 6, 12 |
| Calendar | month, quarter, year |
| Trend | integer time index |
| Expanding mean | disabled by default |

Customise via `FeatureConfig`:

```python
from boa_forecaster.features import FeatureConfig, FeatureEngineer

config = FeatureConfig(
    lag_periods=[1, 2, 3, 12],
    rolling_windows=[3, 6],
    include_calendar=True,
    include_trend=True,
    include_expanding=False,
)
spec = RandomForestSpec(feature_config=config)
```

### Available models

| Model | Extra required | Key hyperparameters searched |
|-------|---------------|------------------------------|
| `SARIMASpec` | none | p, d, q, P, D, Q |
| `RandomForestSpec` | none (scikit-learn is core) | n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features |
| `XGBoostSpec` | `xgboost` | n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda |
| `LightGBMSpec` | `lightgbm` | n_estimators, num_leaves, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda |

### Adding a new model

Implement the `ModelSpec` protocol (~50 lines):

```python
from boa_forecaster.models.base import ModelSpec, OptimizationResult, IntParam, FloatParam

class MyModelSpec:
    name = "my_model"
    needs_features = True

    @property
    def search_space(self):
        return {"n_estimators": IntParam(50, 500)}

    @property
    def warm_starts(self):
        return [{"n_estimators": 100}]

    def suggest_params(self, trial):
        from boa_forecaster.models.base import suggest_from_space
        return suggest_from_space(trial, self.search_space)

    def evaluate(self, series, params, metric_fn, feature_config=None):
        # walk-forward CV, return float score
        ...

    def build_forecaster(self, params, feature_config=None):
        # return callable: series -> pd.Series of length horizon
        ...
```

Then register it:

```python
from boa_forecaster.models import register_model
register_model("my_model", MyModelSpec)
```

---

## Input Data Format

### Sales Excel workbook (`data/input/sales.xlsx`)

| Layer | Content |
|-------|---------|
| Row 0 | Blank / metadata (skipped) |
| Row 1 | Blank / metadata (skipped) |
| Row 2 | **Column headers** |
| Row 3+ | Data rows |

> If your file has no extra header rows, set `skip_rows: 0` in `config.yaml`.

#### Required columns

| Column | Type | Format |
|--------|------|--------|
| `Date` | string | `YYYYMM` — e.g. `"202201"` = January 2022 |
| `CS` | float | Target variable (units, cases, revenue, or any numeric measure) |

#### Optional columns

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `SKU` | integer | `1` | Series identifier — omit for single-series use |
| `Country` | string | `"_"` | Market / region code (e.g. `"US"`, `"MX"`) |

> `SKU` and `Country` are **auto-injected** with their default values if not present.

---

## Configuration

Copy `config.example.yaml` to `config.yaml` and adjust:

```yaml
data:
  input_path:  "data/input/sales.xlsx"
  sheet_name:  "Data"
  skip_rows:   2
  date_format: "%Y%m"
  freq:        "MS"

models:
  active: sarima          # sarima | random_forest | xgboost | lightgbm

  sarima:
    enabled: true
    seasonal_period: 12

  random_forest:
    enabled: true
    forecast_horizon: 12

  xgboost:
    enabled: true
    forecast_horizon: 12
    early_stopping_rounds: 20

  lightgbm:
    enabled: true
    forecast_horizon: 12
    early_stopping_rounds: 20

features:
  lag_periods:       [1, 2, 3, 6, 12]
  rolling_windows:   [3, 6, 12]
  include_calendar:  true
  include_trend:     true
  include_expanding: false

forecast:
  n_periods: 12
  alpha:     0.05

output:
  output_path: "data/output/"
  run_id:      "RUN-2026-01"
```

### Supported frequencies

| `freq` | Sampling rate | Recommended `seasonal_period` |
|--------|--------------|-------------------------------|
| `"MS"` | Monthly (default) | `12` |
| `"W"` | Weekly | `52` or `4` |
| `"D"` | Daily | `7` or `365` |
| `"h"` | Hourly | `24` or `168` |
| `"QS"` | Quarterly | `4` |

---

## Configurable Metric

### Available metrics

| Name | Formula | Best suited for |
|------|---------|-----------------|
| `smape` | `100 × mean(|y-ŷ| / ((|y|+|ŷ|)/2 + ε))` | Intermittent / zero-heavy demand |
| `rmsle` | `√mean((log(1+y) − log(1+ŷ))²)` | Series spanning multiple orders of magnitude |
| `mae` | `mean(|y − ŷ|)` | Revenue — absolute scale matters |
| `rmse` | `√mean((y − ŷ)²)` | Penalises large deviations more than MAE |
| `mape` | `100 × mean(|y − ŷ| / (|y| + ε))` | Clean series without zeros |

### Programmatic usage

```python
from boa_forecaster import optimize_model, build_combined_metric
from boa_forecaster.models import SARIMASpec

result = optimize_model(
    series=series,
    model_spec=SARIMASpec(),
    n_trials=30,
    metric_components=[
        {"metric": "mae",  "weight": 0.6},
        {"metric": "rmse", "weight": 0.4},
    ],
)
```

---

## Validation & Benchmarks

Walk-forward (expanding window) cross-validation evaluates models on true
out-of-sample periods, preventing look-ahead bias.

### Baseline models

| Model | Description |
|-------|-------------|
| `seasonal_naive` | Repeats value from same period last year |
| `ets_model` | Holt-Winters additive trend+seasonal |
| `auto_arima_nixtla` | AutoARIMA via statsforecast |

### Running a comparison

```python
from boa_forecaster import run_model_comparison
from boa_forecaster.models import SARIMASpec, RandomForestSpec, XGBoostSpec

summary = run_model_comparison(
    df=df,
    group_cols=["SKU", "Country"],
    target_col="CS",
    date_col="Date",
    model_specs=[SARIMASpec(), RandomForestSpec(), XGBoostSpec()],
    n_calls_per_model=50,
    n_folds=3,
    test_size=6,
)
print(summary)
```

---

## Running the Demo Notebook

```bash
pip install -e ".[dev,notebooks]"
jupyter notebook notebooks/demo.ipynb
```

---

## Output Files

| File | Contents |
|------|---------|
| `{run_id} forecast boa.xlsx` | Point forecasts: Date, Pred, Country, Sku |
| `{run_id} data estandar.xlsx` | Standardised series: raw + adjusted demand |
| `{run_id} data metricas.xlsx` | Optimisation results: best params and score per series |
| `{run_id} data logs.xlsx` | Execution log: timestamps, status, errors |

---

## Backward Compatibility

The `sarima_bayes` package remains fully functional in v2.0 as a **deprecated
compatibility shim**. Existing code will continue to work — you will only see
a `DeprecationWarning` on import:

```python
# Still works — emits DeprecationWarning
from sarima_bayes import optimize_arima, forecast_arima

# Recommended v2.0 equivalent
from boa_forecaster import optimize_model
from boa_forecaster.models import SARIMASpec
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```
