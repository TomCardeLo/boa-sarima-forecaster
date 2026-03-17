# Methodology

## Overview

`boa-sarima-forecaster` is a monthly demand-forecasting pipeline that
combines two well-established techniques:

1. **ARIMA / SARIMA** — a classical statistical model for univariate time
   series based on autoregression, integration (differencing), and
   moving-average components.
2. **Bayesian Optimisation via Optuna TPE** — an efficient probabilistic
   search strategy that finds the optimal ARIMA hyperparameters
   (p, d, q) by learning from past evaluations.

The pipeline also includes an **outlier-robust pre-processing** step based
on weighted moving averages, which generates a smoothed demand series as
an alternative modelling input.

---

## 1. Data Preparation

### 1.1 Loading

Data is read from an Excel workbook (see `data/README.md` for the expected
column structure).  The loader skips two metadata rows, fills blank cells
with zero, removes invalid SKU markers (`"##"` subtotal rows), parses the
`Date` column from `YYYYMM` string format, and casts `SKU` to `int` and
`CS` to `float`.

### 1.2 Representative-SKU Consolidation (optional)

When multiple detailed SKUs are forecasted under a single "representative"
SKU, a mapping table is used to aggregate demand before modelling.  This
reduces the number of individual time series and ensures that low-volume
child SKUs benefit from pooled demand signal.

### 1.3 Zero-Series Removal

Time series whose cumulative demand is zero are dropped before fitting,
as they contain no signal and waste optimisation budget.

### 1.4 Missing-Month Fill

SARIMA requires a complete, gap-free time series.  Any missing calendar
month / SKU combination is filled with zero demand using a vectorised
MultiIndex reindex operation.  The series is then extended up to the
configured `end_date`.

---

## 2. Outlier-Robust Standardisation

Before model fitting, each time series undergoes a custom weighted
moving-average smoother (`standardization.weighted_moving_stats`).

For every observation at index *i*, the algorithm:

1. Extracts the window of up to 3 neighbours on each side.
2. Assigns decaying weights:
   - Distance 1 → 0.3
   - Distance 2 → 0.2
   - Distance 3 → 0.1
3. Computes the **weighted mean** μ and **weighted standard deviation** σ
   of the neighbourhood (excluding the centre point itself).
4. **Clips** the original value to [μ − σ, μ + σ].

The resulting `adjusted_value` column is modelled in parallel with the
original `CS` column.  The optimiser independently finds the best ARIMA
parameters for each, and the forecast with the lower combined metric score
is selected as the final output.

---

## 3. ARIMA Model

The forecasting model used is `ARIMA(p, d, q)`, a special case of SARIMA
without seasonal components.

| Parameter | Meaning                                        | Default search range |
|-----------|------------------------------------------------|----------------------|
| `p`       | Autoregressive order — number of lagged values | 0 – 6                |
| `d`       | Integration order — number of differences      | 0 – 2                |
| `q`       | Moving-average order — number of lagged errors | 0 – 6                |

The model is implemented via `statsmodels.tsa.statespace.sarimax.SARIMAX`
with `enforce_stationarity=False` and `enforce_invertibility=False`.  These
flags allow the optimiser to explore the full parameter space without hard
failures on numerically challenging combinations; the cost function
naturally penalises unstable fits.

Forecasts are clipped to zero from below (demand cannot be negative).

> **Note on seasonality**: Seasonal ARIMA components `(P, D, Q, s)` are
> intentionally omitted.  Monthly seasonality is handled implicitly by
> choosing an appropriate `p` order.  Seasonal components can be activated
> by passing a non-`None` `s_order` tuple to `model.pred_arima`.

---

## 4. Cost Function

The objective minimised during optimisation is a **weighted hybrid metric**:

```
combined = 0.7 × sMAPE + 0.3 × RMSLE
```

### 4.1 sMAPE — Symmetric Mean Absolute Percentage Error

```
sMAPE = 100 × mean( |y_true − y_pred| / ((|y_true| + |y_pred|) / 2 + ε) )
```

- **Bounded** in [0, 200] and symmetric around zero.
- `ε = 1e-10` prevents division-by-zero for intermittent demand.
- Captures *relative* forecasting accuracy.

### 4.2 RMSLE — Root Mean Squared Logarithmic Error

```
RMSLE = sqrt( mean( (log(1 + y_true) − log(1 + y_pred))² ) )
```

- Less sensitive to large absolute errors than RMSE.
- Works on a logarithmic scale, suitable for series spanning orders of
  magnitude.
- Penalises *under-predictions* more than *over-predictions*.

### 4.3 Rationale for the 0.7 / 0.3 weights

The higher weight on sMAPE prioritises relative percentage accuracy, which
is the KPI most relevant to demand planners.  The RMSLE component provides
a secondary anchor on absolute scale, preventing the optimiser from
selecting models that are percentage-accurate but miss large-volume peaks.

---

## 5. Bayesian Optimisation (Optuna TPE)

### 5.1 Why Bayesian Optimisation?

The brute-force grid over `p ∈ [0,6], d ∈ [0,2], q ∈ [0,6]` contains
**7 × 3 × 7 = 147** candidate combinations.  Each combination requires
fitting a full SARIMA model (O(n²) in the series length).  Bayesian
Optimisation with a probabilistic surrogate model explores this space
much more efficiently than grid search or random search.

### 5.2 Tree-structured Parzen Estimator (TPE)

The **TPE** sampler (Bergstra et al., 2011) models the distribution of
good and bad parameter configurations separately using kernel density
estimates.  At each step it proposes the parameter vector that maximises
the ratio *p(good) / p(bad)*, focusing sampling on promising regions.

Configuration:

```python
TPESampler(seed=42, multivariate=True)
```

- `multivariate=True` captures **correlations between p and q** (high AR
  order and high MA order tend to compete), producing better proposals than
  independent univariate TPE.
- `seed=42` ensures **reproducibility** across runs.

### 5.3 Warm Start

Two trials are enqueued before the probabilistic model takes over:

| Trial           | Rationale                                          |
|-----------------|----------------------------------------------------|
| `(p=1, d=1, q=1)` | Robust baseline for trending monthly demand      |
| `(p=1, d=0, q=0)` | Simple AR(1) for stationary series               |

These warm starts prevent wasting the early budget on trivial
`(p=0, d=0, q=0)` combinations.

### 5.4 In-sample Evaluation

For each `(p, d, q)` trial:

1. Fit `ARIMA(p, d, q)` on the full historical series.
2. Generate in-sample predictions for all observed periods.
3. Compute `combined_metric(y_true, y_pred)`.
4. Return the score to Optuna.

On model failure (numerical divergence, etc.), a penalty of `1e6` is
returned so the TPE sampler steers away from the failing region.

### 5.5 Study Configuration

```python
study = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=n_calls, n_jobs=n_jobs)
```

- **Direction**: minimize (lower combined metric = better).
- **Storage**: in-memory (no disk I/O overhead per trial).
- **n_trials**: 50 by default (configurable via `config.yaml`).

---

## 6. Parallel Execution

The pipeline processes multiple `(Country, SKU)` combinations in parallel
using `joblib.Parallel` with the threading backend:

```python
Parallel(n_jobs=-1, backend="threading", verbose=0)(
    delayed(process_sku)(country, sku, data) for country, sku in combinations
)
```

Using threads (rather than processes) avoids serialisation overhead and
is compatible with Python's free-threaded builds (3.14t+), where NumPy and
Pandas operations release the GIL natively.

---

## 7. Output Files

| File                        | Contents                                                |
|-----------------------------|---------------------------------------------------------|
| `{run_id} forecast boa.xlsx` | Point forecasts: Date, Pred, Country, Sku, Column, Model |
| `{run_id} data estandar.xlsx` | Standardised series: Date, CS, SKU, Country, prom_movil, desv_movil, valor_ajustado |
| `{run_id} data metricas.xlsx` | Optimisation results: Country, SKU, Col, Score, Model, Params |
| `{run_id} data logs.xlsx`   | Execution log: Timestamp, Country, SKU, Type, Column, Message |

---

## 8. References

- Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
  *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
- Bergstra, J., Bardenet, R., Bengio, Y., & Kégl, B. (2011).
  Algorithms for hyper-parameter optimization. *NeurIPS*.
- Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).
  Optuna: A next-generation hyperparameter optimization framework. *KDD*.
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2020).
  The M4 Competition: 100,000 time series and 61 forecasting methods.
  *International Journal of Forecasting*.
