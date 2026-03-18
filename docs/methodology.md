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
4. **Clips** the original value to [μ − threshold·σ, μ + threshold·σ]
   (default threshold = 2.5).

The resulting `adjusted_value` column is modelled in parallel with the
original `CS` column.  The optimiser independently finds the best SARIMA
parameters for each, and the forecast with the lower combined metric score
is selected as the final output.

---

## 2a. Outlier Treatment

### Why clipping is necessary

SARIMA models are sensitive to extreme values because their parameter
estimates are derived from autocovariance calculations.  A single large
spike — a one-off promotion, a data-entry error, or a supply disruption —
can inflate variance estimates, shift the mean, and produce forecasts that
overfit the anomaly rather than capturing the true demand pattern.  Pre-processing
the series to dampen extreme values before fitting gives the model a cleaner
signal to learn from.

### Why ±1σ is too aggressive

Clipping at ±1σ around the local mean is equivalent to treating any value
in the upper or lower 16 % of a normal distribution as an outlier.
Across the full series, a ±1σ rule clips approximately **32 % of all
observations**.  In demand-planning contexts, this threshold is far too
conservative: it routinely removes legitimate demand events such as
promotional lifts, seasonal peaks (e.g. Q4 holiday demand), and new-product
launch spikes.  Discarding these events distorts the estimated mean and
variance, causing the model to systematically under-forecast periods with
elevated demand.

### Why 2.5σ is the recommended default

A threshold of ±2.5σ retains **98.8 % of values** from a normal
distribution, clipping only observations that are statistically extreme
by a clear margin.  In practice, values beyond 2.5σ almost always represent
genuine data quality issues (duplicate entries, unit-of-measure errors) or
truly anomalous one-off events that should not be extrapolated into the
forecast horizon.  The 2.5σ boundary strikes the right balance between
robustness and signal preservation for monthly demand series.

### Why IQR×1.5 (Tukey fences) is better for heavy-tailed distributions

Many SKU-level demand series are far from normally distributed.  They
frequently exhibit a high proportion of zero or near-zero periods
interspersed with occasional large spikes — a pattern common in
intermittent or lumpy demand.  For such series, σ-based clipping is
unreliable because the standard deviation is inflated by the spikes it is
supposed to remove, producing a circular dependency.  The interquartile
range (IQR = Q3 − Q1) is a rank-based statistic unaffected by extreme
values.  Tukey fences at Q1 − 1.5·IQR and Q3 + 1.5·IQR are the
industry-standard choice for detecting outliers in heavy-tailed or
non-symmetric distributions.

**Practical recommendation**: use `method="sigma"` with `threshold=2.5` as
the default for series with a coefficient of variation (CV = std/mean) below
1.0.  Switch to `method="iqr"` with `threshold=1.5` for series with
CV > 1.0 or series with a high proportion of zero periods, as these are
most likely to have heavy-tailed distributions where σ-based clipping is
unreliable.

### Why 2.5σ instead of 1σ?

A 1σ threshold clips ~32% of normally distributed observations, which in
demand planning means removing legitimate spikes from promotions, new
product launches, or seasonal peaks. The 2.5σ default balances noise
removal with signal preservation. For very clean data (no promotions),
lower thresholds (1.5–2.0) may be appropriate. Configure via
`standardization.sigma_threshold` in `config.yaml`.

---

## 2b. Seasonal Parameter Optimisation

### Why m = 12 is fixed

For monthly demand planning, the seasonal period `m = 12` corresponds to
the annual cycle and is one of the most well-established empirical facts in
retail and consumer-goods forecasting (M4 Competition, 2020).  Allowing the
optimiser to search over `m` would conflate model selection with data-frequency
assumptions and would dramatically expand the search space, adding very high-
cardinality trials (e.g. `m = 6, 12, 24`) that are expensive to fit and
unlikely to outperform `m = 12` on monthly series.  Fixing `m = 12` is the
standard approach used in academic benchmarks, commercial forecasting platforms,
and the ``auto.arima`` literature.

### Why P and Q are worth searching

The seasonal AR order `P` and seasonal MA order `Q` control how strongly the
model uses lagged seasonal observations (P) and lagged seasonal forecast errors
(Q) to predict the next period.  In demand planning, these components can
meaningfully improve accuracy for SKUs with strong and regular seasonal patterns
(e.g. a product that consistently sells more in December than in July).  Unlike
varying `m`, searching over `P ∈ [0,2]` and `Q ∈ [0,2]` is computationally
inexpensive: it adds only a 3 × 3 = 9-element sub-grid per trial, and the
complexity constraints `(P+Q) ≤ 3` ensure that over-parameterised seasonal
components are pruned cheaply before fitting.

### Why D ∈ [0, 1] is sufficient

Seasonal differencing with order `D = 1` removes deterministic seasonal trends
(e.g. a steadily growing Q4 peak year-over-year).  `D = 2` would apply
seasonal differencing twice, which is rarely needed for monthly demand data and
substantially increases the minimum series length required for a stable fit.
The search space `D ∈ [0,1]` covers the practically relevant range without
the risk of over-differencing.

### Complexity constraints

Two hard constraints are enforced inside the Optuna objective function before
any model is fitted:

| Constraint | Rationale |
|------------|-----------|
| `(p + q) ≤ 4` | Prevents ARMA overfitting on the non-seasonal component.  An ARIMA(3,d,3) already has six free parameters on top of the differencing order; adding more rarely improves generalisation. |
| `(P + Q) ≤ 3` | Keeps the seasonal component parsimonious.  SARIMA(P=2,D,Q=2) with M=12 requires the model to estimate four lags 12 months apart; beyond `P+Q=3` the fit typically deteriorates on series shorter than 60 months. |

Both constraints return :data:`~sarima_bayes.optimizer.OPTIMIZER_PENALTY`
(``1e6``) immediately, so the TPE sampler learns to steer away from
over-parameterised regions without wasting a full model-fitting evaluation.

---

## 3. SARIMA Model

The forecasting model used is `SARIMA(p, d, q)(P, D, Q, m)` — a Seasonal
ARIMA model that explicitly captures both short-term autocorrelation and
annual seasonal patterns.

| Parameter | Meaning                                                     | Search range |
|-----------|-------------------------------------------------------------|--------------|
| `p`       | Autoregressive order — number of lagged values              | 0 – 3        |
| `d`       | Integration order — number of non-seasonal differences      | 0 – 2        |
| `q`       | Moving-average order — number of lagged forecast errors     | 0 – 3        |
| `P`       | Seasonal AR order — lagged values at multiples of m         | 0 – 2        |
| `D`       | Seasonal differencing order                                 | 0 – 1        |
| `Q`       | Seasonal MA order — lagged seasonal forecast errors         | 0 – 2        |
| `m`       | Seasonal period — **fixed at 12** (monthly annual cycle)    | —            |

The model is implemented via `statsmodels.tsa.statespace.sarimax.SARIMAX`
with `enforce_stationarity=False` and `enforce_invertibility=False`.  These
flags allow the optimiser to explore the full parameter space without hard
failures on numerically challenging combinations; the cost function
naturally penalises unstable fits.

Forecasts are clipped to zero from below (demand cannot be negative).

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
Parallel(n_jobs=-1, backend="loky", verbose=0)(  # Changed from "threading" to "loky" — better for CPU-bound tasks (SARIMA fitting)
    delayed(process_sku)(country, sku, data) for country, sku in combinations
)
```

Using the `loky` process-based backend is better suited for CPU-bound tasks
like SARIMA fitting, as each worker runs in a separate process and avoids
GIL contention.

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
