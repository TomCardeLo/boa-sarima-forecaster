"""Global configuration constants for the boa-forecaster library.

All values defined here are defaults that can be overridden at call sites or
via ``config.example.yaml``.  Import individual constants in other modules
rather than importing the module as a whole to keep namespaces clean.
"""

# ── Data Loading ──────────────────────────────────────────────────────────────
# The input Excel workbook has two meta-header rows above the actual column
# names.  Row indices 0 and 1 are skipped; row 2 becomes the header.
DEFAULT_SKIP_ROWS: int = 2

# Name of the worksheet that contains the time-series sales table.
DEFAULT_SHEET_NAME: str = "Data"

# Date format used in the raw "Date" column of the Excel source.
# Example cell value: "202201"  →  January 2022.
DEFAULT_DATE_FORMAT: str = "%Y%m"

# ── Date Range ────────────────────────────────────────────────────────────────
# Start and end dates used when filling missing monthly periods.
# Both bounds are inclusive; frequency is month-start ("MS").
DEFAULT_START_DATE: str = "2022-01-01"
DEFAULT_END_DATE: str = "2026-01-01"
DEFAULT_FREQ: str = "MS"  # Pandas offset alias for month start

# ── ARIMA Search Space ────────────────────────────────────────────────────────
# Inclusive integer ranges explored by the Bayesian optimiser for each order.
#   p – autoregressive order  (how many lagged observations to include)
#   d – degree of differencing (how many times to difference for stationarity)
#   q – moving-average order   (how many lagged forecast errors to include)
#   P – seasonal AR order
#   D – seasonal differencing order
#   Q – seasonal MA order
DEFAULT_P_RANGE: tuple = (0, 3)
DEFAULT_D_RANGE: tuple = (0, 2)
DEFAULT_Q_RANGE: tuple = (0, 3)
DEFAULT_P_SEASONAL_RANGE: tuple = (0, 2)
DEFAULT_D_SEASONAL_RANGE: tuple = (0, 1)
DEFAULT_Q_SEASONAL_RANGE: tuple = (0, 2)

# ── Seasonal Period ───────────────────────────────────────────────────────────
# Fixed seasonal period (m) — NOT optimised.  12 for monthly data (annual cycle).
# Configurable via config.yaml model.sarima.seasonal_period.
DEFAULT_SEASONAL_PERIOD: int = 12

# ── Optimisation Budget ───────────────────────────────────────────────────────
# Total number of Optuna trials (model evaluations) per time series.
# Increasing this value improves solution quality at the cost of runtime.
DEFAULT_N_CALLS: int = 30

# ── Outlier Clipping ──────────────────────────────────────────────────────────
# Method and threshold used by clip_outliers in standardization.py.
#   method    – "sigma" (mean ± threshold*std) or "iqr" (Tukey fences)
#   threshold – multiplier for the clipping boundary.
#               For "sigma": 2.5 clips ~1.2% of a normal distribution.
#               For "iqr":   1.5 is the classic Tukey fence.
# Configurable via config.yaml standardization.clipping.
DEFAULT_CLIP_METHOD: str = "sigma"
DEFAULT_CLIP_THRESHOLD: float = 2.5

# Outlier clipping
OUTLIER_SIGMA: float = (
    2.5  # ±2.5σ (previously 1.0; 1σ is too aggressive for promo-driven demand)
)

# ── Metric Composition ────────────────────────────────────────────────────────
# Default weighted objective minimised by the Bayesian optimiser.
# Each entry maps to a metric in boa_forecaster.metrics.METRIC_REGISTRY.
# Weights do not need to sum to 1 but it is recommended for interpretability.
# Configurable via config.yaml metrics.components.
DEFAULT_METRIC_COMPONENTS: list = [
    {"metric": "smape", "weight": 0.7},
    {"metric": "rmsle", "weight": 0.3},
]

# ── Optimiser Penalty ─────────────────────────────────────────────────────────
# Score returned when a model evaluation fails (e.g. SARIMAX convergence error).
# Large enough to guide TPE away from infeasible parameter regions.
OPTIMIZER_PENALTY: float = 1e6

# ── Random Forest Defaults ────────────────────────────────────────────────────
# Sensible starting-point hyperparameters for RandomForestSpec warm starts.
# Used by the first warm-start trial before TPE takes over.
RF_DEFAULT_N_ESTIMATORS: int = 100
RF_DEFAULT_MAX_DEPTH: int = 10

# ── XGBoost Defaults ──────────────────────────────────────────────────────────
# Sensible starting-point hyperparameters for XGBoostSpec warm starts.
# Used by the first warm-start trial before TPE takes over.
XGB_DEFAULT_N_ESTIMATORS: int = 100
XGB_DEFAULT_MAX_DEPTH: int = 6

# ── LightGBM Defaults ─────────────────────────────────────────────────────────
# Sensible starting-point hyperparameters for LightGBMSpec warm starts.
# num_leaves is the primary complexity control in LightGBM (leaf-wise growth).
LGBM_DEFAULT_N_ESTIMATORS: int = 100
LGBM_DEFAULT_NUM_LEAVES: int = 31
