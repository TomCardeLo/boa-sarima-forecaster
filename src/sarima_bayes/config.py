"""Global configuration constants for the sarima-bayes forecasting library.

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
OUTLIER_SIGMA: float = 2.5  # ±2.5σ (previously 1.0; 1σ is too aggressive for promo-driven demand)