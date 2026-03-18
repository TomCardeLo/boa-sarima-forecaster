"""Generate example forecast plot for the README.

Produces ``docs/img/forecast_example.png`` showing:
- Training series (all historical observations)
- Last 24 months actuals highlighted
- SARIMA+BO point forecast
- 80% and 95% confidence interval bands

Run from the repository root::

    python scripts/generate_plots.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sarima_bayes.model import pred_arima
from sarima_bayes.optimizer import optimize_arima

# ---------------------------------------------------------------------------
# Reproducible synthetic series
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_total = 60  # 5 years of monthly data

# Base trend + annual seasonality + noise
t = np.arange(n_total)
seasonal = 30 * np.sin(2 * np.pi * t / 12)
trend = 0.3 * t
noise = rng.normal(0, 8, n_total)
series_values = 100 + trend + seasonal + noise
series_values = np.maximum(series_values, 0)

dates = pd.date_range("2020-01-01", periods=n_total, freq="MS")
full_series = pd.Series(series_values, index=dates)

# ---------------------------------------------------------------------------
# Fit SARIMA+BO on the first 48 months, forecast the remaining 12
# ---------------------------------------------------------------------------
train = full_series.iloc[:48]
actuals_last24 = full_series.iloc[24:48]

best_params, best_score = optimize_arima(
    train.values,
    n_calls=30,
    m=12,
)
print(f"Best params: {best_params}  |  Score: {best_score:.4f}")

bd = train.reset_index()
bd.columns = ["Date", "CS"]

forecast_df, conf_int, order, s_order, _ = pred_arima(
    bd,
    col_x="Date",
    col_y="CS",
    order=(best_params["p"], best_params["d"], best_params["q"]),
    s_order=(best_params["P"], best_params["D"], best_params["Q"], best_params["m"]),
    n_per=12,
    alpha=0.05,
)

_, ci80, _, _, _ = pred_arima(
    bd,
    col_x="Date",
    col_y="CS",
    order=(best_params["p"], best_params["d"], best_params["q"]),
    s_order=(best_params["P"], best_params["D"], best_params["Q"], best_params["m"]),
    n_per=12,
    alpha=0.20,
)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)

# Full training series (grey)
ax.plot(
    train.index,
    train.values,
    color="#aaaaaa",
    linewidth=1.0,
    label="Training history",
)

# Last 24 months actuals (dark blue)
ax.plot(
    actuals_last24.index,
    actuals_last24.values,
    color="#1f4e79",
    linewidth=2.0,
    label="Last 24 months (actuals)",
)

# Forecast (orange)
if forecast_df is not None and not forecast_df.empty:
    ax.plot(
        forecast_df.index,
        forecast_df["Predictions"].clip(lower=0),
        color="#e06c00",
        linewidth=2.0,
        linestyle="--",
        label=f"SARIMA+BO forecast  SARIMA{order}{s_order}",
    )

    # 95% CI band
    if conf_int is not None:
        ax.fill_between(
            conf_int.index,
            conf_int.iloc[:, 0].clip(lower=0),
            conf_int.iloc[:, 1].clip(lower=0),
            alpha=0.15,
            color="#e06c00",
            label="95% CI",
        )

    # 80% CI band (narrower, more opaque)
    if ci80 is not None:
        ax.fill_between(
            ci80.index,
            ci80.iloc[:, 0].clip(lower=0),
            ci80.iloc[:, 1].clip(lower=0),
            alpha=0.30,
            color="#e06c00",
            label="80% CI",
        )

# Vertical line at forecast origin
ax.axvline(
    x=train.index[-1],
    color="#555555",
    linewidth=1.0,
    linestyle=":",
)

ax.set_title(
    "SARIMA + Bayesian Optimization — Forecast vs Actuals",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Date")
ax.set_ylabel("Demand (units)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
out_path = Path(__file__).parent.parent / "docs" / "img" / "forecast_example.png"
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")
