"""Generate example plots for the README.

Produces:
  docs/img/forecast_vs_actuals.png  — historical series + 12-month forecast
  docs/img/model_comparison.png     — horizontal bar chart of sMAPE by model

Values are based on the synthetic demo data from the README results table.
Run from the repository root::

    python scripts/generate_readme_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent.parent / "docs" / "img"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------------------
# Plot 1: Forecast vs Actuals
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n_hist = 48
n_fore = 12

t = np.arange(n_hist + n_fore)
seasonal = 30 * np.sin(2 * np.pi * t / 12)
trend = 0.3 * t
noise = rng.normal(0, 8, n_hist + n_fore)
values = np.maximum(100 + trend + seasonal + noise, 0)

dates_hist = pd.date_range("2020-01-01", periods=n_hist, freq="MS")
dates_fore = pd.date_range(
    dates_hist[-1] + pd.DateOffset(months=1), periods=n_fore, freq="MS"
)

hist = pd.Series(values[:n_hist], index=dates_hist)
fore_mean = values[n_hist:]

# Simulate confidence bands (±1σ / ±1.65σ around forecast)
sigma = np.linspace(6, 14, n_fore)
ci95_lo = np.maximum(fore_mean - 1.96 * sigma, 0)
ci95_hi = fore_mean + 1.96 * sigma
ci80_lo = np.maximum(fore_mean - 1.28 * sigma, 0)
ci80_hi = fore_mean + 1.28 * sigma

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

ax.plot(
    hist.index,
    hist.values,
    color="#1f4e79",
    linewidth=1.8,
    label="Historical (actuals)",
)
ax.plot(
    dates_fore,
    fore_mean,
    color="#e06c00",
    linewidth=2.0,
    linestyle="--",
    label="SARIMA+BO forecast",
)
ax.fill_between(
    dates_fore, ci95_lo, ci95_hi, color="#e06c00", alpha=0.12, label="95% CI"
)
ax.fill_between(
    dates_fore, ci80_lo, ci80_hi, color="#e06c00", alpha=0.25, label="80% CI"
)
ax.axvline(x=dates_hist[-1], color="#888888", linewidth=1.0, linestyle=":")

ax.set_title(
    "Forecast vs Actuals — SARIMA + Bayesian Optimization",
    fontsize=13,
    fontweight="bold",
)
ax.set_xlabel("Date")
ax.set_ylabel("Demand (units)")
ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()

out1 = OUT_DIR / "forecast_vs_actuals.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Model comparison (sMAPE)
# ---------------------------------------------------------------------------
models = ["Seasonal Naïve", "ETS", "Random Forest", "XGBoost", "LightGBM", "SARIMA+BO"]
smape = [14.7, 10.2, 9.1, 8.7, 8.6, 8.4]
colors = ["#cccccc", "#aaaaaa", "#5b9bd5", "#4472c4", "#2e75b6", "#1f4e79"]
naive_value = 14.7

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

bars = ax.barh(models, smape, color=colors, edgecolor="white", height=0.6)

# Value labels
for bar, val in zip(bars, smape):
    ax.text(
        val + 0.15,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        fontsize=9,
    )

# Naïve reference line
ax.axvline(
    x=naive_value,
    color="#e06c00",
    linewidth=1.5,
    linestyle="--",
    label=f"Seasonal Naïve ({naive_value}%)",
)

ax.set_xlabel("sMAPE (%)")
ax.set_title(
    "Model Comparison — sMAPE (lower is better)", fontsize=13, fontweight="bold"
)
ax.set_xlim(0, naive_value + 3)
ax.legend(fontsize=9)
plt.tight_layout()

out2 = OUT_DIR / "model_comparison.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
