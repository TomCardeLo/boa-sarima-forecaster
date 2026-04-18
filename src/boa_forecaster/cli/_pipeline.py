"""Shared helpers for CLI subcommands.

Centralises the "load YAML → read Excel → aggregate to series" path so
``run``, ``validate``, and ``compare`` stay thin.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from boa_forecaster.config_schema import BoaConfig
from boa_forecaster.data_loader import load_data
from boa_forecaster.models import MODEL_REGISTRY, get_model_spec
from boa_forecaster.models.base import ModelSpec
from boa_forecaster.preprocessor import fill_blanks

logger = logging.getLogger(__name__)


def load_series_from_config(cfg: BoaConfig) -> pd.Series:
    """Load input data and aggregate it into a single univariate series.

    The CLI runs a single-series pipeline: rows are summed across SKU /
    Country so ``optimize_model`` gets a scalar target.  Multi-group runs
    should use :func:`boa_forecaster.benchmarks.run_model_comparison`
    directly from Python.

    Raises:
        ValueError: When ``cfg.data`` is missing.
        FileNotFoundError: When ``cfg.data.input_path`` does not exist.
    """
    if cfg.data is None:
        raise ValueError("config is missing the 'data' block — cannot load input")

    data_path = Path(cfg.data.input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input data file not found: {data_path}")

    df = load_data(
        str(data_path),
        sheet_name=cfg.data.sheet_name,
        skip_rows=cfg.data.skip_rows,
        date_format=cfg.data.date_format,
    )

    end_date = cfg.data.end_date or df["Date"].max().strftime("%Y-%m-%d")
    df = fill_blanks(
        df,
        date_col="Date",
        group_cols=["SKU", "Country"],
        value_col="CS",
        end_date=end_date,
        freq=cfg.data.freq,
    )

    agg = df.groupby("Date", as_index=True)["CS"].sum().sort_index()
    agg.index = pd.DatetimeIndex(agg.index, freq=cfg.data.freq)
    return agg.astype(float)


def build_active_spec(cfg: BoaConfig) -> ModelSpec:
    """Instantiate the ``ModelSpec`` named in ``cfg.models.active``.

    Raises:
        KeyError: The active model is not registered (e.g. missing extra).
    """
    name = cfg.models.active
    if name not in MODEL_REGISTRY or MODEL_REGISTRY[name] is None:
        raise KeyError(
            f"Active model '{name}' is not registered. "
            f"Available: {sorted(n for n, c in MODEL_REGISTRY.items() if c is not None)}"
        )
    return get_model_spec(name)


def ensure_output_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if needed and return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def metric_components_as_dicts(cfg: BoaConfig) -> list[dict]:
    """Convert ``cfg.metrics.components`` into the list-of-dicts the
    optimiser and metric builder expect."""
    return [{"metric": c.metric, "weight": c.weight} for c in cfg.metrics.components]


def summarise_folds(folds: pd.DataFrame, metric_cols: Iterable[str]) -> pd.DataFrame:
    """Collapse a walk-forward fold table to a one-row mean/std summary."""
    metric_cols = list(metric_cols)
    if folds.empty:
        return pd.DataFrame(columns=[f"{c}_mean" for c in metric_cols])
    row: dict[str, float] = {}
    for col in metric_cols:
        if col in folds.columns:
            row[f"{col}_mean"] = float(folds[col].mean())
            row[f"{col}_std"] = float(folds[col].std(ddof=0))
    return pd.DataFrame([row])


def write_forecast_outputs(
    out_dir: Path,
    forecast: pd.Series,
    best_params: dict,
    best_score: float,
    metrics_df: pd.DataFrame | None = None,
) -> None:
    """Persist ``forecast.csv``, ``params.json``, and (optional) ``metrics.csv``.

    ``plot.png`` is skipped when matplotlib is not installed, so the core
    dep set stays lean.
    """
    import json

    forecast_df = forecast.rename("forecast").to_frame()
    forecast_df.index.name = "date"
    forecast_df.to_csv(out_dir / "forecast.csv")

    with (out_dir / "params.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {"best_params": best_params, "best_score": float(best_score)},
            fh,
            indent=2,
            default=str,
        )

    if metrics_df is not None:
        metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        forecast.plot(ax=ax, title="Forecast")
        fig.tight_layout()
        fig.savefig(out_dir / "plot.png", dpi=120)
        plt.close(fig)
    except ImportError:
        logger.info("matplotlib not installed — skipping plot.png")
