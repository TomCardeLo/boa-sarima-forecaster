"""Genera graficas (historico + forecast) por cada par (SKU, Country).

- Un PNG individual por grupo en ``<output-dir>/plots/``
- Un grid resumen ``all_groups_grid.png`` con todos los grupos
- Requiere haber corrido ``forecast_by_group.py`` antes

Uso:
    python scripts/plot_forecasts_by_group.py --help
    python scripts/plot_forecasts_by_group.py             # defaults
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from boa_forecaster.data_loader import load_data
from boa_forecaster.preprocessor import fill_blanks

warnings.filterwarnings("ignore")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot historical + forecast per (SKU, Country) group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="data/input/Base ventas KCP.xlsx",
        help="Path to input Excel workbook (historicals)",
    )
    p.add_argument("--sheet", default="Data", help="Worksheet name")
    p.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Header rows to skip before column-name row",
    )
    p.add_argument(
        "--date-format", default="%Y%m", help="strptime format for Date column"
    )
    p.add_argument("--freq", default="MS", help="Pandas frequency alias (MS, W, D, QS)")
    p.add_argument(
        "--forecast-csv",
        type=Path,
        default=Path("data/output/by_group/forecast_by_group.csv"),
        help="CSV produced by forecast_by_group.py",
    )
    p.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("data/output/by_group/metrics_by_group.csv"),
        help="Metrics CSV produced by forecast_by_group.py",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/by_group/plots"),
        help="Directory for individual PNGs",
    )
    return p.parse_args()


def _plot_single(
    ax, hist: pd.Series, fcst: pd.Series, title: str, score: float | None
) -> None:
    ax.plot(hist.index, hist.values, color="#1f77b4", linewidth=1.5, label="Histórico")
    ax.plot(
        fcst.index,
        fcst.values,
        color="#d62728",
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Forecast",
    )
    # conector visual entre último punto histórico y primer forecast
    if len(hist) and len(fcst):
        ax.plot(
            [hist.index[-1], fcst.index[0]],
            [hist.values[-1], fcst.values[0]],
            color="#d62728",
            linewidth=1.0,
            linestyle=":",
            alpha=0.6,
        )
    ax.axvspan(fcst.index[0], fcst.index[-1], color="#d62728", alpha=0.05)
    score_txt = f"  (score={score:.2f})" if score is not None else ""
    ax.set_title(f"{title}{score_txt}", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Históricos
    df = load_data(
        args.input,
        sheet_name=args.sheet,
        skip_rows=args.skip_rows,
        date_format=args.date_format,
    )
    end_date = df["Date"].max().strftime("%Y-%m-%d")
    df = fill_blanks(
        df,
        date_col="Date",
        group_cols=["SKU", "Country"],
        value_col="CS",
        end_date=end_date,
        freq=args.freq,
    )

    # Forecasts + métricas
    fcst_df = pd.read_csv(args.forecast_csv, parse_dates=["Date"])
    metrics = pd.read_csv(args.metrics_csv).set_index(["SKU", "Country"])

    groups = sorted(fcst_df.groupby(["SKU", "Country"]).groups.keys())
    print(f"Generando {len(groups)} graficas individuales + 1 grid...")

    # -- individual PNGs --
    for sku, country in groups:
        hist = (
            df.query("SKU == @sku and Country == @country")
            .set_index("Date")["CS"]
            .sort_index()
            .asfreq(args.freq)
            .astype(float)
        )
        fcst = (
            fcst_df.query("SKU == @sku and Country == @country")
            .set_index("Date")["forecast"]
            .sort_index()
        )
        try:
            score = float(metrics.loc[(sku, country), "best_score"])
        except Exception:
            score = None

        fig, ax = plt.subplots(figsize=(9, 4))
        _plot_single(ax, hist, fcst, f"SKU {sku} - {country}", score)
        ax.set_xlabel("Fecha")
        ax.set_ylabel("CS")
        ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        out = args.output_dir / f"SKU_{sku}_{country.replace(' ', '_')}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)

    # -- grid resumen --
    n = len(groups)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 2.8))
    axes = axes.flatten() if nrows > 1 else [axes] if ncols == 1 else axes

    for ax, (sku, country) in zip(axes, groups):
        hist = (
            df.query("SKU == @sku and Country == @country")
            .set_index("Date")["CS"]
            .sort_index()
            .asfreq(args.freq)
            .astype(float)
        )
        fcst = (
            fcst_df.query("SKU == @sku and Country == @country")
            .set_index("Date")["forecast"]
            .sort_index()
        )
        try:
            score = float(metrics.loc[(sku, country), "best_score"])
        except Exception:
            score = None
        _plot_single(ax, hist, fcst, f"SKU {sku} - {country}", score)

    # vaciar ejes sobrantes
    for ax in list(axes)[len(groups) :]:
        ax.axis("off")

    fig.suptitle("Historico + Forecast SARIMA por SKU x Country", fontsize=12, y=1.00)
    fig.tight_layout()
    grid_path = args.output_dir.parent / "all_groups_grid.png"
    fig.savefig(grid_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGraficas individuales: {args.output_dir} ({len(groups)} PNGs)")
    print(f"Grid resumen:          {grid_path}")


if __name__ == "__main__":
    main()
