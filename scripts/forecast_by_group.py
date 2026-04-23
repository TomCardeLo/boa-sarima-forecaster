"""Forecast SARIMA por cada combinacion (SKU, Country) del archivo de ventas.

Corre SARIMA + Optuna TPE por cada grupo y concatena los resultados en
un unico CSV y Excel. Grupos con menos de ``--min-months`` observaciones
se omiten (SARIMA con m=12 necesita al menos ~24 meses).

Uso:
    python scripts/forecast_by_group.py --help
    python scripts/forecast_by_group.py                   # defaults
    python scripts/forecast_by_group.py --n-trials 50     # mas trials
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd

from boa_forecaster import optimize_model
from boa_forecaster.data_loader import load_data
from boa_forecaster.models.sarima import SARIMASpec
from boa_forecaster.preprocessor import fill_blanks

warnings.filterwarnings("ignore")  # statsmodels convergence noise
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-(SKU, Country) SARIMA forecast with Optuna TPE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        default="data/input/Base ventas KCP.xlsx",
        help="Path to input Excel workbook",
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
    p.add_argument(
        "--freq",
        default="MS",
        help="Pandas frequency alias (MS monthly, W weekly, D daily, QS quarterly)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/by_group"),
        help="Output directory for CSV/Excel files",
    )
    p.add_argument("--n-trials", type=int, default=30, help="Optuna trials per group")
    p.add_argument(
        "--horizon", type=int, default=12, help="Forecast horizon in periods"
    )
    p.add_argument(
        "--min-months",
        type=int,
        default=24,
        help="Minimum observations required (SARIMA m=12 needs >=24)",
    )
    return p.parse_args()


def _forecast_group(
    series: pd.Series, n_trials: int, horizon: int
) -> tuple[pd.Series, dict, float]:
    """Optimizar y generar forecast para una serie individual."""
    spec = SARIMASpec()
    result = optimize_model(series, spec, n_calls=n_trials)
    forecaster = spec.build_forecaster(result.best_params)
    forecast = forecaster(series).iloc[:horizon]
    return forecast, result.best_params, float(result.best_score)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    groups = list(df.groupby(["SKU", "Country"], as_index=False))
    print(f"Procesando {len(groups)} grupos (SKU x Country)...")

    all_forecasts: list[pd.DataFrame] = []
    metrics_rows: list[dict] = []

    for i, ((sku, country), sub) in enumerate(groups, 1):
        series = (
            sub.set_index("Date")["CS"].sort_index().asfreq(args.freq).astype(float)
        )
        n_obs = int(series.notna().sum())

        if n_obs < args.min_months:
            print(
                f"[{i}/{len(groups)}] SKU={sku} {country}  -> SKIP ({n_obs} meses < {args.min_months})"
            )
            metrics_rows.append(
                {
                    "SKU": sku,
                    "Country": country,
                    "n_obs": n_obs,
                    "status": "skipped_too_short",
                    "best_score": None,
                }
            )
            continue

        try:
            forecast, params, score = _forecast_group(
                series, args.n_trials, args.horizon
            )
            print(
                f"[{i}/{len(groups)}] SKU={sku} {country}  -> score={score:.3f}  n={n_obs}"
            )

            fdf = forecast.rename("forecast").to_frame().reset_index()
            fdf.columns = ["Date", "forecast"]
            fdf["SKU"] = sku
            fdf["Country"] = country
            all_forecasts.append(fdf)

            metrics_rows.append(
                {
                    "SKU": sku,
                    "Country": country,
                    "n_obs": n_obs,
                    "status": "ok",
                    "best_score": score,
                    **{f"p_{k}": v for k, v in params.items()},
                }
            )
        except Exception as exc:
            print(f"[{i}/{len(groups)}] SKU={sku} {country}  -> ERROR: {exc}")
            metrics_rows.append(
                {
                    "SKU": sku,
                    "Country": country,
                    "n_obs": n_obs,
                    "status": f"error:{type(exc).__name__}",
                    "best_score": None,
                }
            )

    if all_forecasts:
        forecast_df = pd.concat(all_forecasts, ignore_index=True)
        forecast_df = forecast_df[["SKU", "Country", "Date", "forecast"]]
        forecast_df.to_csv(args.output_dir / "forecast_by_group.csv", index=False)
        forecast_df.to_excel(args.output_dir / "forecast_by_group.xlsx", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(args.output_dir / "metrics_by_group.csv", index=False)

    print("\n--- Resumen ---")
    print(f"Total grupos: {len(groups)}")
    print(f"OK:      {(metrics_df['status'] == 'ok').sum()}")
    print(f"Skipped: {(metrics_df['status'] == 'skipped_too_short').sum()}")
    print(f"Error:   {metrics_df['status'].str.startswith('error').sum()}")
    print(
        f"\nArchivos:"
        f"\n  {args.output_dir / 'forecast_by_group.csv'}"
        f"\n  {args.output_dir / 'forecast_by_group.xlsx'}"
        f"\n  {args.output_dir / 'metrics_by_group.csv'}"
    )


if __name__ == "__main__":
    main()
