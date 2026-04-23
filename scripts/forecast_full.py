"""Forecast completo por grupo (SKU, Country) con 4 mejoras:

1. Recorte de ceros iniciales (maneja SKUs con lanzamiento tardio).
2. Comparacion multi-modelo: SARIMA, XGBoost, LightGBM superpuestos.
   (Prophet descartado porque requiere MinGW/cmdstan en Windows.)
3. Bandas probabilisticas P10/P50/P90 via QuantileMLSpec (LightGBM).
4. Suavizado WMA opcional (weighted moving average) - cada modelo se
   optimiza sobre la serie cruda Y la suavizada, y se queda con el
   variant de menor score (logica del pipeline v1.x).

Genera: CSV combinado, metricas, graficas individuales y grid resumen.

Uso:
    python scripts/forecast_full.py --help
    python scripts/forecast_full.py                          # defaults
    python scripts/forecast_full.py --wma-threshold 2.5      # mas conservador
    python scripts/forecast_full.py --n-jobs 8 --n-trials 50 # ajustar runtime
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed

from boa_forecaster import optimize_model
from boa_forecaster.data_loader import load_data
from boa_forecaster.models.lightgbm import LightGBMSpec
from boa_forecaster.models.quantile import QuantileMLSpec
from boa_forecaster.models.sarima import SARIMASpec
from boa_forecaster.models.xgboost import XGBoostSpec
from boa_forecaster.preprocessor import fill_blanks
from boa_forecaster.standardization import weighted_moving_stats_series

warnings.filterwarnings("ignore")

MODEL_STYLES = {
    "SARIMA": {"color": "#d62728", "linestyle": "-"},
    "XGBoost": {"color": "#2ca02c", "linestyle": "--"},
    "LightGBM": {"color": "#9467bd", "linestyle": "-."},
}


def parse_args() -> argparse.Namespace:
    default_jobs = min(15, os.cpu_count() or 4)
    p = argparse.ArgumentParser(
        description="Multi-model forecast (SARIMA + XGBoost + LightGBM + QuantileML) "
        "per (SKU, Country) with WMA smoothing and parallelization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # -- data loading --
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
    p.add_argument("--freq", default="MS", help="Pandas frequency alias (MS, W, D, QS)")
    # -- output --
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/output/full"),
        help="Output directory (CSVs + plots/ subfolder)",
    )
    # -- optimisation --
    p.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Optuna trials per model per variant (raw + smoothed)",
    )
    p.add_argument(
        "--horizon", type=int, default=12, help="Forecast horizon in periods"
    )
    p.add_argument(
        "--min-months",
        type=int,
        default=24,
        help="Minimum observations required (SARIMA m=12 needs >=24)",
    )
    # -- smoothing --
    p.add_argument(
        "--wma-window",
        type=int,
        default=3,
        help="WMA window (neighbours on each side, centered smoother)",
    )
    p.add_argument(
        "--wma-threshold",
        type=float,
        default=1.5,
        help="WMA clipping threshold in sigmas "
        "(lower = more aggressive; 3.5 ~0.04%% clipped, 1.5 ~13%% clipped)",
    )
    # -- parallelism --
    p.add_argument(
        "--n-jobs",
        type=int,
        default=default_jobs,
        help="Parallel workers (joblib loky backend)",
    )
    return p.parse_args()


def _trim_leading_zeros(series: pd.Series) -> pd.Series:
    nonzero = series[series > 0]
    if nonzero.empty:
        return series
    return series.loc[nonzero.index[0] :]


def _smooth_wma(series: pd.Series, window: int, threshold: float) -> pd.Series:
    """Aplica WMA clipping punto por punto y devuelve una nueva Series con mismo indice."""
    _, _, clipped = weighted_moving_stats_series(
        series.values,
        window_size=window,
        threshold=threshold,
    )
    return pd.Series(clipped, index=series.index, dtype=float)


def _fit_point(series: pd.Series, spec, n_trials: int, horizon: int):
    result = optimize_model(series, spec, n_calls=n_trials)
    forecaster = spec.build_forecaster(result.best_params)
    return forecaster(series).iloc[:horizon], float(result.best_score)


def _fit_point_best_variant(
    raw: pd.Series, smoothed: pd.Series, spec_factory, n_trials: int, horizon: int
):
    """Corre el modelo sobre raw y smoothed, devuelve el variant con mejor score."""
    raw_fcst, raw_score = _fit_point(raw, spec_factory(), n_trials, horizon)
    sm_fcst, sm_score = _fit_point(smoothed, spec_factory(), n_trials, horizon)
    if sm_score < raw_score:
        return sm_fcst, sm_score, "smoothed", raw_score
    return raw_fcst, raw_score, "raw", sm_score


def _fit_quantile(series: pd.Series, n_trials: int, horizon: int):
    spec = QuantileMLSpec(base="lightgbm", quantiles=(0.1, 0.5, 0.9))
    result = optimize_model(series, spec, n_calls=n_trials)
    qfcster = spec.build_quantile_forecaster(result.best_params)
    qf = qfcster(series)
    return (
        qf.lower.iloc[:horizon],
        qf.median.iloc[:horizon],
        qf.upper.iloc[:horizon],
        float(result.best_score),
    )


def _fit_quantile_best_variant(
    raw: pd.Series, smoothed: pd.Series, n_trials: int, horizon: int
):
    raw_res = _fit_quantile(raw, n_trials, horizon)
    sm_res = _fit_quantile(smoothed, n_trials, horizon)
    if sm_res[3] < raw_res[3]:
        return (*sm_res, "smoothed", raw_res[3])
    return (*raw_res, "raw", sm_res[3])


def _plot_group(
    ax,
    hist: pd.Series,
    forecasts: dict,
    quantile: tuple | None,
    title: str,
    smoothed: pd.Series | None = None,
    wma_threshold: float = 1.5,
) -> None:
    ax.plot(hist.index, hist.values, color="#1f77b4", linewidth=1.4, label="Historico")
    if smoothed is not None:
        ax.plot(
            smoothed.index,
            smoothed.values,
            color="#7f7f7f",
            linewidth=1.0,
            linestyle=":",
            alpha=0.7,
            label=f"WMA {wma_threshold}sigma",
        )

    if quantile is not None:
        lower, median, upper = quantile
        ax.fill_between(
            lower.index,
            lower.values,
            upper.values,
            color="#ff7f0e",
            alpha=0.18,
            label="P10-P90 (QuantileML)",
        )
        ax.plot(
            median.index,
            median.values,
            color="#ff7f0e",
            linewidth=1.3,
            linestyle=":",
            label="P50 (QuantileML)",
        )

    for name, fcst in forecasts.items():
        style = MODEL_STYLES[name]
        ax.plot(
            fcst.index,
            fcst.values,
            linewidth=1.5,
            marker="o",
            markersize=2.5,
            color=style["color"],
            linestyle=style["linestyle"],
            label=name,
        )
        if len(hist):
            ax.plot(
                [hist.index[-1], fcst.index[0]],
                [hist.values[-1], fcst.values[0]],
                color=style["color"],
                linewidth=0.7,
                alpha=0.4,
            )

    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", labelsize=7, rotation=30)
    ax.tick_params(axis="y", labelsize=7)


def _process_group(
    sku, country, sub, freq, min_months, wma_window, wma_threshold, n_trials, horizon
):
    import warnings as _w

    _w.filterwarnings("ignore")

    raw = sub.set_index("Date")["CS"].sort_index().asfreq(freq).astype(float)
    trimmed = _trim_leading_zeros(raw)
    n_obs = int(trimmed.notna().sum())
    n_trimmed = len(raw) - len(trimmed)
    tag = f"SKU={sku} {country}"
    row_base = {
        "SKU": sku,
        "Country": country,
        "n_obs": n_obs,
        "n_trimmed_zeros": n_trimmed,
    }

    if n_obs < min_months:
        print(f"[{tag}] SKIP n_obs={n_obs} < {min_months}", flush=True)
        return {
            "results_row": {**row_base, "status": "skipped_too_short"},
            "forecasts_rows": [],
            "payload_key": None,
            "payload": None,
        }

    smoothed = _smooth_wma(trimmed, wma_window, wma_threshold)

    forecasts: dict[str, pd.Series] = {}
    scores: dict[str, float] = {}
    variants: dict[str, str] = {}

    for spec_name, spec_factory in [
        ("SARIMA", SARIMASpec),
        ("XGBoost", XGBoostSpec),
        ("LightGBM", LightGBMSpec),
    ]:
        try:
            fcst, score, variant, _ = _fit_point_best_variant(
                trimmed,
                smoothed,
                spec_factory,
                n_trials,
                horizon,
            )
            forecasts[spec_name] = fcst
            scores[spec_name] = score
            variants[spec_name] = variant
        except Exception as exc:
            print(f"[{tag}] {spec_name} ERROR: {type(exc).__name__}: {exc}", flush=True)
            scores[spec_name] = None
            variants[spec_name] = None

    quantile_payload = None
    try:
        lower, median, upper, q_score, q_variant, _ = _fit_quantile_best_variant(
            trimmed,
            smoothed,
            n_trials,
            horizon,
        )
        quantile_payload = (lower, median, upper)
        scores["QuantileML_P50"] = q_score
        variants["QuantileML_P50"] = q_variant
    except Exception as exc:
        print(f"[{tag}] QuantileML ERROR: {type(exc).__name__}: {exc}", flush=True)
        scores["QuantileML_P50"] = None
        variants["QuantileML_P50"] = None

    score_str = "  ".join(
        f"{k}={v:.2f}({variants[k][:3]})" for k, v in scores.items() if v is not None
    )
    print(f"[{tag}] n={n_obs} trimmed={n_trimmed}  {score_str}", flush=True)

    fcst_rows: list[dict] = []
    for model_name, fcst in forecasts.items():
        for dt, val in fcst.items():
            fcst_rows.append(
                {
                    "SKU": sku,
                    "Country": country,
                    "Date": dt,
                    "model": model_name,
                    "forecast": float(val),
                }
            )
    if quantile_payload is not None:
        lower, median, upper = quantile_payload
        for dt in median.index:
            fcst_rows.append(
                {
                    "SKU": sku,
                    "Country": country,
                    "Date": dt,
                    "model": "QuantileML_P10",
                    "forecast": float(lower.loc[dt]),
                }
            )
            fcst_rows.append(
                {
                    "SKU": sku,
                    "Country": country,
                    "Date": dt,
                    "model": "QuantileML_P50",
                    "forecast": float(median.loc[dt]),
                }
            )
            fcst_rows.append(
                {
                    "SKU": sku,
                    "Country": country,
                    "Date": dt,
                    "model": "QuantileML_P90",
                    "forecast": float(upper.loc[dt]),
                }
            )

    return {
        "results_row": {
            **row_base,
            "status": "ok",
            **{f"score_{k}": v for k, v in scores.items()},
            **{f"variant_{k}": v for k, v in variants.items()},
        },
        "forecasts_rows": fcst_rows,
        "payload_key": (sku, country),
        "payload": {
            "hist": trimmed,
            "smoothed": smoothed,
            "forecasts": forecasts,
            "quantile": quantile_payload,
            "scores": scores,
            "variants": variants,
        },
    }


def main() -> None:
    import time

    args = parse_args()
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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
    print(
        f"Procesando {len(groups)} grupos con SARIMA + XGBoost + LightGBM + QuantileML"
    )
    print(
        f"  WMA threshold={args.wma_threshold}  |  n_trials={args.n_trials}  |  n_jobs={args.n_jobs}"
    )
    print(f"  cpu_count={os.cpu_count()}")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
        delayed(_process_group)(
            sku,
            country,
            sub,
            args.freq,
            args.min_months,
            args.wma_window,
            args.wma_threshold,
            args.n_trials,
            args.horizon,
        )
        for (sku, country), sub in groups
    )
    elapsed = time.time() - t0
    print(f"\n--- Paralelizacion terminada en {elapsed:.1f}s ---")

    results_rows = [r["results_row"] for r in results]
    forecasts_rows = [row for r in results for row in r["forecasts_rows"]]
    group_payloads = {
        r["payload_key"]: r["payload"] for r in results if r["payload"] is not None
    }

    # -- Save tables --
    pd.DataFrame(results_rows).to_csv(args.output_dir / "metrics_full.csv", index=False)
    if forecasts_rows:
        pd.DataFrame(forecasts_rows).to_csv(
            args.output_dir / "forecasts_full.csv", index=False
        )

    # -- Individual plots --
    for (sku, country), payload in group_payloads.items():
        fig, ax = plt.subplots(figsize=(10, 4.2))
        best_model = min(
            (
                k
                for k in ("SARIMA", "XGBoost", "LightGBM")
                if payload["scores"].get(k) is not None
            ),
            key=lambda k: payload["scores"][k],
            default=None,
        )
        title_suffix = (
            f"  best={best_model} ({payload['scores'][best_model]:.2f}, "
            f"{payload['variants'].get(best_model,'?')})"
            if best_model
            else ""
        )
        _plot_group(
            ax,
            payload["hist"],
            payload["forecasts"],
            payload["quantile"],
            f"SKU {sku} - {country}{title_suffix}",
            smoothed=payload["smoothed"],
            wma_threshold=args.wma_threshold,
        )
        ax.set_xlabel("Fecha")
        ax.set_ylabel("CS")
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plots_dir / f"SKU_{sku}_{country.replace(' ', '_')}.png", dpi=120)
        plt.close(fig)

    # -- Grid --
    if group_payloads:
        n = len(group_payloads)
        ncols = 3
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 3.2))
        axes_flat = axes.flatten() if nrows > 1 else list(axes)
        for ax, ((sku, country), payload) in zip(axes_flat, group_payloads.items()):
            _plot_group(
                ax,
                payload["hist"],
                payload["forecasts"],
                payload["quantile"],
                f"SKU {sku} - {country}",
                smoothed=payload["smoothed"],
                wma_threshold=args.wma_threshold,
            )
        for ax in axes_flat[len(group_payloads) :]:
            ax.axis("off")
        # legend once
        handles, labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=6,
            fontsize=9,
            bbox_to_anchor=(0.5, -0.01),
        )
        fig.suptitle(
            "Historico + SARIMA / XGBoost / LightGBM / QuantileML (P10-P90) por SKU x Country",
            fontsize=12,
            y=1.00,
        )
        fig.tight_layout()
        fig.savefig(
            args.output_dir / "all_groups_grid.png", dpi=130, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"\n--- Outputs en {args.output_dir} ---")
    print("  forecasts_full.csv   (long format: Date, SKU, Country, model, forecast)")
    print("  metrics_full.csv     (scores por modelo y grupo)")
    print(f"  plots/               ({len(group_payloads)} PNGs individuales)")
    print("  all_groups_grid.png  (vista resumen)")


if __name__ == "__main__":
    main()
