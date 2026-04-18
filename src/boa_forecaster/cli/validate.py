"""``boa-forecaster validate`` — walk-forward CV for the active model."""

from __future__ import annotations

import logging

import click

from boa_forecaster.cli._pipeline import (
    build_active_spec,
    ensure_output_dir,
    load_series_from_config,
    metric_components_as_dicts,
    summarise_folds,
    write_forecast_outputs,
)
from boa_forecaster.config_schema import BoaConfig
from boa_forecaster.optimizer import optimize_model
from boa_forecaster.validation import walk_forward_validation

logger = logging.getLogger(__name__)


@click.command(name="validate")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the YAML config file.",
)
@click.option(
    "--output",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False),
    help="Directory where metrics.csv and forecast.csv are written.",
)
@click.option("--n-folds", type=int, default=3, show_default=True)
@click.option("--test-size", type=int, default=12, show_default=True)
@click.option("--min-train-size", type=int, default=24, show_default=True)
def validate(
    config_path: str,
    output_dir: str,
    n_folds: int,
    test_size: int,
    min_train_size: int,
) -> None:
    """Run walk-forward validation for the active model."""
    cfg = BoaConfig.load(config_path)
    out = ensure_output_dir(output_dir)

    series = load_series_from_config(cfg)
    spec = build_active_spec(cfg)

    result = optimize_model(
        series,
        spec,
        n_calls=cfg.optimization.n_calls,
        n_jobs=cfg.optimization.n_jobs,
        metric_components=metric_components_as_dicts(cfg),
    )
    forecaster = spec.build_forecaster(result.best_params)

    folds = walk_forward_validation(
        series,
        forecaster,
        n_folds=n_folds,
        test_size=test_size,
        min_train_size=min_train_size,
    )
    metric_cols = [
        c
        for c in folds.columns
        if c not in {"fold", "train_start", "train_end", "test_start", "test_end"}
    ]
    summary = summarise_folds(folds, metric_cols)

    click.echo(f"[validate] folds={len(folds)}")
    click.echo(summary.to_string(index=False))

    folds.to_csv(out / "folds.csv", index=False)
    write_forecast_outputs(
        out_dir=out,
        forecast=forecaster(series),
        best_params=result.best_params,
        best_score=result.best_score,
        metrics_df=summary,
    )
