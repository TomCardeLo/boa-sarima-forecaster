"""``boa-forecaster run`` — optimise the active model and write a forecast."""

from __future__ import annotations

import logging

import click

from boa_forecaster.cli._pipeline import (
    build_active_spec,
    ensure_output_dir,
    load_series_from_config,
    metric_components_as_dicts,
    write_forecast_outputs,
)
from boa_forecaster.config_schema import BoaConfig
from boa_forecaster.optimizer import optimize_model

logger = logging.getLogger(__name__)


@click.command(name="run")
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
    help="Directory where forecast.csv / params.json / metrics.csv / plot.png are written.",
)
@click.option(
    "--n-trials",
    type=int,
    default=None,
    help="Override optimization.n_calls from the config.",
)
def run(config_path: str, output_dir: str, n_trials: int | None) -> None:
    """Optimise the active model and emit a point forecast."""
    cfg = BoaConfig.load(config_path)
    out = ensure_output_dir(output_dir)

    series = load_series_from_config(cfg)
    spec = build_active_spec(cfg)

    n_calls = n_trials if n_trials is not None else cfg.optimization.n_calls
    result = optimize_model(
        series,
        spec,
        n_calls=n_calls,
        n_jobs=cfg.optimization.n_jobs,
        metric_components=metric_components_as_dicts(cfg),
    )

    forecaster = spec.build_forecaster(result.best_params)
    forecast = forecaster(series)

    click.echo(
        f"[run] model={result.model_name} "
        f"score={result.best_score:.4f} "
        f"trials={result.n_trials} "
        f"fallback={result.is_fallback}"
    )

    write_forecast_outputs(
        out_dir=out,
        forecast=forecast,
        best_params=result.best_params,
        best_score=result.best_score,
    )
