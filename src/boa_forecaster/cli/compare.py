"""``boa-forecaster compare`` — head-to-head comparison of enabled models."""

from __future__ import annotations

import logging

import click
import pandas as pd

from boa_forecaster.benchmarks import ets_model, seasonal_naive
from boa_forecaster.cli._pipeline import (
    ensure_output_dir,
    load_series_from_config,
    metric_components_as_dicts,
    summarise_folds,
)
from boa_forecaster.config_schema import BoaConfig
from boa_forecaster.metrics import rmsle, smape
from boa_forecaster.models import MODEL_REGISTRY, get_model_spec
from boa_forecaster.optimizer import optimize_model
from boa_forecaster.validation import walk_forward_validation

logger = logging.getLogger(__name__)


def _enabled_model_names(cfg: BoaConfig) -> list[str]:
    """Return the names of ``models.*`` entries with ``enabled: true``."""
    out: list[str] = []
    for name in ("sarima", "random_forest", "xgboost", "lightgbm"):
        entry = getattr(cfg.models, name, None)
        if entry is None or not entry.enabled:
            continue
        if name not in MODEL_REGISTRY or MODEL_REGISTRY[name] is None:
            logger.warning("Model '%s' enabled but not installed — skipping.", name)
            continue
        out.append(name)
    if not out:
        out.append(cfg.models.active)
    return out


@click.command(name="compare")
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
    help="Directory where metrics.csv is written.",
)
@click.option("--n-folds", type=int, default=3, show_default=True)
@click.option("--test-size", type=int, default=12, show_default=True)
@click.option("--min-train-size", type=int, default=24, show_default=True)
@click.option(
    "--baselines/--no-baselines",
    default=True,
    show_default=True,
    help="Include Seasonal-Naïve and ETS as zero-budget baselines.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help=(
        "Load config in strict mode: unknown keys raise ValidationError. "
        "Default is off for v2.x back-compat."
    ),
)
def compare(
    config_path: str,
    output_dir: str,
    n_folds: int,
    test_size: int,
    min_train_size: int,
    baselines: bool,
    strict: bool,
) -> None:
    """Optimise every enabled model and rank them on walk-forward folds."""
    cfg = BoaConfig.load(config_path, strict=strict)
    out = ensure_output_dir(output_dir)

    series = load_series_from_config(cfg)
    metrics_fn = {"sMAPE": smape, "RMSLE": rmsle}

    rows: list[pd.DataFrame] = []

    for name in _enabled_model_names(cfg):
        spec = get_model_spec(name)
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
            metrics_fn=metrics_fn,
        )
        summary = summarise_folds(folds, metrics_fn.keys())
        summary.insert(0, "model", name)
        rows.append(summary)

    if baselines:
        for label, fn in (
            ("seasonal_naive", lambda tr: seasonal_naive(tr, test_size)),
            ("ets", lambda tr: ets_model(tr, test_size)),
        ):
            folds = walk_forward_validation(
                series,
                fn,
                n_folds=n_folds,
                test_size=test_size,
                min_train_size=min_train_size,
                metrics_fn=metrics_fn,
            )
            summary = summarise_folds(folds, metrics_fn.keys())
            summary.insert(0, "model", label)
            rows.append(summary)

    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out / "metrics.csv", index=False)
    click.echo(combined.to_string(index=False))
