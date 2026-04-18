"""Smoke tests for the ``boa-forecaster`` Click CLI.

Each subcommand gets a happy-path invocation driven by ``click.testing.CliRunner``.
The fixture creates a synthetic Excel file that mimics the real input layout
and a config pointing at it, so no fixtures on disk are required.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from boa_forecaster.cli import cli

# ── Fixture: synthetic input + config ────────────────────────────────────────


def _write_synthetic_input(xlsx_path: Path, n_periods: int = 60) -> None:
    """Create a mini YYYYMM Excel file with two skip rows and one SKU."""
    dates = pd.date_range("2020-01-01", periods=n_periods, freq="MS")
    rng = np.random.default_rng(42)
    rows = pd.DataFrame(
        {
            "Date": [d.strftime("%Y%m") for d in dates],
            "SKU": 1,
            "CS": 100
            + 20 * np.sin(2 * np.pi * np.arange(n_periods) / 12)
            + rng.normal(0, 5, n_periods),
            "Country": "MX",
        }
    )

    # Prepend two dummy header rows to match the real workbook layout.
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [["meta header"] * 4, ["meta header 2"] * 4],
            columns=list(rows.columns),
        ).to_excel(writer, sheet_name="Data", index=False, header=True)
        # Re-open and append data rows so the layout is:
        #   row 0: header (column names)
        #   rows 1..2: meta text (skip_rows=2 jumps past these)
        #   row 3: blank
        #   rows 4+: data (after pandas reads, iloc[2:] strips the meta rows)
        rows.to_excel(writer, sheet_name="Data", index=False, startrow=3)


def _write_config(cfg_path: Path, xlsx_path: Path, output_path: Path) -> None:
    cfg = {
        "data": {
            "input_path": str(xlsx_path),
            "sheet_name": "Data",
            "skip_rows": 2,
            "date_format": "%Y%m",
            "freq": "MS",
        },
        "optimization": {"n_calls": 3, "n_jobs": 1},
        "metrics": {
            "components": [
                {"metric": "smape", "weight": 0.7},
                {"metric": "rmsle", "weight": 0.3},
            ]
        },
        "forecast": {"n_periods": 6, "alpha": 0.05},
        "output": {"output_path": str(output_path), "run_id": "TEST"},
        "logging": {"level": "WARNING"},
        "models": {
            "active": "sarima",
            "sarima": {
                "enabled": True,
                "seasonal_period": 12,
                "search_space": {
                    "p": {"low": 0, "high": 1},
                    "d": {"low": 0, "high": 1},
                    "q": {"low": 0, "high": 1},
                    "P": {"low": 0, "high": 0},
                    "D": {"low": 0, "high": 0},
                    "Q": {"low": 0, "high": 0},
                },
                "warm_starts": [
                    {"p": 1, "d": 0, "q": 0, "P": 0, "D": 0, "Q": 0},
                ],
            },
        },
        "features": {
            "lag_periods": [1, 2],
            "rolling_windows": [3],
            "include_calendar": True,
            "include_trend": True,
            "include_expanding": False,
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


@pytest.fixture
def cli_workspace(tmp_path: Path) -> dict[str, Path]:
    xlsx = tmp_path / "sales.xlsx"
    _write_synthetic_input(xlsx)

    cfg = tmp_path / "config.yaml"
    out = tmp_path / "out"
    _write_config(cfg, xlsx, out)

    return {"config": cfg, "output": out, "xlsx": xlsx}


# ── Tests ────────────────────────────────────────────────────────────────────


def test_cli_help_lists_subcommands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for name in ("run", "validate", "compare"):
        assert name in result.output


def test_cli_version_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "boa-forecaster" in result.output


def test_cli_run_produces_forecast(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "run",
            "--config",
            str(cli_workspace["config"]),
            "--output",
            str(cli_workspace["output"]),
            "--n-trials",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output

    forecast_csv = cli_workspace["output"] / "forecast.csv"
    params_json = cli_workspace["output"] / "params.json"
    assert forecast_csv.exists()
    assert params_json.exists()

    df = pd.read_csv(forecast_csv)
    assert not df.empty
    assert "forecast" in df.columns

    params = json.loads(params_json.read_text())
    assert "best_params" in params
    assert "best_score" in params


def test_cli_validate_writes_metrics(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "validate",
            "--config",
            str(cli_workspace["config"]),
            "--output",
            str(cli_workspace["output"]),
            "--n-folds",
            "3",
            "--test-size",
            "6",
            "--min-train-size",
            "24",
        ],
    )
    assert result.exit_code == 0, result.output

    folds_csv = cli_workspace["output"] / "folds.csv"
    metrics_csv = cli_workspace["output"] / "metrics.csv"
    assert folds_csv.exists()
    assert metrics_csv.exists()

    folds = pd.read_csv(folds_csv)
    assert len(folds) == 3


def test_cli_compare_ranks_models(cli_workspace: dict[str, Path]) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "compare",
            "--config",
            str(cli_workspace["config"]),
            "--output",
            str(cli_workspace["output"]),
            "--n-folds",
            "3",
            "--test-size",
            "6",
            "--min-train-size",
            "24",
        ],
    )
    assert result.exit_code == 0, result.output

    metrics_csv = cli_workspace["output"] / "metrics.csv"
    assert metrics_csv.exists()

    metrics = pd.read_csv(metrics_csv)
    # At least the active model + 2 baselines.
    assert "model" in metrics.columns
    assert len(metrics) >= 1
    assert "sarima" in set(metrics["model"].tolist())
