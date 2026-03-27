"""Tests that sarima_bayes emits DeprecationWarning and preserves backward compat."""

import importlib
import sys
import warnings

import pytest


def _reimport_sarima_bayes():
    """Force a fresh import by evicting the module from sys.modules cache."""
    sys.modules.pop("sarima_bayes", None)
    return importlib.import_module("sarima_bayes")


def test_import_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="boa_forecaster"):
        _reimport_sarima_bayes()


def test_optimize_arima_still_callable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _reimport_sarima_bayes()
        from sarima_bayes import optimize_arima  # noqa: PLC0415

    assert callable(optimize_arima)


def test_all_v1_symbols_available():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        sb = _reimport_sarima_bayes()

    expected = [
        "optimize_arima",
        "forecast_arima",
        "pred_arima",
        "combined_metric",
        "build_combined_metric",
        "METRIC_REGISTRY",
        "smape",
        "rmsle",
        "mae",
        "rmse",
        "mape",
        "walk_forward_validation",
        "validate_by_group",
        "run_benchmark_comparison",
        "summary_table",
    ]
    for sym in expected:
        assert hasattr(sb, sym), f"sarima_bayes.{sym} missing after rename"
