"""End-to-end integration test for seasonal bias correction.

Uses the ``long_series`` fixture (60-month synthetic series) to verify that
enabling ``apply_bias_correction=True`` in ``optimize_model`` never degrades
the corrected forecast metric by more than 2% relative to the uncorrected run.

Marked ``integration``; deselect with ``pytest -m "not integration"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.metrics import smape
from boa_forecaster.models.sarima import SARIMASpec
from boa_forecaster.optimizer import optimize_model
from boa_forecaster.postprocess import apply_seasonal_bias

pytestmark = pytest.mark.integration


# ─── helpers ──────────────────────────────────────────────────────────────────


def _score_on_last_fold(
    series: pd.Series,
    spec: SARIMASpec,
    best_params: dict,
    bias: np.ndarray | None,
) -> float:
    """Re-score the last hold-out fold (12 months) with optional bias correction.

    Mirrors the fold math used internally by ``_compute_bias_from_last_fold``
    so the comparison is apples-to-apples.
    """
    min_train_size = 24
    n_folds = 3
    test_size = 12
    train_end = min_train_size + (n_folds - 1) * test_size

    train = series.iloc[:train_end]
    test = series.iloc[train_end : train_end + test_size]

    forecaster = spec.build_forecaster(best_params)
    predictions = forecaster(train)

    y_true = test.values[: len(predictions)]
    y_pred = predictions.values[: len(y_true)]

    if bias is not None:
        test_idx = test.index[: len(y_true)]
        y_pred_s = pd.Series(y_pred, index=test_idx)
        corrected = apply_seasonal_bias(y_pred_s, bias)
        y_pred = corrected.values

    return float(smape(y_true, y_pred))


# ─── test ─────────────────────────────────────────────────────────────────────


class TestBiasCorrectionE2E:
    """Bias-corrected forecast is never more than 2% worse than the baseline."""

    @pytest.fixture
    def results(self, long_series):
        spec = SARIMASpec(forecast_horizon=12)

        result_without = optimize_model(
            long_series,
            spec,
            n_calls=5,
            seed=0,
            apply_bias_correction=False,
        )
        result_with = optimize_model(
            long_series,
            spec,
            n_calls=5,
            seed=0,
            apply_bias_correction=True,
        )
        return {
            "without": result_without,
            "with": result_with,
            "series": long_series,
            "spec": spec,
        }

    def test_bias_correction_field_is_none_without_flag(self, results):
        assert results["without"].bias_correction is None

    def test_bias_correction_field_populated_with_flag(self, results):
        assert results["with"].bias_correction is not None
        assert isinstance(results["with"].bias_correction, np.ndarray)
        assert results["with"].bias_correction.shape == (12,)

    def test_bias_factors_within_clip_range(self, results):
        bias = results["with"].bias_correction
        assert (bias >= 0.5).all()
        assert (bias <= 2.0).all()

    def test_corrected_score_not_worse_by_more_than_two_percent(self, results):
        series = results["series"]
        spec = results["spec"]

        assert (
            results["without"].best_params == results["with"].best_params
        ), "apply_bias_correction must not perturb the optimizer's best_params"

        score_without = _score_on_last_fold(
            series, spec, results["without"].best_params, bias=None
        )
        score_with = _score_on_last_fold(
            series,
            spec,
            results["with"].best_params,
            bias=results["with"].bias_correction,
        )

        assert score_with <= score_without * 1.02, (
            f"Bias-corrected sMAPE {score_with:.4f} is more than 2% worse "
            f"than baseline {score_without:.4f}."
        )
