"""Tests for sarima_bayes.metrics: smape, rmsle, mae, rmse, mape, combined_metric, build_combined_metric."""

import math

import numpy as np
import pytest

from sarima_bayes.metrics import (
    METRIC_REGISTRY,
    build_combined_metric,
    combined_metric,
    mae,
    mape,
    rmse,
    rmsle,
    smape,
)


class TestSmape:
    def test_perfect_prediction(self):
        assert smape([100], [100]) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self):
        # |100-150| / ((100+150)/2) * 100 = 50/125*100 = 40.0
        assert smape([100], [150]) == pytest.approx(40.0, abs=1e-4)

    def test_all_zeros(self):
        # epsilon prevents NaN
        result = smape([0, 0], [0, 0])
        assert not math.isnan(result)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_multiple_values(self):
        # mean of (10/105, 10/195) * 100 = (0.09524 + 0.05128) / 2 * 100 ≈ 7.326
        result = smape([100.0, 200.0], [110.0, 190.0])
        assert result == pytest.approx(7.326, abs=1e-2)


class TestRmsle:
    def test_perfect_prediction(self):
        assert rmsle([100], [100]) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self):
        # sqrt((log1p(100) - log1p(110))^2) ≈ 0.0944
        expected = abs(math.log(101) - math.log(111))
        assert rmsle([100], [110]) == pytest.approx(expected, abs=1e-4)

    def test_all_zeros(self):
        # log1p(0) = 0, result should be 0.0
        result = rmsle([0], [0])
        assert not math.isnan(result)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_negative_values_no_crash(self):
        # clips to 0 internally via np.maximum
        result = rmsle([-5], [-10])
        assert not math.isnan(result)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_all_zeros_arrays(self):
        result = rmsle([0, 0, 0], [0, 0, 0])
        assert not math.isnan(result)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestCombinedMetric:
    def test_known_value(self):
        # smape=40.0, rmsle=|log(101)-log(151)|
        s = 40.0
        r = abs(math.log(101) - math.log(151))
        expected = 0.7 * s + 0.3 * r
        assert combined_metric([100], [150]) == pytest.approx(expected, abs=1e-4)

    def test_perfect_prediction(self):
        assert combined_metric([100], [100]) == pytest.approx(0.0, abs=1e-4)

    def test_all_zeros_no_nan(self):
        result = combined_metric([0, 0], [0, 0])
        assert not math.isnan(result)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestMae:
    def test_perfect_prediction(self):
        assert mae([100], [100]) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self):
        # mean(|100-110|, |200-190|) = 10.0
        assert mae([100.0, 200.0], [110.0, 190.0]) == pytest.approx(10.0, abs=1e-4)

    def test_all_zeros(self):
        assert mae([0, 0], [0, 0]) == pytest.approx(0.0, abs=1e-4)


class TestRmse:
    def test_perfect_prediction(self):
        assert rmse([100], [100]) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self):
        # sqrt(mean(100, 100)) = sqrt(100) = 10.0
        assert rmse([100.0, 200.0], [110.0, 190.0]) == pytest.approx(10.0, abs=1e-4)

    def test_all_zeros(self):
        assert rmse([0, 0], [0, 0]) == pytest.approx(0.0, abs=1e-4)


class TestMape:
    def test_perfect_prediction(self):
        assert mape([100], [100]) == pytest.approx(0.0, abs=1e-4)

    def test_known_value(self):
        # mean(|100-110|/100, |200-190|/200) * 100 = mean(0.1, 0.05) * 100 = 7.5
        assert mape([100.0, 200.0], [110.0, 190.0]) == pytest.approx(7.5, abs=1e-2)

    def test_zeros_no_nan(self):
        result = mape([0, 0], [0, 0])
        assert not math.isnan(result)
        assert result == pytest.approx(0.0, abs=1e-4)


class TestBuildCombinedMetric:
    def test_default_components_match_combined_metric(self):
        # build_combined_metric with default components must equal combined_metric
        fn = build_combined_metric(
            [
                {"metric": "smape", "weight": 0.7},
                {"metric": "rmsle", "weight": 0.3},
            ]
        )
        y_true = np.array([100.0, 200.0, 150.0])
        y_pred = np.array([110.0, 190.0, 160.0])
        assert fn(y_true, y_pred) == pytest.approx(
            combined_metric(y_true, y_pred), abs=1e-6
        )

    def test_custom_mae_rmse_composition(self):
        fn = build_combined_metric(
            [
                {"metric": "mae", "weight": 0.6},
                {"metric": "rmse", "weight": 0.4},
            ]
        )
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        expected = 0.6 * mae(y_true, y_pred) + 0.4 * rmse(y_true, y_pred)
        assert fn(y_true, y_pred) == pytest.approx(expected, abs=1e-6)

    def test_single_metric_weight_one(self):
        fn = build_combined_metric([{"metric": "smape", "weight": 1.0}])
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        assert fn(y_true, y_pred) == pytest.approx(smape(y_true, y_pred), abs=1e-6)

    def test_perfect_prediction_returns_zero(self):
        fn = build_combined_metric(
            [
                {"metric": "mae", "weight": 0.5},
                {"metric": "rmse", "weight": 0.5},
            ]
        )
        assert fn([100.0], [100.0]) == pytest.approx(0.0, abs=1e-4)

    def test_unknown_metric_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown metric"):
            build_combined_metric([{"metric": "nonexistent", "weight": 1.0}])

    def test_registry_contains_expected_keys(self):
        assert set(METRIC_REGISTRY.keys()) == {"smape", "rmsle", "mae", "rmse", "mape"}
