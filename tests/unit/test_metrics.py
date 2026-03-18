"""Tests for sarima_bayes.metrics: smape, rmsle, combined_metric."""

import math

import pytest

from sarima_bayes.metrics import combined_metric, rmsle, smape


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
