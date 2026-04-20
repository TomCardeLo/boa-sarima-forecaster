"""Tests for sarima_bayes.metrics: smape, rmsle, mae, rmse, mape, combined_metric, build_combined_metric."""

import math

import numpy as np
import pytest

from boa_forecaster.metrics import (
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
        # hit_rate is registered by the G2 work (v2.3.0). We assert a subset
        # here so that runtime-registered user metrics don't break the check.
        expected = {"smape", "rmsle", "mae", "rmse", "mape", "hit_rate"}
        assert expected.issubset(set(METRIC_REGISTRY.keys()))


class TestHitRate:
    """hit_rate computes the fraction of predictions landing in the same
    bucket as the truth.  Requires caller-supplied bucket edges.
    """

    def test_hit_rate_basic(self):
        """Simple edges [0, 10, 20]: true=[5, 15, 25], pred=[7, 14, 100]
        → first two buckets match (1, 2), third also matches (both above 20).
        """
        from boa_forecaster.metrics import hit_rate

        y_true = np.array([5.0, 15.0, 25.0])
        y_pred = np.array([7.0, 14.0, 100.0])
        # np.digitize default (right=False) places y in bucket i iff
        # edges[i-1] <= y < edges[i]. Our three values bucket as (1, 2, 3)
        # for both true and pred → 100% hit rate.
        result = hit_rate(y_true, y_pred, edges=[0, 10, 20])
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_hit_rate_partial_miss(self):
        from boa_forecaster.metrics import hit_rate

        y_true = np.array([5.0, 15.0, 25.0])
        y_pred = np.array([5.0, 25.0, 25.0])  # second prediction moves bucket
        result = hit_rate(y_true, y_pred, edges=[0, 10, 20])
        # Buckets: true=(1,2,3), pred=(1,3,3) → 2/3 correct.
        assert result == pytest.approx(2.0 / 3.0, abs=1e-6)

    def test_hit_rate_all_miss(self):
        from boa_forecaster.metrics import hit_rate

        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([15.0, 25.0, 100.0])
        result = hit_rate(y_true, y_pred, edges=[0, 10, 20])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_hit_rate_registered_in_registry(self):
        assert "hit_rate" in METRIC_REGISTRY

    def test_hit_rate_in_build_combined_metric_with_edges(self):
        """build_combined_metric must forward 'edges' into hit_rate."""
        fn = build_combined_metric(
            [{"metric": "hit_rate", "weight": 1.0, "edges": [0, 10, 20]}]
        )
        y_true = np.array([5.0, 15.0, 25.0])
        y_pred = np.array([7.0, 14.0, 100.0])
        result = fn(y_true, y_pred)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_build_combined_metric_ignores_unknown_kwargs_silently(self):
        """Extra keys in a component dict that are not accepted by the
        underlying metric callable are filtered out via inspect.signature
        — the old {metric, weight} contract must keep working unchanged.
        """
        fn = build_combined_metric(
            [
                # 'edges' is not accepted by smape; must be filtered out.
                {"metric": "smape", "weight": 1.0, "edges": [0, 1, 2]},
            ]
        )
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        assert fn(y_true, y_pred) == pytest.approx(smape(y_true, y_pred), abs=1e-6)


class TestCombinedMetricDelegatesToFactory:
    """Regression tests proving combined_metric == build_combined_metric.

    F4 refactors combined_metric to delegate to build_combined_metric so
    every metric path honours register_metric().  Numerical output must
    match the legacy implementation byte-for-byte on fixed inputs.
    """

    def test_combined_metric_matches_legacy_numerics(self):
        """Fixed numpy-seeded inputs produce the same scalar output as the
        hand-rolled 0.7 * sMAPE + 0.3 * RMSLE composition.
        """
        rng = np.random.default_rng(2026)
        y_true = rng.uniform(50.0, 500.0, size=200)
        y_pred = y_true + rng.normal(0.0, 25.0, size=200)

        expected = 0.7 * smape(y_true, y_pred) + 0.3 * rmsle(y_true, y_pred)
        got = combined_metric(y_true, y_pred)
        assert got == pytest.approx(expected, rel=0.0, abs=0.0)

    def test_combined_metric_matches_factory_numerics(self):
        """combined_metric must equal the factory-built equivalent bit-for-bit."""
        rng = np.random.default_rng(7)
        y_true = rng.uniform(10.0, 1000.0, size=300)
        y_pred = y_true * rng.uniform(0.8, 1.2, size=300)

        factory_fn = build_combined_metric(
            [
                {"metric": "smape", "weight": 0.7},
                {"metric": "rmsle", "weight": 0.3},
            ]
        )
        assert combined_metric(y_true, y_pred) == pytest.approx(
            factory_fn(y_true, y_pred), rel=0.0, abs=1e-12
        )

    def test_combined_metric_honours_custom_weights(self):
        """Non-default weights still compose correctly through the factory."""
        rng = np.random.default_rng(123)
        y_true = rng.uniform(1.0, 100.0, size=50)
        y_pred = y_true + rng.normal(0.0, 5.0, size=50)

        expected = 0.4 * smape(y_true, y_pred) + 0.6 * rmsle(y_true, y_pred)
        got = combined_metric(y_true, y_pred, w_smape=0.4, w_rmsle=0.6)
        assert got == pytest.approx(expected, rel=0.0, abs=1e-12)

    def test_combined_metric_honours_register_metric(self):
        """Register a custom metric and verify it flows through
        build_combined_metric (proves the factory is the single source of
        truth for metric composition)."""
        from boa_forecaster.metrics import register_metric

        def always_42(y_true, y_pred):  # noqa: ARG001
            return 42.0

        register_metric("always_42_f4", always_42)
        fn = build_combined_metric([{"metric": "always_42_f4", "weight": 0.5}])

        # A custom metric registered at runtime must be resolvable through
        # build_combined_metric after registration.
        assert fn(np.array([1.0]), np.array([2.0])) == pytest.approx(21.0)
