"""Tests for hit_rate_weighted and f1_by_bucket bucket metrics (H7-core)."""

from __future__ import annotations

import numpy as np
import pytest

from boa_forecaster.metrics import (
    METRIC_DICT_REGISTRY,
    METRIC_REGISTRY,
    f1_by_bucket,
    hit_rate,
    hit_rate_weighted,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

EDGES_THREE = [0.0, 10.0, 20.0]  # buckets: (−∞,0], (0,10], (10,20], (20,+∞)
EDGES_TWO = [10.0, 20.0]  # buckets: low / med / high


# ===========================================================================
# hit_rate_weighted
# ===========================================================================


class TestHitRateWeightedNoneWeights:
    """weights=None must reproduce plain hit_rate exactly."""

    def test_hit_rate_weighted_none_weights_equals_hit_rate(self):
        y_true = np.array([5.0, 15.0, 25.0, 8.0])
        y_pred = np.array([7.0, 14.0, 30.0, 12.0])  # bucket mismatch on last
        edges = [10.0, 20.0]
        assert hit_rate_weighted(y_true, y_pred, edges, weights=None) == pytest.approx(
            hit_rate(y_true, y_pred, edges)
        )


class TestHitRateWeightedPenalisesHighStakes:
    """Heavy weight on the missed bucket pulls weighted HR below unweighted."""

    def test_hit_rate_weighted_penalizes_high_stakes_misses(self):
        # 5 hits in low bucket (bucket index 1), 1 miss in high bucket (bucket index 3)
        # edges = [0, 10, 20]  → buckets: ≤0 → 0, (0,10] → 1, (10,20] → 2, >20 → 3
        edges = [0.0, 10.0, 20.0]
        y_true = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 25.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 15.0])  # last misses: 3 vs 2
        weights = [1.0, 1.0, 1.0, 10.0]  # heavy penalty for bucket 3 (high)

        unweighted = hit_rate(y_true, y_pred, edges)
        weighted = hit_rate_weighted(y_true, y_pred, edges, weights=weights)

        assert weighted < unweighted, (
            f"Expected weighted ({weighted}) < unweighted ({unweighted}) "
            "but heavy-penalty miss did not drag the score down"
        )


class TestHitRateWeightedBoundaryMiss:
    """Boundary crossing (9.9 truth vs 10.1 pred across edge 10.0) is a miss."""

    def test_hit_rate_weighted_boundary_miss(self):
        y_true = np.array([10.1])  # bucket 2 (>10)
        y_pred = np.array([9.9])  # bucket 1 (≤10)
        edges = [10.0]
        # With weights=None the result must match plain hit_rate (0.0 here)
        assert hit_rate_weighted(y_true, y_pred, edges, weights=None) == pytest.approx(
            hit_rate(y_true, y_pred, edges)
        )
        assert hit_rate(y_true, y_pred, edges) == pytest.approx(0.0)


class TestHitRateWeightedValidation:
    """ValueError for invalid weight arrays."""

    def test_hit_rate_weighted_rejects_wrong_weights_length(self):
        edges = [10.0, 20.0]  # len(edges)+1 = 3 buckets
        weights_bad = [1.0, 1.0]  # only 2 → wrong
        with pytest.raises(ValueError, match="weights"):
            hit_rate_weighted(
                np.array([5.0]), np.array([5.0]), edges, weights=weights_bad
            )

    def test_hit_rate_weighted_rejects_negative_weights(self):
        edges = [10.0, 20.0]
        weights_bad = [1.0, -0.5, 1.0]
        with pytest.raises(ValueError, match="negative"):
            hit_rate_weighted(
                np.array([5.0]), np.array([5.0]), edges, weights=weights_bad
            )

    def test_hit_rate_weighted_rejects_all_zero_weights(self):
        edges = [10.0, 20.0]
        weights_zero = [0.0, 0.0, 0.0]
        with pytest.raises(ValueError, match="zero"):
            hit_rate_weighted(
                np.array([5.0]), np.array([5.0]), edges, weights=weights_zero
            )


# ===========================================================================
# f1_by_bucket
# ===========================================================================


class TestF1ByBucketShape:
    def test_f1_by_bucket_shape(self):
        edges = [10.0, 20.0]
        y_true = np.array([5.0, 15.0, 25.0, 8.0, 22.0])
        y_pred = np.array([6.0, 14.0, 26.0, 12.0, 21.0])
        result = f1_by_bucket(y_true, y_pred, edges)
        assert len(result) == len(edges) + 1


class TestF1ByBucketValuesInRange:
    def test_f1_by_bucket_values_in_range(self):
        edges = [10.0, 20.0]
        y_true = np.array([5.0, 15.0, 25.0, 8.0, 22.0])
        y_pred = np.array([6.0, 14.0, 26.0, 12.0, 21.0])
        result = f1_by_bucket(y_true, y_pred, edges)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"F1 for bucket {key!r} out of range: {val}"


class TestF1ByBucketLabels:
    def test_f1_by_bucket_labels_respected(self):
        edges = [10.0, 20.0]
        labels = ["low", "med", "high"]
        y_true = np.array([5.0, 15.0, 25.0])
        y_pred = np.array([5.0, 15.0, 25.0])
        result = f1_by_bucket(y_true, y_pred, edges, labels=labels)
        assert set(result.keys()) == {"low", "med", "high"}

    def test_f1_by_bucket_labels_wrong_length_raises(self):
        edges = [10.0, 20.0]  # needs 3 labels
        bad_labels = ["low", "high"]  # only 2
        with pytest.raises(ValueError, match="labels"):
            f1_by_bucket(np.array([5.0]), np.array([5.0]), edges, labels=bad_labels)


class TestF1ByBucketPerfectPrediction:
    """y_true == y_pred: every occupied bucket gets F1=1.0; empty buckets get 0.0."""

    def test_f1_by_bucket_perfect_prediction(self):
        edges = [10.0, 20.0]
        # Values only in low and high; nothing in med → med bucket has no support
        y_true = np.array([5.0, 5.0, 25.0, 25.0])
        y_pred = y_true.copy()
        result = f1_by_bucket(y_true, y_pred, edges)

        # low (bucket_0) and high (bucket_2) must be 1.0
        assert result["bucket_0"] == pytest.approx(1.0)
        assert result["bucket_2"] == pytest.approx(1.0)
        # med (bucket_1) has no support → convention: 0.0
        assert result["bucket_1"] == pytest.approx(0.0)


# ===========================================================================
# Registry tests
# ===========================================================================


class TestRegistry:
    def test_hit_rate_weighted_in_metric_registry(self):
        assert "hit_rate_weighted" in METRIC_REGISTRY

    def test_f1_by_bucket_not_in_scalar_registry(self):
        """f1_by_bucket returns dict, not float — kept out of METRIC_REGISTRY
        to avoid breaking build_combined_metric which expects float returns.
        It is registered in METRIC_DICT_REGISTRY instead."""
        assert "f1_by_bucket" not in METRIC_REGISTRY

    def test_f1_by_bucket_in_dict_registry(self):
        assert "f1_by_bucket" in METRIC_DICT_REGISTRY
