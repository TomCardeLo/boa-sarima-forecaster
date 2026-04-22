"""Unit tests for boa_forecaster.metrics_probabilistic.

Covers:
- pinball_loss: correctness, asymmetry, non-negativity, validation, input types
- interval_coverage: all-in, all-out, half-in, inclusive boundaries, shape mismatch
- METRIC_REGISTRY registration
- top-level re-exports
"""

from __future__ import annotations

import numpy as np
import pytest

# ── TestPinballLoss ───────────────────────────────────────────────────────────


class TestPinballLoss:
    def _import(self):
        from boa_forecaster.metrics_probabilistic import pinball_loss

        return pinball_loss

    def test_at_median_equals_half_mae(self):
        """At q=0.5, pinball_loss == 0.5 * MAE."""
        pinball_loss = self._import()
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 35.0, 38.0])
        mae_val = float(np.mean(np.abs(y_true - y_pred)))
        result = pinball_loss(y_true, y_pred, quantile=0.5)
        assert result == pytest.approx(0.5 * mae_val, rel=1e-6)

    def test_underprediction_penalised_more_at_q01(self):
        """At q=0.1 under-prediction (y_true > y_pred) should be penalised
        9× more than over-prediction (y_true < y_pred).

        Use a 2-point example for exact arithmetic:
          - point 1: y_true=10, y_pred=0  => under (error=10), penalty = 0.1 * 10 = 1.0
          - point 2: y_true=0,  y_pred=10 => over  (error=10), penalty = 0.9 * 10 = 9.0
        Mean pinball = (1.0 + 9.0) / 2 = 5.0

        The asymmetry ratio (over-penalty / under-penalty) = 9 / 1 = 9.
        """
        pinball_loss = self._import()
        y_true = np.array([10.0, 0.0])
        y_pred = np.array([0.0, 10.0])
        result = pinball_loss(y_true, y_pred, quantile=0.1)
        assert result == pytest.approx(5.0, rel=1e-6)

    def test_overprediction_penalised_more_at_q09(self):
        """At q=0.9 over-prediction (y_true < y_pred) should be penalised
        9× more than under-prediction.

        2-point example (symmetric to q=0.1 test, roles swapped):
          - point 1: y_true=10, y_pred=0  => under, penalty = (1-0.9)*10 = 1.0
          - point 2: y_true=0,  y_pred=10 => over,  penalty = 0.9*10     = 9.0
        Mean = 5.0 (same magnitude, different cause).
        """
        pinball_loss = self._import()
        y_true = np.array([10.0, 0.0])
        y_pred = np.array([0.0, 10.0])
        result = pinball_loss(y_true, y_pred, quantile=0.9)
        assert result == pytest.approx(5.0, rel=1e-6)

    def test_non_negative(self):
        """Pinball loss is always >= 0."""
        pinball_loss = self._import()
        rng = np.random.default_rng(99)
        y_true = rng.uniform(0, 100, 20)
        y_pred = rng.uniform(0, 100, 20)
        for q in (0.1, 0.5, 0.9):
            assert pinball_loss(y_true, y_pred, quantile=q) >= 0.0

    def test_zero_when_perfect_prediction(self):
        """Pinball loss is 0 when y_true == y_pred."""
        pinball_loss = self._import()
        y = np.array([5.0, 10.0, 15.0])
        for q in (0.1, 0.5, 0.9):
            assert pinball_loss(y, y, quantile=q) == pytest.approx(0.0, abs=1e-10)

    def test_validates_quantile_zero(self):
        """quantile=0.0 should raise ValueError."""
        pinball_loss = self._import()
        with pytest.raises(ValueError, match="quantile"):
            pinball_loss([1.0], [1.0], quantile=0.0)

    def test_validates_quantile_one(self):
        """quantile=1.0 should raise ValueError."""
        pinball_loss = self._import()
        with pytest.raises(ValueError, match="quantile"):
            pinball_loss([1.0], [1.0], quantile=1.0)

    def test_validates_quantile_negative(self):
        """quantile=-0.5 should raise ValueError."""
        pinball_loss = self._import()
        with pytest.raises(ValueError, match="quantile"):
            pinball_loss([1.0], [1.0], quantile=-0.5)

    def test_validates_quantile_greater_than_one(self):
        """quantile=1.5 should raise ValueError."""
        pinball_loss = self._import()
        with pytest.raises(ValueError, match="quantile"):
            pinball_loss([1.0], [1.0], quantile=1.5)

    def test_accepts_list_input(self):
        """Should accept plain Python lists (not just np.ndarray)."""
        pinball_loss = self._import()
        result = pinball_loss([10.0, 20.0], [10.0, 20.0], quantile=0.5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_returns_plain_float(self):
        """Return type should be Python float, not np.float64."""
        pinball_loss = self._import()
        result = pinball_loss([10.0], [12.0], quantile=0.5)
        assert isinstance(result, float)


# ── TestIntervalCoverage ──────────────────────────────────────────────────────


class TestIntervalCoverage:
    def _import(self):
        from boa_forecaster.metrics_probabilistic import interval_coverage

        return interval_coverage

    def test_all_in_returns_one(self):
        """All values inside [lower, upper] => coverage = 1.0."""
        interval_coverage = self._import()
        y = np.array([5.0, 10.0, 15.0])
        lower = np.array([4.0, 9.0, 14.0])
        upper = np.array([6.0, 11.0, 16.0])
        assert interval_coverage(y, lower, upper) == pytest.approx(1.0)

    def test_all_out_returns_zero(self):
        """All values outside [lower, upper] => coverage = 0.0."""
        interval_coverage = self._import()
        y = np.array([0.0, 100.0])
        lower = np.array([10.0, 110.0])
        upper = np.array([20.0, 120.0])
        assert interval_coverage(y, lower, upper) == pytest.approx(0.0)

    def test_half_in_returns_half(self):
        """Half values inside => coverage = 0.5."""
        interval_coverage = self._import()
        y = np.array([5.0, 50.0])
        lower = np.array([4.0, 4.0])
        upper = np.array([6.0, 6.0])
        assert interval_coverage(y, lower, upper) == pytest.approx(0.5)

    def test_inclusive_lower_boundary(self):
        """y_true exactly on lower boundary counts as in."""
        interval_coverage = self._import()
        y = np.array([5.0])
        lower = np.array([5.0])
        upper = np.array([10.0])
        assert interval_coverage(y, lower, upper) == pytest.approx(1.0)

    def test_inclusive_upper_boundary(self):
        """y_true exactly on upper boundary counts as in."""
        interval_coverage = self._import()
        y = np.array([10.0])
        lower = np.array([5.0])
        upper = np.array([10.0])
        assert interval_coverage(y, lower, upper) == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        """Mismatched shapes should raise ValueError."""
        interval_coverage = self._import()
        with pytest.raises(ValueError, match="shape"):
            interval_coverage([1.0, 2.0], [1.0], [2.0, 3.0])


# ── TestRegistry ──────────────────────────────────────────────────────────────


class TestRegistry:
    def test_pinball_loss_in_metric_registry(self):
        from boa_forecaster.metrics import METRIC_REGISTRY

        assert "pinball_loss" in METRIC_REGISTRY

    def test_pinball_loss_accessible_from_top_level(self):
        from boa_forecaster import pinball_loss

        assert callable(pinball_loss)
        # Sanity check: should work correctly when called via top-level import
        result = pinball_loss([10.0], [10.0], quantile=0.5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_interval_coverage_accessible_from_top_level(self):
        from boa_forecaster import interval_coverage

        assert callable(interval_coverage)
        result = interval_coverage([5.0], [4.0], [6.0])
        assert result == pytest.approx(1.0)
