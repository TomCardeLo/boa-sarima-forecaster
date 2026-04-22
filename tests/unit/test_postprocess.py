"""Unit tests for ``boa_forecaster.postprocess``.

Test strategy
-------------
* Sinusoidal recovery: inject a known per-month bias and verify that
  ``compute_seasonal_bias`` recovers it within 5%.
* Perfect forecast → all factors ≈ 1.0.
* Clipping: zero-prediction months are dropped; empty bucket returns 1.0.
* No-op apply: applying all-1.0 bias leaves the forecast unchanged.
* Alignment: starting month != 1 uses the correct bucket both for DatetimeIndex
  and for raw ndarray (position-based, start_period-adjusted).
* periods != 12: weekly seasonality on a synthetic 7-cycle series.
* Input types: both ndarray and Series accepted; Series → Series for apply.
* NaN handling: NaN in either array drops the pair before computing median.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.postprocess import (
    apply_seasonal_bias,
    compute_seasonal_bias,
)

# ─── helpers ─────────────────────────────────────────────────────────────────


def _monthly_series(n_years: int = 5, start: str = "2019-01-01") -> pd.Series:
    """Pure-positive monthly series spanning *n_years* full years."""
    n = n_years * 12
    rng = np.random.default_rng(0)
    values = 100.0 + rng.normal(0, 5, n)
    values = np.abs(values) + 1.0  # ensure strictly positive
    idx = pd.date_range(start, periods=n, freq="MS")
    return pd.Series(values, index=idx)


def _true_factors(n_years: int = 5) -> np.ndarray:
    """Known per-month multiplicative factors: 1 + 0.3*sin(2π·(m-1)/12)."""
    months = np.arange(1, 13)
    return 1.0 + 0.3 * np.sin(2 * np.pi * (months - 1) / 12.0)


# ─── sinusoidal recovery ──────────────────────────────────────────────────────


class TestSinusoidalRecovery:
    """Bias computed from injected sinusoidal pattern recovers within 5%."""

    def test_factors_within_five_percent(self):
        n_years = 5
        series = _monthly_series(n_years)
        true_f = _true_factors(n_years)

        # y_pred is the "perfect" forecast; y_true adds the known bias.
        y_pred = series.copy()
        monthly_idx = (series.index.month - 1) % 12  # 0..11
        multiplier = true_f[monthly_idx]
        y_true = series * multiplier

        bias = compute_seasonal_bias(y_true, y_pred, periods=12, start_period=1)
        assert bias.shape == (12,)

        for m_idx in range(12):
            recovered = bias[m_idx]
            expected = true_f[m_idx]
            assert (
                abs(recovered - expected) / expected < 0.05
            ), f"Month {m_idx + 1}: expected {expected:.4f}, got {recovered:.4f}"

    def test_output_shape_is_periods(self):
        series = _monthly_series(3)
        y_pred = series.copy()
        y_true = series * 1.1
        bias = compute_seasonal_bias(y_true, y_pred, periods=12)
        assert bias.shape == (12,)


# ─── perfect forecasts ────────────────────────────────────────────────────────


class TestPerfectForecast:
    """When y_true == y_pred, every factor must be ≈ 1.0."""

    def test_factors_are_unity(self):
        series = _monthly_series(4)
        bias = compute_seasonal_bias(series, series.copy(), periods=12)
        np.testing.assert_allclose(bias, 1.0, atol=1e-10)

    def test_works_with_ndarray_input(self):
        values = np.linspace(10, 200, 48)
        bias = compute_seasonal_bias(values, values.copy(), periods=12)
        np.testing.assert_allclose(bias, 1.0, atol=1e-10)


# ─── clipping and zero-prediction handling ────────────────────────────────────


class TestClippingAndZeros:
    """Zero-prediction pairs are dropped; empty bucket returns 1.0."""

    def test_empty_bucket_returns_one(self):
        """A period where every y_pred == 0 → that bucket is empty → factor 1.0."""
        series = _monthly_series(3)  # 36 months, Jan..Dec × 3
        y_true = series.copy()
        y_pred = series.copy()
        # Zero out all January y_pred values — months 0, 12, 24 (0-indexed)
        y_pred.iloc[0::12] = 0.0

        bias = compute_seasonal_bias(y_true, y_pred, periods=12, start_period=1)
        # January bucket (index 0) must default to 1.0
        assert bias[0] == pytest.approx(1.0, abs=1e-10)
        # Other months should still have factors
        assert bias.shape == (12,)

    def test_factors_remain_in_clip_range(self):
        """Extreme ratios must be clipped to (0.5, 2.0)."""
        series = _monthly_series(3)
        y_true = series.copy()
        y_pred = series.copy()
        # Make February y_true 100× larger → ratio >> 2.0 → clips to 2.0
        y_pred.iloc[1::12] = series.iloc[1::12] / 100.0  # y_true/y_pred >> 2

        bias = compute_seasonal_bias(y_true, y_pred, periods=12, start_period=1)
        assert (bias >= 0.5).all()
        assert (bias <= 2.0).all()

    def test_custom_clip_range(self):
        """clip_range parameter is respected."""
        series = _monthly_series(3)
        y_pred = series * 0.1  # ratios ~ 10 → clips to upper bound
        bias = compute_seasonal_bias(series, y_pred, periods=12, clip_range=(0.8, 1.5))
        assert (bias <= 1.5).all()
        assert (bias >= 0.8).all()


# ─── no-op application ────────────────────────────────────────────────────────


class TestNoOpApply:
    """Applying all-1.0 bias must leave the forecast unchanged."""

    def test_ndarray_noop(self):
        forecast = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0] * 2)
        bias = np.ones(12)
        result = apply_seasonal_bias(forecast, bias, start_period=1)
        np.testing.assert_allclose(result, forecast)

    def test_series_noop_returns_series(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="MS")
        forecast = pd.Series(np.arange(1.0, 13.0), index=idx)
        bias = np.ones(12)
        result = apply_seasonal_bias(forecast, bias, start_period=1)
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, forecast)


# ─── alignment ────────────────────────────────────────────────────────────────


class TestAlignment:
    """Correct bucket is applied based on DatetimeIndex month or position."""

    def test_datetimeindex_july_gets_july_factor(self):
        """A forecast starting in July must receive the July factor (index 6)."""
        idx = pd.date_range("2024-07-01", periods=12, freq="MS")
        forecast = pd.Series(np.ones(12), index=idx)

        # bias[6] = 2.0 (July, 0-based), rest = 1.0
        bias = np.ones(12)
        bias[6] = 2.0

        result = apply_seasonal_bias(forecast, bias, start_period=1)
        # July is month 7 → 0-based index 6 → factor 2.0
        assert result.iloc[0] == pytest.approx(2.0)
        # August is month 8 → 0-based index 7 → factor 1.0
        assert result.iloc[1] == pytest.approx(1.0)

    def test_ndarray_position_alignment_start_period_7(self):
        """With start_period=7, position 0 maps to bucket 6 (July, 0-based)."""
        forecast = np.ones(12)
        bias = np.ones(12)
        bias[6] = 3.0  # July factor

        result = apply_seasonal_bias(forecast, bias, start_period=7)
        # Position 0 → (0 + 7 - 1) % 12 = 6 → factor 3.0
        assert result[0] == pytest.approx(3.0)
        # Position 1 → (1 + 7 - 1) % 12 = 7 → factor 1.0
        assert result[1] == pytest.approx(1.0)

    def test_compute_bias_datetimeindex_buckets(self):
        """compute_seasonal_bias uses DatetimeIndex.month when available."""
        # Start in July so months rotate July … June
        idx = pd.date_range("2020-07-01", periods=36, freq="MS")
        rng = np.random.default_rng(1)
        base = 100.0 + rng.normal(0, 1, 36)
        y_pred = pd.Series(base, index=idx)

        # Inject factor 1.5 for September (month=9, bucket index 8)
        factors = np.ones(12)
        factors[8] = 1.5
        month_buckets = (idx.month - 1) % 12
        y_true = pd.Series(base * factors[month_buckets], index=idx)

        bias = compute_seasonal_bias(y_true, y_pred, periods=12, start_period=1)
        assert bias[8] == pytest.approx(1.5, rel=0.05)


# ─── periods != 12 ────────────────────────────────────────────────────────────


class TestNonMonthlyPeriods:
    """periods=7 works for weekly seasonality on a synthetic series."""

    def test_weekly_seasonality_recovery(self):
        rng = np.random.default_rng(42)
        n_weeks = 7 * 4  # 4 full cycles
        base = 50.0 + rng.normal(0, 2, n_weeks)
        base = np.abs(base) + 1.0

        # True factors for 7 day-of-week buckets
        true_f = np.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.3, 0.95])
        bucket = np.arange(n_weeks) % 7
        y_true = base * true_f[bucket]
        y_pred = base.copy()

        bias = compute_seasonal_bias(y_true, y_pred, periods=7, start_period=1)
        assert bias.shape == (7,)
        for i in range(7):
            assert abs(bias[i] - true_f[i]) / true_f[i] < 0.05


# ─── input types ──────────────────────────────────────────────────────────────


class TestInputTypes:
    """Both np.ndarray and pd.Series are accepted; Series in → Series out."""

    def test_compute_accepts_series(self):
        series = _monthly_series(3)
        bias = compute_seasonal_bias(series, series * 1.1, periods=12)
        assert isinstance(bias, np.ndarray)

    def test_compute_accepts_ndarray(self):
        arr = np.linspace(10, 200, 36)
        bias = compute_seasonal_bias(arr, arr * 1.1, periods=12)
        assert isinstance(bias, np.ndarray)

    def test_apply_series_in_series_out(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="MS")
        forecast = pd.Series(np.ones(12), index=idx)
        bias = np.ones(12) * 1.1
        result = apply_seasonal_bias(forecast, bias)
        assert isinstance(result, pd.Series)
        assert result.index.equals(forecast.index)

    def test_apply_ndarray_in_ndarray_out(self):
        forecast = np.ones(12)
        bias = np.ones(12) * 1.1
        result = apply_seasonal_bias(forecast, bias)
        assert isinstance(result, np.ndarray)


# ─── NaN handling ─────────────────────────────────────────────────────────────


class TestNaNHandling:
    """NaN in y_true or y_pred drops that pair from the bucket's median."""

    def test_nan_in_y_true_is_dropped(self):
        series = _monthly_series(3)
        y_true = series.copy()
        y_pred = series.copy()
        # Inject NaN in all January y_true values
        y_true.iloc[0::12] = np.nan

        # Should not raise; January bucket is empty → defaults to 1.0
        bias = compute_seasonal_bias(y_true, y_pred, periods=12, start_period=1)
        assert np.isfinite(bias).all()
        assert bias[0] == pytest.approx(1.0, abs=1e-10)

    def test_nan_in_y_pred_is_dropped(self):
        series = _monthly_series(3)
        y_true = series * 1.2
        y_pred = series.copy()
        # NaN in March y_pred
        y_pred.iloc[2::12] = np.nan

        bias = compute_seasonal_bias(y_true, y_pred, periods=12, start_period=1)
        # March bucket (index 2) is empty → 1.0
        assert np.isfinite(bias).all()
        assert bias[2] == pytest.approx(1.0, abs=1e-10)

    def test_partial_nan_does_not_corrupt_other_buckets(self):
        """NaN in one bucket must not affect median of other buckets."""
        series = _monthly_series(4)
        y_pred = series.copy()
        y_true = series * 1.3  # all factors should be ~1.3
        # Corrupt half of January pairs with NaN in y_pred
        y_pred.iloc[0] = np.nan  # first January only

        bias = compute_seasonal_bias(y_true, y_pred, periods=12, start_period=1)
        # Non-January months should still show ~1.3
        for m_idx in range(1, 12):
            assert bias[m_idx] == pytest.approx(1.3, rel=0.05)
