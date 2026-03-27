"""Tests for sarima_bayes.standardization: weighted_moving_stats, clip_outliers."""

import math

import pandas as pd
import pytest

from boa_forecaster.standardization import clip_outliers, weighted_moving_stats

# ---------------------------------------------------------------------------
# weighted_moving_stats (existing tests, preserved)
# ---------------------------------------------------------------------------


def test_no_nan_in_results(raw_series_with_outliers):
    data = raw_series_with_outliers
    results = [weighted_moving_stats(i, data) for i in range(len(data))]
    clipped_values = [r[2] for r in results]
    for cv in clipped_values:
        assert not math.isnan(cv)


def test_output_length(raw_series_with_outliers):
    data = raw_series_with_outliers
    results = [weighted_moving_stats(i, data) for i in range(len(data))]
    assert len(results) == len(data)


def test_spike_is_dampened(raw_series_with_outliers):
    # Index 2 has value 800 — should be clipped below 800 even at threshold=2.5
    data = raw_series_with_outliers
    _, _, clipped = weighted_moving_stats(2, data)
    assert clipped < 800


def test_single_element_series():
    # No neighbours → returns (0.0, 0.0, original_value)
    result = weighted_moving_stats(0, [42])
    assert result == (0.0, 0.0, 42.0)


def test_constant_series_clipped_values():
    data = [5, 5, 5, 5, 5]
    for i in range(len(data)):
        _, _, clipped = weighted_moving_stats(i, data)
        assert clipped == 5.0


def test_returns_three_tuple(raw_series_with_outliers):
    data = raw_series_with_outliers
    result = weighted_moving_stats(0, data)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# clip_outliers — new tests
# ---------------------------------------------------------------------------


@pytest.fixture
def series_with_outlier():
    """Series where index 5 is a clear outlier."""
    values = [100.0, 102.0, 98.0, 101.0, 99.0, 900.0, 103.0, 97.0, 100.0, 101.0]
    return pd.Series(values)


class TestClipOutliers:
    def test_sigma_clips_outlier(self, series_with_outlier):
        result = clip_outliers(series_with_outlier, method="sigma", threshold=2.5)
        assert result[5] < 900

    def test_iqr_no_nan(self, series_with_outlier):
        result = clip_outliers(series_with_outlier, method="iqr", threshold=1.5)
        assert not result.isna().any()

    def test_iqr_no_negatives(self, series_with_outlier):
        result = clip_outliers(series_with_outlier, method="iqr", threshold=1.5)
        assert (result >= 0).all()

    def test_sigma_threshold_1_clips_more_than_25(self, series_with_outlier):
        # Lower threshold → more values reach the boundary
        clipped_1 = clip_outliers(series_with_outlier, method="sigma", threshold=1.0)
        clipped_25 = clip_outliers(series_with_outlier, method="sigma", threshold=2.5)
        # At threshold=1.0 the upper bound is lower, so more values are clipped
        assert clipped_1.max() <= clipped_25.max()
        # And at least one value differs between the two clippings
        assert not clipped_1.equals(clipped_25)

    def test_nan_raises(self):
        s = pd.Series([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN"):
            clip_outliers(s)

    def test_unknown_method_raises(self, series_with_outlier):
        with pytest.raises(ValueError, match="unknown method"):
            clip_outliers(series_with_outlier, method="median")

    def test_output_never_negative(self):
        # Even if input has negative values, output is clipped to 0
        s = pd.Series([-50.0, -10.0, 0.0, 100.0, 200.0])
        result = clip_outliers(s, method="sigma", threshold=2.5)
        assert (result >= 0).all()

    def test_output_length_preserved(self, series_with_outlier):
        result = clip_outliers(series_with_outlier, method="sigma", threshold=2.5)
        assert len(result) == len(series_with_outlier)

    def test_output_index_preserved(self, series_with_outlier):
        result = clip_outliers(series_with_outlier, method="sigma", threshold=2.5)
        pd.testing.assert_index_equal(result.index, series_with_outlier.index)

    def test_sigma_warns_on_constant_series(self):
        s = pd.Series([5.0] * 10)
        with pytest.warns(UserWarning, match="zero standard deviation"):
            clip_outliers(s, method="sigma")

    def test_iqr_warns_on_constant_series(self):
        s = pd.Series([5.0] * 10)
        with pytest.warns(UserWarning, match="zero IQR"):
            clip_outliers(s, method="iqr")
