"""Tests for sarima_bayes.standardization: weighted_moving_stats, clip_outliers."""

import math

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.standardization import (
    clip_outliers,
    weighted_moving_stats,
    weighted_moving_stats_batch,
    weighted_moving_stats_series,
)

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
# weighted_moving_stats_batch — vectorized equivalence tests
# ---------------------------------------------------------------------------

_DATASETS = [
    [100, 110, 800, 105, 95, 0, 102, 98, 107, 0, 103, 99],  # outliers + zeros
    [42],  # single element
    [5, 5, 5, 5, 5],  # constant series
    [0, 0, 0],  # all-zero series
    list(range(1, 51)),  # 50-point ramp
]


def _loop_ref(data, **kwargs):
    """Reference: row-by-row results as three lists."""
    ref = [weighted_moving_stats(i, data, **kwargs) for i in range(len(data))]
    means = [r[0] for r in ref]
    stds = [r[1] for r in ref]
    clips = [r[2] for r in ref]
    return means, stds, clips


def test_batch_matches_loop_default():
    """Batch results must be numerically equivalent to the row-by-row loop.

    Means and stds use assert_allclose (rtol=1e-10) because different FP
    operation ordering causes ~1e-14 differences between the scalar loop and
    the vectorised path — mathematically identical, not bit-identical.
    Clipped values are rounded integers so they must match exactly.
    """
    for data in _DATASETS:
        means_ref, stds_ref, clips_ref = _loop_ref(data)
        means_b, stds_b, clips_b = weighted_moving_stats_batch(data)
        np.testing.assert_allclose(means_b, means_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(stds_b, stds_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(clips_b, clips_ref)


def test_batch_window_size_variations():
    data = [10, 20, 30, 40, 50, 60, 70]
    for ws in (1, 2, 3):
        means_ref, stds_ref, clips_ref = _loop_ref(data, window_size=ws)
        means_b, stds_b, clips_b = weighted_moving_stats_batch(data, window_size=ws)
        np.testing.assert_allclose(means_b, means_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(stds_b, stds_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(clips_b, clips_ref)


def test_batch_custom_threshold():
    data = [100, 110, 800, 105, 95]
    for thr in (1.0, 2.0, 3.5):
        means_ref, stds_ref, clips_ref = _loop_ref(data, threshold=thr)
        means_b, stds_b, clips_b = weighted_moving_stats_batch(data, threshold=thr)
        np.testing.assert_allclose(means_b, means_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(stds_b, stds_ref, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(clips_b, clips_ref)


def test_batch_return_types():
    means, stds, clips = weighted_moving_stats_batch([1, 2, 3])
    assert isinstance(means, np.ndarray)
    assert isinstance(stds, np.ndarray)
    assert isinstance(clips, np.ndarray)
    assert means.shape == (3,)
    assert stds.shape == (3,)
    assert clips.shape == (3,)


def test_batch_empty_input():
    means, stds, clips = weighted_moving_stats_batch([])
    assert means.shape == (0,)
    assert stds.shape == (0,)
    assert clips.shape == (0,)


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


# ---------------------------------------------------------------------------
# weighted_moving_stats_series — vectorized bulk version
# ---------------------------------------------------------------------------


def _per_row_reference(data, window_size=3, threshold=2.5):
    """Reference implementation: loop the per-row function."""
    means, stds, clipped = [], [], []
    for i in range(len(data)):
        m, s, c = weighted_moving_stats(
            i, data, window_size=window_size, threshold=threshold
        )
        means.append(m)
        stds.append(s)
        clipped.append(c)
    return np.array(means), np.array(stds), np.array(clipped, dtype=float)


class TestWeightedMovingStatsSeries:
    def test_equivalence_with_outliers(self, raw_series_with_outliers):
        data = raw_series_with_outliers
        ref_m, ref_s, ref_c = _per_row_reference(data)
        m, s, c = weighted_moving_stats_series(data)
        np.testing.assert_allclose(m, ref_m, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s, ref_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(c, ref_c)

    def test_equivalence_custom_threshold(self, raw_series_with_outliers):
        ref_m, ref_s, ref_c = _per_row_reference(
            raw_series_with_outliers, threshold=1.0
        )
        m, s, c = weighted_moving_stats_series(raw_series_with_outliers, threshold=1.0)
        np.testing.assert_allclose(m, ref_m, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s, ref_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(c, ref_c)

    def test_equivalence_large_window_size(self, raw_series_with_outliers):
        # window_size > len(_REFERENCE_WEIGHTS): far neighbours get weight 0
        ref_m, ref_s, ref_c = _per_row_reference(
            raw_series_with_outliers, window_size=5
        )
        m, s, c = weighted_moving_stats_series(raw_series_with_outliers, window_size=5)
        np.testing.assert_allclose(m, ref_m, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s, ref_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(c, ref_c)

    def test_equivalence_long_random_series(self):
        rng = np.random.default_rng(123)
        data = rng.integers(0, 1000, size=200).tolist()
        ref_m, ref_s, ref_c = _per_row_reference(data)
        m, s, c = weighted_moving_stats_series(data)
        np.testing.assert_allclose(m, ref_m, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(s, ref_s, rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(c, ref_c)

    def test_empty_input(self):
        m, s, c = weighted_moving_stats_series([])
        assert m.shape == (0,)
        assert s.shape == (0,)
        assert c.shape == (0,)

    def test_single_element(self):
        # No neighbours → (0, 0, original_value) not rounded
        m, s, c = weighted_moving_stats_series([42])
        assert m.tolist() == [0.0]
        assert s.tolist() == [0.0]
        assert c.tolist() == [42.0]

    def test_two_elements(self):
        ref_m, ref_s, ref_c = _per_row_reference([10, 20])
        m, s, c = weighted_moving_stats_series([10, 20])
        np.testing.assert_allclose(m, ref_m, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(s, ref_s, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(c, ref_c)

    def test_constant_series_clipped_to_constant(self):
        m, s, c = weighted_moving_stats_series([5, 5, 5, 5, 5])
        np.testing.assert_array_equal(c, np.array([5.0, 5.0, 5.0, 5.0, 5.0]))
        np.testing.assert_allclose(s, np.zeros(5), atol=1e-12)

    def test_accepts_numpy_array(self, raw_series_with_outliers):
        ref = _per_row_reference(raw_series_with_outliers)
        got = weighted_moving_stats_series(np.asarray(raw_series_with_outliers))
        for r, g in zip(ref, got):
            np.testing.assert_allclose(g, r, rtol=1e-12, atol=1e-12)

    def test_accepts_pandas_series(self, raw_series_with_outliers):
        ref = _per_row_reference(raw_series_with_outliers)
        got = weighted_moving_stats_series(pd.Series(raw_series_with_outliers))
        for r, g in zip(ref, got):
            np.testing.assert_allclose(g, r, rtol=1e-12, atol=1e-12)

    def test_output_shapes_match_input(self):
        data = list(range(50))
        m, s, c = weighted_moving_stats_series(data)
        assert m.shape == (50,)
        assert s.shape == (50,)
        assert c.shape == (50,)
