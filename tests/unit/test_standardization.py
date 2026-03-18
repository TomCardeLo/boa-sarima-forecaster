"""Tests for sarima_bayes.standardization: weighted_moving_stats."""

import math

from sarima_bayes.standardization import weighted_moving_stats


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
    # Index 2 has value 800 — should be clipped below 800
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
