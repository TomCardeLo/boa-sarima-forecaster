"""Tests for WMA_THRESHOLD_HIGH_VOLATILITY constant (Track H9b)."""

import inspect

import pandas as pd


def test_constant_is_importable_from_package():
    """The constant must be accessible via the top-level package."""
    from boa_forecaster import WMA_THRESHOLD_HIGH_VOLATILITY

    assert WMA_THRESHOLD_HIGH_VOLATILITY == 3.5


def test_constant_is_importable_from_module():
    """The constant must be accessible directly from the standardization module."""
    from boa_forecaster.standardization import WMA_THRESHOLD_HIGH_VOLATILITY

    assert WMA_THRESHOLD_HIGH_VOLATILITY == 3.5


def test_constant_produces_same_result_as_literal():
    """clip_outliers(series, threshold=WMA_THRESHOLD_HIGH_VOLATILITY) must
    return the same result as passing the literal 3.5."""
    from boa_forecaster.standardization import (
        WMA_THRESHOLD_HIGH_VOLATILITY,
        clip_outliers,
    )

    series = pd.Series([10.0, 12.0, 11.0, 500.0, 9.0, 1000.0, 13.0, 10.0])

    result_constant = clip_outliers(series, threshold=WMA_THRESHOLD_HIGH_VOLATILITY)
    result_literal = clip_outliers(series, threshold=3.5)

    pd.testing.assert_series_equal(result_constant, result_literal)


def test_default_threshold_unchanged():
    """The default threshold for clip_outliers must remain 2.5, not 3.5."""
    from boa_forecaster.standardization import clip_outliers

    # Verify via signature introspection.
    sig = inspect.signature(clip_outliers)
    assert sig.parameters["threshold"].default == 2.5

    # Also verify empirically: a series with a moderate spike that is clipped
    # by 2.5σ but NOT by 3.5σ must differ between the two calls.
    # Build a series where a spike sits between 2.5σ and 3.5σ so the two
    # thresholds produce different outputs.
    base = [10.0] * 20
    # Insert a spike that is ~3.0σ above the mean — clipped at 2.5, kept at 3.5.
    spike_series = pd.Series(base + [80.0])

    result_default = clip_outliers(spike_series)
    result_3_5 = clip_outliers(spike_series, threshold=3.5)

    assert not result_default.equals(result_3_5), (
        "Default (2.5σ) and 3.5σ thresholds should differ on a series with a "
        "moderate spike, but they returned identical results."
    )
