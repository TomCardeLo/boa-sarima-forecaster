"""Shared pytest fixtures for the sarima_bayes test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_series():
    """50-point monthly time series with trend + noise, no nulls, no zeros."""
    rng = np.random.default_rng(42)
    trend = np.arange(50) * 0.5
    noise = rng.normal(0, 2, 50)
    values = 100 + trend + noise
    index = pd.date_range("2020-01-01", periods=50, freq="MS")
    return pd.Series(values, index=index)


@pytest.fixture
def raw_series_with_outliers():
    """Plain list with a spike at index 2 and some zeros."""
    return [100, 110, 800, 105, 95, 0, 102, 98, 107, 0, 103, 99]


@pytest.fixture
def long_series():
    """60-point monthly time series with trend + noise, no nulls, no zeros."""
    rng = np.random.default_rng(7)
    trend = np.arange(60) * 0.5
    noise = rng.normal(0, 2, 60)
    values = 100 + trend + noise
    index = pd.date_range("2019-01-01", periods=60, freq="MS")
    return pd.Series(values, index=index)
