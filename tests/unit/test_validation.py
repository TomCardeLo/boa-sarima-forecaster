"""Unit tests for sarima_bayes.validation."""

import numpy as np
import pandas as pd
import pytest

from sarima_bayes.validation import walk_forward_validation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _naive_model(train: pd.Series) -> pd.Series:
    """Always returns a constant forecast equal to the last training value."""
    # We don't know test_size here, but WFV passes train only.
    # Use 12 as a safe upper bound — the test parametrisation keeps test_size<=12.
    horizon = 12
    freq = train.index.freq or "MS"
    idx = pd.date_range(start=train.index[-1], periods=horizon + 1, freq=freq)[1:]
    return pd.Series([train.iloc[-1]] * horizon, index=idx)


def _fixed_horizon_model(horizon):
    """Factory: returns a model_fn that always predicts 'horizon' steps."""

    def _fn(train: pd.Series) -> pd.Series:
        freq = train.index.freq or "MS"
        idx = pd.date_range(start=train.index[-1], periods=horizon + 1, freq=freq)[1:]
        return pd.Series([train.iloc[-1]] * horizon, index=idx)

    return _fn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWalkForwardValidation:
    def test_wfv_returns_3_rows(self, long_series):
        result = walk_forward_validation(
            long_series,
            _fixed_horizon_model(6),
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        assert len(result) == 3

    def test_wfv_columns(self, long_series):
        result = walk_forward_validation(
            long_series,
            _fixed_horizon_model(6),
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        required = {
            "fold",
            "train_start",
            "train_end",
            "test_start",
            "test_end",
            "sMAPE",
            "RMSLE",
        }
        assert required.issubset(result.columns)

    def test_wfv_smape_nonneg(self, long_series):
        result = walk_forward_validation(
            long_series,
            _fixed_horizon_model(6),
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        assert (result["sMAPE"] >= 0).all()

    def test_wfv_raises_on_1_fold(self, long_series):
        with pytest.raises(ValueError):
            walk_forward_validation(long_series, _fixed_horizon_model(6), n_folds=1)

    def test_wfv_raises_if_too_short(self):
        rng = np.random.default_rng(0)
        short = pd.Series(
            rng.normal(100, 5, 30),
            index=pd.date_range("2020-01-01", periods=30, freq="MS"),
        )
        with pytest.raises(ValueError):
            walk_forward_validation(
                short,
                _fixed_horizon_model(12),
                n_folds=3,
                test_size=12,
                min_train_size=24,
            )

    def test_wfv_custom_metrics(self, long_series):
        custom = {"MAE": lambda a, b: float(np.mean(np.abs(np.array(a) - np.array(b))))}
        result = walk_forward_validation(
            long_series,
            _fixed_horizon_model(6),
            n_folds=3,
            test_size=6,
            min_train_size=24,
            metrics_fn=custom,
        )
        assert "MAE" in result.columns

    def test_wfv_fold_numbers_are_1indexed(self, long_series):
        result = walk_forward_validation(
            long_series,
            _fixed_horizon_model(6),
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        assert list(result["fold"]) == [1, 2, 3]

    def test_wfv_nan_on_failing_model(self, long_series):
        def _bad_model(train):
            raise RuntimeError("intentional failure")

        result = walk_forward_validation(
            long_series,
            _bad_model,
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        assert result["sMAPE"].isna().all()
