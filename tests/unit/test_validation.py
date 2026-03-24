"""Unit tests for sarima_bayes.validation."""

import numpy as np
import pandas as pd
import pytest

from sarima_bayes.validation import validate_by_group, walk_forward_validation

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


# ---------------------------------------------------------------------------
# Tests for validate_by_group (previously uncovered)
# ---------------------------------------------------------------------------


def _make_monthly_df(n_skus: int = 2, n_periods: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    rows = []
    for sku in range(1, n_skus + 1):
        dates = pd.date_range("2019-01-01", periods=n_periods, freq="MS")
        values = 100.0 + rng.normal(0, 2, n_periods)
        for d, v in zip(dates, values):
            rows.append({"SKU": sku, "Date": d, "CS": float(v)})
    return pd.DataFrame(rows)


def _naive_group_fn(train: pd.Series) -> pd.Series:
    """Return a constant forecast of length 6 starting after the last training obs."""
    _f = train.index.freq or "MS"
    idx = pd.date_range(start=train.index[-1], periods=7, freq=_f)[1:]
    return pd.Series([train.iloc[-1]] * 6, index=idx)


class TestValidateByGroup:
    """validate_by_group was previously untested; these are the first coverage tests."""

    def test_returns_correct_row_count(self):
        df = _make_monthly_df(n_skus=2)
        result = validate_by_group(
            df,
            group_cols=["SKU"],
            target_col="CS",
            date_col="Date",
            model_fn=_naive_group_fn,
            freq="MS",
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        # 2 groups × 3 folds = 6 rows
        assert len(result) == 6

    def test_group_col_present_in_output(self):
        df = _make_monthly_df(n_skus=2)
        result = validate_by_group(
            df,
            group_cols=["SKU"],
            target_col="CS",
            date_col="Date",
            model_fn=_naive_group_fn,
            freq="MS",
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        assert "SKU" in result.columns

    def test_default_freq_unchanged(self):
        # Calling without freq= must default to "MS" and not raise.
        df = _make_monthly_df(n_skus=1)
        result = validate_by_group(
            df,
            group_cols=["SKU"],
            target_col="CS",
            date_col="Date",
            model_fn=_naive_group_fn,
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        assert len(result) == 3

    def test_metric_columns_present(self):
        df = _make_monthly_df(n_skus=1)
        result = validate_by_group(
            df,
            group_cols=["SKU"],
            target_col="CS",
            date_col="Date",
            model_fn=_naive_group_fn,
            freq="MS",
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )
        assert {"sMAPE", "RMSLE"}.issubset(result.columns)
