"""Unit tests for sarima_bayes.validation."""

import logging

import numpy as np
import pandas as pd
import pytest

from boa_forecaster.validation import validate_by_group, walk_forward_validation

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

    def test_wfv_truncates_overlong_predictions(self, long_series):
        """model_fn returning ``test_size + 2`` predictions must be truncated
        to ``test_size`` (uses first ``test_size``); sMAPE must be non-NaN."""
        test_size = 6

        result = walk_forward_validation(
            long_series,
            _fixed_horizon_model(test_size + 2),
            n_folds=3,
            test_size=test_size,
            min_train_size=24,
        )
        assert len(result) == 3
        assert result["sMAPE"].notna().all()
        assert (result["sMAPE"] >= 0).all()

    def test_wfv_nan_on_undersized_predictions(self, long_series, caplog):
        """model_fn returning fewer predictions than ``test_size`` must raise
        internally and surface as NaN metrics (with a WARNING logged)."""
        test_size = 6

        caplog.set_level(logging.WARNING, logger="boa_forecaster.validation")
        result = walk_forward_validation(
            long_series,
            _fixed_horizon_model(test_size - 2),
            n_folds=3,
            test_size=test_size,
            min_train_size=24,
        )
        assert result["sMAPE"].isna().all()
        assert any(
            "predictions" in rec.message.lower() for rec in caplog.records
        ), "Expected a WARNING about undersized predictions."


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


class TestValidateByGroupFailureIsolation:
    """Per-group failures must be isolated: one group's exception cannot
    take down the others, and when *every* group fails the function must
    return an empty DataFrame rather than raise.

    We trigger *group-level* failures (inside ``validate_by_group``'s outer
    try/except) by giving a group too-short a series — this makes
    ``walk_forward_validation`` itself raise ``ValueError`` rather than
    catching the failure per-fold.
    """

    @staticmethod
    def _naive_fn(train: pd.Series) -> pd.Series:
        freq = train.index.freq or "MS"
        idx = pd.date_range(start=train.index[-1], periods=7, freq=freq)[1:]
        return pd.Series([float(train.iloc[-1])] * 6, index=idx)

    @staticmethod
    def _make_mixed_length_df(
        ok_skus: list[int], short_skus: list[int]
    ) -> pd.DataFrame:
        """Build a df where ``ok_skus`` have 60 periods (WFV passes) and
        ``short_skus`` have only 10 periods (WFV raises)."""
        rng = np.random.default_rng(7)
        rows: list[dict] = []
        for sku in ok_skus:
            dates = pd.date_range("2019-01-01", periods=60, freq="MS")
            values = 100.0 + rng.normal(0, 2, 60)
            for d, v in zip(dates, values):
                rows.append({"SKU": sku, "Date": d, "CS": float(v)})
        for sku in short_skus:
            dates = pd.date_range("2019-01-01", periods=10, freq="MS")
            values = 100.0 + rng.normal(0, 2, 10)
            for d, v in zip(dates, values):
                rows.append({"SKU": sku, "Date": d, "CS": float(v)})
        return pd.DataFrame(rows)

    def test_one_failing_group_other_groups_returned(self, caplog):
        df = self._make_mixed_length_df(ok_skus=[1, 3], short_skus=[2])
        caplog.set_level(logging.WARNING, logger="boa_forecaster.validation")

        result = validate_by_group(
            df,
            group_cols=["SKU"],
            target_col="CS",
            date_col="Date",
            model_fn=self._naive_fn,
            freq="MS",
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )

        assert not result.empty
        returned_skus = {int(v) for v in result["SKU"].unique()}
        assert returned_skus == {
            1,
            3,
        }, f"Expected non-failing groups only, got {returned_skus}"

    def test_one_failing_group_logs_warning(self, caplog):
        df = self._make_mixed_length_df(ok_skus=[1, 3], short_skus=[2])
        caplog.set_level(logging.WARNING, logger="boa_forecaster.validation")

        validate_by_group(
            df,
            group_cols=["SKU"],
            target_col="CS",
            date_col="Date",
            model_fn=self._naive_fn,
            freq="MS",
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )

        # Look for the group-level warning emitted by validate_by_group
        # ("Group ... failed: ...") — distinct from WFV's per-fold warnings.
        group_level = [
            rec
            for rec in caplog.records
            if rec.message.startswith("Group") and "failed" in rec.message
        ]
        assert group_level, (
            "Expected a group-level WARNING from validate_by_group's outer "
            "except block."
        )

    def test_all_groups_failing_returns_empty_df(self, caplog):
        df = self._make_mixed_length_df(ok_skus=[], short_skus=[1, 2, 3])
        caplog.set_level(logging.WARNING, logger="boa_forecaster.validation")

        result = validate_by_group(
            df,
            group_cols=["SKU"],
            target_col="CS",
            date_col="Date",
            model_fn=self._naive_fn,
            freq="MS",
            n_folds=3,
            test_size=6,
            min_train_size=24,
        )

        assert isinstance(result, pd.DataFrame)
        assert result.empty
