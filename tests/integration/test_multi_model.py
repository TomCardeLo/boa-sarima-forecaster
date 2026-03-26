"""Integration tests for multi-model benchmark comparison.

These tests exercise the full pipeline end-to-end:
  synthetic data → Bayesian optimisation → benchmark comparison → summary_table

They are slower than unit tests because they run actual Optuna studies.
Mark with ``pytest -m integration`` to run in isolation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn", reason="scikit-learn not installed")

from boa_forecaster.benchmarks import run_model_comparison, summary_table
from boa_forecaster.models.random_forest import RandomForestSpec
from boa_forecaster.models.sarima import SARIMASpec

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_group_df(n: int = 60, groups: list[str] | None = None) -> pd.DataFrame:
    if groups is None:
        groups = ["A"]
    rng = np.random.default_rng(99)
    rows = []
    for g in groups:
        t = np.arange(n)
        values = 100 + 0.5 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, n)
        index = pd.date_range("2019-01-01", periods=n, freq="MS")
        for date, val in zip(index, values):
            rows.append({"group": g, "date": date, "demand": val})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestEndToEndPipeline:
    """Full pipeline: optimise → compare → summarise."""

    @pytest.fixture(scope="class")
    def comparison_results(self):
        df = _make_group_df(n=60, groups=["A"])
        return run_model_comparison(
            df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec(), RandomForestSpec()],
            n_calls_per_model=5,
            n_folds=3,
            test_size=6,
            min_train_size=24,
            seed=42,
        )

    def test_returns_non_empty_dataframe(self, comparison_results):
        assert isinstance(comparison_results, pd.DataFrame)
        assert len(comparison_results) > 0

    def test_sarima_present(self, comparison_results):
        assert "sarima" in comparison_results["model"].unique()

    def test_random_forest_present(self, comparison_results):
        assert "random_forest" in comparison_results["model"].unique()

    def test_seasonal_naive_present(self, comparison_results):
        assert "seasonal_naive" in comparison_results["model"].unique()

    def test_optimized_flag_correct(self, comparison_results):
        for model in ("sarima", "random_forest"):
            rows = comparison_results[comparison_results["model"] == model]
            assert (rows["optimized"] == True).all()  # noqa: E712
        for baseline in ("seasonal_naive", "ETS"):
            rows = comparison_results[comparison_results["model"] == baseline]
            if len(rows) > 0:
                assert (rows["optimized"] == False).all()  # noqa: E712

    def test_smape_values_are_finite_positive(self, comparison_results):
        assert comparison_results["sMAPE"].notna().all()
        assert (comparison_results["sMAPE"] >= 0).all()

    def test_rmsle_values_are_finite_positive(self, comparison_results):
        assert comparison_results["RMSLE"].notna().all()
        assert (comparison_results["RMSLE"] >= 0).all()

    def test_fold_column_has_expected_values(self, comparison_results):
        folds = set(comparison_results["fold"].unique())
        assert folds == {1, 2, 3}


class TestSummaryTableMultiModel:
    """summary_table correctness on multi-model comparison output."""

    @pytest.fixture(scope="class")
    def summary(self):
        df = _make_group_df(n=60, groups=["A"])
        results = run_model_comparison(
            df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec(), RandomForestSpec()],
            n_calls_per_model=5,
            n_folds=3,
            test_size=6,
            min_train_size=24,
            seed=42,
        )
        return summary_table(results, group_cols=["group"])

    def test_returns_dataframe(self, summary):
        assert isinstance(summary, pd.DataFrame)

    def test_has_beats_naive_column(self, summary):
        assert "beats_naive" in summary.columns

    def test_seasonal_naive_never_beats_itself(self, summary):
        naive_row = summary[summary["model"] == "seasonal_naive"]
        assert (naive_row["beats_naive"] == False).all()  # noqa: E712

    def test_one_row_per_model(self, summary):
        # sarima + random_forest + seasonal_naive + ETS + AutoARIMA = 5
        assert len(summary) == 5

    def test_sorted_ascending_by_smape(self, summary):
        smapes = summary["sMAPE_mean"].tolist()
        assert smapes == sorted(smapes)

    def test_optimized_column_present(self, summary):
        assert "optimized" in summary.columns

    def test_optimized_models_flagged_correctly(self, summary):
        for model in ("sarima", "random_forest"):
            row = summary[summary["model"] == model]
            assert bool(row["optimized"].values[0]) is True


class TestReproducibility:
    """Same seed → same best_params → same forecast results."""

    def test_same_seed_produces_same_results(self):
        df = _make_group_df(n=60, groups=["A"])
        common_kwargs = dict(
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=5,
            n_folds=3,
            test_size=6,
            min_train_size=24,
            seed=42,
        )
        r1 = run_model_comparison(df, **common_kwargs)
        r2 = run_model_comparison(df, **common_kwargs)
        sarima1 = r1[r1["model"] == "sarima"]["sMAPE"].values
        sarima2 = r2[r2["model"] == "sarima"]["sMAPE"].values
        np.testing.assert_allclose(sarima1, sarima2)


class TestMultiGroupComparison:
    """Multi-group DataFrame produces independent results per group."""

    @pytest.fixture(scope="class")
    def multi_group_results(self):
        df = _make_group_df(n=60, groups=["A", "B"])
        return run_model_comparison(
            df,
            group_cols=["group"],
            target_col="demand",
            date_col="date",
            model_specs=[SARIMASpec()],
            n_calls_per_model=3,
            n_folds=3,
            test_size=6,
            min_train_size=24,
            seed=42,
        )

    def test_both_groups_present(self, multi_group_results):
        assert set(multi_group_results["group"].unique()) == {"A", "B"}

    def test_each_group_has_all_models(self, multi_group_results):
        for g in ("A", "B"):
            group_rows = multi_group_results[multi_group_results["group"] == g]
            assert "sarima" in group_rows["model"].unique()
            assert "seasonal_naive" in group_rows["model"].unique()
