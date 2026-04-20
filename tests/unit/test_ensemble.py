"""Tests for ``EnsembleSpec`` + ``build_ensemble``.

The ensemble wraps two pre-optimised forecasters and weight-averages their
outputs.  These tests cover:

1. Top-level import path — ``from boa_forecaster import EnsembleSpec``.
2. Registry entry — ``"ensemble"`` is registered after Track A's proxy pattern.
3. ``evaluate`` raises ``NotImplementedError`` (not a TPE candidate).
4. The ``inverse_cv_loss`` weighting beats the worse of two members on a
   synthetic series.
5. Explicit-weight construction and empty-members guard.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import pytest

from boa_forecaster import EnsembleSpec, build_ensemble
from boa_forecaster.metrics import smape
from boa_forecaster.models import MODEL_REGISTRY

# ── Helper: minimal ModelSpec wrappers ────────────────────────────────────────


class _ConstantSpec:
    """Trivial ``ModelSpec`` that always predicts ``value`` for a 6-step horizon.

    Not intended for production — it provides a deterministic member we can
    reason about inside the test.  Two instances with different values let us
    verify that the ensemble blend sits between them.
    """

    needs_features = False
    forecast_horizon = 6

    def __init__(self, name: str, value: float) -> None:
        self.name = name
        self.value = value

    @property
    def search_space(self) -> dict:
        return {}

    @property
    def warm_starts(self) -> list[dict]:
        return [{}]

    def suggest_params(self, trial) -> dict:
        return {}

    def evaluate(self, series, params, metric_fn, feature_config=None) -> float:
        return float(metric_fn(series.values, np.full_like(series.values, self.value)))

    def build_forecaster(
        self, params: dict, feature_config=None
    ) -> Callable[[pd.Series], pd.Series]:
        value = self.value
        horizon = self.forecast_horizon

        def forecaster(train: pd.Series) -> pd.Series:
            freq = train.index.freq or "MS"
            idx = pd.date_range(train.index[-1], periods=horizon + 1, freq=freq)[1:]
            return pd.Series(np.full(horizon, value, dtype=float), index=idx)

        return forecaster


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_series() -> pd.Series:
    """Flat series at y=100 — any constant forecast maps cleanly to sMAPE."""
    idx = pd.date_range("2020-01-01", periods=36, freq="MS")
    return pd.Series(np.full(36, 100.0), index=idx)


# ── 1–2. Import + registry ────────────────────────────────────────────────────


def test_ensemble_is_registered() -> None:
    assert "ensemble" in MODEL_REGISTRY
    assert MODEL_REGISTRY["ensemble"] is EnsembleSpec


def test_ensemble_reexported_from_top_level() -> None:
    # Already imported at module top; this asserts the symbol is truly there.
    import boa_forecaster

    assert boa_forecaster.EnsembleSpec is EnsembleSpec
    assert callable(boa_forecaster.build_ensemble)


# ── 3. evaluate() guard ───────────────────────────────────────────────────────


def test_evaluate_raises_not_implemented() -> None:
    ens = EnsembleSpec([_ConstantSpec("a", 100.0)], weighting="equal")
    with pytest.raises(NotImplementedError):
        ens.evaluate(pd.Series([1.0]), {}, smape)


# ── Input-validation guards ──────────────────────────────────────────────────


def test_empty_members_raises() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        EnsembleSpec([], weighting="equal")


def test_mismatched_explicit_weights_raises() -> None:
    specs = [_ConstantSpec("a", 100.0), _ConstantSpec("b", 80.0)]
    with pytest.raises(ValueError, match="weighting list"):
        EnsembleSpec(specs, weighting=[0.5])  # 1 weight, 2 members


def test_build_forecaster_missing_member_params_raises() -> None:
    specs = [_ConstantSpec("a", 100.0), _ConstantSpec("b", 80.0)]
    ens = EnsembleSpec(specs, weighting="equal")
    with pytest.raises(KeyError, match="Missing params"):
        ens.build_forecaster({"a": {}})  # "b" missing


# ── 4. Weighting strategies ──────────────────────────────────────────────────


def test_equal_weighting_averages_members(synthetic_series: pd.Series) -> None:
    a = _ConstantSpec("a", 120.0)
    b = _ConstantSpec("b", 80.0)
    ens = EnsembleSpec([a, b], weighting="equal")

    forecaster = ens.build_forecaster({"a": {}, "b": {}})
    pred = forecaster(synthetic_series)

    # Equal average of 120 and 80 is 100.
    assert len(pred) == 6
    assert np.allclose(pred.values, 100.0)


def test_explicit_weights_are_normalised(synthetic_series: pd.Series) -> None:
    a = _ConstantSpec("a", 120.0)
    b = _ConstantSpec("b", 80.0)
    # 3:1 in favour of a → weighted avg = (3*120 + 80) / 4 = 110.
    ens = EnsembleSpec([a, b], weighting=[0.75, 0.25])
    forecaster = ens.build_forecaster({"a": {}, "b": {}})
    pred = forecaster(synthetic_series)
    assert np.allclose(pred.values, 110.0)


def test_inverse_cv_loss_favours_better_member(
    synthetic_series: pd.Series,
) -> None:
    """``inverse_cv_loss`` should steer the average toward the lower-loss member."""
    good = _ConstantSpec("good", 100.0)  # predicts truth exactly
    bad = _ConstantSpec("bad", 50.0)  # way off
    scores = {"good": 0.01, "bad": 1.0}  # good has much lower loss
    ens = EnsembleSpec([good, bad], weighting="inverse_cv_loss", member_scores=scores)
    forecaster = ens.build_forecaster({"good": {}, "bad": {}})
    pred = forecaster(synthetic_series)

    # Weights: 1/0.01 vs 1/1.0 → 100 vs 1 → ~99% good
    # Average ≈ 100 * (100/101) + 50 * (1/101) ≈ 99.5
    assert pred.mean() > 99.0
    # Must also beat the worse member on sMAPE vs truth=100.
    bad_score = smape(np.full(6, 100.0), np.full(6, 50.0))
    ens_score = smape(np.full(6, 100.0), pred.values)
    assert ens_score < bad_score


def test_inverse_cv_loss_falls_back_when_scores_missing(
    synthetic_series: pd.Series,
) -> None:
    """No ``member_scores`` → fall back to equal weighting with a warning."""
    a = _ConstantSpec("a", 120.0)
    b = _ConstantSpec("b", 80.0)
    ens = EnsembleSpec([a, b], weighting="inverse_cv_loss")  # no scores

    forecaster = ens.build_forecaster({"a": {}, "b": {}})
    pred = forecaster(synthetic_series)

    assert np.allclose(pred.values, 100.0)  # equal blend of 120 and 80


# ── 5. build_ensemble helper ─────────────────────────────────────────────────


def test_build_ensemble_without_optimise(synthetic_series: pd.Series) -> None:
    """``optimise=False`` skips TPE and uses warm-start defaults."""
    specs = [_ConstantSpec("a", 120.0), _ConstantSpec("b", 80.0)]
    spec, params = build_ensemble(
        synthetic_series, specs, weighting="equal", optimise=False
    )

    assert isinstance(spec, EnsembleSpec)
    assert set(params) == {"a", "b"}
    forecaster = spec.build_forecaster(params)
    pred = forecaster(synthetic_series)
    assert np.allclose(pred.values, 100.0)


# ── E1: parallel build_ensemble reproducibility ─────────────────────────────


def test_build_ensemble_parallel_matches_sequential() -> None:
    """``n_jobs=1`` vs ``n_jobs=2`` must produce identical optimisation output.

    Each member is optimised by an Optuna study with the same seed, so the
    sequential and parallel paths must converge on bit-identical
    ``best_params`` and ``best_score``.  We use SARIMA (fast, core-only) to
    keep the test cheap; two members exercise the Parallel dispatch.
    """
    import inspect

    from boa_forecaster.models.sarima import SARIMASpec

    # ``n_jobs`` must be an **explicit** parameter on build_ensemble, not
    # silently forwarded via **optimize_kwargs.  Guard against regression:
    sig = inspect.signature(build_ensemble)
    assert "n_jobs" in sig.parameters, "build_ensemble must expose n_jobs kwarg"

    rng = np.random.default_rng(123)
    idx = pd.date_range("2019-01-01", periods=48, freq="MS")
    series = pd.Series(100.0 + np.arange(48) * 0.3 + rng.normal(0, 1.0, 48), index=idx)

    specs_seq = [SARIMASpec(), SARIMASpec()]
    specs_par = [SARIMASpec(), SARIMASpec()]
    # Differentiate member names so the params_per_member dict has distinct
    # keys (both default to "sarima" which collapses in the dict — not a
    # fair parallel-vs-sequential comparison).
    specs_seq[1].name = "sarima_b"
    specs_par[1].name = "sarima_b"

    spec_seq, params_seq = build_ensemble(
        series, specs_seq, weighting="equal", optimise=True, n_calls=5, n_jobs=1
    )
    spec_par, params_par = build_ensemble(
        series, specs_par, weighting="equal", optimise=True, n_calls=5, n_jobs=2
    )

    assert params_seq == params_par
    assert spec_seq.member_scores == spec_par.member_scores


# ── E3: needs_features reflects members ──────────────────────────────────────


def test_needs_features_false_when_no_ml_members() -> None:
    """Pure-SARIMA ensemble does not need feature engineering."""
    from boa_forecaster.models.sarima import SARIMASpec

    only_sarima = EnsembleSpec([SARIMASpec()], weighting="equal")
    assert only_sarima.needs_features is False


@pytest.mark.requires_sklearn
def test_needs_features_true_when_ml_member_present() -> None:
    """``EnsembleSpec.needs_features`` must be the OR across its members.

    Regression for Track E (v2.3) bug: the hardcoded class attribute
    ``needs_features = False`` lied when the ensemble contained tabular-ML
    members.  Downstream code that gates feature engineering on this
    attribute would skip it and hand the raw series to an ML model that
    needs a ``FeatureEngineer``.
    """
    pytest.importorskip("sklearn")
    from boa_forecaster.models.random_forest import RandomForestSpec
    from boa_forecaster.models.sarima import SARIMASpec

    mixed = EnsembleSpec([SARIMASpec(), RandomForestSpec()], weighting="equal")
    assert mixed.needs_features is True
