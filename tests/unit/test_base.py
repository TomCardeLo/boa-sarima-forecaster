"""Tests for boa_forecaster.models.base: param types, suggest_from_space, Protocol."""

from __future__ import annotations

import optuna
import pytest

from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ModelSpec,
    OptimizationResult,
    SearchSpaceParam,
    suggest_from_space,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_trial(search_space: dict) -> optuna.Trial:
    """Create a real Optuna trial for the given search space."""
    sampler = optuna.samplers.RandomSampler(seed=0)
    study = optuna.create_study(sampler=sampler)
    return study.ask()


class _ValidSpec:
    """Minimal class satisfying the ModelSpec protocol."""

    name = "test_model"
    needs_features = False

    @property
    def search_space(self) -> dict:
        return {}

    @property
    def warm_starts(self) -> list[dict]:
        return []

    def suggest_params(self, trial):
        return {}

    def evaluate(self, series, params, metric_fn, feature_config=None) -> float:
        return 0.0

    def build_forecaster(self, params, feature_config=None):
        return lambda train: train


# ── IntParam ──────────────────────────────────────────────────────────────────


def test_int_param_defaults():
    p = IntParam(0, 3)
    assert p.low == 0
    assert p.high == 3
    assert p.step == 1
    assert p.log is False


def test_int_param_custom():
    p = IntParam(2, 20, step=2, log=False)
    assert p.step == 2


# ── FloatParam ────────────────────────────────────────────────────────────────


def test_float_param_defaults():
    p = FloatParam(0.01, 1.0)
    assert p.low == 0.01
    assert p.high == 1.0
    assert p.log is False


def test_float_param_log():
    p = FloatParam(1e-4, 1.0, log=True)
    assert p.log is True


# ── CategoricalParam ──────────────────────────────────────────────────────────


def test_categorical_param_mixed_types():
    p = CategoricalParam(["sqrt", 0.5, "log2"])
    assert len(p.choices) == 3
    assert "sqrt" in p.choices


# ── OptimizationResult ────────────────────────────────────────────────────────


def test_optimization_result_fields():
    r = OptimizationResult(
        best_params={"p": 1},
        best_score=0.5,
        n_trials=10,
        model_name="sarima",
    )
    assert r.best_params == {"p": 1}
    assert r.best_score == 0.5
    assert r.n_trials == 10
    assert r.model_name == "sarima"


# ── suggest_from_space ────────────────────────────────────────────────────────


def test_suggest_from_space_int():
    space = {"k": IntParam(1, 5)}
    trial = _make_trial(space)
    params = suggest_from_space(trial, space)
    assert "k" in params
    assert isinstance(params["k"], int)
    assert 1 <= params["k"] <= 5


def test_suggest_from_space_float():
    space = {"lr": FloatParam(0.001, 0.1)}
    trial = _make_trial(space)
    params = suggest_from_space(trial, space)
    assert "lr" in params
    assert isinstance(params["lr"], float)
    assert 0.001 <= params["lr"] <= 0.1


def test_suggest_from_space_categorical():
    space = {"feat": CategoricalParam(["sqrt", "log2", 0.5])}
    trial = _make_trial(space)
    params = suggest_from_space(trial, space)
    assert "feat" in params
    assert params["feat"] in ["sqrt", "log2", 0.5]


def test_suggest_from_space_int_log():
    """IntParam with log=True must not raise (step is ignored)."""
    space = {"n": IntParam(10, 1000, log=True)}
    trial = _make_trial(space)
    params = suggest_from_space(trial, space)
    assert "n" in params
    assert 10 <= params["n"] <= 1000


def test_suggest_from_space_unsupported_type():
    trial = _make_trial({})
    with pytest.raises(TypeError):
        suggest_from_space(trial, {"bad": "not_a_param"})


def test_suggest_from_space_multiple_params():
    space = {
        "p": IntParam(0, 3),
        "lr": FloatParam(0.01, 0.3),
        "method": CategoricalParam(["a", "b"]),
    }
    trial = _make_trial(space)
    params = suggest_from_space(trial, space)
    assert set(params.keys()) == {"p", "lr", "method"}


# ── ModelSpec Protocol ────────────────────────────────────────────────────────


def test_model_spec_isinstance_valid():
    assert isinstance(_ValidSpec(), ModelSpec)


def test_model_spec_isinstance_missing_method():
    class _Incomplete:
        name = "x"
        needs_features = False

        @property
        def search_space(self):
            return {}

        @property
        def warm_starts(self):
            return []

        def suggest_params(self, trial):
            return {}

        def evaluate(self, series, params, metric_fn, feature_config=None):
            return 0.0

        # build_forecaster is missing

    assert not isinstance(_Incomplete(), ModelSpec)


def test_search_space_param_union():
    """Verify SearchSpaceParam is usable for isinstance checks."""
    assert isinstance(IntParam(0, 1), (IntParam, FloatParam, CategoricalParam))
    assert isinstance(FloatParam(0.0, 1.0), (IntParam, FloatParam, CategoricalParam))
    assert isinstance(
        CategoricalParam([1, 2]), (IntParam, FloatParam, CategoricalParam)
    )
