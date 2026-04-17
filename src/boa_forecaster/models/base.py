"""Core types and Protocol for the boa-forecaster model plugin system.

Every model that implements ``ModelSpec`` can be optimised by the generic
``optimize_model`` TPE engine without touching the optimiser internals.

Design notes
------------
* ``SearchSpaceParam`` uses ``Union[...]`` (not ``|``) so it evaluates at
  runtime on Python 3.9 where ``X | Y`` as a value requires 3.10+.
* ``ModelSpec`` is ``@runtime_checkable`` so ``isinstance(obj, ModelSpec)``
  works for structural duck-type checks in tests and user code.
* ``FeatureConfig`` is imported only under ``TYPE_CHECKING`` to avoid a
  circular import: ``features.py`` imports from this module, so this module
  must not import ``features.py`` at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, Union, runtime_checkable

import joblib
import optuna

if TYPE_CHECKING:
    import pandas as pd

    from boa_forecaster.features import FeatureConfig


# ── Search-space parameter descriptors ───────────────────────────────────────


@dataclass
class IntParam:
    """Integer hyper-parameter for Optuna's ``suggest_int``."""

    low: int
    high: int
    step: int = 1
    log: bool = False


@dataclass
class FloatParam:
    """Continuous hyper-parameter for Optuna's ``suggest_float``."""

    low: float
    high: float
    log: bool = False


@dataclass
class CategoricalParam:
    """Categorical hyper-parameter for Optuna's ``suggest_categorical``.

    ``choices`` may mix types (e.g. ``["sqrt", 0.5, "log2"]``).
    """

    choices: list


# Runtime type alias — must use Union on Python 3.9
SearchSpaceParam = Union[IntParam, FloatParam, CategoricalParam]


# ── Optimisation result ───────────────────────────────────────────────────────


@dataclass
class OptimizationResult:
    """Returned by ``optimize_model`` after TPE search completes.

    Attributes
    ----------
    best_params
        Best hyper-parameters found by the search, or the first ``warm_starts``
        entry (or ``{}``) when ``is_fallback`` is ``True``.
    best_score
        Best metric value, or ``OPTIMIZER_PENALTY`` when ``is_fallback`` is ``True``.
    n_trials
        Number of completed trials, or ``0`` when ``is_fallback`` is ``True``.
    model_name
        Name of the ``ModelSpec`` that was optimised.
    is_fallback
        ``True`` when the Optuna study crashed and this result carries default
        (warm-start) parameters and ``OPTIMIZER_PENALTY`` instead of real search
        output.  Callers that need to distinguish a successful optimisation
        from a soft-failure should branch on this flag.
    """

    best_params: dict
    best_score: float
    n_trials: int
    model_name: str
    is_fallback: bool = False

    def save(self, path: str | Path) -> None:
        """Persist this result to *path* using joblib serialisation."""
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> OptimizationResult:
        """Load and return an ``OptimizationResult`` previously saved to *path*."""
        return joblib.load(path)


# ── Helper ────────────────────────────────────────────────────────────────────


def suggest_from_space(
    trial: optuna.Trial, search_space: dict[str, SearchSpaceParam]
) -> dict[str, object]:
    """Translate a ``search_space`` dict into Optuna trial suggestions.

    Args:
        trial: Active Optuna trial.
        search_space: Mapping of parameter name → ``SearchSpaceParam``.

    Returns:
        Dict of sampled parameter values.

    Raises:
        TypeError: If an unsupported param type is encountered.

    Note:
        Optuna does not allow ``log=True`` together with ``step != 1`` for
        ``IntParam``.  When ``IntParam.log is True`` the ``step`` field is
        silently ignored so the call does not raise.
    """
    params: dict[str, object] = {}
    for name, param in search_space.items():
        if isinstance(param, IntParam):
            if param.log:
                # log scale and custom step are mutually exclusive in Optuna
                params[name] = trial.suggest_int(name, param.low, param.high, log=True)
            else:
                params[name] = trial.suggest_int(
                    name, param.low, param.high, step=param.step
                )
        elif isinstance(param, FloatParam):
            params[name] = trial.suggest_float(
                name, param.low, param.high, log=param.log
            )
        elif isinstance(param, CategoricalParam):
            params[name] = trial.suggest_categorical(name, param.choices)
        else:
            raise TypeError(f"Unsupported search-space param type: {type(param)}")
    return params


# ── Protocol ──────────────────────────────────────────────────────────────────


@runtime_checkable
class ModelSpec(Protocol):
    """Structural contract for any model optimisable with the TPE engine.

    Implementing this protocol (structural sub-typing) is sufficient — no
    explicit inheritance is required.  The four abstract members are:

    Attributes
    ----------
    name
        Unique identifier used in ``MODEL_REGISTRY`` and ``OptimizationResult``.
    needs_features
        ``True`` → the model requires a ``FeatureEngineer`` (tabular ML).
        ``False`` → the model consumes a raw ``pd.Series`` (SARIMA, ETS…).

    Properties
    ----------
    search_space
        Mapping of parameter name → ``SearchSpaceParam`` that describes the
        Optuna search space for this model.
    warm_starts
        List of parameter dicts pre-seeded into the Optuna study before TPE
        takes over.

    Methods
    -------
    suggest_params(trial)
        Sample one point from the search space.
    evaluate(series, params, metric_fn, feature_config)
        Fit the model with ``params`` and return the scalar metric score.
        Must return ``OPTIMIZER_PENALTY`` on failure — never raise.
    build_forecaster(params, feature_config)
        Return a callable ``forecaster(train: pd.Series) -> pd.Series``.
    """

    name: str
    needs_features: bool

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]: ...

    @property
    def warm_starts(self) -> list[dict]: ...

    def suggest_params(self, trial: optuna.Trial) -> dict: ...

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn,
        feature_config: FeatureConfig | None = None,
    ) -> float: ...

    def build_forecaster(
        self,
        params: dict,
        feature_config: FeatureConfig | None = None,
    ) -> Callable[[pd.Series], pd.Series]: ...
