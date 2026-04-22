"""Model registry and base types for boa-forecaster v2.0.

The ``MODEL_REGISTRY`` maps model names to their ``ModelSpec`` classes.
Use ``register_model`` to add new models and ``get_model_spec`` to
instantiate a registered model by name.

Phase 1 ships with ``SARIMASpec`` auto-registered as ``"sarima"``.
Phase 2 adds ``RandomForestSpec`` auto-registered as ``"random_forest"``.
Phase 3 adds ``XGBoostSpec`` auto-registered as ``"xgboost"`` (optional dep).
Phase 4 adds ``LightGBMSpec`` auto-registered as ``"lightgbm"`` (optional dep).
"""

from __future__ import annotations

from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    IntParam,
    ModelSpec,
    OptimizationResult,
    SearchSpaceParam,
    suggest_from_space,
)
from boa_forecaster.models.random_forest import RandomForestSpec
from boa_forecaster.models.sarima import SARIMASpec

MODEL_REGISTRY: dict[str, type] = {}


def register_model(name: str, spec_cls: type) -> None:
    """Register a ``ModelSpec`` class under *name* in the global registry.

    Args:
        name: Unique identifier (e.g. ``"random_forest"``).
        spec_cls: Class implementing the ``ModelSpec`` protocol.
    """
    MODEL_REGISTRY[name] = spec_cls


def get_model_spec(name: str, **kwargs) -> ModelSpec:
    """Instantiate a registered model by name.

    Args:
        name: Registered model name.
        **kwargs: Passed to the model class constructor.

    Returns:
        New ``ModelSpec`` instance.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Model '{name}' not registered. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](**kwargs)


class _MissingExtra:
    """Sentinel standing in for an optional ``ModelSpec`` when its extra is not installed.

    Instances are callable; invoking them raises ``ImportError`` with a message
    pointing at the correct ``pip install`` extra. This replaces the earlier
    ``XGBoostSpec = None`` fallback so that ``XGBoostSpec()`` produces a clear,
    actionable error instead of a cryptic ``TypeError: 'NoneType' not callable``.
    """

    def __init__(self, pkg: str, extras: str) -> None:
        self._pkg = pkg
        self._extras = extras

    def __call__(self, *args: object, **kwargs: object) -> object:
        raise ImportError(
            f"{self._pkg} is not installed. "
            f"Install with: pip install 'sarima-bayes[{self._extras}]'"
        )


# Auto-register built-in models
register_model("sarima", SARIMASpec)
register_model("random_forest", RandomForestSpec)

# Optional: XGBoost (requires xgboost package)
try:
    from boa_forecaster.models.xgboost import XGBoostSpec  # noqa: F401

    register_model("xgboost", XGBoostSpec)
except ImportError:
    XGBoostSpec = _MissingExtra("xgboost", "xgboost")  # type: ignore[assignment,misc]

# Optional: LightGBM (requires lightgbm package)
try:
    from boa_forecaster.models.lightgbm import LightGBMSpec  # noqa: F401

    register_model("lightgbm", LightGBMSpec)
except ImportError:
    LightGBMSpec = _MissingExtra("lightgbm", "lightgbm")  # type: ignore[assignment,misc]

# Optional: Prophet (requires prophet package)
try:
    from boa_forecaster.models.prophet import ProphetSpec  # noqa: F401

    register_model("prophet", ProphetSpec)
except ImportError:
    ProphetSpec = _MissingExtra("prophet", "prophet")  # type: ignore[assignment,misc]

# Optional: QuantileML (requires lightgbm OR xgboost — uses whichever is installed)
try:
    from boa_forecaster.models.quantile import (  # noqa: F401
        QuantileForecast,
        QuantileMLSpec,
    )

    register_model("quantile_ml", QuantileMLSpec)
except ImportError:
    QuantileMLSpec = _MissingExtra("lightgbm or xgboost", "ml")  # type: ignore[assignment,misc]
    QuantileForecast = None  # type: ignore[assignment,misc]

# Ensemble (Track D / X3) — registered last, after optional-ML imports are resolved.
from boa_forecaster.models.ensemble import (  # noqa: E402
    EnsembleSpec,
    build_ensemble,
)

register_model("ensemble", EnsembleSpec)

__all__ = [
    "ModelSpec",
    "OptimizationResult",
    "IntParam",
    "FloatParam",
    "CategoricalParam",
    "SearchSpaceParam",
    "suggest_from_space",
    "SARIMASpec",
    "RandomForestSpec",
    "XGBoostSpec",
    "LightGBMSpec",
    "ProphetSpec",
    "QuantileMLSpec",
    "QuantileForecast",
    "EnsembleSpec",
    "build_ensemble",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_spec",
]
