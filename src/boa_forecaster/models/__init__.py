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


# Auto-register built-in models
register_model("sarima", SARIMASpec)
register_model("random_forest", RandomForestSpec)

# Optional: XGBoost (requires xgboost package)
try:
    from boa_forecaster.models.xgboost import XGBoostSpec  # noqa: F401

    register_model("xgboost", XGBoostSpec)
except ImportError:
    XGBoostSpec = None  # type: ignore[assignment,misc]

# Optional: LightGBM (requires lightgbm package)
try:
    from boa_forecaster.models.lightgbm import LightGBMSpec  # noqa: F401

    register_model("lightgbm", LightGBMSpec)
except ImportError:
    LightGBMSpec = None  # type: ignore[assignment,misc]

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
    "MODEL_REGISTRY",
    "register_model",
    "get_model_spec",
]
