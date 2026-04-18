"""Tests for optional-dependency fallbacks in boa_forecaster.models.

When an optional extra (``xgboost`` or ``lightgbm``) is not installed, the
corresponding spec class is replaced by a ``_MissingExtra`` sentinel.
Calling the sentinel must raise ``ImportError`` with a clear hint at the
correct ``pip install`` extra — never a cryptic ``NoneType`` error.

The test strategy splits into two layers:

* Unit test of the sentinel itself (no import games).
* Integration test that reimports ``boa_forecaster.models`` in an isolated
  ``sys.modules`` snapshot so the real package state is untouched for the
  rest of the session.
"""

from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import contextmanager

import pytest

from boa_forecaster.models import _MissingExtra

# ── Direct sentinel behaviour (no module gymnastics) ────────────────────────


def test_missing_extra_instance_raises_on_call() -> None:
    sentinel = _MissingExtra("xgboost", "xgboost")
    with pytest.raises(ImportError, match="xgboost is not installed"):
        sentinel()


def test_missing_extra_message_points_at_extra_slot() -> None:
    sentinel = _MissingExtra("lightgbm", "lightgbm")
    with pytest.raises(ImportError) as excinfo:
        sentinel()
    assert "sarima-bayes[lightgbm]" in str(excinfo.value)


def test_missing_extra_swallows_positional_and_keyword_args() -> None:
    """The sentinel mimics a class constructor: any call shape raises."""
    sentinel = _MissingExtra("xgboost", "xgboost")
    with pytest.raises(ImportError):
        sentinel(forecast_horizon=6, seasonal_period=12)


# ── Integration: simulate missing optional dep at module load ───────────────


@contextmanager
def _simulated_missing(pkg: str):
    """Reimport ``boa_forecaster.models`` as if *pkg* failed to import.

    Takes a full snapshot of ``sys.modules`` and restores it on exit so that
    no stale module references leak into subsequent tests.
    """
    snapshot = dict(sys.modules)
    real_import = builtins.__import__
    target = f"boa_forecaster.models.{pkg}"

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == target:
            raise ImportError(f"simulated missing: {pkg}")
        return real_import(name, globals, locals, fromlist, level)

    # Purge only the submodule and the models package so the try/except
    # in models/__init__ runs fresh. Leave everything else intact.
    for mod in (target, "boa_forecaster.models"):
        sys.modules.pop(mod, None)

    builtins.__import__ = fake_import
    try:
        reloaded = importlib.import_module("boa_forecaster.models")
        yield reloaded
    finally:
        builtins.__import__ = real_import
        # Full snapshot restore: anything we replaced during the context
        # gets put back exactly as it was before the test.
        sys.modules.clear()
        sys.modules.update(snapshot)


def test_missing_xgboost_swaps_in_sentinel() -> None:
    with _simulated_missing("xgboost") as models:
        # The reloaded ``_MissingExtra`` is a different class object than the
        # one imported at test-module load time, so check by class name.
        assert type(models.XGBoostSpec).__name__ == "_MissingExtra"
        with pytest.raises(ImportError, match="xgboost is not installed"):
            models.XGBoostSpec()


def test_missing_lightgbm_swaps_in_sentinel() -> None:
    with _simulated_missing("lightgbm") as models:
        assert type(models.LightGBMSpec).__name__ == "_MissingExtra"
        with pytest.raises(ImportError, match="lightgbm is not installed"):
            models.LightGBMSpec()


def test_sentinel_is_not_none_regression() -> None:
    """Lock in the v2.1 behaviour: missing extras must NOT yield ``None``.

    Prior to v2.1 the fallback was ``XGBoostSpec = None``, producing the
    cryptic ``TypeError: 'NoneType' not callable`` when users tried to
    instantiate it. The sentinel gives a clear, actionable ImportError.
    """
    with _simulated_missing("xgboost") as models:
        assert models.XGBoostSpec is not None
