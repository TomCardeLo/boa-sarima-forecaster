"""Property-based invariants for forecasting metrics (Hypothesis).

Asserts mathematical properties that must hold for *any* valid input,
complementing the example-based tests in ``test_metrics.py``.

Invariants verified
-------------------

* ``smape``
    - Bounded in ``[0, 200]`` (percentage form) for all finite inputs.
    - Symmetric: ``smape(a, b) == smape(b, a)``.
    - Identity: ``smape(x, x) == 0``.

* ``rmsle``
    - Non-negative for *any* finite input (clipping handles negatives).
    - Symmetric.
    - Identity for non-negative inputs: ``rmsle(x, x) == 0``.

* ``mae`` / ``rmse``
    - Non-negative, symmetric, identity.
    - ``rmse(a, b) >= mae(a, b)`` (Jensen's inequality).

* ``combined_metric``
    - Identity at perfect prediction.
    - Exactly linear in the two weights.

* ``build_combined_metric``
    - Monotone-non-decreasing in each weight when the component metric
      is non-negative (true for all registered metrics).
    - Linear: output equals ``Σ wᵢ · metricᵢ``.
    - Zero at perfect prediction regardless of weights.

Hypothesis profile
------------------

The module registers a ``property_tests`` profile with ``deadline=None``
to avoid flakiness on slow CI runners (numpy ops can briefly stall on
Windows). ``max_examples=100`` is the Hypothesis default and is
sufficient for these low-dimensional float invariants.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from boa_forecaster.metrics import (
    build_combined_metric,
    combined_metric,
    mae,
    rmse,
    rmsle,
    smape,
)

# ---------------------------------------------------------------------------
# Settings profile — deadline=None keeps Windows CI stable.
# ---------------------------------------------------------------------------

_PROPERTY_SETTINGS = settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Finite, non-negative floats with bounded magnitude so that squaring
# stays in float64's well-conditioned range and epsilon-guarded ratios
# remain comparable.
_FINITE_NONNEG = st.floats(
    min_value=0.0,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

# Finite floats (may be negative) — used for metrics that clip internally.
_FINITE = st.floats(
    min_value=-1e6,
    max_value=1e6,
    allow_nan=False,
    allow_infinity=False,
)

_ARRAY_LEN = st.integers(min_value=1, max_value=50)


@st.composite
def _paired_arrays(draw, elements=_FINITE_NONNEG):
    """Two 1-D float64 arrays of identical length."""
    n = draw(_ARRAY_LEN)
    shape = (n,)
    y_true = draw(arrays(np.float64, shape, elements=elements))
    y_pred = draw(arrays(np.float64, shape, elements=elements))
    return y_true, y_pred


# ---------------------------------------------------------------------------
# sMAPE
# ---------------------------------------------------------------------------


class TestSmapeProperties:
    @_PROPERTY_SETTINGS
    @given(_paired_arrays())
    def test_bounded_in_0_200(self, pair):
        y_true, y_pred = pair
        out = smape(y_true, y_pred)
        # Theoretical bound: |a-b| / ((|a|+|b|)/2 + ε) ≤ 2 for non-neg inputs.
        assert 0.0 <= out <= 200.0 + 1e-9

    @_PROPERTY_SETTINGS
    @given(_paired_arrays())
    def test_symmetric(self, pair):
        y_true, y_pred = pair
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true), abs=1e-9)

    @_PROPERTY_SETTINGS
    @given(arrays(np.float64, _ARRAY_LEN, elements=_FINITE_NONNEG))
    def test_identity_is_zero(self, x):
        assert smape(x, x) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# RMSLE
# ---------------------------------------------------------------------------


class TestRmsleProperties:
    @_PROPERTY_SETTINGS
    @given(_paired_arrays(elements=_FINITE))
    def test_nonneg_for_any_finite_input(self, pair):
        """RMSLE clips negatives to zero internally, so it is non-negative
        on the full real line — not only on non-negative inputs."""
        y_true, y_pred = pair
        assert rmsle(y_true, y_pred) >= 0.0

    @_PROPERTY_SETTINGS
    @given(_paired_arrays())
    def test_symmetric(self, pair):
        y_true, y_pred = pair
        assert rmsle(y_true, y_pred) == pytest.approx(rmsle(y_pred, y_true), abs=1e-9)

    @_PROPERTY_SETTINGS
    @given(arrays(np.float64, _ARRAY_LEN, elements=_FINITE_NONNEG))
    def test_identity_is_zero(self, x):
        assert rmsle(x, x) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# MAE / RMSE
# ---------------------------------------------------------------------------


class TestMaeRmseProperties:
    @_PROPERTY_SETTINGS
    @given(_paired_arrays(elements=_FINITE))
    def test_mae_nonneg(self, pair):
        y_true, y_pred = pair
        assert mae(y_true, y_pred) >= 0.0

    @_PROPERTY_SETTINGS
    @given(_paired_arrays(elements=_FINITE))
    def test_rmse_nonneg(self, pair):
        y_true, y_pred = pair
        assert rmse(y_true, y_pred) >= 0.0

    @_PROPERTY_SETTINGS
    @given(_paired_arrays(elements=_FINITE))
    def test_mae_symmetric(self, pair):
        y_true, y_pred = pair
        assert mae(y_true, y_pred) == pytest.approx(mae(y_pred, y_true), abs=1e-9)

    @_PROPERTY_SETTINGS
    @given(_paired_arrays(elements=_FINITE))
    def test_rmse_symmetric(self, pair):
        y_true, y_pred = pair
        assert rmse(y_true, y_pred) == pytest.approx(
            rmse(y_pred, y_true), rel=1e-9, abs=1e-9
        )

    @_PROPERTY_SETTINGS
    @given(_paired_arrays(elements=_FINITE))
    def test_rmse_ge_mae(self, pair):
        # Jensen: sqrt(mean(x²)) ≥ mean(|x|).
        y_true, y_pred = pair
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred) - 1e-9

    @_PROPERTY_SETTINGS
    @given(arrays(np.float64, _ARRAY_LEN, elements=_FINITE))
    def test_mae_identity(self, x):
        assert mae(x, x) == pytest.approx(0.0, abs=1e-9)

    @_PROPERTY_SETTINGS
    @given(arrays(np.float64, _ARRAY_LEN, elements=_FINITE))
    def test_rmse_identity(self, x):
        assert rmse(x, x) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# combined_metric
# ---------------------------------------------------------------------------


class TestCombinedMetricProperties:
    @_PROPERTY_SETTINGS
    @given(arrays(np.float64, _ARRAY_LEN, elements=_FINITE_NONNEG))
    def test_identity_is_zero(self, x):
        assert combined_metric(x, x) == pytest.approx(0.0, abs=1e-9)

    @_PROPERTY_SETTINGS
    @given(
        _paired_arrays(),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_linear_in_weights(self, pair, w_smape, w_rmsle):
        y_true, y_pred = pair
        expected = w_smape * smape(y_true, y_pred) + w_rmsle * rmsle(y_true, y_pred)
        got = combined_metric(y_true, y_pred, w_smape=w_smape, w_rmsle=w_rmsle)
        assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)


# ---------------------------------------------------------------------------
# build_combined_metric
# ---------------------------------------------------------------------------


class TestBuildCombinedMetricProperties:
    @_PROPERTY_SETTINGS
    @given(
        _paired_arrays(),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_weight_monotonicity_smape(self, pair, w_base, dw):
        """Increasing the weight of a non-negative metric cannot decrease
        the combined score."""
        y_true, y_pred = pair
        fn_low = build_combined_metric([{"metric": "smape", "weight": w_base}])
        fn_high = build_combined_metric([{"metric": "smape", "weight": w_base + dw}])
        assert fn_high(y_true, y_pred) >= fn_low(y_true, y_pred) - 1e-9

    @_PROPERTY_SETTINGS
    @given(
        _paired_arrays(elements=_FINITE),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_weight_monotonicity_rmse(self, pair, w_base, dw):
        y_true, y_pred = pair
        fn_low = build_combined_metric([{"metric": "rmse", "weight": w_base}])
        fn_high = build_combined_metric([{"metric": "rmse", "weight": w_base + dw}])
        assert fn_high(y_true, y_pred) >= fn_low(y_true, y_pred) - 1e-9

    @_PROPERTY_SETTINGS
    @given(
        _paired_arrays(elements=_FINITE),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_linear_matches_manual_sum(self, pair, w1, w2):
        y_true, y_pred = pair
        fn = build_combined_metric(
            [
                {"metric": "mae", "weight": w1},
                {"metric": "rmse", "weight": w2},
            ]
        )
        expected = w1 * mae(y_true, y_pred) + w2 * rmse(y_true, y_pred)
        assert fn(y_true, y_pred) == pytest.approx(expected, rel=1e-9, abs=1e-9)

    @_PROPERTY_SETTINGS
    @given(
        arrays(np.float64, _ARRAY_LEN, elements=_FINITE_NONNEG),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    )
    def test_zero_at_perfect_prediction(self, x, w_smape, w_rmsle):
        fn = build_combined_metric(
            [
                {"metric": "smape", "weight": w_smape},
                {"metric": "rmsle", "weight": w_rmsle},
            ]
        )
        assert fn(x, x) == pytest.approx(0.0, abs=1e-9)
