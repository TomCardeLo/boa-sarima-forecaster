"""Tests for boa_forecaster.presets.air_quality."""

import numpy as np
import pytest

from boa_forecaster.metrics import hit_rate
from boa_forecaster.presets.air_quality import (
    ICA_EDGES_PM25_CO2017,
    ICA_EDGES_PM25_USAQI,
    ICA_LABELS_6,
    ICA_WEIGHTS_HEALTH,
    hit_rate_ica,
    hit_rate_ica_weighted,
)


def test_hit_rate_ica_matches_core() -> None:
    """hit_rate_ica(..., "CO2017") must equal hit_rate(..., ICA_EDGES_PM25_CO2017)."""
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0, 300, size=50)
    y_pred = y_true + rng.normal(0, 20, size=50)

    expected = hit_rate(y_true, y_pred, ICA_EDGES_PM25_CO2017)
    result = hit_rate_ica(y_true, y_pred, standard="CO2017")

    assert result == pytest.approx(expected)


def test_co2017_vs_usaqi_differ_on_boundary() -> None:
    """Values in [35.4, 37) fall in different buckets under CO2017 and USAQI.

    CO2017 boundary between bucket 2 and 3 is at 37  → 36 stays in bucket 2.
    USAQI  boundary between bucket 2 and 3 is at 35.4 → 36 moves into bucket 3.

    Build a 3-observation series: y_true=36, y_pred=13.
      np.digitize(36,  CO2017) = 2  and  np.digitize(13, CO2017) = 2  → HIT
      np.digitize(36,  USAQI)  = 3  and  np.digitize(13, USAQI)  = 2  → MISS
    So CO2017 hit-rate = 1.0 and USAQI hit-rate = 0.0.
    """
    y_true = np.array([36.0, 36.0, 36.0])
    y_pred = np.array([13.0, 13.0, 13.0])

    score_co2017 = hit_rate_ica(y_true, y_pred, standard="CO2017")
    score_usaqi = hit_rate_ica(y_true, y_pred, standard="USAQI")

    assert score_co2017 == pytest.approx(
        1.0
    ), f"CO2017: expected 1.0 (both in bucket 2), got {score_co2017}"
    assert score_usaqi == pytest.approx(
        0.0
    ), f"USAQI: expected 0.0 (true=bucket3 vs pred=bucket2), got {score_usaqi}"
    assert score_co2017 != score_usaqi


def test_weighted_penalizes_dangerous_bucket() -> None:
    """A miss in 'Peligrosa' (weight=10, bucket 6) yields a lower score than a miss in
    'Buena' (weight=1, bucket 1).

    ``ICA_WEIGHTS_HEALTH`` maps np.digitize bucket indices (0–7) to:
      0 → <0 µg/m³         weight  0
      1 → "Buena"          weight  1
      2 → "Aceptable"      weight  1
      3 → "Dañina grupos"  weight  2
      4 → "Dañina"         weight  3
      5 → "Muy dañina"     weight  5
      6 → "Peligrosa"      weight 10
      7 → >500 µg/m³       weight 10

    Series A: 9 hits all in "Buena" (true=5, pred=5) + 1 miss also in "Buena"
              (true=5 bucket1, pred=200 bucket4).  Miss weight = 1.
    Series B: 9 hits all in "Buena" (true=5, pred=5) + 1 miss in "Peligrosa"
              (true=300 bucket6, pred=5 bucket1).  Miss weight = 10.

    weighted_hr(A) > weighted_hr(B).
    """
    # 9 hits in "Buena"
    y_true_hits = np.array([5.0] * 9)
    y_pred_hits = np.array([5.0] * 9)

    # Series A: miss in "Buena" bucket (true=5, pred=200 → different bucket)
    y_true_a = np.concatenate([y_true_hits, [5.0]])
    y_pred_a = np.concatenate([y_pred_hits, [200.0]])

    # Series B: miss in "Peligrosa" bucket (true=300, pred=5 → different bucket)
    y_true_b = np.concatenate([y_true_hits, [300.0]])
    y_pred_b = np.concatenate([y_pred_hits, [5.0]])

    score_buena = hit_rate_ica_weighted(y_true_a, y_pred_a, standard="CO2017")
    score_peligrosa = hit_rate_ica_weighted(y_true_b, y_pred_b, standard="CO2017")

    assert (
        score_peligrosa < score_buena
    ), f"Expected peligrosa_score ({score_peligrosa:.4f}) < buena_score ({score_buena:.4f})"


def test_hit_rate_ica_weighted_sub_zero_predictions() -> None:
    """Sub-zero y_pred lands in bucket 0; weights come from buckets_true, not pred.

    With ``y_true=[300, 5]``, ``y_pred=[-5, -1]``:
      buckets_true = [6, 1] ("Peligrosa", "Buena")
      buckets_pred = [0, 0] (both below 0 — impossible in reality)
      hits         = [0, 0]       (both miss)
      obs_weights  = [10, 1]      (weights from buckets_true)
      result       = dot([10,1], [0,0]) / (10+1) = 0.0

    Pins behaviour against future changes to how ``hit_rate_weighted``
    handles the weight=0 bucket 0.  Model artifacts that produce negative
    predictions still register as misses — they just don't accrue extra
    weight, because the weight is sourced from the (impossible) bucket of
    the *truth*, not the prediction.
    """
    y_true = np.array([300.0, 5.0])
    y_pred = np.array([-5.0, -1.0])

    result = hit_rate_ica_weighted(y_true, y_pred, standard="CO2017")

    assert result == pytest.approx(0.0)


def test_hit_rate_ica_rejects_invalid_standard() -> None:
    """Typo in ``standard`` must raise ValueError, not silently fall through."""
    y_true = np.array([5.0, 40.0])
    y_pred = np.array([5.0, 40.0])

    with pytest.raises(ValueError, match="standard must be one of"):
        hit_rate_ica(y_true, y_pred, standard="USEPA")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="standard must be one of"):
        hit_rate_ica_weighted(y_true, y_pred, standard="USEPA")  # type: ignore[arg-type]


def test_constants_shape() -> None:
    """Verify that constants have the expected shapes and that edges are strictly monotonic."""
    assert len(ICA_EDGES_PM25_CO2017) == 7
    assert len(ICA_EDGES_PM25_USAQI) == 7
    assert len(ICA_LABELS_6) == 6
    # ICA_WEIGHTS_HEALTH must have len(edges)+1 = 8 entries (one per np.digitize bucket)
    assert len(ICA_WEIGHTS_HEALTH) == 8

    # Edges must be strictly increasing
    co2017 = ICA_EDGES_PM25_CO2017
    for i in range(len(co2017) - 1):
        assert (
            co2017[i] < co2017[i + 1]
        ), f"CO2017 edges not strictly increasing at index {i}"

    usaqi = ICA_EDGES_PM25_USAQI
    for i in range(len(usaqi) - 1):
        assert (
            usaqi[i] < usaqi[i + 1]
        ), f"USAQI edges not strictly increasing at index {i}"
