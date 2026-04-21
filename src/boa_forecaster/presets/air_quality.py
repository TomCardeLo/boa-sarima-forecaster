"""Air-quality presets: ICA (Colombia Res. 2254/2017) and US EPA AQI."""

from collections.abc import Sequence
from typing import Literal

import numpy as np

from boa_forecaster.metrics import hit_rate, hit_rate_weighted

ICA_EDGES_PM25_CO2017: list[float] = [0, 12, 37, 55, 150, 250, 500]
ICA_EDGES_PM25_USAQI: list[float] = [0, 12, 35.4, 55.4, 150.4, 250.4, 500.4]
ICA_LABELS_6: list[str] = [
    "Buena",
    "Aceptable",
    "Dañina grupos sensibles",
    "Dañina",
    "Muy dañina",
    "Peligrosa",
]
ICA_WEIGHTS_HEALTH: list[float] = [0, 1, 1, 2, 3, 5, 10, 10]
"""Per-bucket weights for :func:`hit_rate_weighted`.

``np.digitize`` with a 7-element edge array returns indices 0–7 (8 buckets):
  0 → below 0 µg/m³ (physically impossible, weight 0)
  1 → "Buena"                (0–12 µg/m³,   weight  1)
  2 → "Aceptable"            (12–37 µg/m³,  weight  1)
  3 → "Dañina grupos"        (37–55 µg/m³,  weight  2)
  4 → "Dañina"               (55–150 µg/m³, weight  3)
  5 → "Muy dañina"           (150–250 µg/m³,weight  5)
  6 → "Peligrosa"            (250–500 µg/m³,weight 10)
  7 → above 500 µg/m³ (extreme / instrument error, weight 10)

The tail plateau (buckets 6 and 7 both weighted 10) treats values above
500 µg/m³ as equivalent to "Peligrosa" — both already hit the regulatory
alarm ceiling.  If downstream consumers need to differentiate, pass a
custom ``weights`` list to :func:`hit_rate_ica_weighted`.
"""

_VALID_STANDARDS = ("CO2017", "USAQI")


def hit_rate_ica(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    standard: Literal["CO2017", "USAQI"] = "CO2017",
) -> float:
    """ICA PM2.5 hit-rate. Dispatches to core hit_rate with air-quality edges.

    Args:
        y_true: Array of observed PM2.5 values (µg/m³).
        y_pred: Array of predicted PM2.5 values (µg/m³).
        standard: Edge set to use. ``"CO2017"`` uses Colombia Res. 2254/2017
            boundaries; ``"USAQI"`` uses US EPA AQI breakpoints.

    Returns:
        Fraction of predictions that fall in the same ICA bucket as the truth.

    Raises:
        ValueError: If ``standard`` is not ``"CO2017"`` or ``"USAQI"``.
    """
    if standard not in _VALID_STANDARDS:
        raise ValueError(
            f"standard must be one of {_VALID_STANDARDS}, got {standard!r}"
        )
    edges = ICA_EDGES_PM25_CO2017 if standard == "CO2017" else ICA_EDGES_PM25_USAQI
    return hit_rate(y_true, y_pred, edges)


def hit_rate_ica_weighted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    standard: Literal["CO2017", "USAQI"] = "CO2017",
    weights: Sequence[float] = ICA_WEIGHTS_HEALTH,
) -> float:
    """Weighted ICA PM2.5 hit-rate, default weights bias toward dangerous buckets.

    Each observation's hit or miss is weighted by the health-risk weight
    of its true ICA bucket.  ``ICA_WEIGHTS_HEALTH`` assigns weight 10 to
    "Peligrosa" and weight 1 to "Buena", so misses in high-risk tiers
    penalise the score far more than misses in low-risk tiers.

    Args:
        y_true: Array of observed PM2.5 values (µg/m³).
        y_pred: Array of predicted PM2.5 values (µg/m³).
        standard: Edge set to use — ``"CO2017"`` or ``"USAQI"``.
        weights: Per-bucket weights.  Must have ``len(edges) + 1`` elements
            (one per bucket produced by ``np.digitize``).  Defaults to
            ``ICA_WEIGHTS_HEALTH``.

    Returns:
        Weighted fraction of predictions in the correct ICA bucket.

    Raises:
        ValueError: If ``standard`` is not ``"CO2017"`` or ``"USAQI"``.
    """
    if standard not in _VALID_STANDARDS:
        raise ValueError(
            f"standard must be one of {_VALID_STANDARDS}, got {standard!r}"
        )
    edges = ICA_EDGES_PM25_CO2017 if standard == "CO2017" else ICA_EDGES_PM25_USAQI
    return hit_rate_weighted(y_true, y_pred, edges, weights=list(weights))
