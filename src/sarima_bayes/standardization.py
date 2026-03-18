"""Outlier-robust standardisation via weighted moving averages.

This module implements a custom rolling-window smoother that:

1. Computes a *weighted* average of temporal neighbours (closer neighbours
   receive higher weight).
2. Computes a *weighted* standard deviation around that average.
3. **Clips** the original value to ``[mean − threshold·σ, mean + threshold·σ]``,
   replacing extreme spikes with a more conservative estimate.

The clipped values are stored as an alternative demand column
(``valor_ajustado`` in the pipeline) alongside the raw demand, allowing the
downstream optimiser to choose whichever representation yields the lower
combined metric score.

Public functions
----------------
``clip_outliers``
    Clip a whole series using global sigma or IQR statistics.
``weighted_moving_stats``
    Per-observation local-neighbourhood smoother and clipper.
"""

import numpy as np
import pandas as pd


def clip_outliers(
    series: pd.Series,
    window: int = 6,
    sigma_threshold: float = 2.5,
) -> pd.Series:
    """Clip outliers from a time series using global statistics.

    Args:
        series: Raw demand series.
        window: Rolling window size for moving average.
        sigma_threshold: Number of standard deviations for clipping bounds.
            Default 2.5. Values below 2.0 risk clipping legitimate demand
            spikes from promotions or seasonal events.

    Returns:
        Series with outliers clipped to ±sigma_threshold of the local mean.

    Raises:
        ValueError: If ``method`` is not ``"sigma"`` or ``"iqr"``.
        ValueError: If ``series`` contains NaN (caller must handle nulls first).

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([100.0, 105.0, 800.0, 98.0, 102.0])
        >>> clipped = clip_outliers(s, method="sigma", threshold=2.5)
        >>> clipped[2] < 800
        True
    """
    if series.isna().any():
        raise ValueError(
            "clip_outliers: series contains NaN values. "
            "Handle missing values before clipping."
        )

    if method == "sigma":
        mu = series.mean()
        sigma = series.std(ddof=1)
        lower = mu - sigma_threshold * sigma
        upper = mu + sigma_threshold * sigma
    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - sigma_threshold * iqr
        upper = q3 + sigma_threshold * iqr
    else:
        raise ValueError(
            f"clip_outliers: unknown method '{method}'. " "Choose 'sigma' or 'iqr'."
        )

    # Clip to computed bounds, then enforce non-negativity
    return series.clip(lower=lower, upper=upper).clip(lower=0)


def weighted_moving_stats(
    row_index: int,
    sales_data: list,
    window_size: int = 3,
    threshold: float = 2.5,
) -> tuple[float, float, float]:
    """Compute weighted moving average, std dev, and clipped value.

    For the observation at position ``row_index`` in ``sales_data``, the
    function looks at the ``window_size`` neighbours on each side, assigns
    decaying weights ``[0.3, 0.2, 0.1]`` by distance (distance 1 → 0.3,
    distance 2 → 0.2, distance 3 → 0.1), and computes:

    - A *weighted mean* of the neighbourhood (excluding the centre point).
    - A *weighted standard deviation* of that neighbourhood.
    - The centre value **clipped** to
      ``[mean − threshold·σ, mean + threshold·σ]``.

    Neighbours at distance > 3 receive weight 0 (effectively ignored).
    The function is designed to be called row-by-row within a loop over
    a single SKU time series.

    Args:
        row_index: Integer position of the target observation within
            ``sales_data``.
        sales_data: List or 1-D sequence of numeric demand values ordered
            chronologically (all periods for one SKU/country combination).
        window_size: Number of neighbours to consider on each side of the
            centre point.  Defaults to ``3``.
        threshold: Number of weighted standard deviations used as the
            clipping boundary.  Defaults to ``2.5`` (clips ~1.2 % of a
            normal distribution).  The previous default of ``1.0`` clipped
            ~32 % of values — too aggressive for demand series containing
            promotions or seasonal peaks.

    Returns:
        Tuple ``(weighted_mean, weighted_std, clipped_value)`` where:

        - ``weighted_mean``  – weighted average of neighbours (``float``).
        - ``weighted_std``   – weighted standard deviation (``float``).
        - ``clipped_value``  – original value clipped to
          ``[mean − threshold·σ, mean + threshold·σ]`` (``float``).

    Note:
        If there are no valid neighbours (e.g. a single-element series),
        the function returns ``(0.0, 0.0, original_value)`` unchanged.

    Example:
        >>> data = [100, 120, 300, 110, 105]  # 300 is a spike
        >>> wmean, wstd, clipped = weighted_moving_stats(2, data)
        >>> clipped < 300  # spike is dampened
        True
    """
    # Decaying weights: distance 1 → 0.3, distance 2 → 0.2, distance 3 → 0.1
    reference_weights = np.array([0.3, 0.2, 0.1])
    # Small epsilon prevents division-by-zero on flat-zero series
    epsilon = 1e-10

    # Extract the neighbourhood window (bounded by series edges)
    start = max(0, row_index - window_size)
    end = min(len(sales_data), row_index + window_size + 1)
    neighbourhood = np.array(sales_data[start:end])

    # Absolute distance of each element from the centre point
    window_indices = np.arange(start, end)
    distances = np.abs(window_indices - row_index)

    # Exclude the centre point itself (distance == 0)
    neighbour_mask = distances > 0
    neighbours = neighbourhood[neighbour_mask]
    neighbour_distances = distances[neighbour_mask]

    # Assign weights: distance ≤ 3 → look up in reference_weights (0-indexed by dist-1);
    # distance > 3 → weight = 0.
    weights = np.where(
        neighbour_distances <= len(reference_weights),
        reference_weights[
            np.clip(neighbour_distances.astype(int) - 1, 0, len(reference_weights) - 1)
        ],
        0.0,
    )

    weight_sum = np.sum(weights)

    # Edge case: no usable neighbours (empty window or all-zero weights)
    if weight_sum < epsilon:
        return 0.0, 0.0, float(sales_data[row_index])

    # Weighted mean of the neighbourhood
    weighted_mean = np.sum(neighbours * weights) / weight_sum

    # Weighted population variance (clipped to ≥ 0 for numerical stability)
    weighted_variance = np.sum(weights * (neighbours - weighted_mean) ** 2) / weight_sum
    weighted_std = np.sqrt(max(0.0, weighted_variance))

    # Clip the original observation to ±threshold·σ around the weighted mean
    original_value = sales_data[row_index]
    lower_bound = weighted_mean - threshold * weighted_std
    upper_bound = weighted_mean + threshold * weighted_std
    clipped_value = float(np.clip(original_value, lower_bound, upper_bound))

    return float(weighted_mean), float(weighted_std), round(clipped_value)
