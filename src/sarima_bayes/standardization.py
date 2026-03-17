"""Outlier-robust standardisation via weighted moving averages.

This module implements a custom rolling-window smoother that:

1. Computes a *weighted* average of temporal neighbours (closer neighbours
   receive higher weight).
2. Computes a *weighted* standard deviation around that average.
3. **Clips** the original value to ``[mean − σ, mean + σ]``, replacing
   extreme spikes with a more conservative estimate.

The clipped values are stored as an alternative demand column
(``valor_ajustado`` in the pipeline) alongside the raw demand, allowing the
downstream optimiser to choose whichever representation yields the lower
combined metric score.
"""

import numpy as np


def weighted_moving_stats(
    row_index: int,
    sales_data: list,
    window_size: int = 3,
) -> tuple:
    """Compute weighted moving average, std dev, and clipped value.

    For the observation at position ``row_index`` in ``sales_data``, the
    function looks at the ``window_size`` neighbours on each side, assigns
    decaying weights ``[0.3, 0.2, 0.1]`` by distance (distance 1 → 0.3,
    distance 2 → 0.2, distance 3 → 0.1), and computes:

    - A *weighted mean* of the neighbourhood (excluding the centre point).
    - A *weighted standard deviation* of that neighbourhood.
    - The centre value **clipped** to ``[mean − σ, mean + σ]``.

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

    Returns:
        Tuple ``(weighted_mean, weighted_std, clipped_value)`` where:

        - ``weighted_mean``  – weighted average of neighbours (``float``).
        - ``weighted_std``   – weighted standard deviation (``float``).
        - ``clipped_value``  – original value clipped to
          ``[mean − σ, mean + σ]`` (``float``).

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

    # Clip the original observation to ±1σ around the weighted mean
    original_value = sales_data[row_index]
    lower_bound = weighted_mean - weighted_std
    upper_bound = weighted_mean + weighted_std
    clipped_value = float(np.clip(original_value, lower_bound, upper_bound))

    return float(weighted_mean), float(weighted_std), round(clipped_value)
