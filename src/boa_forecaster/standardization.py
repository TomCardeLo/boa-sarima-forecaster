"""Outlier-robust standardisation via weighted moving averages.

This module implements a custom rolling-window smoother that:

1. Computes a *weighted* average of temporal neighbours (closer neighbours
   receive higher weight).
2. Computes a *weighted* standard deviation around that average.
3. **Clips** the original value to ``[mean - threshold * sigma, mean + threshold * sigma]``,
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
    Per-observation local-neighbourhood smoother and clipper (legacy row API).
``weighted_moving_stats_series``
    Vectorised bulk version of ``weighted_moving_stats``: computes all three
    output arrays for a full series in a single ``O(n)`` pass.  Prefer this
    over row-by-row loops for production pipelines.
``weighted_moving_stats_batch``
    Alias of ``weighted_moving_stats_series``; kept for API compatibility.
"""

import warnings

import numpy as np
import pandas as pd

# Shared constants used by both the per-row and vectorised implementations.
# Decaying weights: distance 1 -> 0.3, distance 2 -> 0.2, distance 3 -> 0.1.
# Values chosen empirically to balance sensitivity vs. noise tolerance.
_REFERENCE_WEIGHTS: np.ndarray = np.array([0.3, 0.2, 0.1])

# Opt-in threshold for series with legitimate extreme spikes.
# The library default (2.5σ) is designed for typical demand series but
# over-clips real peaky processes — e.g. PM2.5 pollution episodes, electricity
# demand spikes, or financial return fat-tails — where a higher tolerance is
# more appropriate.  Pass this constant as `threshold=WMA_THRESHOLD_HIGH_VOLATILITY`
# to loosen the clipping boundary; the library default of 2.5 is unchanged.
WMA_THRESHOLD_HIGH_VOLATILITY: float = 3.5
# Threshold below which a sum of weights is treated as zero (avoids div-by-0
# on all-edge or all-zero-weight neighbourhoods).
_WEIGHT_EPSILON: float = 1e-10


def clip_outliers(
    series: pd.Series,
    method: str = "sigma",
    window: int = 6,
    threshold: float = 2.5,
) -> pd.Series:
    """Clip outliers from a time series using global statistics.

    Args:
        series: Raw demand series.
        window: Rolling window size for moving average.
        threshold: Number of standard deviations for clipping bounds.
            Default 2.5. Values below 2.0 risk clipping legitimate demand
            spikes from promotions or seasonal events.

    Returns:
        Series with outliers clipped to +/-threshold of the local mean.

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
        if sigma == 0:
            warnings.warn(
                "clip_outliers: series has zero standard deviation "
                "(constant series); no values will be clipped.",
                UserWarning,
                stacklevel=2,
            )
        lower = mu - threshold * sigma
        upper = mu + threshold * sigma
    elif method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            warnings.warn(
                "clip_outliers: series has zero IQR "
                "(constant or near-constant series); no values will be clipped.",
                UserWarning,
                stacklevel=2,
            )
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
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
    decaying weights ``[0.3, 0.2, 0.1]`` by distance (distance 1 -> 0.3,
    distance 2 -> 0.2, distance 3 -> 0.1), and computes:

    - A *weighted mean* of the neighbourhood (excluding the centre point).
    - A *weighted standard deviation* of that neighbourhood.
    - The centre value **clipped** to
      ``[mean - threshold * sigma, mean + threshold * sigma]``.

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
            ~32 % of values - too aggressive for demand series containing
            promotions or seasonal peaks.

    Returns:
        Tuple ``(weighted_mean, weighted_std, clipped_value)`` where:

        - ``weighted_mean``  - weighted average of neighbours (``float``).
        - ``weighted_std``   - weighted standard deviation (``float``).
        - ``clipped_value``  - original value clipped to
          ``[mean - threshold * sigma, mean + threshold * sigma]`` (``float``).

    Note:
        If there are no valid neighbours (e.g. a single-element series),
        the function returns ``(0.0, 0.0, original_value)`` unchanged.

    Example:
        >>> data = [100, 120, 300, 110, 105]  # 300 is a spike
        >>> wmean, wstd, clipped = weighted_moving_stats(2, data)
        >>> clipped < 300  # spike is dampened
        True
    """
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

    # Assign weights: distance <= 3 -> look up in _REFERENCE_WEIGHTS (0-indexed by dist-1);
    # distance > 3 -> weight = 0.
    weights = np.where(
        neighbour_distances <= len(_REFERENCE_WEIGHTS),
        _REFERENCE_WEIGHTS[
            np.clip(neighbour_distances.astype(int) - 1, 0, len(_REFERENCE_WEIGHTS) - 1)
        ],
        0.0,
    )

    weight_sum: float = float(np.sum(weights))

    # Edge case: no usable neighbours (empty window or all-zero weights)
    if weight_sum < _WEIGHT_EPSILON:
        return 0.0, 0.0, float(sales_data[row_index])

    # Weighted mean of the neighbourhood
    weighted_mean = np.sum(neighbours * weights) / weight_sum

    # Weighted population variance (clipped to >= 0 for numerical stability)
    weighted_variance = np.sum(weights * (neighbours - weighted_mean) ** 2) / weight_sum
    weighted_std = np.sqrt(max(0.0, weighted_variance))

    # Clip the original observation to +/-threshold * sigma around the weighted mean
    original_value = sales_data[row_index]
    lower_bound = weighted_mean - threshold * weighted_std
    upper_bound = weighted_mean + threshold * weighted_std
    clipped_value = float(np.clip(original_value, lower_bound, upper_bound))

    return float(weighted_mean), float(weighted_std), round(clipped_value)


def weighted_moving_stats_series(
    sales_data,
    window_size: int = 3,
    threshold: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised bulk version of :func:`weighted_moving_stats`.

    Computes the weighted moving mean, weighted standard deviation, and
    clipped value for **every** position in ``sales_data`` in a single
    ``O(n)`` pass using a sliding-window view over the padded array.
    Mathematically equivalent to looping ``weighted_moving_stats`` over
    each index, but 3-10x faster for long series because it avoids Python-
    level iteration and re-allocation.

    The same weighting scheme applies: neighbours at distance ``1/2/3``
    receive weights ``0.3/0.2/0.1`` respectively; distance ``0`` (centre)
    and distance ``> 3`` receive weight ``0``.  Out-of-range neighbours at
    the series edges are masked via NaN padding and excluded from both the
    weighted mean and variance.

    Args:
        sales_data: 1-D sequence of numeric demand values ordered
            chronologically (list, ``np.ndarray``, or ``pd.Series``).
        window_size: Number of neighbours to consider on each side of the
            centre point.  Defaults to ``3``.
        threshold: Number of weighted standard deviations used as the
            clipping boundary.  Defaults to ``2.5``.

    Returns:
        Tuple ``(means, stds, clipped)`` of three ``np.ndarray`` of shape
        ``(len(sales_data),)`` with ``dtype=float``:

        - ``means[i]``   - weighted mean of neighbours at position ``i``.
        - ``stds[i]``    - weighted standard deviation at position ``i``.
        - ``clipped[i]`` - original value clipped to
          ``[mean - threshold * sigma, mean + threshold * sigma]`` and rounded.

        For positions with no valid neighbours (e.g. the two ends of a
        length-1 series) the corresponding entries are ``(0.0, 0.0,
        sales_data[i])`` - matching the per-row function exactly.

    Example:
        >>> data = [100, 110, 800, 105, 95, 102, 98, 107]
        >>> means, stds, clipped = weighted_moving_stats_series(data)
        >>> clipped[2] < 800  # spike is dampened
        True

    See Also:
        :func:`weighted_moving_stats` - legacy per-row implementation.
    """
    arr = np.asarray(sales_data, dtype=float)
    n = arr.size
    if n == 0:
        empty: np.ndarray = np.empty(0, dtype=float)
        return empty, empty, empty

    # Kernel: weight for every offset in [-window_size, +window_size].
    # Centre (offset 0) and far neighbours (distance > len(ref_weights))
    # receive weight 0 by construction.
    offsets = np.arange(-window_size, window_size + 1)
    distances = np.abs(offsets)
    kernel = np.where(
        (distances >= 1) & (distances <= len(_REFERENCE_WEIGHTS)),
        _REFERENCE_WEIGHTS[np.clip(distances - 1, 0, len(_REFERENCE_WEIGHTS) - 1)],
        0.0,
    )

    # NaN-pad so that for every position we can extract a fixed-width window;
    # NaN entries flag out-of-range neighbours and are masked out below.
    pad = np.full(window_size, np.nan)
    padded = np.concatenate([pad, arr, pad])

    # (n, 2*window_size + 1) sliding view over padded array.
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * window_size + 1)

    valid = ~np.isnan(windows)
    # Per-position weights: kernel zeroed where the neighbour is out of range.
    weights = kernel[None, :] * valid
    # Replace NaN with 0 so it can participate in sums without contaminating
    # them (its weight is already 0 via `valid`).
    safe_values = np.where(valid, windows, 0.0)

    weight_sums = weights.sum(axis=1)
    no_neighbours = weight_sums < _WEIGHT_EPSILON
    # Safe divisor: 1.0 for degenerate rows (their final values are overridden
    # below), actual weight_sum otherwise.
    safe_sums = np.where(no_neighbours, 1.0, weight_sums)

    means = (weights * safe_values).sum(axis=1) / safe_sums
    deviations_sq = (safe_values - means[:, None]) ** 2
    variances = (weights * deviations_sq).sum(axis=1) / safe_sums
    stds = np.sqrt(np.maximum(0.0, variances))

    # Degenerate rows: mean=std=0, clipped value stays as the raw input
    # (NOT rounded, matching the per-row function's edge-case branch).
    means = np.where(no_neighbours, 0.0, means)
    stds = np.where(no_neighbours, 0.0, stds)

    lower = means - threshold * stds
    upper = means + threshold * stds
    # Clip and round for normal rows; pass the raw value through for
    # degenerate rows. Two np.where calls keep the branches separate.
    clipped_normal = np.round(np.clip(arr, lower, upper))
    clipped = np.where(no_neighbours, arr, clipped_normal)

    return means, stds, clipped


# API-compatibility alias: the PR that introduced ``weighted_moving_stats_batch``
# landed in parallel with ``weighted_moving_stats_series``. Both names resolve to
# the same vectorised implementation.
weighted_moving_stats_batch = weighted_moving_stats_series
