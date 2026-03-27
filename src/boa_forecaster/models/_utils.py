"""Shared utilities for ML model specs in boa-forecaster v2.0.

This private module holds helpers that are identical across
``RandomForestSpec``, ``XGBoostSpec``, and ``LightGBMSpec`` to avoid
copy-paste duplication.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from boa_forecaster.features import FeatureEngineer

# ── Shared CV constants ───────────────────────────────────────────────────────

MIN_TRAIN_SIZE: int = 24
N_CV_FOLDS: int = 3
VAL_FRACTION: float = 0.2  # fraction of training fold for early-stopping val
MIN_VAL_SIZE: int = 6  # minimum rows for the inner validation split


# ── Shared helpers ────────────────────────────────────────────────────────────


def recursive_forecast(
    model,
    fe: FeatureEngineer,
    train: pd.Series,
    horizon: int,
) -> pd.Series:
    """Predict *horizon* steps ahead using recursive one-step-at-a-time forecasting.

    At each step the latest prediction is appended to the series so that lag
    and rolling features for the next step can be computed causally.

    Args:
        model: Fitted regressor with a ``predict(X) -> array`` method.
        fe: ``FeatureEngineer`` already fitted on *train* via ``fit_transform``.
        train: Training series used to seed the extended series.
        horizon: Number of future steps to predict.

    Returns:
        ``pd.Series`` of length *horizon* with a ``DatetimeIndex`` starting
        one month after the last training date (``freq="MS"``).
    """
    future_index = pd.date_range(
        start=train.index[-1] + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS",
    )
    # Pre-allocate: avoids O(horizon²) pd.concat inside the loop.
    extended = pd.concat([train, pd.Series(np.nan, index=future_index)])
    train_len = len(train)
    preds: list[float] = []

    for i in range(horizon):
        X_all = fe.transform(extended.iloc[: train_len + i])
        y_pred = float(model.predict(X_all.iloc[[-1]])[0])
        extended.iloc[train_len + i] = y_pred
        preds.append(y_pred)

    return pd.Series(preds, index=future_index, name="forecast")


def split_for_early_stopping(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split feature matrix and target into inner train / validation sets.

    The validation set is the last ``VAL_FRACTION`` of rows, with a minimum
    of ``MIN_VAL_SIZE`` rows.  Used by ``XGBoostSpec`` and ``LightGBMSpec``
    to provide an early-stopping signal without leaking test-fold information.

    Args:
        X: Feature matrix.
        y: Target vector.

    Returns:
        Tuple ``(X_tr, y_tr, X_val, y_val)``.
    """
    val_size = max(MIN_VAL_SIZE, int(len(X) * VAL_FRACTION))
    return (
        X.iloc[:-val_size],
        y.iloc[:-val_size],
        X.iloc[-val_size:],
        y.iloc[-val_size:],
    )
