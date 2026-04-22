"""LSTMSpec — PyTorch LSTM plugin for boa-forecaster v2.4.

Implements the ``ModelSpec`` protocol for sequence-to-one forecasting using
a stacked LSTM followed by a linear readout layer.

Shape
-----
LSTM is **SARIMA-shaped**, not feature-based: it consumes a raw 1-D series
that is sliced into overlapping windows of length ``window_size``.
``needs_features`` is therefore ``False``.

Normalisation
-------------
The series is z-score normalised (mean/std) before windowing so the LSTM
sees values near zero regardless of the raw scale.  Predictions are
denormalised before being scored or returned.

Training / early stopping
-------------------------
The supervised window-pairs are split 80 / 20 (last 20 % = validation).
Adam is used with MSE loss.  A patience-5 early stopping scheme restores
the best-validation-loss state dict before handing back control to the
caller.

Availability
------------
Registered in ``MODEL_REGISTRY`` as ``"lstm"`` only when ``torch`` is
importable.  The top-level ``boa_forecaster.LSTMSpec`` re-export falls back
to a ``_MissingExtra`` sentinel when the extra is not installed.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Callable

import numpy as np
import optuna
import pandas as pd

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.models.base import (
    FloatParam,
    IntParam,
    SearchSpaceParam,
    suggest_from_space,
)

logger = logging.getLogger(__name__)

# ── Private LSTM nn.Module ────────────────────────────────────────────────────


class _LSTMModel(nn.Module):  # type: ignore[misc]
    """A thin wrapper: stacked LSTM → linear head (last hidden state)."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # PyTorch emits a UserWarning when dropout > 0 with num_layers == 1
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        last = out[:, -1, :]  # (batch, hidden_size)
        return self.fc(last).squeeze(-1)  # (batch,)


# ── LSTMSpec ──────────────────────────────────────────────────────────────────


class LSTMSpec:
    """``ModelSpec`` implementation for a stacked PyTorch LSTM.

    Args:
        hidden_size: Number of features in the LSTM hidden state.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers (ignored when
            ``num_layers == 1`` to suppress PyTorch warnings).
        learning_rate: Adam learning rate.
        n_epochs: Maximum training epochs.
        batch_size: Mini-batch size used during training.
        window_size: Length of the input window (look-back).
        forecast_horizon: Number of future steps produced by
            ``build_forecaster``.
        device: ``"cpu"`` (default) or ``"auto"`` (use CUDA if available).
            Any other value raises :class:`ValueError`.
        seed: RNG seed set inside every ``fit`` call for reproducibility.

    Raises:
        ImportError: If ``torch`` is not installed.
        ValueError: If ``device`` is not ``"cpu"`` or ``"auto"``.
    """

    name: str = "lstm"
    needs_features: bool = False
    uses_early_stopping: bool = True

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 16,
        window_size: int = 12,
        forecast_horizon: int = 12,
        device: str = "cpu",
        seed: int = 42,
    ) -> None:
        self._check_availability()
        if device not in ("cpu", "auto"):
            raise ValueError(
                f"device must be 'cpu' or 'auto'; got {device!r}. "
                "Use 'auto' to select CUDA when available."
            )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.device = device
        self.seed = seed

    def _check_availability(self) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for LSTMSpec. "
                "Install it with: pip install 'sarima-bayes[deep]'"
            )

    def _resolve_device(self) -> torch.device:  # type: ignore[name-defined]
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    # ── ModelSpec properties ──────────────────────────────────────────────────

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """Five hyperparameters searched by Optuna TPE."""
        return {
            "hidden_size": IntParam(16, 128),
            "num_layers": IntParam(1, 3),
            "dropout": FloatParam(0.0, 0.4),
            "learning_rate": FloatParam(1e-4, 1e-2, log=True),
            "n_epochs": IntParam(10, 100),
        }

    @property
    def warm_starts(self) -> list[dict]:
        """Two sensible starting configurations for Optuna warm-starting."""
        return [
            {
                "hidden_size": 32,
                "num_layers": 1,
                "dropout": 0.1,
                "learning_rate": 1e-3,
                "n_epochs": 30,
            },
            {
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "learning_rate": 5e-4,
                "n_epochs": 50,
            },
        ]

    # ── ModelSpec methods ─────────────────────────────────────────────────────

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Sample a point from the LSTM search space."""
        return suggest_from_space(trial, self.search_space)

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn: Callable,
        feature_config: object = None,
        feature_cache: object = None,
        trial: optuna.Trial | None = None,
    ) -> float:
        """Train an LSTM on *series* and return the in-sample metric score.

        Args:
            series: Training series.
            params: Dict with at minimum the five search-space keys.
            metric_fn: Callable ``(y_true, y_pred) -> float``.
            feature_config: Ignored (LSTM builds its own windows).
            feature_cache: Ignored.
            trial: Active Optuna trial; reported once at step=0 for pruner
                compatibility.

        Returns:
            Scalar metric score, or ``OPTIMIZER_PENALTY`` on failure.
        """
        try:
            raw = np.asarray(series, dtype=np.float64)
            window_size = self.window_size
            if len(raw) <= window_size:
                logger.debug(
                    "LSTMSpec.evaluate: series length %d <= window_size %d; "
                    "returning OPTIMIZER_PENALTY",
                    len(raw),
                    window_size,
                )
                return OPTIMIZER_PENALTY

            values, mu, sigma = _normalise(raw)
            X, y = _make_windows(values.astype(np.float32), window_size)
            if len(X) == 0:
                return OPTIMIZER_PENALTY

            device = self._resolve_device()
            model = self._build_and_train(X, y, params, device)

            # One-step-ahead in-sample rollout (normalised domain)
            model.eval()
            y_pred_norm = np.full(len(values), np.nan, dtype=np.float32)
            with torch.no_grad():
                for i in range(window_size, len(values)):
                    window = values[i - window_size : i].astype(np.float32)
                    x_t = (
                        torch.tensor(window, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(-1)
                        .to(device)
                    )
                    y_pred_norm[i] = model(x_t).item()

            valid = ~np.isnan(y_pred_norm)
            y_pred = _denormalise(y_pred_norm[valid], mu, sigma)
            score = float(metric_fn(raw[valid], y_pred))

        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.debug("LSTMSpec.evaluate failed for params=%s: %s", params, exc)
            return OPTIMIZER_PENALTY

        if trial is not None:
            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    def build_forecaster(
        self,
        params: dict,
        feature_config: object = None,
    ) -> Callable[[pd.Series], pd.Series]:
        """Return a closure ``forecaster(train: pd.Series) -> pd.Series``.

        The closure trains a fresh LSTM on *train*, then performs a recursive
        rollout for ``forecast_horizon`` steps.

        Args:
            params: Dict with the five search-space keys.
            feature_config: Ignored.

        Returns:
            Callable ``(train: pd.Series) -> pd.Series``.
        """
        window_size = self.window_size
        horizon = self.forecast_horizon

        def forecaster(train: pd.Series) -> pd.Series:
            raw = np.asarray(train, dtype=np.float64)
            device = self._resolve_device()

            if len(raw) <= window_size:
                # Cannot build windows — return NaN forecast
                idx = _build_future_index(train, horizon)
                return pd.Series(
                    np.full(horizon, np.nan),
                    index=idx,
                    name=getattr(train, "name", None),
                )

            values, mu, sigma = _normalise(raw)
            X, y = _make_windows(values.astype(np.float32), window_size)
            model = self._build_and_train(X, y, params, device)
            model.eval()

            # Recursive rollout in normalised domain
            history = list(values[-window_size:])
            preds_norm: list[float] = []
            with torch.no_grad():
                for _ in range(horizon):
                    window = np.array(history[-window_size:], dtype=np.float32)
                    x_t = torch.tensor(window).unsqueeze(0).unsqueeze(-1).to(device)
                    step_pred = float(model(x_t).item())
                    preds_norm.append(step_pred)
                    history.append(step_pred)

            preds = _denormalise(np.array(preds_norm, dtype=np.float64), mu, sigma)
            idx = _build_future_index(train, horizon)
            return pd.Series(
                preds,
                index=idx,
                name=getattr(train, "name", None),
            )

        return forecaster

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _seed_everything(self) -> None:
        """Set all relevant RNG seeds (called inside every fit)."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _build_and_train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict,
        device: torch.device,  # type: ignore[name-defined]
    ) -> _LSTMModel:
        """Build and train an LSTM model with early stopping.

        The five searched hyperparameters override ``self`` attributes when
        present in *params*.
        """
        self._seed_everything()

        hidden_size = int(params.get("hidden_size", self.hidden_size))
        num_layers = int(params.get("num_layers", self.num_layers))
        dropout = float(params.get("dropout", self.dropout))
        learning_rate = float(params.get("learning_rate", self.learning_rate))
        n_epochs = int(params.get("n_epochs", self.n_epochs))

        model = _LSTMModel(hidden_size, num_layers, dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        n = len(X)
        split = max(1, int(n * 0.8))
        X_tr, y_tr = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

        batch_size = self.batch_size
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0
        best_state: dict = {}

        for _ in range(n_epochs):
            model.train()
            # Mini-batch loop
            perm = torch.randperm(len(X_tr_t))
            for start in range(0, len(X_tr_t), batch_size):
                idx = perm[start : start + batch_size]
                optimizer.zero_grad()
                pred = model(X_tr_t[idx])
                loss = loss_fn(pred, y_tr_t[idx])
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = loss_fn(val_pred, y_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        return model


# ── Module-level helpers ──────────────────────────────────────────────────────


def _normalise(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Z-score normalise *values*; return (normalised, mean, std).

    When std is near zero (constant series) std is set to 1.0 to avoid
    division by zero.
    """
    mu = float(np.mean(values))
    sigma = float(np.std(values))
    if sigma < 1e-8:
        sigma = 1.0
    return (values - mu) / sigma, mu, sigma


def _denormalise(normed: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Reverse z-score normalisation."""
    return normed * sigma + mu


def _make_windows(
    values: np.ndarray, window_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping (X, y) pairs from a 1-D float32 array.

    Returns:
        X: shape ``(n_samples, window_size, 1)``
        y: shape ``(n_samples,)``
    """
    n = len(values) - window_size
    if n <= 0:
        return np.empty((0, window_size, 1), dtype=np.float32), np.empty(
            0, dtype=np.float32
        )
    X = np.lib.stride_tricks.as_strided(
        values,
        shape=(n, window_size),
        strides=(values.strides[0], values.strides[0]),
    ).copy()
    y = values[window_size:]
    return X[:, :, np.newaxis], y


def _build_future_index(train: pd.Series, horizon: int) -> pd.DatetimeIndex:
    """Build a ``DatetimeIndex`` of length *horizon* immediately after *train*."""
    freq = getattr(train.index, "freq", None)
    if freq is None:
        inferred = pd.infer_freq(train.index)
        freq = inferred or "MS"
    return pd.date_range(
        start=train.index[-1],
        periods=horizon + 1,
        freq=freq,
    )[1:]
