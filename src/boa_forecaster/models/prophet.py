"""ProphetSpec — Meta's Prophet plugin for boa-forecaster v2.4.

Implements the ``ModelSpec`` protocol for trend + seasonality + holidays
decomposition via `Prophet <https://facebook.github.io/prophet/>`_.

Shape
-----
Prophet is **SARIMA-shaped**, not feature-based: it builds its own Fourier
seasonal components from the ``ds`` (datetime) column, so ``needs_features``
is ``False`` and no ``FeatureEngineer`` is wired in.  ``evaluate`` fits one
Prophet model in-sample and scores the fitted values against the training
series, mirroring :class:`~boa_forecaster.models.sarima.SARIMASpec`.

Search space
------------
Four hyperparameters: ``changepoint_prior_scale`` (trend flexibility),
``seasonality_prior_scale`` / ``holidays_prior_scale`` (regularisation on
seasonal / holiday components), and ``seasonality_mode`` ("additive" or
"multiplicative").

Availability
------------
Registered in ``MODEL_REGISTRY`` as ``"prophet"`` only when ``prophet`` is
importable.  The top-level ``boa_forecaster.ProphetSpec`` re-export falls back
to a ``_MissingExtra`` sentinel when the extra is not installed — calling it
raises :class:`ImportError` with a pointer to ``pip install`` instead of the
cryptic ``TypeError: 'NoneType' not callable``.
"""

from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd

try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

from boa_forecaster.config import OPTIMIZER_PENALTY
from boa_forecaster.models.base import (
    CategoricalParam,
    FloatParam,
    SearchSpaceParam,
    suggest_from_space,
)

logger = logging.getLogger(__name__)

# Prophet and its cmdstanpy backend log verbosely at INFO/DEBUG on every fit
# ("Disabling yearly seasonality. Run prophet with yearly_seasonality=True ...",
# "Initial log joint probability = ..."). Silence them globally at module
# import — users who want Prophet's chatter can re-enable per-logger.
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)


_SUPPORTED_FREQS = ("MS", "M", "W", "D", "h", "H")


class ProphetSpec:
    """``ModelSpec`` implementation for Meta's Prophet.

    Args:
        seasonality_mode: Default seasonality mode when the Optuna-sampled
            value is absent.  One of ``"additive"`` or ``"multiplicative"``.
        country_holidays: ISO country code passed to
            ``Prophet.add_country_holidays`` (e.g. ``"US"``, ``"CO"``).
            ``None`` (default) disables holiday regressors.
        growth: Trend model — ``"linear"`` (default) or ``"logistic"``.  The
            logistic path auto-injects a ``cap`` column at 1.5× the training
            max; users who need a tighter cap should fit Prophet directly.
        freq: Pandas offset alias used by ``make_future_dataframe`` when
            building the forecast horizon.  Default ``"MS"``.
        forecast_horizon: Number of future steps to predict.  Default 12.

    Raises:
        ImportError: If ``prophet`` is not installed.
    """

    name: str = "prophet"
    needs_features: bool = False
    uses_early_stopping: bool = False

    def __init__(
        self,
        seasonality_mode: str = "additive",
        country_holidays: str | None = None,
        growth: str = "linear",
        freq: str = "MS",
        forecast_horizon: int = 12,
    ) -> None:
        self._check_availability()
        if seasonality_mode not in ("additive", "multiplicative"):
            raise ValueError(
                f"seasonality_mode must be 'additive' or 'multiplicative'; "
                f"got {seasonality_mode!r}"
            )
        if growth not in ("linear", "logistic"):
            raise ValueError(f"growth must be 'linear' or 'logistic'; got {growth!r}")
        self.seasonality_mode: str = seasonality_mode
        self.country_holidays: str | None = country_holidays
        self.growth: str = growth
        self.freq: str = freq
        self.forecast_horizon: int = forecast_horizon

    def _check_availability(self) -> None:
        if not HAS_PROPHET:
            raise ImportError(
                "prophet is required for ProphetSpec. "
                "Install it with: pip install 'sarima-bayes[prophet]'"
            )

    # ── ModelSpec properties ──────────────────────────────────────────────────

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """Four hyperparameters — scales (log) + seasonality mode (categorical)."""
        return {
            "changepoint_prior_scale": FloatParam(0.001, 0.5, log=True),
            "seasonality_prior_scale": FloatParam(0.01, 10.0, log=True),
            "holidays_prior_scale": FloatParam(0.01, 10.0, log=True),
            "seasonality_mode": CategoricalParam(["additive", "multiplicative"]),
        }

    @property
    def warm_starts(self) -> list[dict]:
        """Two sensible starting configurations.

        - Prophet's documented defaults (``changepoint_prior_scale=0.05``,
          ``seasonality_prior_scale=10``, additive).
        - A tighter-regularised multiplicative alternative for series where
          the seasonal amplitude grows with the level.
        """
        return [
            {
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "seasonality_mode": "additive",
            },
            {
                "changepoint_prior_scale": 0.1,
                "seasonality_prior_scale": 1.0,
                "holidays_prior_scale": 1.0,
                "seasonality_mode": "multiplicative",
            },
        ]

    # ── ModelSpec methods ─────────────────────────────────────────────────────

    def suggest_params(self, trial: optuna.Trial) -> dict:
        """Sample a point from the Prophet search space."""
        return suggest_from_space(trial, self.search_space)

    def evaluate(
        self,
        series: pd.Series,
        params: dict,
        metric_fn,
        feature_config=None,
        feature_cache=None,
        trial: optuna.Trial | None = None,
    ) -> float:
        """Fit Prophet in-sample and return the metric score.

        Any Prophet/cmdstanpy exception returns ``OPTIMIZER_PENALTY`` so the
        outer Optuna study can keep searching.  Matches the soft-failure
        contract of :class:`SARIMASpec.evaluate`.

        Args:
            series: Training series with a ``DatetimeIndex``.
            params: Dict with the four search-space keys.
            metric_fn: Callable ``(y_true, y_pred) -> float``.
            feature_config: Ignored (Prophet builds its own features).
            feature_cache: Ignored.
            trial: Active Optuna trial; reported once at step 0 for pruner
                compatibility (matches SARIMASpec behaviour).

        Returns:
            Scalar metric score, or ``OPTIMIZER_PENALTY`` on failure.
        """
        try:
            model = self._fit_prophet(series, params)
            history = pd.DataFrame({"ds": series.index})
            if self.growth == "logistic":
                history["cap"] = float(np.asarray(series).max()) * 1.5
            pred_df = model.predict(history)
            y_pred = pred_df["yhat"].to_numpy()
            y_true = (
                series.to_numpy()
                if isinstance(series, pd.Series)
                else np.asarray(series)
            )
            score = float(metric_fn(y_true, y_pred))
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            logger.debug("Prophet evaluate failed for params=%s: %s", params, exc)
            return OPTIMIZER_PENALTY

        if trial is not None:
            trial.report(score, step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    def build_forecaster(self, params: dict, feature_config=None):
        """Return a closure ``forecaster(train: pd.Series) -> pd.Series``.

        The returned callable fits Prophet on *train* and produces a
        ``forecast_horizon``-step-ahead forecast as a ``pd.Series`` with a
        ``DatetimeIndex`` at frequency ``self.freq``.

        Args:
            params: Dict with the four search-space keys.
            feature_config: Ignored.

        Returns:
            Callable ``(train: pd.Series) -> pd.Series``.
        """
        horizon = self.forecast_horizon
        freq = self.freq
        growth = self.growth

        def forecaster(train: pd.Series) -> pd.Series:
            model = self._fit_prophet(train, params)
            future = model.make_future_dataframe(
                periods=horizon, freq=freq, include_history=False
            )
            if growth == "logistic":
                future["cap"] = float(train.max()) * 1.5
            pred_df = model.predict(future)
            return pd.Series(
                pred_df["yhat"].to_numpy(),
                index=pd.DatetimeIndex(pred_df["ds"]),
                name=getattr(train, "name", None),
            )

        return forecaster

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _fit_prophet(self, series: pd.Series, params: dict) -> Prophet:
        """Build a DataFrame, instantiate Prophet, add holidays, and fit."""
        idx = series.index if isinstance(series, pd.Series) else None
        values = (
            series.to_numpy() if isinstance(series, pd.Series) else np.asarray(series)
        )
        df = pd.DataFrame({"ds": idx, "y": values})
        if self.growth == "logistic":
            df["cap"] = float(values.max()) * 1.5

        model = Prophet(
            changepoint_prior_scale=params["changepoint_prior_scale"],
            seasonality_prior_scale=params["seasonality_prior_scale"],
            holidays_prior_scale=params["holidays_prior_scale"],
            seasonality_mode=params.get("seasonality_mode", self.seasonality_mode),
            growth=self.growth,
        )
        if self.country_holidays:
            model.add_country_holidays(country_name=self.country_holidays)
        model.fit(df)
        return model

    # ── Frequency-aware factory ───────────────────────────────────────────────

    @classmethod
    def for_frequency(cls, freq: str, **overrides: object) -> ProphetSpec:
        """Return a ``ProphetSpec`` whose ``freq`` matches *freq*.

        Mirrors :meth:`SARIMASpec.for_frequency` in shape.  Supported
        aliases: ``"MS"``, ``"M"``, ``"W"``, ``"D"``, ``"h"``, ``"H"``.

        Args:
            freq: Pandas offset alias.
            **overrides: Constructor keyword arguments that override the
                frequency-derived defaults (e.g. ``forecast_horizon=30``).

        Raises:
            ValueError: If *freq* is not a supported alias.
        """
        if freq not in _SUPPORTED_FREQS:
            raise ValueError(
                f"Unknown frequency {freq!r}; supported: {', '.join(_SUPPORTED_FREQS)}"
            )
        defaults: dict[str, object] = {"freq": freq}
        defaults.update(overrides)
        return cls(**defaults)  # type: ignore[arg-type]
