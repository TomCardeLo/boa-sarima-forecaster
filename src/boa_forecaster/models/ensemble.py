"""EnsembleSpec — post-optimisation weighted average of multiple ``ModelSpec``s.

Philosophy
----------
An ensemble is **not** a TPE candidate: its hyperparameters are fully
determined by the already-optimised members.  ``evaluate`` therefore raises
``NotImplementedError`` — callers build an ensemble via the
:func:`build_ensemble` helper, which optimises each member first and then
composes a closure that weight-averages their forecasts.

Weighting strategies
--------------------
* ``"equal"`` — uniform weights, ``1/N`` per member.
* ``"inverse_cv_loss"`` — weight ∝ ``1 / member_score``; members with a
  lower objective receive more weight.  Requires ``member_scores`` to be
  set on the spec (done automatically by :func:`build_ensemble`).  Falls
  back to equal weights when any score is zero or non-finite.
* ``list[float]`` — explicit weights (normalised to sum to 1).

Weighting caveats
-----------------
The ``"inverse_cv_loss"`` strategy assumes every member's
``best_score`` was produced by a **comparable** cross-validation
protocol.  When members use different CV semantics this assumption
breaks and the weighting becomes biased:

* **Inner-validation early stopping** — ``XGBoostSpec`` and
  ``LightGBMSpec`` carve off the tail of each training fold as an
  internal validation slice so boosting can early-stop.  The final
  score therefore reflects training on roughly 80% of the fold, while
  ``RandomForestSpec`` and ``SARIMASpec`` train on 100% of the fold.
  The XGB/LGB score is usually slightly *worse* than it would be at
  full fold size, so ``inverse_cv_loss`` over-weights RF/SARIMA by a
  small but systematic margin.
* **Different feature configs** — a member with a richer
  ``FeatureConfig`` (more lags, rolling windows, calendar features)
  will tend to score better on clean data and worse on short/sparse
  series.  Mixing feature configs across members can therefore mask
  genuine model-quality differences behind a feature-engineering gap.
* **Different forecast horizons** — all members must share the same
  ``forecast_horizon``; a mismatch would compare scores computed over
  different-length windows and the weighting would be meaningless.

When any of these apply, prefer explicit ``list[float]`` weights (e.g.
learned off-line from a holdout) or fall back to ``"equal"``.  Track a
feature request in the backlog (see ``tasks/feedback_aire.md`` point 4)
for a "normalised CV score" option that re-scores every member on the
same outer-fold protocol before weighting.
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Literal, Union

import numpy as np
import pandas as pd

from boa_forecaster.models.base import ModelSpec, SearchSpaceParam

logger = logging.getLogger(__name__)

WeightingStrategy = Union[Literal["equal", "inverse_cv_loss"], list[float]]


class EnsembleSpec:
    """Weighted-average ensemble of pre-optimised ``ModelSpec`` members.

    Attributes:
        members: The underlying ``ModelSpec`` instances to ensemble.
        weighting: Either a named strategy (``"equal"``,
            ``"inverse_cv_loss"``) or an explicit list of float weights.
        member_scores: Optional ``{member.name: best_score}`` mapping used
            by the ``"inverse_cv_loss"`` strategy.  Set automatically by
            :func:`build_ensemble`.
    """

    name: str = "ensemble"

    def __init__(
        self,
        members: list[ModelSpec],
        weighting: WeightingStrategy = "inverse_cv_loss",
        member_scores: dict[str, float] | None = None,
    ) -> None:
        if not members:
            raise ValueError("EnsembleSpec requires at least one member")
        if isinstance(weighting, list) and len(weighting) != len(members):
            raise ValueError(
                f"weighting list has {len(weighting)} entries "
                f"but ensemble has {len(members)} members"
            )
        self.members = list(members)
        self.weighting: WeightingStrategy = weighting
        self.member_scores = dict(member_scores) if member_scores else {}

    # ── ModelSpec protocol ────────────────────────────────────────────────────

    @property
    def needs_features(self) -> bool:
        """``True`` iff **any** member requires a ``FeatureEngineer``.

        Previously hardcoded to ``False``; an ensemble containing a tabular
        ML member (e.g. ``RandomForestSpec``) would silently lie to any
        downstream code gating feature engineering on this flag.
        """
        return any(getattr(m, "needs_features", False) for m in self.members)

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]:
        """An ensemble has no hyperparameters of its own."""
        return {}

    @property
    def warm_starts(self) -> list[dict]:
        return []

    def suggest_params(self, trial) -> dict:  # pragma: no cover — not TPE-tunable
        return {}

    def evaluate(
        self,
        series,
        params,
        metric_fn,
        feature_config=None,
        feature_cache=None,
        trial=None,
    ) -> float:
        """Ensembles are post-optimisation compositions, not TPE candidates.

        Use :func:`build_ensemble` (which runs ``optimize_model`` per member
        first, then instantiates this class) rather than plugging an
        ``EnsembleSpec`` into ``optimize_model``.
        """
        raise NotImplementedError(
            "EnsembleSpec.evaluate is intentionally unimplemented: an ensemble "
            "is a post-optimisation composition, not a TPE trial candidate. "
            "Build one with boa_forecaster.models.ensemble.build_ensemble()."
        )

    def build_forecaster(
        self,
        params: dict,
        feature_config=None,
    ) -> Callable[[pd.Series], pd.Series]:
        """Return a closure that weight-averages member forecasts.

        Args:
            params: ``{member.name: member_params}`` — one entry per member,
                matching the ``best_params`` each member returned from its
                own ``optimize_model`` run.
            feature_config: Passed through to member ``build_forecaster``
                calls (ignored for non-feature-based specs).

        Returns:
            Callable ``(train: pd.Series) -> pd.Series`` producing the
            weighted-average forecast aligned with the first member's
            output index.
        """
        missing = [m.name for m in self.members if m.name not in params]
        if missing:
            raise KeyError(
                f"Missing params for ensemble members: {missing}. "
                f"Expected keys: {[m.name for m in self.members]}"
            )

        weights = self._resolve_weights()
        members = self.members

        def forecaster(train: pd.Series) -> pd.Series:
            predictions: list[pd.Series] = []
            for member in members:
                member_fn = member.build_forecaster(
                    params[member.name], feature_config=feature_config
                )
                predictions.append(member_fn(train))

            base_index = predictions[0].index
            stacked = np.stack(
                [
                    np.asarray(p.reindex(base_index).values, dtype=float)
                    for p in predictions
                ],
                axis=0,
            )
            averaged = np.average(stacked, axis=0, weights=weights)
            return pd.Series(averaged, index=base_index, name="ensemble")

        return forecaster

    # ── Internal ──────────────────────────────────────────────────────────────

    def _resolve_weights(self) -> np.ndarray:
        """Turn ``self.weighting`` into a normalised numpy weight vector."""
        n = len(self.members)

        if isinstance(self.weighting, list):
            w = np.asarray(self.weighting, dtype=float)
        elif self.weighting == "equal":
            w = np.ones(n, dtype=float)
        elif self.weighting == "inverse_cv_loss":
            has_full_fold = any(
                not getattr(m, "uses_early_stopping", False) for m in self.members
            )
            if has_full_fold:
                flagged = {
                    m.__class__.__name__
                    for m in self.members
                    if getattr(m, "uses_early_stopping", False)
                }
                if flagged:
                    warnings.warn(
                        f"Ensemble mixes early-stopping members ({flagged}) with full-fold "
                        f"members under strategy='inverse_cv_loss'. CV losses are not directly "
                        f"comparable and weighting will be biased. Consider strategy='equal' or "
                        f"explicit weights. (Flagged members set `uses_early_stopping=True` on "
                        f"the spec; unflagged members default to False.)",
                        UserWarning,
                        stacklevel=2,
                    )
            scores = np.asarray(
                [self.member_scores.get(m.name, np.nan) for m in self.members],
                dtype=float,
            )
            if np.any(~np.isfinite(scores)) or np.any(scores <= 0):
                logger.warning(
                    "inverse_cv_loss weighting: missing/invalid scores %s — "
                    "falling back to equal weights.",
                    scores.tolist(),
                )
                w = np.ones(n, dtype=float)
            else:
                w = 1.0 / scores
        else:  # pragma: no cover — guarded by type system
            raise ValueError(f"Unknown weighting strategy: {self.weighting!r}")

        total = float(w.sum())
        if total <= 0:
            raise ValueError(f"Ensemble weights sum to {total}; cannot normalise")
        return w / total


# ── High-level helper ─────────────────────────────────────────────────────────


def _optimize_one_member(
    member: ModelSpec,
    series: pd.Series,
    n_calls: int,
    optimize_kwargs: dict,
) -> tuple[str, dict, float]:
    """Module-level worker: run ``optimize_model`` for one ensemble member.

    Defined at module scope (not as a nested closure) so ``loky`` can pickle
    it by import path — mirrors the pattern used by
    ``boa_forecaster.validation._run_fold_pinned``.  Returns a simple tuple
    to keep the payload pickle-friendly.
    """
    # Local import avoids circular dependency (optimizer imports models).
    from boa_forecaster.optimizer import optimize_model

    result = optimize_model(series, member, n_calls=n_calls, **optimize_kwargs)
    return member.name, dict(result.best_params), float(result.best_score)


def build_ensemble(
    series: pd.Series,
    specs: list[ModelSpec],
    weighting: WeightingStrategy = "inverse_cv_loss",
    optimise: bool = True,
    n_calls: int = 30,
    n_jobs: int = 1,
    **optimize_kwargs,
) -> tuple[EnsembleSpec, dict[str, dict]]:
    """Optimise each member and return a ready-to-use ``EnsembleSpec``.

    Args:
        series: Training series passed to every member's optimisation.
        specs: List of ``ModelSpec`` instances to include.
        weighting: See :class:`EnsembleSpec` — defaults to inverse CV loss.
        optimise: When ``False``, skip TPE and use each spec's first
            ``warm_starts`` entry as its params (test-fixture convenience).
        n_calls: TPE trials per member when ``optimise=True``.
        n_jobs: Parallel workers used to optimise members concurrently.
            Default ``1`` preserves backwards-compatible sequential
            behaviour.  ``-1`` uses all available CPUs.  Ignored when
            ``optimise=False`` (warm-start path is trivial).  Each worker
            runs its own Optuna study — with the default seed each member
            is deterministic, so ``n_jobs=1`` and ``n_jobs>=2`` produce
            identical output.
        **optimize_kwargs: Forwarded to :func:`optimize_model`.

    Returns:
        ``(spec, params_per_member)`` — plug the dict straight into
        ``spec.build_forecaster(params_per_member)(train)``.
    """
    params_per_member: dict[str, dict] = {}
    member_scores: dict[str, float] = {}

    if not optimise:
        # Fast path — no TPE, no parallelism required.
        for member in specs:
            fallback = member.warm_starts[0] if member.warm_starts else {}
            params_per_member[member.name] = dict(fallback)
            member_scores[member.name] = 1.0
        spec = EnsembleSpec(specs, weighting=weighting, member_scores=member_scores)
        return spec, params_per_member

    if n_jobs is None or n_jobs == 1:
        results = [
            _optimize_one_member(member, series, n_calls, optimize_kwargs)
            for member in specs
        ]
    else:
        # joblib is already a core dep (validation.py uses it).  Each member
        # is an independent Optuna study, so loky-backed process parallelism
        # gives a near-linear speedup on multi-member ensembles.
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_optimize_one_member)(member, series, n_calls, optimize_kwargs)
            for member in specs
        )

    # Preserve the caller's spec order in the returned dicts regardless of
    # completion order from the workers.
    result_by_name = {name: (params, score) for name, params, score in results}
    for member in specs:
        params, score = result_by_name[member.name]
        params_per_member[member.name] = params
        member_scores[member.name] = score

    spec = EnsembleSpec(specs, weighting=weighting, member_scores=member_scores)
    return spec, params_per_member
