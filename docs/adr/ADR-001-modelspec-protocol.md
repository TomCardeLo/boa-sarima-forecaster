# ADR-001 — `ModelSpec` as `Protocol`, not `ABC`

- **Status:** Accepted
- **Date:** 2026-03-26
- **Deciders:** core maintainers
- **Related code:** `src/boa_forecaster/models/base.py`, `src/boa_forecaster/models/_ml_base.py`

## Context

In v2.0, `boa-forecaster` grew from a single-model SARIMA pipeline into a
multi-model framework that had to accommodate:

- **Statistical models** consuming raw `pd.Series` — `SARIMASpec`, and
  potentially ETS, Theta, Prophet, etc.
- **Tabular ML models** that require `FeatureEngineer` — `RandomForestSpec`,
  `XGBoostSpec`, `LightGBMSpec`, and future additions.
- **User-defined models** living *outside* this repository.

All had to plug into a single generic TPE engine (`optimize_model`) without
the engine knowing the model's internals.  Python gave us two idiomatic
options for the interface:

| Option | Coupling | Runtime check | Implementation sharing |
|--------|----------|---------------|------------------------|
| `abc.ABC` + `@abstractmethod` | nominal (must `import` base) | yes (`isinstance`) | yes (via base methods) |
| `typing.Protocol` (structural) | zero (duck typing) | opt-in via `@runtime_checkable` | no — by design |

## Decision

Define `ModelSpec` as a `@runtime_checkable` `typing.Protocol`, and factor
*shared implementation* for tree-based ML models into a **separate concrete
base class** (`BaseMLSpec`) that itself satisfies the Protocol.

```python
# models/base.py
@runtime_checkable
class ModelSpec(Protocol):
    name: str
    needs_features: bool

    @property
    def search_space(self) -> dict[str, SearchSpaceParam]: ...
    @property
    def warm_starts(self) -> list[dict]: ...

    def suggest_params(self, trial: optuna.Trial) -> dict: ...
    def evaluate(self, series, params, metric_fn, feature_config=None) -> float: ...
    def build_forecaster(self, params, feature_config=None): ...
```

`SARIMASpec` satisfies the Protocol without inheriting anything;
`RandomForestSpec`, `XGBoostSpec`, and `LightGBMSpec` inherit from
`BaseMLSpec`, which implements the shared CV loop and forecaster closure
(see ADR folder's `_ml_base.py` and the C1 refactor in the v1.6.0 changelog).

## Consequences

### Positive

- **Zero-import extensibility.** A downstream user or third-party library
  can define a class that satisfies `ModelSpec` without depending on
  `boa_forecaster.models.base`.  This is the textbook advantage of
  structural typing.
- **Clean separation of contract and implementation.**  The Protocol
  describes *what* a spec must expose; `BaseMLSpec` describes *how* ML specs
  typically implement it.  Users of non-ML paradigms (state-space,
  hierarchical, Bayesian) are not forced into the ML shape.
- **Runtime checks still work.**  `@runtime_checkable` means tests and
  defensive code can `isinstance(obj, ModelSpec)`; see
  `tests/unit/test_base.py`.
- **Matches mypy's `--strict` ergonomics.**  Errors surface at the call
  site (`optimize_model(..., spec=MyThing())`) rather than at class
  definition.  That is where the mistake is usually made anyway.

### Negative

- **No single-source base implementation.**  If we only had the Protocol,
  every model would duplicate the CV loop.  Mitigated by `BaseMLSpec` —
  but that mitigation is *optional*, which is fine for SARIMA (which
  shares nothing with tree models) but places the onus on maintainers to
  reach for `BaseMLSpec` when adding a new tree model.
- **Type-checker lag.**  Older mypy releases occasionally miss Protocol
  conformance across re-exports.  We pin `mypy>=1.8` to keep this tight.

### Neutral

- `runtime_checkable` checks only the *names* of members, not their
  signatures.  Signature mismatches are caught by mypy and by the
  integration tests (`tests/integration/test_multi_model.py`,
  `test_full_pipeline.py`), not by `isinstance`.

## Alternatives considered

1. **`abc.ABC` with `@abstractmethod`.**  Rejected: forces every model —
   including one-off experimental scripts — to import our base class, and
   couples the contract to a concrete inheritance tree that conflates
   "must expose X" with "inherits Y's default implementation of X".

2. **Plain duck typing (no Protocol).**  Rejected: we lose `isinstance`
   checks in the optimizer, and the contract becomes purely documentary.
   Tests that assert "this object is a valid spec" would be brittle.

3. **Protocol *plus* `ABC` mixin for shared behaviour.**  Rejected as
   over-engineered: the Protocol-plus-concrete-base-class pattern we
   shipped achieves the same thing with less ceremony.

## References

- [PEP 544 — Protocols: Structural subtyping (static duck typing)](https://peps.python.org/pep-0544/)
- `src/boa_forecaster/models/base.py` — Protocol definition
- `src/boa_forecaster/models/_ml_base.py` — concrete shared base
- ADR-002 — soft-failure contract required by every spec's `evaluate`
