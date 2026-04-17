# Architecture Decision Records

This folder captures the non-obvious design choices that shape
`boa-forecaster`.  Each ADR is a short, dated record of **context → decision →
consequences** so future maintainers can reconstruct *why* something is the way
it is — not just *what* the code does.

We follow the format popularised by Michael Nygard
([*Documenting Architecture Decisions*](https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions),
2011): a problem statement, the decision taken, and the trade-offs accepted.

## Index

| ID | Title | Status | Date |
|----|-------|--------|------|
| [ADR-001](ADR-001-modelspec-protocol.md) | `ModelSpec` as `Protocol`, not `ABC` | Accepted | 2026-03-26 |
| [ADR-002](ADR-002-optimizer-soft-failure.md) | Soft-failure in `optimize_model` (`is_fallback`) | Accepted | 2026-04-12 |
| [ADR-003](ADR-003-combined-metric-weights.md) | Combined objective `0.7·sMAPE + 0.3·RMSLE` | Accepted | 2026-03-17 |

## When to add an ADR

Write one whenever you make a choice that a newcomer would reasonably question
("why a Protocol and not an ABC?", "why catch this exception?"), or that
locks in a trade-off downstream code depends on.  Prefer short, opinionated
ADRs over long ones — if a reader cannot finish it in three minutes, split it.

Numbering is monotonic and never reused.  An ADR that becomes obsolete is
marked `Superseded by ADR-NNN`, not deleted.
