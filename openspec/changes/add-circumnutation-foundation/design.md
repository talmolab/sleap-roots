# Design: add-circumnutation-foundation

## Context

This PR is foundational — no compute, only contracts. The detailed theoretical content is in `docs/circumnutation/theory.md`; the umbrella roadmap is in `docs/circumnutation/roadmap.md`. This design doc covers only the implementation choices that affect every subsequent tier PR.

## Goals / Non-Goals

**Goals:**
- Establish the package import-tree so subsequent PRs only need to fill in functions, never restructure.
- Lock the calibration contract (CC-3) by replicating the exact serialization pattern `TrackedTipPipeline` uses.
- Lock the trait CSV row-identity schema (CC-4) so per-genotype / per-plate / per-plant aggregation is feasible without re-engineering.
- Lock the constants pattern (CC-2) so magic numbers don't proliferate.
- Lock the logging convention (CC-9).

**Non-Goals:**
- No trait computation. All compute modules raise `NotImplementedError` until their respective tier PR lands.
- No CLI subcommand (lands PR #17).
- No `CircumnutationPipeline` class (lands PR #14).
- No re-implementation of LOESS or wavelet machinery (lands later PRs).

## Decisions

### D1. Package layout (mirrors `docs/circumnutation/roadmap.md`)

```
sleap_roots/circumnutation/
├── __init__.py           # public API; re-exports CircumnutationInputs only in this PR
├── _constants.py         # all module-level named constants (CC-2)
├── _types.py             # CircumnutationInputs attrs class
├── _io.py                # units-sidecar writer; calibration-pattern reader/writer
├── kinematics.py         # NotImplementedError stub (PR #2)
├── qc.py                 # NotImplementedError stub (PR #3)
├── synthetic.py          # NotImplementedError stub (PR #4)
├── temporal_cwt.py       # NotImplementedError stub (PR #5)
├── psi_g.py              # NotImplementedError stub (PR #7)
├── midline.py            # NotImplementedError stub (PR #8)
├── spatial_cwt.py        # NotImplementedError stub (PR #9)
├── parametric.py         # NotImplementedError stub (PR #11)
├── plotting.py           # NotImplementedError stub (PR #16)
└── pipeline.py           # NotImplementedError stub (PR #14)
```

The stubs exist so subsequent PRs are pure additions, not restructuring. Each stub raises a clear `NotImplementedError("PR #N — see docs/circumnutation/roadmap.md")` on import-time access.

### D2. Calibration pattern — investigation, then replication

`TrackedTipPipeline` (sleap_roots/tracked_tip_pipeline.py, merged in PR #190) already serializes `px_per_mm` somewhere. Possible locations: a sidecar JSON, a CSV header comment, an attrs metadata field, or per-row column. **The implementation MUST start by reading `tracked_tip_pipeline.py`, its integration tests, and a sample output CSV** before writing `_io.py`. The acceptance criterion is: a downstream user can load both pipelines' outputs with the same loader.

This is recorded as task 1.1 in `tasks.md`. The result of that investigation is documented in a docstring comment at the top of `_io.py` and propagates to the units-sidecar / calibration-handling implementation.

**Alternatives considered.**

- *Adopt a fresh pattern* (sidecar JSON + per-row column) without checking what `TrackedTipPipeline` does. Rejected: divergent patterns force every downstream user to write two loaders and break the existing `pixel-unit-invariance` capability.
- *Calibration as required parameter* — fail loudly if `px_per_mm` is missing. Rejected: contradicts theory.md §2.3 contract (NaN-on-missing for `[mm]` traits) and breaks the calibration-omitted scenario in the spec.

### D3. Row-identity schema (CC-4)

The trait CSV has these eight columns at the *front* of every row:

```
series, sample_uid, timepoint, plate_id, plant_id, track_id, genotype, treatment
```

Today `plant_id == track_id` and both columns are populated identically by the foundation. `genotype` and `treatment` are populated from `Series` metadata if available (via the existing `add-sample-uid-timepoint-metadata` capability shipped in PR #171), NaN otherwise.

The trait CSV is sorted lexicographically by `(series, sample_uid, plate_id, plant_id, track_id)` for deterministic ordering across runs.

### D4. Constants strategy (CC-2)

All defaults live in `_constants.py` as module-level UPPER_SNAKE constants with type hints. Trait functions take an optional `constants: Optional[ConstantsT] = None` parameter (a typed dict or attrs class) so callers can override per-call without monkey-patching the module. The `Pipeline` class (PR #14) accepts `constants=` at init and passes through.

`_constants.py` exposes a frozen `ConstantsT` `attrs` class enumerating every overridable parameter, with the table-of-defaults from `roadmap.md` CC-2.

### D5. Logging (CC-9)

Per-module logger via `logger = logging.getLogger(__name__)`, matching `tracked_tip_pipeline.py`. The package does NOT configure logging at import time — that's the application's responsibility. Tests use the `caplog` fixture to verify the expected messages are emitted.

### D6. Test scope for this PR

- Schema test: every row-identity column exists in a function-emitted DataFrame; column dtypes are correct.
- Calibration contract test: `[mm]` traits are NaN when `px_per_mm` is None; `[—]` traits are not affected.
- Constants test: every name from the table-of-defaults (CC-2) is importable and has the documented default.
- Stubs test: importing every stub module succeeds; calling any stub function raises `NotImplementedError` with the documented message format.
- `__init__.py` re-export test: `from sleap_roots import CircumnutationInputs` works.
- Calibration round-trip test (covers D2): create a `CircumnutationInputs` with a known `px_per_mm`, write a CSV via `_io`, read it back via the same loader that handles `TrackedTipPipeline` output, assert `px_per_mm` round-trips.

## Risks / Trade-offs

| Risk | Mitigation |
|---|---|
| `TrackedTipPipeline`'s calibration pattern is hard to discover or undocumented. | The investigation task is explicit (1.1); if the pattern is genuinely unclear, document the finding in `_io.py` and propose a new pattern as a follow-up issue rather than guessing. |
| Stub modules raising `NotImplementedError` confuse Sphinx / mkdocs at import time when generating API docs. | Wrap stub raises in functions, not at module-import time. The module imports cleanly; only function calls raise. |
| Subsequent PRs end up needing to MODIFY (not ADD) the foundational requirements. | Keep this PR's requirements at the *contract* level (calibration, schema, units sidecar). Tier-specific behavior is ADDED by tier PRs. If a tier PR finds the contract too narrow, it's a design bug we want to surface — discuss before MODIFYing. |
| Constants chosen here turn out wrong when first used in a tier PR. | Constants are overridable per-call. Wrong defaults are not blockers; they're tunable. |

## Migration Plan

This is purely additive. No existing behavior changes. Rollback = revert the PR.

## Open Questions

1. Should `CircumnutationInputs` accept the trajectory DataFrame directly, or a path to a `TrackedTipPipeline` output CSV (let it load)? Strawman: accept either, dispatching by type. Resolved during implementation; not blocking.
2. Should `_constants.py` expose constants as a frozen attrs class (`ConstantsT`) with named fields, or as a plain module-level scope? Strawman: frozen attrs class — gives type safety + override semantics in one place. Resolved during implementation.
