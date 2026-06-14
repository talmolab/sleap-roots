# Change: add-circumnutation-pipeline (PR #14)

## Why

The five circumnutation trait tiers (Tier 0 kinematics, QC, Tier 1 nutation, Tier 2 ψ_g,
Tier 3c traveling_wave) each emit a per-plant DataFrame, but nothing composes them into the single
per-plant trait table + CSV + provenance sidecars that downstream work (PR #15 aggregation, PR #16
plots, PR #17 CLI, PR #18 user guide) consumes. PR #14 supplies that composition: a
`CircumnutationPipeline` that runs each tier once, merges their outputs on the per-plant 5-tuple,
assembles a complete units mapping, writes the CSV + two sidecars, and is picklable for future
parallelization.

It also resolves a redundancy the tiers shipped by design: `traveling_wave.compute` recomputes
Tier 0 and a per-track Tier 1 temporal CWT internally to obtain `v_total_median_px_per_frame` and
`T_nutation_median`. In a full pipeline run those tiers are already computed as emitting tiers, so
the pipeline must **dedup** rather than pay that cost twice (issue #232 lineage).

This is a **stub → implementation** transition: `pipeline.compute_traits` currently raises
`NotImplementedError`; PR #14 implements it (implementation modules 9 → 10, stub modules 3 → 2). No
new science is introduced — every trait is already computed and tested by its tier.

## What Changes

- **Implement `sleap_roots.circumnutation.pipeline`** as a sequential merge-orchestrator (NOT the
  per-frame networkx `TraitDef` DAG the stub docstring / roadmap row mention — see Deviations):
  - A picklable `CircumnutationPipeline` `attrs` class whose only field is `constants`.
  - Pure `compute_traits(inputs, constants=None) -> (per_plant_df, trajectory_df, units_dict)`
    (module-level wrapper preserved for the stub's signature) that calls each tier's `compute()`
    once in a documented fixed order (Tier 0 / QC / Tier 1 / Tier 2 independent; Tier 3c depends on
    Tier 0 + Tier 1) and merges the per-plant outputs on `_IDENTITY_5_TUPLE` into a 46-column frame.
  - A separate `save(out_path, per_plant_df, units, *, input_path, run_id=None)` that writes the
    CSV + units + run-metadata sidecars via the existing `_io` writers (I/O kept out of `compute`).
- **Dedup Tier 0/Tier 1** via an optional precomputed-frames fast path on
  `traveling_wave.compute`: new keyword-only `tier0_df` / `tier1_df`. When supplied (both), it skips
  the internal Tier 0/Tier 1 recompute; standalone behavior (kwargs omitted) is byte-identical.
- **Close the #222 units-map gap**: add `_NUTATION_TRAIT_UNITS` to `nutation.py` and
  `_PSIG_TRAIT_UNITS` to `psi_g.py` (additive module constants) so every emitted column has a unit
  for the sidecar writer's 1:1 / vocabulary checks.
- **Coalesce `growth_axis_unreliable`** (emitted, equal-by-construction, by both Tier 0 and QC):
  Tier 0 owns it in the composed output; QC's copy is dropped before merge.
- **Integration test** round-trips the real proofread plate-001 `.slp` through the full pipeline
  (compute → save → read-back) and asserts the composed schema + all five tiers' columns. The
  general `Series → CircumnutationInputs` adapter is deferred to PR #17 (CLI), where the
  metadata-sourcing design belongs.

## Deviations (recorded per deviation discipline)

- **Sequential merge-orchestrator, not a `TraitDef` DAG.** The `pipeline.py` stub docstring and the
  roadmap row state PR #14 builds "a TraitDef DAG matching the `Pipeline` base class pattern". It
  deliberately does not. **Why:** `trait_pipelines.Pipeline` is a *per-frame* networkx `TraitDef`
  DAG; the circumnutation tiers are *per-track DataFrame → DataFrame* functions that do not fit that
  node model. A sequential merge-orchestrator expresses the same dependency structure without the
  impedance mismatch. The stub docstring + roadmap row are corrected as tasks.
- **`_NUTATION_TRAIT_UNITS` / `_PSIG_TRAIT_UNITS` added now.** These tiers predate the
  `_*_TRAIT_UNITS` convention (#222). PR #14 surfaces the gap (the pipeline must supply units for
  every column) and closes it, additively.

## Capabilities / Spec deltas

- **MODIFIED — Package layout** (`specs/circumnutation/spec.md`): `pipeline` transitions stub →
  implementation (impl 9 → 10, stub 3 → 2); removed from the stub-callable table; PR #14 scope note
  + a `pipeline.compute_traits` callability scenario added; the now-empty "stubs accept
  `constants=None`" scenario dropped (`pipeline` was the last constants-bearing stub).
- **MODIFIED — Tier 3c traveling-wave trait emission API**: add the optional keyword-only
  `tier0_df` / `tier1_df` precomputed-frames dedup fast path (both-or-neither; validated;
  result-identical to the recompute; standalone byte-identical).
- **ADDED — Circumnutation pipeline composition API**: the `CircumnutationPipeline` class +
  `compute_traits` return contract + the 46-column composed schema + merge-on-5-tuple +
  `growth_axis_unreliable` coalescing + units-dict assembly (incl. the new nutation/psi_g maps) +
  `save()` sidecar writing + picklability + determinism + the plate-001 integration contract.

## Impact

- **Affected specs**: `circumnutation` (2 MODIFIED requirements, 1 ADDED requirement).
- **Affected code**: `sleap_roots/circumnutation/pipeline.py` (implement), `traveling_wave.py`
  (additive fast-path kwargs), `nutation.py` + `psi_g.py` (additive units maps),
  `tests/test_circumnutation_foundation.py` (stub→impl migration),
  `tests/test_circumnutation_pipeline.py` (new).
- **Docs**: `docs/circumnutation/roadmap.md` (row #14, correct the DAG claim),
  `docs/changelog.md`.
- **Constants**: none added — `_CONSTANTS_VERSION` stays 6.
- **Backward compatibility**: the `traveling_wave.compute` change is additive (optional kwargs);
  the nutation/psi_g additions are new module constants; no existing signature changes.

## References

- Design doc: `docs/superpowers/specs/2026-06-14-add-circumnutation-pipeline-design.md`
- Epic #197; closes the PR #14 roadmap row; surfaces #222 (units maps), #232 (dedup lineage).
