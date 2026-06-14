# Design: add-circumnutation-pipeline (PR #14)

**Date**: 2026-06-14
**Author**: eberrigan (+ Claude)
**Epic**: #197 — circumnutation analysis program
**Upstream (merged)**: every trait tier #1–#10 (Tier 0 kinematics, QC, Tier 1 nutation,
Tier 2 psi_g, Tier 3c traveling_wave) + the PR #1 foundation (`_types`, `_io`, `_constants`)
**Promotes ahead of**: PR #11 (Tier 4 parametric, deferred — blocked on #230 / Phase 2)
**Surfaces**: #222 (the nutation/psi_g `_*_TRAIT_UNITS` gap); #232 (the Tier 0/1 dedup lineage)

## Context

PR #14 is the **composition** PR: it wires the five trait-emitting tiers into one
`CircumnutationPipeline` that produces a single per-plant trait DataFrame, a CSV, and the two
provenance sidecars — the artifact a downstream user (PR #15 aggregation / PR #16 plots / PR #17
CLI / PR #18 guide) actually consumes. It introduces **no new science**: every trait is already
computed and tested by its tier. The work is correct composition (merge on the per-plant 5-tuple),
units aggregation, the headline **Tier 0/Tier 1 dedup**, provenance writing, pickle-ability, and a
real-data plate-001 integration test.

This is a **stub → implementation** transition. `pipeline.compute_traits` currently raises
`NotImplementedError`; the foundation tests track it in `STUB_MODULES` /
`STUBS_WITH_CONSTANTS_KWARG`. PR #14 moves it to the implemented set (impl 9→10, stub 3→2).

## The five tiers being composed

All five are per-track, per-plant `DataFrame → DataFrame` functions with the 8 row-identity
columns and a 5-tuple (`series, sample_uid, plate_id, plant_id, track_id`) groupby, mergeable on
`_IDENTITY_5_TUPLE`:

| Tier | Module | Signature | Cadence? | Trait cols | Has `_*_TRAIT_UNITS`? |
|---|---|---|---|---|---|
| 0 | `kinematics.compute(df, constants=None)` | no cadence | 10 | ✓ `_TIER0_TRAIT_UNITS` |
| QC | `qc.compute(df, constants=None)` | no cadence | 11 | ✓ `_QC_TRAIT_UNITS` |
| 1 | `nutation.compute(df, cadence_s, coordinate="lateral", constants=None)` | needs | 8 | ✗ **add** |
| 2 | `psi_g.compute(df, cadence_s, constants=None)` | needs | 4 | ✗ **add** |
| 3c | `traveling_wave.compute(df, cadence_s, constants=None)` | needs | 6 | ✓ `_TRAVELING_WAVE_TRAIT_UNITS` |

Tier 3c internally **recomputes** Tier 0 (`kinematics.compute`) and a per-track Tier 1 temporal
CWT (`nutation.compute`, lateral) to obtain `v_total_median_px_per_frame` and `T_nutation_median`
for its `lambda_expected_px` / `traveling_wave_residual` composition — redundant by design for a
self-contained tier, and the dedup target here.

## Decisions (D1–D8)

### D1 — Pipeline shape: SEQUENTIAL MERGE-ORCHESTRATOR (not the per-frame networkx DAG)

`CircumnutationPipeline` is a purpose-built `attrs` class that calls each tier's `compute()` once
and merges the per-plant outputs on `_IDENTITY_5_TUPLE`. It honors the DAG *intent* — a documented
fixed tier order (Tier 0 / QC / Tier 1 / Tier 2 independent; Tier 3c depends on Tier 0 + Tier 1),
pure, picklable — **without** the per-frame node model of `trait_pipelines.Pipeline`.

> **⚠ DEVIATION (record in spec + design):** the `pipeline.py` stub docstring and the roadmap row
> say PR #14 builds a "TraitDef DAG matching the `Pipeline` base class pattern". PR #14
> **deliberately does not**. **Why:** `trait_pipelines.Pipeline` is a *per-frame* networkx
> `TraitDef` DAG; the circumnutation tiers are *per-track DataFrame → DataFrame* functions that do
> not fit that node model cleanly. A sequential merge-orchestrator expresses the same dependency
> structure with far less impedance mismatch. The stub docstring + roadmap row are corrected as
> tasks.

### D2 — Tier 0/Tier 1 dedup: PRECOMPUTED-FRAMES FAST PATH (Option A)

Add optional keyword-only `tier0_df=None` / `tier1_df=None` to `traveling_wave.compute`. When the
pipeline supplies them (it has already computed Tier 0 and Tier 1 as emitting tiers), `compute`
skips the internal recompute (current lines ~367–378) and uses them directly. When omitted
(standalone path), `compute` recomputes exactly as today.

- The spatial chain (`_compute_one_track`, calibration, COI gating) and the delicate
  `lambda_expected_px` / `traveling_wave_residual` composition (current lines ~389–411) stay
  entirely inside `traveling_wave` — **single source of truth**; the pipeline never re-derives that
  logic.
- **Standalone behavior is byte-identical when the kwargs are omitted** → the merged module's 37
  tests + cross-OS canary stay green untouched.
- Reusability is sound: `kinematics.compute` has no coordinate axis, and `nutation.compute`
  **defaults to `coordinate="lateral"`** ([nutation.py:562](../../../sleap_roots/circumnutation/nutation.py#L562))
  — exactly the coordinate Tier 3c needs — so the pipeline's *emitted* Tier 1 frame is the correct
  operand frame to pass in.

**Kwarg semantics (pin in the spec):**
- **Both-or-neither.** Supplying exactly one of `tier0_df` / `tier1_df` is a `ValueError`
  (ambiguous: half-deduped would silently recompute the other half).
- **Same `constants`.** The pipeline passes the same resolved `constants` to all tiers and to the
  fast path, so an override is honored consistently.
- **Validated, not trusted blindly.** The fast path verifies each passed frame carries
  `_IDENTITY_5_TUPLE` + the single operand column it consumes
  (`v_total_median_px_per_frame` from `tier0_df`, `T_nutation_median` from `tier1_df`) and raises a
  named `ValueError` if missing — cheaper than a recompute and prevents a silent all-NaN join.
- The standalone docstring `Note` (currently telling batch callers to route through the pipeline)
  is updated to document the new fast-path kwargs.

### D3 — Output contract: PURE `compute_traits` + separate `save()` (Option 1)

`compute_traits(inputs, constants=None)` stays **pure** (no I/O) and returns the stub's tuple
`(per_plant_df, trajectory_df, units_dict)`. Writing is a separate
`CircumnutationPipeline.save(out_path, per_plant_df, units, *, input_path, run_id=None)` that calls
`_io.gather_run_metadata(input_path, run_id, constants)` + `_io.write_per_plant_csv(out_path, df,
units, run_metadata)` (which writes the CSV + `<stem>.units.json` + `run_metadata.json`).

- **Why separate:** the stub return tuple is honored verbatim (no signature deviation); compute is
  trivially unit-testable with no filesystem; picklability stays clean. Critically,
  `gather_run_metadata` requires `input_path` (the source `.slp` path), which
  `CircumnutationInputs` does **not** carry — the `save` boundary is where the caller actually knows
  that path, so the provenance gap is resolved there instead of leaking into the compute signature.
- The integration test exercises compute → save → `_io.read_per_plant_csv` round-trip.

### D4 — Units assembly: ADD `_*_TRAIT_UNITS` TO `nutation` + `psi_g` (Option X)

`write_per_plant_csv` hard-enforces (before writing anything) that the units dict 1:1-covers the
DataFrame columns **and** that every value is in `PIPELINE_UNIT_VOCABULARY`. Two of five tiers lack
a units map. PR #14 closes the **#222** gap properly: add `_NUTATION_TRAIT_UNITS` to `nutation.py`
and `_PSIG_TRAIT_UNITS` to `psi_g.py`, co-located with their `_*_TRAIT_COLUMNS` (the established
kinematics/qc/traveling_wave convention) — **additive only**, no behavior change, existing tier
tests stay green.

The pipeline assembles one composed units dict by merging, in tier order: `ROW_IDENTITY_UNITS`
(restricted to the identity columns) ∪ `_TIER0_TRAIT_UNITS` ∪ `_QC_TRAIT_UNITS` ∪
`_NUTATION_TRAIT_UNITS` ∪ `_PSIG_TRAIT_UNITS` ∪ `_TRAVELING_WAVE_TRAIT_UNITS`. The
`growth_axis_unreliable` key appears in both Tier 0 and QC maps with the **same** value (`"bool"`),
so the dict-merge dedups it; the pipeline asserts the two source values are equal before merging
(guards a future divergence). Each new per-column unit follows the column's existing semantics
(`px`, `px/frame`, `s`, `rad`, `—`, `bool`, `string`) drawn only from `PIPELINE_UNIT_VOCABULARY`;
the writer's membership + 1:1 checks validate the result.

### D5 — `growth_axis_unreliable` coalescing: TIER 0 OWNS IT

Both Tier 0 and QC emit `growth_axis_unreliable` (QC recomputes it locally with the same formula
and inputs as Tier 0 — [qc.py:207–231](../../../sleap_roots/circumnutation/qc.py#L207-L231) — and
an existing cross-tier equality test guarantees element-wise equality). A naive per-plant merge
would collide them into `_x` / `_y`.

**Resolution:** Tier 0 owns the column in the composed output (it sits in the Tier 0 block next to
`principal_axis_angle`, the computation it flags). QC's copy is **dropped before QC merges in**, so
QC contributes 10 columns instead of 11. The plate-001 integration test re-asserts cross-tier
equality at the composed level (defense-in-depth; cheap; documents the invariant).

### D6 — Adapter + integration test: INLINE IN TEST, DEFER ADAPTER TO PR #17 (Option 6a)

The plate-001 integration test builds the trajectory DataFrame inline using the proven pattern from
[test_circumnutation_traveling_wave.py:562–576](../../../tests/test_circumnutation_traveling_wave.py#L562-L576)
(`Series.load` → `get_tracked_tips()` → coerce `track_id` `"track_N"`→int → attach the 8
row-identity columns with hardcoded plate-001 metadata), constructs `CircumnutationInputs`, and
round-trips compute → save → read-back.

A general `Series → CircumnutationInputs` adapter is **deferred to PR #17 (CLI)**, where the
metadata-sourcing design (sample_uid / timepoint / genotype / treatment come from CLI args or a
metadata file, **not** the `.slp`) is properly motivated. Building it now would pull
metadata-provenance design into PR #14 and risk locking an interface before the CLI's needs are
known. The integration test still exercises the real public entry point
(`CircumnutationInputs` → `compute_traits` → `save`).

### D7 — Validation & cadence threading

`compute_traits(inputs, constants=None)` receives an already-validated `CircumnutationInputs` (its
`attrs` validators run `_validate_trajectory_df` + cadence/`R_px` checks at construction), and each
tier re-validates `trajectory_df` internally. The pipeline therefore adds **no redundant validation
of its own**. It reads `inputs.trajectory_df` + `inputs.cadence_s`, passes `trajectory_df` to all
five tiers, and threads `cadence_s` **only** to nutation / psi_g / traveling_wave (Tier 0 / QC take
no cadence). `R_px` is unused by these five (pure-pixel, CC-3) and is carried in provenance only.

### D8 — Constants / determinism / picklability

- **Constants:** no new *physical* constant. Tier order and composed column order are module-level
  tuples in `pipeline.py`, not `ConstantsT` fields. **`_CONSTANTS_VERSION` stays 6.**
- **Picklability:** `CircumnutationPipeline` holds only `constants` (an optional, picklable
  `ConstantsT`). A `pickle.loads(pickle.dumps(pipeline))` round-trip test guards it.
- **Determinism:** **no composed-output canary.** Each tier already owns its canary; a composed
  canary would re-pin values the tiers already pin and add brittleness. Guards instead: a cheap
  in-process two-run bit-identical determinism test, plus the plate-001 integration test (schema +
  all-five-tiers'-columns present + the QPB band from the traveling_wave real-data test) as the
  scientific cross-check.

## Architecture

```
inputs: CircumnutationInputs(trajectory_df, cadence_s, R_px=None, run_id=None)
   │
CircumnutationPipeline(constants=None)            # picklable attrs; only field = constants
   │  .compute_traits(inputs) -> (per_plant_df, trajectory_df, units_dict)   # PURE
   │     1. tier0 = kinematics.compute(df, constants)            # 10 cols
   │     2. qc    = qc.compute(df, constants)                    # 11 cols → drop growth_axis_unreliable → 10
   │     3. tier1 = nutation.compute(df, cadence_s, constants)   # 8 cols  (coordinate="lateral" default)
   │     4. tier2 = psi_g.compute(df, cadence_s, constants)      # 4 cols
   │     5. tier3c = traveling_wave.compute(df, cadence_s, constants,
   │                                        tier0_df=tier0, tier1_df=tier1)   # 6 cols — DEDUP fast path
   │     6. per_plant_df = reduce(merge on _IDENTITY_5_TUPLE) in fixed tier order
   │     7. units_dict   = ROW_IDENTITY_UNITS ∪ all five tier _*_TRAIT_UNITS
   │
   │  .save(out_path, per_plant_df, units, *, input_path, run_id=None)        # I/O
   │     run_metadata = _io.gather_run_metadata(input_path, run_id, constants)
   │     _io.write_per_plant_csv(out_path, per_plant_df, units, run_metadata) # CSV + 2 sidecars

module-level compute_traits(inputs, constants=None) = CircumnutationPipeline(constants).compute_traits(inputs)
   # thin wrapper preserving the stub's module-level signature
```

### Composed schema (per-plant CSV)

8 `ROW_IDENTITY_COLUMNS` + 38 trait columns = **46 columns**, in fixed tier order:

| Block | Source | # cols | Notes |
|---|---|---|---|
| Row identity | `ROW_IDENTITY_COLUMNS` | 8 | series … treatment |
| Tier 0 | `_TIER0_TRAIT_COLUMNS` | 10 | **owns `growth_axis_unreliable`** |
| QC | `_QC_TRAIT_COLUMNS` − `growth_axis_unreliable` | 10 | `track_is_clean`, `qc_failure_reason`, … |
| Tier 1 | `_NUTATION_TRAIT_COLUMNS` | 8 | `is_nutating` (bool), … |
| Tier 2 | `_PSIG_TRAIT_COLUMNS` | 4 | `handedness`, … |
| Tier 3c | `_TRAVELING_WAVE_TRAIT_COLUMNS` | 6 | all float64 |

Mixed dtypes across blocks are expected (each tier emits its own; bool/string flags survive the
left-merge). The 5-tuple merge is `how="left"` onto the shared per-plant template
(`_build_per_plant_template_from_df`), with the int64 coercion-with-raise guard on
`track_id`/`plant_id` that every tier already applies — so a dtype mismatch raises instead of
silently producing all-NaN.

## Files touched

- **`sleap_roots/circumnutation/pipeline.py`** — implement `CircumnutationPipeline` + `compute_traits`
  + `save`; module-level tier-order + column-order tuples; `logger.debug` per CC-9.
- **`sleap_roots/circumnutation/traveling_wave.py`** — add the `tier0_df`/`tier1_df` fast-path kwargs
  (additive, standalone path unchanged); update the docstring `Note`.
- **`sleap_roots/circumnutation/nutation.py`** — add `_NUTATION_TRAIT_UNITS` (additive).
- **`sleap_roots/circumnutation/psi_g.py`** — add `_PSIG_TRAIT_UNITS` (additive).
- **`tests/test_circumnutation_foundation.py`** — stub→impl migration for `pipeline`: remove from
  `STUB_MODULES` / `STUBS_WITH_CONSTANTS_KWARG`; add to the implemented + namespaced-logger lists +
  an `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG` entry + a dedicated
  `test_implementation_accepts_constants_kwarg` branch (verify exact locations at impl time).
- **`tests/test_circumnutation_pipeline.py`** (new) — unit + integration tests.
- **Docs/specs:** `openspec/specs/circumnutation/spec.md` (Package layout 9→10 / 3→2 + new
  composition-API requirement + import scenario); `docs/circumnutation/roadmap.md` (row #14 ⬜→
  in-progress, correct the "TraitDef DAG" claim, note the dedup + units additions);
  `docs/changelog.md`.

## Testing strategy (TDD)

- **Composition unit tests** (synthetic multi-track, ≥2 plates with overlapping `track_id` to
  exercise the 5-tuple merge): composed schema = exactly the 46 columns in declared order; every
  tier's columns present; one row per 5-tuple; dtypes preserved per block.
- **Dedup equivalence:** `traveling_wave.compute(df, cadence_s)` (standalone) vs the pipeline's
  fast-path call (`tier0_df`/`tier1_df` supplied) produce **identical** Tier 3c columns — the dedup
  changes performance, not results. Plus: supplying one of two kwargs raises; a frame missing its
  operand column raises.
- **Units:** the assembled units dict 1:1-covers the 46 columns and passes the writer's vocabulary
  check; `growth_axis_unreliable` present exactly once; `_NUTATION_TRAIT_UNITS` /
  `_PSIG_TRAIT_UNITS` cover their tiers' columns.
- **`growth_axis_unreliable` coalescing:** exactly one such column post-merge (no `_x`/`_y`); equals
  both source tiers' values.
- **save() round-trip:** compute → save → `read_per_plant_csv` recovers the DataFrame, units, and
  run_metadata (with provenance keys: git SHA, versions, ISO timestamp, `_schema_version=1`,
  `_constants_version=6`, `_constants_snapshot`).
- **Picklability:** `pickle.loads(pickle.dumps(CircumnutationPipeline()))` round-trips and the
  unpickled instance computes identically.
- **Determinism:** two in-process `compute_traits` runs are bit-identical on the float columns.
- **Real plate-001 integration:** load the proofread `.slp`, run the full pipeline, write the CSV +
  sidecars; assert 6 rows, the 46-column schema, all five tiers' columns present and finite where
  expected, the QPB residual band (matching the traveling_wave real-data test), and cross-tier
  `growth_axis_unreliable` equality. Skipped if the Git-LFS fixture is absent.

## Risks / Trade-offs

- **Touching the merged `traveling_wave.py`** (dedup fast path) → mitigated: additive optional
  kwargs; standalone path byte-identical; the dedup-equivalence test pins identical results; the 37
  existing tests + canary must stay green (verification gate).
- **Touching merged `nutation.py` / `psi_g.py`** (units maps) → mitigated: additive module
  constants, no behavior change.
- **Mixed-dtype composed frame** → expected and tested; the left-merge preserves bool/string flags;
  no global float64 coercion (unlike a single-tier frame).

## Open questions (resolve at impl)

- Exact per-column unit strings for `_NUTATION_TRAIT_UNITS` / `_PSIG_TRAIT_UNITS` (derive from each
  column's semantics; e.g. `handedness` and `is_nutating` unit choice from `PIPELINE_UNIT_VOCABULARY`
  — likely `—` / `bool`; pin in the proposal tasks after reading each column's definition).
- Exact foundation-test edit locations (verify line numbers at impl, per the PR #10 CR2-2 lesson).

## Decisions log

- **D1** Sequential merge-orchestrator, not the per-frame networkx DAG (deviation from stub/roadmap;
  recorded with Why).
- **D2** Tier 0/1 dedup via optional `tier0_df`/`tier1_df` fast-path kwargs on `traveling_wave.compute`
  (Option A); both-or-neither; validated; standalone byte-identical.
- **D3** Pure `compute_traits` returning the stub tuple + separate `save()` (Option 1); `input_path`
  provenance handled at the write boundary.
- **D4** Add `_NUTATION_TRAIT_UNITS` / `_PSIG_TRAIT_UNITS` to their tiers (Option X, the #222
  follow-up); pipeline merges row-identity + five tier maps.
- **D5** Tier 0 owns `growth_axis_unreliable`; QC's copy dropped pre-merge; cross-tier equality
  re-asserted.
- **D6** Integration test builds the df inline (plate-001 pattern); `Series → Inputs` adapter
  deferred to PR #17.
- **D7** No redundant pipeline validation; cadence only to Tier 1/2/3c; `R_px` provenance-only.
- **D8** `_CONSTANTS_VERSION` stays 6; picklable attrs (only `constants`); no composed canary —
  integration test is the cross-check.
