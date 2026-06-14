# Design: add-circumnutation-pipeline (PR #14)

> Full design narrative + the one-at-a-time brainstorm record (D1–D8) lives in
> `docs/superpowers/specs/2026-06-14-add-circumnutation-pipeline-design.md`. This file captures the
> architecture decisions and trade-offs that bear on the spec deltas.

## Context

PR #14 composes five already-implemented, per-plant trait tiers into one `CircumnutationPipeline`.
No new science. The work is correct composition, units aggregation, the Tier 0/Tier 1 dedup,
provenance writing, picklability, and a real-data integration test.

The five tiers (all per-track, 5-tuple groupby, mergeable on `_IDENTITY_5_TUPLE`):

| Tier | Callable | Cadence? | Trait cols | `_*_TRAIT_UNITS`? |
|---|---|---|---|---|
| 0 | `kinematics.compute(df, constants=None)` | no | 10 | ✓ |
| QC | `qc.compute(df, constants=None)` | no | 11 | ✓ |
| 1 | `nutation.compute(df, cadence_s, coordinate="lateral", constants=None)` | yes | 8 | ✗ add |
| 2 | `psi_g.compute(df, cadence_s, constants=None)` | yes | 4 | ✗ add |
| 3c | `traveling_wave.compute(df, cadence_s, constants=None)` | yes | 6 | ✓ |

## Decision D1 — Sequential merge-orchestrator (not a per-frame TraitDef DAG)

`CircumnutationPipeline` is a purpose-built `attrs` class that calls each tier's `compute()` once
and merges per-plant outputs on `_IDENTITY_5_TUPLE`. It honors the DAG *intent* (documented fixed
tier order; Tier 3c depends on Tier 0 + Tier 1; pure; picklable) without the per-frame node model.

**Deviation:** the stub docstring + roadmap row say "TraitDef DAG matching the `Pipeline` base
class pattern". PR #14 does not. **Why:** `trait_pipelines.Pipeline` is a *per-frame* networkx DAG;
the circumnutation tiers are *per-track DataFrame → DataFrame* functions that do not fit that node
model. The merge-orchestrator expresses the same dependency structure with far less impedance
mismatch. Corrected in the stub docstring + roadmap as tasks.

## Decision D2 — Tier 0/Tier 1 dedup: precomputed-frames fast path (Option A)

`traveling_wave.compute` gains optional keyword-only `tier0_df` / `tier1_df`. When the pipeline
supplies them (it has already computed Tier 0 and Tier 1 as emitting tiers), `compute` uses them
directly and skips the internal recompute. The spatial chain and the delicate
`lambda_expected_px` / `traveling_wave_residual` composition stay entirely inside `traveling_wave`
— single source of truth; the pipeline never re-derives that logic.

- **Reusability is sound:** `kinematics.compute` has no coordinate axis, and `nutation.compute`
  defaults to `coordinate="lateral"` — exactly the operand frame Tier 3c needs — so the pipeline's
  *emitted* Tier 1 frame is the correct one to pass in.
- **Both-or-neither.** Supplying exactly one kwarg raises `ValueError` (a half-deduped call would
  silently recompute the other half).
- **Validated, not blindly trusted.** Each frame is checked to carry `_IDENTITY_5_TUPLE` + the
  single operand column it supplies (`v_total_median_px_per_frame` / `T_nutation_median`); missing
  → `ValueError`. Cheaper than a recompute; prevents a silent all-NaN join.
- **Result-identical + standalone byte-identical.** The fast path produces the same Tier 3c columns
  as the recompute (dedup changes performance, not results); omitting the kwargs reproduces today's
  behavior exactly, so the merged module's 37 tests + cross-OS canary stay green.

Rejected: (B) relocating the spatial chain + composition into the pipeline (duplicates the delicate
residual logic; refactor risk to a merged module); (C) leaving the redundancy (does not deliver the
headline dedup).

## Decision D3 — Pure `compute_traits` + separate `save()` (Option 1)

`compute_traits` stays pure and returns the stub's `(per_plant_df, trajectory_df, units_dict)`.
`CircumnutationPipeline.save(out_path, per_plant_df, units, *, input_path, run_id=None)` calls
`_io.gather_run_metadata(input_path, run_id, constants)` + `_io.write_per_plant_csv(...)`.
**Why separate:** honors the stub tuple (no signature deviation); compute is filesystem-free /
trivially testable; picklability stays clean. Critically, `gather_run_metadata` needs `input_path`
(the source `.slp`), which `CircumnutationInputs` does not carry — the `save` boundary is where the
caller knows that path, so the provenance gap is resolved there, not leaked into `compute`.

## Decision D4 — Units assembly: add `_NUTATION_TRAIT_UNITS` / `_PSIG_TRAIT_UNITS` (Option X)

`write_per_plant_csv` hard-enforces 1:1 column coverage + `PIPELINE_UNIT_VOCABULARY` membership.
Two of five tiers lack a units map. PR #14 adds them, co-located with each tier's
`_*_TRAIT_COLUMNS` (the kinematics/qc/traveling_wave convention) — additive, no behavior change.
The pipeline assembles one composed dict: `ROW_IDENTITY_UNITS` ∪ the five tier maps. The exact
strings are **pinned in the spec** (the writer validates vocabulary membership but NOT semantic
correctness — an in-vocabulary-but-wrong string would silently mislabel a published column). Traps
the review surfaced: `noise_floor_estimate` → `"px"` (a median FFT amplitude, NOT `"—"`);
`helix_signed_area_px2` → `"px²"` (superscript glyph, not ASCII `"px2"`); `handedness` → `"int"`
(integer sign, matching the type-token convention for non-float columns). The `"s"` period units
arrive pre-converted from the temporal CWT (`periods_s = scale2frequency(...) / cadence_s`); the
pipeline does no unit conversion.

**#222 scope (deferred).** #222 is a program-wide *suffix convention* requiring `T_nutation_median`
→ `T_nutation_median_s` / `T_nutation_iqr` → `T_nutation_iqr_s`, a documented rule, and a foundation
suffix-gate — NOT just the units maps. PR #14 adds ONLY the maps (keyed on current column names) and
explicitly defers the rename to #222 (the rename ripples into `traveling_wave`'s internal
`T_nutation_median` read, the dedup, theory.md, and the nutation tests). The maps re-key when #222
lands.

Locked as scenarios under the composition-API requirement (the maps are added to *enable*
composition), rather than reproducing the two large tier requirements.

## Decision D5 — `growth_axis_unreliable` coalescing: Tier 0 owns it

Both Tier 0 and QC emit it; QC recomputes it locally with the same formula/inputs as Tier 0, and an
existing cross-tier equality test guarantees element-wise equality. A naive merge collides them into
`_x`/`_y`. **Resolution:** Tier 0 owns the column (it sits next to `principal_axis_angle`, the
computation it flags); QC's copy is dropped before QC merges in (QC contributes 10 cols, not 11).
The integration test re-asserts cross-tier equality at the composed level.

## Decision D6 — Integration test inline; adapter deferred to PR #17

The plate-001 integration test builds the trajectory DataFrame inline (the proven
`traveling_wave` real-data pattern: `Series.load` → `get_tracked_tips()` → `track_id` int coercion
→ attach the 8 row-identity columns with hardcoded plate-001 metadata) → `CircumnutationInputs` →
compute → save → read-back. The general `Series → CircumnutationInputs` adapter — and its
metadata-sourcing design (metadata comes from CLI args / a file, not the `.slp`) — is deferred to
PR #17 (CLI), where its requirements are defined. Avoids prematurely locking that interface.

## Decision D7 — Validation & cadence threading

`compute_traits` receives an already-validated `CircumnutationInputs` (its `attrs` validators run
`_validate_trajectory_df` + cadence/`R_px` checks at construction); each tier re-validates
internally. The pipeline adds no redundant validation. It threads `cadence_s` only to nutation /
psi_g / traveling_wave; Tier 0 / QC take none. `R_px` is unused by these five (pure-pixel, CC-3),
carried in provenance only.

## Decision D8 — Constants / determinism / picklability

- **Constants:** no new physical constant; tier order + composed column order are module-level
  tuples, not `ConstantsT` fields → `_CONSTANTS_VERSION` stays 6.
- **Picklability:** `CircumnutationPipeline` holds only `constants` (picklable `ConstantsT`); a
  pickle round-trip test guards it.
- **Determinism:** no composed-output canary — each tier already owns its canary; a composed canary
  would re-pin tier values and add brittleness. Guards: a two-run in-process bit-identical test +
  the plate-001 integration test (schema, all-tiers-present, QPB band) as the cross-check.

## Composed schema

8 `ROW_IDENTITY_COLUMNS` + 38 trait columns = **46 columns**, fixed tier order: Tier 0 (10, owns
`growth_axis_unreliable`) → QC (10, GAU dropped) → Tier 1 (8) → Tier 2 (4) → Tier 3c (6). Mixed
dtypes across blocks are expected (bool `is_nutating`, string `qc_failure_reason`, etc.). The merge
is `how="left"` onto the shared `_build_per_plant_template_from_df` template with the int64
coercion-with-raise guard on `track_id`/`plant_id` every tier already applies.

## Risks / Trade-offs

- **Touching merged `traveling_wave.py`** (fast path) → additive optional kwargs; standalone
  byte-identical; a dedup-equivalence test pins identical results; the 37 tests + canary stay green.
- **Touching merged `nutation.py` / `psi_g.py`** (units maps) → additive constants, no behavior
  change.
- **Mixed-dtype composed frame** → expected + tested; the left-merge preserves bool/string flags.

## Open questions (resolve at impl)

- Exact `tests/test_circumnutation_foundation.py` edit locations (verify line numbers at impl, per
  the PR #10 CR2-2 lesson — the stub→impl migration touches `STUB_MODULES`,
  `STUBS_WITH_CONSTANTS_KWARG`, the namespaced-logger list, `IMPLEMENTATIONS_WITH_CONSTANTS_KWARG`,
  and the `test_implementation_accepts_constants_kwarg` branch).
