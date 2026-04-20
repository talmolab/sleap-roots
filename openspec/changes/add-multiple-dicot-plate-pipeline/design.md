# Design: `add-multiple-dicot-plate-pipeline`

**Architectural source of truth**: [`docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md`](../../../docs/superpowers/specs/2026-04-16-multiple-dicot-plate-pipeline-design.md) (committed on this branch; corrections committed as `8b02698`).

That document describes:

- **Decision framework D1–D7** (cylinder-vs-plate semantics, count-filter skip, deterministic `plant_id`, tertiary→primary direct association for PR 2, raw points in JSON, CSV column order, no base-class changes).
- **TraitDef DAG** for the plate pipeline.
- **Compute flow** (frame loop + nested DicotPipeline per plant).
- **SLEAP instance index mapping** (validity mask computed before `filter_roots_with_nans` so original indices survive the filter's index collapse).
- **Per-plant trait set** — reuses `DicotPipeline.csv_traits` unchanged; no renaming.
- **Output structures** (per-series JSON, per-plant CSV, batch DataFrame / JSON list).
- **Follow-up issues** (PR 2, PR 3, A, B, C, D, E).

## Why this proposal defers to the design doc

The design doc was produced via the `superpowers:brainstorming` skill with the same user, captures 7 architectural decisions with rationale, and is committed to the feature branch. Duplicating it here would create two sources of truth and invite silent divergence. The proposal and spec delta instead:

- Declare the **scope** of PR 1 (primary + lateral only).
- Enumerate the **acceptance criteria** as spec scenarios.
- Break the implementation into ordered **TDD tasks**.
- Reference the design doc for every architectural "why".

## OpenSpec-specific design notes

### New capability, not a modification

`multiple-dicot-plate-pipeline` is introduced as a **new capability** (orthogonal addition) rather than as `MODIFIED Requirements` on the existing `multiple-dicot-pipeline` spec. Rationale:

- The existing `multiple-dicot-pipeline` spec is scoped to cylinder-scan semantics (count-filter drops frames on mismatch, cross-frame aggregation, summary stats output).
- The plate pipeline has **inverted semantics** on count-filtering (keeps plants, emits expected_count/detected_count) and **different output shape** (per-plant-per-frame rows, not cross-frame summary).
- Mixing the two into one spec would require ambiguous qualifiers ("for cylinders, X; for plates, Y") that obscure the contract.
- A clean separation means the cylinder spec keeps its regression scenarios (pinned by PR #155) intact, and the plate spec can evolve independently in PR 2 (tertiary) and PR 3 (filter config).

### Spec delta format

All requirements live under `## ADDED Requirements` in `specs/multiple-dicot-plate-pipeline/spec.md`. No `MODIFIED` or `REMOVED` sections — PR 1 is purely additive.

### Task ordering (TDD)

`tasks.md` enforces tests-first strictly: sections 1 and 3 write failing tests with named verification criteria, and each implementation step (sections 2 and 4) is bounded by the tests it makes pass. This matches the successful structure of the archived [`2026-04-16-make-expected-count-optional`](../archive/2026-04-16-make-expected-count-optional/tasks.md) proposal.

### Synthetic `.slp` round-trip for integration tests

Integration tests go through `sio.save_slp` → `Series.load` rather than constructing `initial_frame_traits` dicts in memory. This exercises the full Series interface (including `get_primary_points`'s user_instances + unused_predictions stacking and NaN placeholder injection). Same idiom as `tests/test_pixel_units.py`. Real plate `.slp` fixtures are deferred to follow-up issue D (MK22 dataset).
