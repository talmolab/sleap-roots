# Design: `add-sample-uid-timepoint-metadata`

**Architectural source of truth**: [`docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md`](../../../docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md) § Workstream 1 (PR #171).

That document describes the rationale for the `sample_uid` rename (the role previously implicit in `plant_qr_code`), the `timepoint` first-class concept, and the `get_metadata(column, plant_id=None)` accessor shape with `plant_id` fallback semantics.

## OpenSpec-specific design notes

### New capability + MODIFIED delta on plate-pipeline spec

**Two spec files ship in this change:**

1. `specs/series-metadata/spec.md` — new capability with ADDED requirements for the Series-layer plumbing (`sample_uid`, `get_metadata`, `timepoint`, refactored existing properties, CSV-builder helpers). The per-pipeline emission contract is explicitly NOT covered here — a dedicated "pipeline emission is per-pipeline" requirement directs readers to the appropriate pipeline capability's spec.
2. `specs/multiple-dicot-plate-pipeline/spec.md` — MODIFIED delta on THREE existing requirements in the plate-pipeline capability (per-series dict shape, CSV column order, JSON output with RFC-8259 + schema_version) plus one ADDED requirement (schema_version bump policy). The MODIFIED pattern is required because:
   - The existing `multiple-dicot-plate-pipeline/spec.md` already specifies the exact top-level dict keys and CSV column positions. Adding `sample_uid` / `timepoint` changes those pinned contracts.
   - Per `openspec/AGENTS.md`, MODIFIED requirements MUST paste the full updated text — which this change does.
   - Without the MODIFIED delta, post-archive drift leaves the plate-pipeline spec stale (seven top-level keys instead of nine; six CSV columns instead of eight).

Rationale for the split: Series-layer plumbing is genuinely orthogonal (a cross-cutting capability any pipeline can consume), but the observable contract change on the plate pipeline IS a modification of that pipeline's existing observable contract. Splitting keeps each capability's spec internally coherent.

### `schema_version` bump to 2

The plate JSON's `schema_version` bumps from 1 to 2 because this change inserts two CSV columns at positions 1 and 2, shifting every other column right. Positional shift is NOT an additive change. The bump policy is codified as an ADDED requirement on the plate-pipeline capability: any non-additive change (column shift, rename, removal) bumps `schema_version`; purely additive changes (new scalar trait in `DicotPipeline.csv_traits`) do not.

### Backward compatibility

- `Series.load` kwarg defaults preserve existing behavior.
- Existing properties (`expected_count`, `group`, `qc_fail`) keep their observable contracts; `get_metadata()` is a refactor, not a rewrite.
- CSV lookup key stays `plant_qr_code`. Follow-up #163 introduces `sample_uid`/`qc`/`expected_count` column names with `plant_qr_code`/`qc_cylinder`/`number_of_plants_cylinder` fallback.

### Plate pipeline output: column positioning

New CSV columns `sample_uid` and `timepoint` go at positions 1 and 2 (right after `series`). Rationale: both are identity-metadata columns belonging with `series`; placing them there keeps the existing metadata block grouped and pushes only the downstream trait columns by 2 positions. Not at position 0 because `series` is already in position 0 and renaming it would create churn.

### Why re-export from `sleap_roots.__init__`

`build_metadata_csv` and `infer_timepoints_from_filenames` are user-facing utilities (users run them before calling any pipeline). Re-exporting matches the existing pattern for `Series`, `DicotPipeline`, etc.

### `infer_timepoints_from_filenames` return-dict key

The return dict keys on the matched portion of the stem (`m.group(0)`), not the `series_name` named group alone. Rationale: for a stem like `plant_1_0` with a pattern `(?P<series_name>.+?)_(?P<timepoint>\d+)`, the `series_name` group is `plant` (lazy match) and the full match is `plant_1_0`. Keying on the full match ensures the dict values align with what the user can later match against `Series.series_name`.
