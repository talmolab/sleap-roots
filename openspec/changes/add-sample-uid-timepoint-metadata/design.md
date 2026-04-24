# Design: `add-sample-uid-timepoint-metadata`

**Architectural source of truth**: [`docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md`](../../../docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md) § Workstream 1 (PR #171).

That document describes the rationale for the `sample_uid` rename (the role previously implicit in `plant_qr_code`), the `timepoint` first-class concept, and the `get_metadata(column, plant_id=None)` accessor shape with `plant_id` fallback semantics.

## OpenSpec-specific design notes

### New capability, not modification

`series-metadata` is introduced as a NEW capability rather than MODIFIED on an existing one. Rationale: the existing `multiple-dicot-pipeline` spec is scoped to cylinder-scan semantics; `multiple-dicot-plate-pipeline` is scoped to plate per-plant-per-frame output. Neither covers the Series-level metadata lookup contract, which is shared across all pipelines. A dedicated `series-metadata` capability gives the new concepts a clean home and avoids polluting pipeline-specific specs with cross-cutting plumbing.

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
