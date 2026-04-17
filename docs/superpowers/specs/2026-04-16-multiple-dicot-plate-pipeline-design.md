# Design: `MultipleDicotPlatePipeline`

**Date**: 2026-04-16
**Issue**: https://github.com/talmolab/sleap-roots/issues/126
**Depends on**: #125 (`filter_plants_with_unexpected_ct` accepts `Optional[float]`) — landed in PR #155
**Status**: Brainstorm complete, spec drafted

## Summary

Create a new pipeline class `MultipleDicotPlatePipeline` for plate images with multiple dicot plants per frame. Distinct from the existing `MultipleDicotPipeline` (cylinder) because plates have **one frame per timepoint** (not ~72 rotational views) and require **per-plant scalar output** (not cross-frame summary stats). Output is a flat list of per-plant-per-frame rows, usable for a single plate today and a timelapse series tomorrow without code changes.

## Scope

### In scope (this PR — "PR 1")

- New class `MultipleDicotPlatePipeline` in `sleap_roots/trait_pipelines.py`
- Primary + lateral root handling only (tertiary deferred to PR 2)
- Deterministic `plant_id` assignment via left-to-right sort on primary root base x-coordinate (node index 0)
- Optional `expected_count` (None or NaN both tolerated) — no error when missing, no silent filtering on mismatch
- `expected_count` and `detected_count` columns in output (no boolean flags — consumers derive mismatch from raw values)
- Per-plant output: flat list of per-plant-per-frame rows
- JSON (self-contained, includes raw points + SLEAP instance indices) + CSV (scalars only) output
- Frame loop that supports future timelapse (>1 frame per series) without architecture changes
- Synthetic-data tests for all new code paths

### Explicitly deferred (separate follow-up issues — MUST be filed before PR merges)

| # | Issue | Scope |
|---|-------|-------|
| PR 2 | Tertiary root support | `Series.get_tertiary_points`, `tertiary_path` attribute, reuse `associate_lateral_to_primary` with tertiary input (tertiary→primary direct; lateral-level granularity deferred), extend pipeline to emit tertiary columns |
| PR 3 | Configurable filtering thresholds | `min_primary_length_px`, `min_lateral_length_px`, `node_score_threshold`, `primary_angle_filter` as constructor kwargs with plate-specific defaults |
| A | Standardize multi-plant pipeline JSON output + deterministic plant_id | Apply plate JSON format (raw points, association indices, sorted plant_id) to `MultipleDicotPipeline` and `MultiplePrimaryRootPipeline` |
| B | Include raw points in single-plant pipeline JSON | Apply to `DicotPipeline`, `YoungerMonocotPipeline`, `OlderMonocotPipeline` for self-contained analysis artifacts |
| C | Plate visualization / viewer | Consumes plate pipeline JSON to render colored per-plant overlays; related to #128 |
| D | Real plate `.slp` fixture tests | MK22 dataset once available |

## Key design decisions

### D1: Cylinder vs plate — difference is aggregation strategy, not frame count

The `MultipleDicotPipeline` aggregates across ~72 rotational frames (each is a noisy view of the same plant state) and produces summary stats per series. The plate pipeline treats each frame as a distinct physical state (e.g., a timepoint in a timelapse) and emits per-plant rows without cross-frame summarization. Because of this, the plate pipeline's frame loop naturally supports future timelapse: more frames → more rows, no architecture changes.

### D2: Skip `filter_plants_with_unexpected_ct` entirely

`MultipleDicotPipeline` uses this function to drop frames with mismatched count. Plates cannot afford to drop data (one frame = one timepoint = potentially the whole series). The plate pipeline's TraitDef DAG does NOT include a count-filter step. The pipeline keeps all detected plants regardless of `expected_count`, and surfaces the discrepancy via `expected_count` / `detected_count` columns in the output. Per the #125 scope split comment, this is the designated home for "keep all plants on mismatch" semantics.

### D3: Deterministic plant_id by sorting on primary root base x-coordinate

`plant_id` is 0-indexed, assigned left-to-right by the x-coordinate of the primary root's base node (node 0). Uses stable `np.argsort` so identical x-values fall back to original instance order (tiebreaker is arbitrary but deterministic). **Known limitation**: if SLEAP misses a primary root prediction, all downstream plant_ids shift by one. This fragility is accepted for PR 1. Visualization tooling (follow-up issue C) is the mitigation — users can see which plant maps to which ID on the image.

### D4: Tertiary→primary direct association (PR 2)

Tertiary roots branch from laterals biologically, but for per-plant trait aggregation the association chain tertiary → lateral → primary is unnecessary indirection. PR 2 will reuse `associate_lateral_to_primary` with tertiary input to directly associate tertiaries to the nearest primary root. **This decision MUST be documented in PR 2's docstrings** — future code readers should see why we're not tracking tertiary → lateral granularity. If per-lateral tertiary counts become a real need, a separate follow-up can add that.

### D5: JSON includes raw points for self-contained artifacts

The per-plant JSON includes `primary_points`, `lateral_points`, `primary_instance_idx`, `lateral_instance_idxs`, and the full `traits` dict. This makes the JSON a self-sufficient analysis/visualization artifact independent of the `.slp` file. File size is negligible (~30KB per plate uncompressed). CSV remains scalars-only (arrays don't fit tabular format).

### D6: Metadata columns first in CSV, trait columns last

Follow the existing `MultiplePrimaryRootPipeline` flattened CSV convention: `series, frame, plant_id, expected_count, detected_count, <traits>`. Consumers derive `count_validated` / `count_mismatch` booleans from the raw `expected_count` / `detected_count` pair. No redundant flag columns.

### D7: No `Pipeline` base class changes, no refactor of other pipelines

Approach A from brainstorming: mirror `MultipleDicotPipeline`'s compute method structure in a new `compute_plate_traits(series)` method on the new class. Accept some code duplication (frame loop + per-plant inner loop) in exchange for a clean parallel between cylinder and plate pipelines. Shared helper refactor is out of scope for PR 1 — can be done later if duplication pain accumulates across 3+ pipelines.

## Architecture

### Class structure

```python
@attrs.define
class MultipleDicotPlatePipeline(Pipeline):
    """Pipeline for multi-plant dicot plate images."""

    def define_traits(self) -> List[TraitDef]:
        return [
            TraitDef(name="primary_pts_no_nans", fn=filter_roots_with_nans,
                     input_traits=["primary_pts"], scalar=False, include_in_csv=False),
            TraitDef(name="lateral_pts_no_nans", fn=filter_roots_with_nans,
                     input_traits=["lateral_pts"], scalar=False, include_in_csv=False),
            TraitDef(name="detected_count", fn=get_count,
                     input_traits=["primary_pts_no_nans"], scalar=True, include_in_csv=True),
            TraitDef(name="plant_associations_dict", fn=associate_lateral_to_primary,
                     input_traits=["primary_pts_no_nans", "lateral_pts_no_nans"],
                     scalar=False, include_in_csv=False),
            TraitDef(name="plant_id_order", fn=argsort_primaries_by_base_x,
                     input_traits=["plant_associations_dict"],
                     scalar=False, include_in_csv=False),
        ]

    def get_initial_frame_traits(self, plant, frame_idx):
        return {
            "primary_pts": plant.get_primary_points(frame_idx),
            "lateral_pts": plant.get_lateral_points(frame_idx),
            "expected_count": plant.expected_count,
        }

    def compute_plate_traits(self, series, write_csv=False, write_json=False,
                             output_dir=".", ...) -> dict:
        """Run the plate pipeline on a Series (one or more frames).

        Returns a per-series dict with a flat `plants` list.
        """
        ...

    def compute_batch_plate_traits(self, all_series, ...) -> pd.DataFrame:
        """Run the plate pipeline across multiple Series, concatenate per-plant rows."""
        ...
```

### New helper function in `sleap_roots/points.py`

```python
def argsort_primaries_by_base_x(plant_associations_dict: dict) -> List[int]:
    """Return primary instance indices sorted left-to-right by base node x-coordinate.

    Stable sort: identical x-values keep original instance order.

    Args:
        plant_associations_dict: Output of associate_lateral_to_primary, keyed by
            primary instance index.

    Returns:
        List of primary instance indices in left-to-right order. Length matches
        len(plant_associations_dict). Empty list if input is empty.
    """
```

### Compute flow (per series)

```
series (N frames)
 └─ for frame_idx in range(N):
     ├─ initial = get_initial_frame_traits(series, frame_idx)
     │   → {"primary_pts", "lateral_pts", "expected_count"}
     ├─ frame_traits = compute_frame_traits(initial)
     │   → adds "primary_pts_no_nans", "lateral_pts_no_nans",
     │     "detected_count", "plant_associations_dict", "plant_id_order"
     └─ for plant_id, primary_idx in enumerate(frame_traits["plant_id_order"]):
         assoc = frame_traits["plant_associations_dict"][primary_idx]
         dicot_pipeline = DicotPipeline()
         per_plant_traits = dicot_pipeline.compute_frame_traits({
             "primary_pts": assoc["primary_points"][None, ...],  # (1, n_nodes, 2)
             "lateral_pts": assoc["lateral_points"],
         })
         extract scalar traits from per_plant_traits → plant_row_dict
         append plant_row_dict to output["plants"]
```

Per-plant scalar traits computed via nested `DicotPipeline` on each plant's isolated primary + lateral points — same pattern as `MultipleDicotPipeline.compute_multiple_dicots_traits` at [trait_pipelines.py:453](../../../sleap_roots/trait_pipelines.py#L453).

## Output structures

### Per-series dict (returned by `compute_plate_traits`)

```json
{
  "series": "MK22_plate1",
  "group": "MK22",
  "qc_fail": 0,
  "expected_count": 3,
  "plants": [
    {
      "frame": 0,
      "plant_id": 0,
      "detected_count": 3,
      "primary_instance_idx": 2,
      "primary_points": [[x, y], ...],
      "lateral_instance_idxs": [0, 3, 7],
      "lateral_points": [[[x, y], ...], ...],
      "traits": {
        "primary_root_length": 234.5,
        "primary_root_depth": 198.7,
        "lateral_root_count": 3,
        "avg_lateral_root_length": 56.2,
        "max_lateral_root_length": 89.4,
        "total_lateral_root_length": 168.6,
        "total_root_network_length": 403.1
      }
    }
  ]
}
```

### Per-series CSV (one row per plant per frame)

Columns: `series, frame, plant_id, expected_count, detected_count, primary_root_length, primary_root_depth, lateral_root_count, avg_lateral_root_length, max_lateral_root_length, total_lateral_root_length, total_root_network_length`

`expected_count` is nullable (empty cell if None or NaN); `detected_count` is always populated.

### Batch output

- `compute_batch_plate_traits(all_series)` → `pd.DataFrame` concatenating all per-series CSV rows.
- Batch JSON (if requested) → list of per-series dicts at the top level.

### Serialization

- JSON: reuse existing `NumpyArrayEncoder` at [trait_pipelines.py:123](../../../sleap_roots/trait_pipelines.py#L123).
- CSV: `pd.DataFrame.to_csv(path, index=False)`.

## Per-plant scalar traits (PR 1)

All in pixel units. Computed via nested `DicotPipeline` per plant. Output column names differ from DicotPipeline trait names for clarity in the plate context; the plate pipeline renames them on emit.

| Output column | Type | Source from DicotPipeline | Computation |
|---------------|------|----------------------------|-------------|
| `primary_root_length` | float | `primary_length` | Polyline length of the primary root (uses `get_root_lengths` on primary points only, NOT the `network_length` trait which includes laterals) |
| `primary_root_depth` | float | computed in plate pipeline | `primary_tip_pt_y - primary_base_pt_y` (depth in pixels; valid in y-down image coords) |
| `lateral_root_count` | int | `lateral_count` | Number of lateral roots associated to this plant |
| `avg_lateral_root_length` | float | derived from `lateral_lengths` array | `np.nanmean(lateral_lengths)` — returns NaN if no laterals |
| `max_lateral_root_length` | float | derived from `lateral_lengths` array | `np.nanmax(lateral_lengths)` — returns NaN if no laterals |
| `total_lateral_root_length` | float | derived from `lateral_lengths` array | `np.nansum(lateral_lengths)` — returns 0 if no laterals |
| `total_root_network_length` | float | computed in plate pipeline | `primary_root_length + total_lateral_root_length` (tertiary added in PR 2) |

**Implementation note**: The plate pipeline does NOT reuse DicotPipeline's `network_length` trait because that trait computes the total network length across all roots (primary + lateral) treated as a single network via convex-hull math, whereas the plate output wants primary and lateral lengths separated. `primary_length` is the correct DicotPipeline trait for the primary-only measurement.

## Test plan

### Unit tests (`tests/test_points.py`)

- `test_argsort_primaries_by_base_x_basic` — three primaries at x=[100, 50, 200] → [1, 0, 2]
- `test_argsort_primaries_by_base_x_single_plant` — one primary → [0]
- `test_argsort_primaries_by_base_x_empty` — empty dict → []
- `test_argsort_primaries_by_base_x_identical_x` — identical x-values → stable tiebreak (original order)

### Unit tests (`tests/test_trait_pipelines.py`)

- `test_multiple_dicot_plate_pipeline_define_traits` — TraitDef names, inputs, and DAG are correct
- `test_multiple_dicot_plate_pipeline_get_initial_frame_traits` — returns `{primary_pts, lateral_pts, expected_count}`

### Integration tests (`tests/test_trait_pipelines.py`)

All use synthetic numpy arrays for precise control (real `.slp` fixtures deferred to issue D):

- `test_multiple_dicot_plate_pipeline_basic` — 3 synthetic plants with known geometry, verify plant_id order, instance index round-trip, 7 scalar traits present, detected_count == 3 on every plant row
- `test_multiple_dicot_plate_pipeline_expected_count_none` — no CSV, `expected_count is None` in output, pipeline runs without error
- `test_multiple_dicot_plate_pipeline_expected_count_mismatch` — 3 detected primaries, expected_count=2 forced, verify all 3 plants still in output AND `expected_count=2, detected_count=3`
- `test_multiple_dicot_plate_pipeline_empty_frame` — all-NaN primary predictions → empty `plants` list, no crash
- `test_multiple_dicot_plate_pipeline_timelapse_shape` — 2-frame synthetic series, verify N_plants × 2 rows
- `test_multiple_dicot_plate_pipeline_csv_output` — CSV round-trip, column order verified
- `test_multiple_dicot_plate_pipeline_json_output` — JSON round-trip, `primary_points` as nested list
- `test_compute_batch_plate_traits` — two synthetic series, batch DataFrame concatenation + batch JSON list structure

### Acceptance criteria (mapped from issue #126)

| Criterion | Status |
|---|---|
| New class `MultipleDicotPlatePipeline` in trait_pipelines.py | in PR 1 |
| Extends `Pipeline` via `@attrs.define` | in PR 1 |
| `define_traits()` returns TraitDef list | in PR 1 (tertiary deferred to PR 2) |
| `get_initial_frame_traits()` returns primary + lateral (+ tertiary) | primary/lateral in PR 1, tertiary in PR 2 |
| New `associate_tertiary_to_lateral()` function | superseded by D4 — reuse `associate_lateral_to_primary` with tertiary input in PR 2 |
| Filtering parameters configurable via constructor kwargs | PR 3 |
| `expected_count` optional, no error when missing | in PR 1 |
| `count_mismatch` flag in output | superseded by D6 — use raw `expected_count` / `detected_count` columns instead |
| Per-plant output: one dict per plant with scalar trait values in pixels | in PR 1 |
| All traits in pixel units (no DPI conversion) | in PR 1 |
| Tests: synthetic multi-plant point arrays with known geometry | in PR 1 |
| Tests: expected_count present vs missing | in PR 1 |
| Tests: expected_count mismatch (detected != expected) | in PR 1 |
| Tests: plant indexing deterministic (sorted by primary base x) | in PR 1 |
| Tests: tertiary-to-lateral association | PR 2 |
| Tests: filtering thresholds | PR 3 |

## Implementation sequencing

1. **Write failing tests first** (TDD): unit tests for `argsort_primaries_by_base_x`, integration tests for the pipeline class
2. **Implement `argsort_primaries_by_base_x`** in `sleap_roots/points.py`
3. **Implement `MultipleDicotPlatePipeline`** class in `sleap_roots/trait_pipelines.py`
4. **Implement `compute_plate_traits`** and `compute_batch_plate_traits` methods
5. **Run full test suite** to verify no regressions in existing pipelines
6. **File follow-up issues** (PR 2, PR 3, A, B, C, D) before the PR is marked ready for review
7. **Run pre-merge checks** (black, pydocstyle, openspec validate --strict, full pytest suite)
8. **Pre-PR self-review** via `/review-pr` on the branch
9. **Create PR** with follow-up issue links in the body

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `plant_id` fragility if SLEAP misses a primary prediction | Accepted for PR 1; visualization (issue C) is the follow-up mitigation. Document the limitation in the class docstring. |
| Code duplication between cylinder and plate compute methods | Accepted; both compute methods share only the frame loop skeleton, aggregation strategies differ enough that a shared helper would paper over real differences. Refactor later if duplication pain grows. |
| Scope creep into tertiary / filtering config during implementation | Strict PR boundaries enforced by the 3-PR decomposition. Any tertiary or filtering work that sneaks in must be moved back to PR 2 / PR 3. |
| Breaking changes to `MultipleDicotPipeline` or other existing pipelines | None proposed — PR 1 only adds a new class and a new helper function. `Pipeline` base class is untouched. |
| Real plate `.slp` fixtures not yet available | Tests use synthetic arrays; real-data regression tests deferred to issue D (MK22 dataset). |

## Open questions

None blocking. All design decisions are resolved through the brainstorming dialogue (D1-D7 above).
