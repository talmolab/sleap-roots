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
- Original SLEAP instance indices preserved in output (`primary_sleap_idx`, `lateral_sleap_idxs`) for traceability back to the `.slp` file; `plant_id` is an additional left-to-right ordering for consumer convenience
- Optional `expected_count` (None or NaN both tolerated) — no error when missing, no silent filtering on mismatch
- `expected_count` and `detected_count` columns in output (no boolean flags — consumers derive mismatch from raw values)
- Per-plant output: flat list of per-plant-per-frame rows
- JSON (self-contained, includes raw points + SLEAP instance indices) + CSV (scalars only) output
- Per-plant scalar traits reuse existing `DicotPipeline` trait names unchanged (no renaming)
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
| E | Generalize cylinder-conventional CSV column names for plates | Three `Series` properties read cylinder-named CSV columns that plates also have to use: `qc_fail` reads `qc_cylinder` ([series.py:196-208](../../../sleap_roots/series.py#L196-L208)), `expected_count` reads `number_of_plants_cylinder` ([series.py:165-180](../../../sleap_roots/series.py#L165-L180)), `group` reads `genotype` (plate-agnostic, OK). PR 1 documents this constraint in the class docstring and a spec scenario but does not change it. Follow-up E adds plate-aware column-name resolution (e.g., a `qc_plate` / `number_of_plants_plate` column, or a constructor kwarg). |
| F | Plate-specific depth trait (`primary_depth`) | Issue #126 defined `primary_root_depth` as max y-extent; PR 1 substitutes `primary_base_tip_dist` (Euclidean) as the closest existing DicotPipeline trait. Follow-up F introduces a dedicated max-y-extent or pure-y-depth scalar after real plate data informs which variant is correct. See § "Deviation from #126 on `primary_root_depth`". |

## Key design decisions

### D1: Cylinder vs plate — difference is aggregation strategy, not frame count

The `MultipleDicotPipeline` aggregates across ~72 rotational frames (each is a noisy view of the same plant state) and produces summary stats per series. The plate pipeline treats each frame as a distinct physical state (e.g., a timepoint in a timelapse) and emits per-plant rows without cross-frame summarization. Because of this, the plate pipeline's frame loop naturally supports future timelapse: more frames → more rows, no architecture changes.

### D2: Skip `filter_plants_with_unexpected_ct` entirely

`MultipleDicotPipeline` uses this function to drop frames with mismatched count. Plates cannot afford to drop data (one frame = one timepoint = potentially the whole series). The plate pipeline's TraitDef DAG does NOT include a count-filter step. The pipeline keeps all detected plants regardless of `expected_count`, and surfaces the discrepancy via `expected_count` / `detected_count` columns in the output. Per the #125 scope split comment, this is the designated home for "keep all plants on mismatch" semantics.

**Zero-laterals handling** (verified against [points.py:588-591](../../../sleap_roots/points.py#L588-L591)): `associate_lateral_to_primary` returns a `(1, n_nodes, 2)` all-NaN placeholder when a primary has zero associated laterals. Passing this placeholder into the nested `DicotPipeline` causes `get_count(lateral_pts)` to return `1` (silently wrong — should be `0`) and propagates NaN through `lateral_lengths` → `network_length` → `network_solidity`, corrupting downstream traits. The plate pipeline MUST detect this placeholder in `compute_plate_traits` (via `assoc["lateral_points"].shape[0] == 1 and not is_line_valid(assoc["lateral_points"][0])`) and pass `np.empty((0, n_nodes, 2))` to the nested `DicotPipeline` instead. Test `test_multiple_dicot_plate_pipeline_zero_laterals` MUST assert `plants[0]["traits"]["lateral_count"] == 0` (not 1) for a plant with no laterals.

### D3: Deterministic plant_id by sorting on primary root base x-coordinate, paired with original SLEAP indices

`plant_id` is 0-indexed, assigned left-to-right by the x-coordinate of the primary root's base node (node 0). Uses stable `np.argsort` so identical x-values fall back to original instance order (tiebreaker is arbitrary but deterministic). **Known limitation**: if SLEAP misses a primary root prediction, all downstream plant_ids shift by one. This fragility is accepted for PR 1. Visualization tooling (follow-up issue C) is the mitigation — users can see which plant maps to which ID on the image.

The output ALSO emits `primary_sleap_idx` (scalar int) and `lateral_sleap_idxs` (list of ints) per plant, which are the **original SLEAP instance indices** into `Series.get_primary_points(frame_idx)` and `Series.get_lateral_points(frame_idx)` — pre-`filter_roots_with_nans`. This means both views of "which plant" are available: (a) `plant_id` for user-facing left-to-right order, (b) `primary_sleap_idx` / `lateral_sleap_idxs` for back-mapping to the source `.slp` file during debugging or visualization overlays. Because `filter_roots_with_nans` at [points.py:338-361](../../../sleap_roots/points.py#L338-L361) collapses indices without preserving a mapping, the pipeline computes a per-frame validity mask BEFORE filtering, then maps post-filter association-dict keys back to the original indices.

### D4: Tertiary→primary direct association (PR 2)

Tertiary roots branch from laterals biologically, but for per-plant trait aggregation the association chain tertiary → lateral → primary is unnecessary indirection. PR 2 will reuse `associate_lateral_to_primary` with tertiary input to directly associate tertiaries to the nearest primary root. **This decision MUST be documented in PR 2's docstrings** — future code readers should see why we're not tracking tertiary → lateral granularity. If per-lateral tertiary counts become a real need, a separate follow-up can add that.

### D5: JSON includes raw points for self-contained artifacts

The per-plant JSON includes `primary_points`, `lateral_points`, `primary_sleap_idx`, `lateral_sleap_idxs`, and the full `traits` dict. This makes the JSON a self-sufficient analysis/visualization artifact independent of the `.slp` file. File size is negligible (~30KB per plate uncompressed). CSV remains scalars-only (arrays don't fit tabular format).

The JSON MUST emit `NaN` values as JSON `null` (not the Python-only bare `NaN` literal that `json.dumps` emits by default). Python's default `json.dumps` with `allow_nan=True` produces strings like `{"x": NaN}` that are **not valid JSON per RFC 8259** and are rejected by strict parsers (JavaScript `JSON.parse`, Go `encoding/json`, Jackson strict mode, `jq`). Because the plate JSON is intended as a cross-language analysis artifact (consumed by the plate viewer follow-up C), the plate pipeline MUST use an encoder that converts float NaN / numpy NaN to JSON `null`. Implementation: extend `NumpyArrayEncoder` locally (or add a plate-specific encoder) to intercept `float('nan')` and `np.float*('nan')` values and emit `None`. Add a test that grep's the written file content for the literal string `"NaN"` and fails if found.

The top-level JSON dict MUST include `"schema_version": 1` and `"units": "pixels"` fields so downstream consumers can gracefully handle future schema growth (PR 2 adds tertiary columns, PR 3 may add filter-config provenance) and know the coordinate/length units without out-of-band documentation.

### D5b: Restore `count_mismatch` / `count_validated` flags in JSON (transferred from #125)

The #125 scope-split comment transferred these acceptance criteria from #125 to #126:
- `count_mismatch: True` on mismatch
- `count_validated: True/False` depending on whether expected_count was resolvable and matched
- Warning log on mismatch

The original design draft's D6 rejected these as "redundant flag columns" — but that argument only applies to CSV where column-count bloat is real. JSON does not have that constraint. PR 1 MUST emit `count_mismatch` (bool) and `count_validated` (bool) fields in each per-plant JSON entry AND log a warning (via `warnings.warn` or `print`) on mismatch. These fields do NOT appear as CSV columns (consumers can derive them from raw `expected_count` / `detected_count`). Semantics:

- `count_validated = expected_count is not None and not math.isnan(expected_count) and int(round(expected_count)) == detected_count`
- `count_mismatch = expected_count is not None and not math.isnan(expected_count) and int(round(expected_count)) != detected_count`
- When `expected_count` is None or NaN, both flags are `False` (not validated, not mismatched — just "unknown"). This is the "skip-filter" semantics from #125.
- On mismatch (any plant row where `count_mismatch=True`), emit one warning per series via `warnings.warn(f"MultipleDicotPlatePipeline: {series.series_name} detected {detected_count} primaries but expected {expected_count}; no frames dropped", UserWarning)`.

### D6: Metadata columns first in CSV, trait columns last; reuse DicotPipeline's CSV trait set unchanged

Follow the existing `MultiplePrimaryRootPipeline` flattened CSV convention: `series, frame, plant_id, primary_sleap_idx, expected_count, detected_count, <traits>`. Consumers derive `count_validated` / `count_mismatch` booleans from the raw `expected_count` / `detected_count` pair. No redundant flag columns.

**Trait columns come from `DicotPipeline.csv_traits` unchanged** — no renaming. This preserves the project-wide naming convention (`primary_length`, `lateral_count`, `lateral_lengths_{min,max,mean,...}`, `network_length`, `primary_base_tip_dist`, angle/base/tip traits, etc.) and avoids introducing a parallel `primary_root_length`-style dialect. `lateral_sleap_idxs` is JSON-only (variable-length list; does not belong in a flat CSV cell).

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
        primary_pts_raw = plant.get_primary_points(frame_idx)
        lateral_pts_raw = plant.get_lateral_points(frame_idx)
        # Preserve original SLEAP instance indices through filter_roots_with_nans
        primary_sleap_idxs = [
            i for i, r in enumerate(primary_pts_raw) if not np.isnan(r).any()
        ]
        lateral_sleap_idxs = [
            i for i, r in enumerate(lateral_pts_raw) if not np.isnan(r).any()
        ]
        return {
            "primary_pts": primary_pts_raw,
            "lateral_pts": lateral_pts_raw,
            "primary_sleap_idxs": primary_sleap_idxs,
            "lateral_sleap_idxs": lateral_sleap_idxs,
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
     │   → {"primary_pts", "lateral_pts", "primary_sleap_idxs",
     │      "lateral_sleap_idxs", "expected_count"}
     ├─ frame_traits = compute_frame_traits(initial)
     │   → adds "primary_pts_no_nans", "lateral_pts_no_nans",
     │     "detected_count", "plant_associations_dict", "plant_id_order"
     └─ for plant_id, primary_idx in enumerate(frame_traits["plant_id_order"]):
         # primary_idx is the POST-FILTER index (into plant_associations_dict)
         assoc = frame_traits["plant_associations_dict"][primary_idx]
         primary_sleap_idx = initial["primary_sleap_idxs"][primary_idx]
         # lateral_sleap_idxs for this plant: map the laterals associated to this
         # primary back through initial["lateral_sleap_idxs"] via is_line_valid
         # matching against lateral_pts_no_nans (see implementation notes)
         dicot_pipeline = DicotPipeline()
         per_plant_traits = dicot_pipeline.compute_frame_traits({
             "primary_pts": assoc["primary_points"][None, ...],  # (1, n_nodes, 2)
             "lateral_pts": assoc["lateral_points"],
         })
         extract full DicotPipeline CSV trait set (scalars + summary of non-scalars)
         from per_plant_traits → plant_row_dict
         append plant_row_dict to output["plants"]
```

Per-plant trait computation uses a nested `DicotPipeline` on each plant's isolated primary + lateral points — same pattern as `MultipleDicotPipeline.compute_multiple_dicots_traits` at [trait_pipelines.py:453](../../../sleap_roots/trait_pipelines.py#L453). The plate pipeline emits the full `DicotPipeline.csv_traits` set per plant, unchanged — no trait renaming.

### SLEAP instance index mapping

`filter_roots_with_nans` collapses indices. `associate_lateral_to_primary` returns a dict keyed by post-filter primary indices. To recover original SLEAP indices:

- **Primary**: `initial["primary_sleap_idxs"][post_filter_primary_idx]` gives the original index.
- **Lateral**: after filtering, each association dict's `"lateral_points"` entry is a `(k, n_nodes, 2)` array of the post-filter laterals associated to that primary. The mapping back to original SLEAP lateral indices is computed by matching each associated lateral against `lateral_pts_no_nans` via `is_line_valid` + coordinate equality (Shapely-distance-based association is deterministic, so post-filter identity is stable). Concretely: for each lateral in `assoc["lateral_points"]`, find its index in `lateral_pts_no_nans` (array-equal), then look up that index in `initial["lateral_sleap_idxs"]`.

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
      "expected_count": 3,
      "primary_sleap_idx": 2,
      "primary_points": [[x, y], ...],
      "lateral_sleap_idxs": [0, 3, 7],
      "lateral_points": [[[x, y], ...], ...],
      "traits": {
        "primary_length": 234.5,
        "primary_base_tip_dist": 198.7,
        "lateral_count": 3,
        "lateral_lengths": [45.2, 56.1, 67.3],
        "network_length": 402.9,
        "primary_angle_proximal": 12.3,
        "primary_angle_distal": 8.7,
        "...": "...full DicotPipeline trait set..."
      }
    }
  ]
}
```

**Important**: the `traits` dict contains the **full DicotPipeline output** (every trait produced by `DicotPipeline.compute_frame_traits` — scalars AND non-scalars). The CSV emits scalars directly and non-scalars via the automatic `{name}_{min,max,mean,median,std,p5,p25,p75,p95}` expansion at [trait_pipelines.py:262-274](../../../sleap_roots/trait_pipelines.py#L262-L274).

### Per-series CSV (one row per plant per frame)

Columns: `series, frame, plant_id, primary_sleap_idx, expected_count, detected_count, <DicotPipeline.csv_traits unchanged>`

`expected_count` is nullable (empty cell if None or NaN); `detected_count` is always populated. `lateral_sleap_idxs` is NOT in CSV (variable-length list; JSON-only).

The DicotPipeline CSV trait set includes (non-exhaustive; authoritative list comes from `DicotPipeline().csv_traits`): `primary_length`, `primary_base_tip_dist`, `primary_angle_proximal`, `primary_angle_distal`, `primary_tip_pt_y`, `lateral_count`, `lateral_lengths_{min,max,mean,median,std,p5,p25,p75,p95}`, `lateral_angles_distal_{...}`, `lateral_angles_proximal_{...}`, `lateral_base_xs_{...}`, `lateral_base_ys_{...}`, `lateral_tip_xs_{...}`, `lateral_tip_ys_{...}`, `network_length`, `network_length_lower`, `network_distribution_ratio`, `network_solidity`, `network_width_depth_ratio`, and others.

### Batch output

- `compute_batch_plate_traits(all_series)` → `pd.DataFrame` concatenating all per-series CSV rows.
- Batch JSON (if requested) → list of per-series dicts at the top level.

### Serialization

- JSON: reuse existing `NumpyArrayEncoder` at [trait_pipelines.py:123](../../../sleap_roots/trait_pipelines.py#L123).
- CSV: `pd.DataFrame.to_csv(path, index=False)`.

## Per-plant traits (PR 1)

All in pixel units. Computed via nested `DicotPipeline` per plant. **No trait renaming** — the plate pipeline emits DicotPipeline trait names unchanged, preserving the project-wide naming convention (`primary_*`, `lateral_*`, `network_*` prefixes; never `_root_` infix).

**What the plate pipeline emits per plant:**

- **JSON** (`plants[i].traits`): the full `DicotPipeline.compute_frame_traits` output — every trait (scalars and non-scalar arrays).
- **CSV** (per-plant row): `DicotPipeline.csv_traits` — scalars emitted as-is, non-scalars expanded to `{name}_{min,max,mean,median,std,p5,p25,p75,p95}` via the existing summary mechanism at [trait_pipelines.py:262-274](../../../sleap_roots/trait_pipelines.py#L262-L274).

**Core traits of interest for plate analysis** (mapping from this PR's conceptual requirements to existing DicotPipeline trait names):

| Conceptual measurement | Existing trait name | Type | Notes |
|------------------------|---------------------|------|-------|
| Primary root length | `primary_length` | scalar | Polyline length of the primary root. Defined at [trait_pipelines.py:1425](../../../sleap_roots/trait_pipelines.py#L1425). |
| Primary base-to-tip distance (proxy for "depth") | `primary_base_tip_dist` | scalar | Euclidean distance from base to tip (NOT pure y-depth, NOT max y-extent). Defined at [trait_pipelines.py:1658](../../../sleap_roots/trait_pipelines.py#L1658). `primary_tip_pt_y` is in `DicotPipeline.csv_traits` ([trait_pipelines.py:1538](../../../sleap_roots/trait_pipelines.py#L1538)); `primary_base_pt_y` is NOT (`include_in_csv=False` at [trait_pipelines.py:1529](../../../sleap_roots/trait_pipelines.py#L1529)), so CSV-only consumers cannot reconstruct `tip_y - base_y`. See design-doc § "Deviation from #126 on `primary_root_depth`" below for the recorded substitution and follow-up issue F. |
| Lateral root count | `lateral_count` | scalar | Defined at [trait_pipelines.py:1264](../../../sleap_roots/trait_pipelines.py#L1264). |
| Individual lateral lengths | `lateral_lengths` | non-scalar (array) | Defined at [trait_pipelines.py:1291](../../../sleap_roots/trait_pipelines.py#L1291). CSV auto-expands to `lateral_lengths_{min,max,mean,median,std,p5,p25,p75,p95}`. |
| Total network length (primary + sum of laterals) | `network_length` | scalar | Defined at [trait_pipelines.py:1520](../../../sleap_roots/trait_pipelines.py#L1520); `fn=get_network_length` with inputs `["primary_length", "lateral_lengths"]` — simple sum, NOT convex-hull math. Reusable directly. |

**Correction from earlier design drafts**: An earlier version of this document claimed `network_length` uses convex-hull math and should not be reused. That was incorrect — `get_network_length` computes `primary_length + sum(lateral_lengths)`. The plate pipeline reuses it unchanged.

**Derived statistics** (e.g., "avg lateral length") are produced by the existing CSV stat-suffix mechanism (`lateral_lengths_mean`, `lateral_lengths_max`). No new trait-name synonyms are introduced.

### Deviation from #126 on `primary_root_depth` (recorded substitution)

Issue #126's original acceptance criteria define `primary_root_depth` as "max y-extent (deepest node y − base node y)". No existing `DicotPipeline` trait computes that quantity exactly. The closest three candidates are:

1. `primary_base_tip_dist` — Euclidean distance between node 0 (base) and node N−1 (tip). Correct magnitude for straight-down roots; over-/under-counts for curved roots.
2. `primary_tip_pt_y − primary_base_pt_y` — signed y-distance from base to tip. Correct only when tip is the deepest node (usually true but not guaranteed).
3. True max-y-extent `max(primary_pts[:, 1]) − primary_pts[0, 1]` — matches #126 semantically but requires a NEW trait.

PR 1 adopts substitution (1): emit `primary_base_tip_dist` as the depth-proxy and DO NOT introduce a new trait. Rationale: (a) `primary_base_tip_dist` is an existing tested DicotPipeline trait, (b) for plate primary roots that grow roughly straight down, it matches max-y-extent to within rounding error, (c) introducing a new trait in PR 1 expands scope beyond "port DicotPipeline to plates". The pure-y and max-y alternatives are tracked as follow-up issue F so the maintainer can decide between them based on real plate data.

**Action**: a comment MUST be posted on #126 documenting this substitution and linking follow-up issue F before PR 1 merges (see tasks.md § 5.7b).

## Test plan

### Unit tests (`tests/test_points.py`)

- `test_argsort_primaries_by_base_x_basic` — three primaries at x=[100, 50, 200] → [1, 0, 2]
- `test_argsort_primaries_by_base_x_single_plant` — one primary → [0]
- `test_argsort_primaries_by_base_x_empty` — empty dict → []
- `test_argsort_primaries_by_base_x_identical_x` — identical x-values → stable tiebreak (original order)

### Unit tests (`tests/test_trait_pipelines.py`)

- `test_multiple_dicot_plate_pipeline_define_traits` — TraitDef names, inputs, and DAG are correct (including `primary_pts_no_nans`, `lateral_pts_no_nans`, `detected_count`, `plant_associations_dict`, `plant_id_order`)
- `test_multiple_dicot_plate_pipeline_get_initial_frame_traits` — returns `{primary_pts, lateral_pts, primary_sleap_idxs, lateral_sleap_idxs, expected_count}`

### Integration tests (`tests/test_trait_pipelines.py`)

Integration tests round-trip through **synthetic `.slp` files** written via `sio.save_slp` and loaded via `Series.load` — same idiom used in `tests/test_pixel_units.py`. This exercises the full `compute_plate_traits(series)` frame loop including the Series interface. Real plate `.slp` fixtures (MK22 dataset) deferred to follow-up issue D.

- `test_multiple_dicot_plate_pipeline_basic` — 3 synthetic plants with known geometry, verify `plant_id` order is left-to-right, `primary_sleap_idx` values match the original SLEAP instance indices pre-filter, full DicotPipeline CSV trait set present on each plant row, `detected_count == 3` on every plant row
- `test_multiple_dicot_plate_pipeline_sleap_idx_traceability` — inject a NaN primary at SLEAP index 1 among 3 predictions so post-filter indices {0, 1} must map back to original {0, 2}; verify `primary_sleap_idx` in output is `[0, 2]` (not `[0, 1]`) in left-to-right sorted order
- `test_multiple_dicot_plate_pipeline_expected_count_none` — no CSV attached to Series, `expected_count` is `NaN` (or `None`) in output, pipeline runs without error
- `test_multiple_dicot_plate_pipeline_expected_count_mismatch` — 3 detected primaries, `expected_count=2`, verify all 3 plants still in output AND `expected_count=2, detected_count=3` (no filtering)
- `test_multiple_dicot_plate_pipeline_empty_frame` — all-NaN primary predictions → empty `plants` list, no crash
- `test_multiple_dicot_plate_pipeline_timelapse_shape` — 2-frame synthetic `.slp`, verify `len(result["plants"]) == N_plants × 2`, each plant row carries correct `frame` value
- `test_multiple_dicot_plate_pipeline_csv_output` — CSV round-trip, column order starts with `series, frame, plant_id, primary_sleap_idx, expected_count, detected_count`, then full DicotPipeline csv_traits
- `test_multiple_dicot_plate_pipeline_json_output` — JSON round-trip, `primary_points` as nested list, `lateral_sleap_idxs` as list of ints, top-level `series`/`group`/`qc_fail` fields present
- `test_compute_batch_plate_traits` — two synthetic `.slp` series, batch DataFrame concatenation + batch JSON list structure

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
| Tests: `primary_sleap_idx` / `lateral_sleap_idxs` traceability through filter collapse | in PR 1 |
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
