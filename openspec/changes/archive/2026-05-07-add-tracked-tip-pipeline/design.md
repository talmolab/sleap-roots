# Design: `add-tracked-tip-pipeline`

**Architectural source of truth**: [`docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md`](../../../docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md) § Workstream 2.

That document captures the brainstorm decisions: substrate-only scope (no velocity/curvature in v1, no circumnutation traits ever), pure DAG-A composition reusing existing trait functions, single-frame edge-case asymmetry trade-off, real-fixture choice and synthesis, and the four follow-up issues (#186 / #187 / #188 / #189) the substrate work-stream surfaced.

## OpenSpec-specific design notes

### Single new capability, no MODIFIED deltas

This change introduces ONE new capability `tracked-tip-pipeline` with all ADDED requirements. No existing capabilities are modified.

Reasoning:
- The new `Series.get_tracked_tips` method and the two `validate_*` helpers are additive to `Series` — no observable contract change to `series-metadata` or any existing pipeline.
- Bundling the Series-side accessor + validators alongside the pipeline class in ONE capability keeps the substrate's contract internally coherent. Splitting "Series accessor" from "pipeline that consumes the accessor" would scatter the spec across capabilities and obscure the substrate-as-a-whole.
- `series-metadata` (PR #171) is scoped to CSV-metadata-lookup concerns — `get_tracked_tips` is a track-aware data-extraction method that doesn't fit that scope.

### Why a NEW pipeline file (`tracked_tip_pipeline.py`)

Existing `trait_pipelines.py` is 3763 lines with 8 pipeline classes. Adding a 9th would push it further. The user explicitly chose to start the per-pipeline-file pattern with this PR; the megafile split for the existing 8 is tracked in **#189**.

Backward compatibility: top-level `from sleap_roots import TrackedTipPipeline` works via `__init__.py` re-export. Code reviewers and grep-for-pipelines users find it where they'd expect (a file named after the pipeline).

### Why DAG-A composition with existing trait functions

The user-visible decision: **no new trait module** (`tip_kinematics.py` was originally proposed and dropped). Existing functions plug in directly via TraitDef DAG composition:

- `tip_displacement_net` ← `bases.get_base_tip_dist` (with DAG-provided slicing of `track_first_xy` / `track_last_xy`)
- `tip_trajectory_length` ← `lengths.get_root_lengths` (consumes `track_xy` directly)
- `tracking_coverage` ← inline lambda

Reasoning:
- DRY by maximum reuse — the existing functions have full unit-test coverage and known NaN semantics. The new pipeline's tests cover **composition** (does the DAG correctly route `track_first_xy` and `track_last_xy` into `get_base_tip_dist`), not the underlying numpy.
- No wrapper-module overhead. The pipeline file itself holds the small lambdas inline in TraitDef definitions.
- Trade-off accepted: single-frame `tip_trajectory_length=NaN` (codebase convention from `get_root_lengths`'s NaN-on-empty-segments guard) instead of the `0.0` originally proposed in brainstorm. Asymmetry with `tip_displacement_net=0.0` (which is naturally 0.0 from `get_base_tip_dist` when `xy[0]==xy[-1]`) is documented; users `fillna(0.0)` or filter on `n_frames_tracked > 1`.

### Why per-track DAG iteration (not per-frame)

Existing pipelines (DicotPipeline, plate pipeline, etc.) iterate per-frame: the DAG runs once per frame on `(instances, nodes, 2)` arrays. TrackedTipPipeline operates on a fundamentally different axis — per-track time series (Nx2 array of one tip's positions across frames).

The pipeline's `compute_tracked_tip_traits`:
1. Calls `series.get_tracked_tips()` to get long-format DataFrame (sorted by `(track_id, frame)`).
2. Iterates `for track_id, group in df.groupby('track_id'):` and runs the DAG once per track group, with `track_xy = group[['tip_x','tip_y']].values`.
3. Aggregates per-track summary rows.

The DAG topology IS the same idea (TraitDef nodes with dependencies, networkx-driven topo sort), just with a different iteration unit. No changes to the `Pipeline` base class needed — `compute_tracked_tip_traits` is the entry point and it controls iteration.

### Why `Series.get_tracked_tips` returns frame-sorted rows

`tip_displacement_net = ||xy[-1] - xy[0]||` is correct only if `xy[0]` is the first-tracked-frame tip and `xy[-1]` is the last. Tracker output is NOT in frame-sorted order naturally — verified during brainstorm: frame 0 instances are `[track_0, 1, 2, 3, 4, 5]` but frame 1 is `[track_0, 3, 4, 2, 1, 5]`. Sorting at the accessor (`df.sort_values(['track_id', 'frame']).reset_index(drop=True)`) is the cheapest place to enforce the invariant — every downstream consumer gets the right ordering for free.

The spec codifies this as a normative requirement with a regression test that constructs a synthetic .slp where instance positional order is randomized per frame, then asserts the accessor's output is frame-sorted.

### Why iterate per-instance (not per-frame stack) in `get_tracked_tips`

The existing `get_primary_points` does `np.stack([inst.numpy() for inst in lf.instances])` — collapses to a `(n_inst, n_nodes, 2)` array per frame, losing track identity since you can't pair `inst.track.name` with array rows after stacking.

`get_tracked_tips` instead:
```python
for frame_idx in range(len(labels)):
    for inst in labels[frame_idx].instances:
        if inst.track is None: raise ValueError(...)
        rows.append({
            "track_id": inst.track.name,
            "frame": frame_idx,
            "tip_x": inst.numpy()[-1, 0],
            "tip_y": inst.numpy()[-1, 1],
        })
return pd.DataFrame(rows).sort_values(["track_id", "frame"]).reset_index(drop=True)
```

Slightly slower than vectorized stack for large frame counts, but correct. At realistic data scale (~2,000 rows for a 311-frame plate × 6 tracks) the difference is negligible (~10ms).

### Real fixture provenance and minimum required content

The fixture (`tests/data/circumnutation_plate/`) ships these files:
- `plate_001_greyscale.tracked.slp` (184 KB, Git LFS) — copied verbatim from the source.
- `fixture_metadata.csv` — synthesized by hand: ONE row, `plant_qr_code="plate_001"`, `genotype="KitaakeX"`, `treatment="MOCK"`, `number_of_plants_cylinder=6`, `timepoint=0`. Uses the existing repo CSV convention so `Series.get_metadata` lookups work without modification.
- `README.md` — provenance + the legacy-name caveat.

Why no per-frame CSV: per-frame metadata is deferred to #186. This PR's substrate uses integer `frame` only.

Why one plate (not all 4): one plate exercises the full pipeline path. Multi-plate batch testing happens with synthetic `.slp` files in unit tests.

### Test strategy: synthetic-first, real-fixture for integration

Synthetic tests (primary surface):
- `Series.get_tracked_tips` accessor — auto-detect, frame-sorted output, untracked-instance error, single-node + multi-node skeleton paths.
- TrackedTipPipeline DAG composition — exact-value assertions on known geometric shapes (straight line, right-angle, square).
- CSV / JSON emission — empty-input, single-frame, partial-tracking, batch concat, `emit_trajectories=False` skip path.

Real-fixture tests (integration surface — two paths):
1. WITH `csv_path=fixture_metadata.csv`: assert `sample_uid="plate_001"`, `timepoint=0.0`, all output columns present, `tracking_coverage ∈ [0, 1]`, summary row count = 6, trajectory row count = 1866 (311 frames × 6 tracks).
2. WITHOUT `csv_path`: same pipeline call, assert `sample_uid` defaults to `series_name`, `timepoint` is NaN, all other column values identical to (1).

No exact trait-value assertions on real-fixture tests — those belong in synthetic tests where the geometry is controlled.
