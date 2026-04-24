# Design: Timelapse diffs, tracked-tip circumnutation, and metadata generalization

**Date**: 2026-04-23
**Related issues**: #112 (close as obsolete), #129 (rewrite), #163 (broaden), #159 (related)
**New issues to file**: TimeDiffPipeline, sample_uid/timepoint metadata layer, circumnutation traits
**Depends on**: PR #165 (MultipleDicotPlatePipeline) — merged 2026-04-21
**Status**: Brainstorm complete, spec drafted

## Summary

Three coordinated workstreams that replace the obsolete `PrimaryRootTimelapsePipeline` idea (#112) and supersede the original `TrackedTipPipeline` scope (#129) with:

1. A generalized metadata layer on `Series` that reads arbitrary CSV columns — enables `sample_uid` (cross-scan stable identity) and `timepoint` (time-axis value) as first-class concepts.
2. A rewritten `TrackedTipPipeline` that consumes tracked `.slp` predictions and emits tip-kinematics AND circumnutation traits in one pipeline (not two).
3. A new `TimeDiffPipeline` wrapper class that takes any inner pipeline and emits the inner output plus a parallel between-timepoint delta table. Works uniformly for plate intra-series timelapses (frame-based time axis), plate inter-series scans (CSV-timepoint-based), and cylinder inter-series scans (per-series aggregate diffs).

Consolidates the "timelapse" concern into existing per-frame pipelines + one post-processing wrapper. Closes #112 as "obsolete — superseded by the plate pipeline's frame loop + `TimeDiffPipeline`".

## Scope decomposition — three workstreams, one design

### Workstream 1 — Metadata layer (`Series` generalization)

Small extension of the existing metadata pattern (`Series.expected_count`, `.group`, `.qc_fail`). Unblocks the other two workstreams and most of follow-up #163.

**New on `Series.load`:**

- `sample_uid: Optional[str]` kwarg — cross-scan stable identity. Defaults to `series_name` when unset (preserves today's behavior).
- `plant_qr_code` column remains the CSV lookup key for the existing `expected_count` / `group` / `qc_fail` properties. (`sample_uid` as a CSV column name is deferred to #163, where it will have `plant_qr_code` fallback.)

**New `Series` properties:**

- `Series.sample_uid: str` — returns the kwarg value.
- `Series.timepoint: Union[float, int]` — thin wrapper around `get_metadata("timepoint")`.
- `Series.get_metadata(column: str, plant_id: Optional[int] = None) -> Any` — generic CSV-column accessor. Looks up `df[df["plant_qr_code"] == self.sample_uid]`. If `plant_id` is given AND the CSV has a `plant_id` column, composite lookup returns the (sample_uid, plant_id) row. If `plant_id` is given but the CSV has no `plant_id` column, the `plant_id` argument is silently ignored and the sample_uid-only lookup is used (this keeps single-row-per-sample CSVs working unchanged). Returns `np.nan` if the requested column is missing from the CSV, or if no row matches.
- `Series.expected_count` / `.group` / `.qc_fail` — kept as thin wrappers around `get_metadata(...)` for backward compatibility.

**CSV builder helpers** (new module `sleap_roots/metadata.py`):

- `build_metadata_csv(rows: List[Dict], path: Union[str, Path]) -> Path` — writes a CSV from row dicts, validating that every row has `plant_qr_code`. Canonical column ordering: `plant_qr_code, genotype, number_of_plants_cylinder, qc_cylinder, qc_code, timepoint, <extras...>`.
- `infer_timepoints_from_filenames(slp_paths: List[Path], pattern: str) -> Dict[str, float]` — regex-parse `(series_name, timepoint)` from filename stems. Convenience for common conventions; user can skip and build CSVs any other way.

**Pipeline output changes** (plate + cylinder + future pipelines):

- Emit `sample_uid` and `timepoint` on every output row (CSV) and in the top-level dict (JSON). NaN/None when not set. This is what makes `TimeDiffPipeline`'s `identity_cols` / `time_col` work without extra plumbing.

**Why this ordering matters:** `sample_uid` is the rename for a role that was previously implicit and entangled with `plant_qr_code`. Naming the role separately makes the other two workstreams cleaner — the diff pipeline doesn't need to care what row-granularity the user is diffing (plant, cylinder aggregate, tracked tip).

### Workstream 2 — `TrackedTipPipeline` (rewrite of #129)

Root-agnostic pipeline consuming a **tracked** `.slp` file. One pipeline emits both tip-kinematics and circumnutation traits. Replaces the original `TrackedTipPipeline` scope (which was kinematics-only) by absorbing circumnutation as part of the same trajectory analysis.

**Inputs:**

- `Series` with ONE of `primary_path` / `lateral_path` / `crown_path` populated. Pipeline is root-agnostic — works on whichever root type the SLEAP model tracked.
- Predictions in the `.slp` MUST carry SLEAP track identities (`instance.track is not None`). Pipeline raises `ValueError("TrackedTipPipeline requires tracked .slp predictions; see sleap.ai/tracking")` if any instance lacks a track.

**Output (two tables, emitted together):**

1. **Trajectory rows** (per-track-per-frame, raw positions):
   ```
   series, sample_uid, track_id, frame, timepoint, tip_x, tip_y
   ```
   One row per `(track_id, frame)`. This is the trajectory data circumnutation traits are derived from.

2. **Track summary rows** (per-track, all kinematic + circumnutation scalars):
   ```
   series, sample_uid, track_id, n_frames_tracked, n_frames_total, tracking_coverage,
   # Kinematics
   tip_trajectory_length, tip_displacement_net, tip_velocity_mean, tip_velocity_max,
   tip_curvature_mean, tip_curvature_max,
   # Circumnutation
   nutation_period_mean, nutation_amplitude_mean, nutation_amplitude_max,
   angular_velocity_mean, rotation_direction, n_nutation_cycles
   ```
   One row per `track_id`. Summary scalars over the whole trajectory.

Both tables emit in one call. CSV/JSON contract inherits from the plate pipeline: `schema_version=1`, structured `units` dict, NaN→null JSON via `_json_sanitize`.

**Circumnutation trait module** (`sleap_roots/circumnutation.py`, new):

Pure functions operating on `(t, x, y)` arrays, parallel to `sleap_roots/angle.py` and `sleap_roots/lengths.py`. The pipeline calls these in its TraitDef DAG.

Trait definitions (all per-track):

| Trait | Definition | Units family |
|---|---|---|
| `nutation_period_mean` | Mean time between zero-crossings of detrended `dx(t)` (or peak-to-peak; user kwarg) | times (user-defined) |
| `nutation_amplitude_mean` | Mean peak magnitude of detrended `sqrt(dx² + dy²)` | lengths (pixels) |
| `nutation_amplitude_max` | Max peak magnitude of detrended `sqrt(dx² + dy²)` | lengths (pixels) |
| `angular_velocity_mean` | Mean `d/dt atan2(dy, dx)` — signed | angles / time |
| `rotation_direction` | `sign(median(angular_velocity))` → +1 CCW, -1 CW, 0 none | unitless |
| `n_nutation_cycles` | Completed 2π rotations | counts |
| `tip_trajectory_length` | Cumulative arclength of raw `(x, y)` | lengths |
| `tip_displacement_net` | Euclidean distance first → last point | lengths |
| `tip_velocity_mean` | Mean `dt`-normalized step length | lengths / time |
| `tip_velocity_max` | Max step length / `dt` | lengths / time |
| `tip_curvature_mean` | Mean `abs(dθ/ds)` along trajectory | inverse_lengths |
| `tip_curvature_max` | Max `abs(dθ/ds)` | inverse_lengths |
| `tracking_coverage` | `n_frames_tracked / n_frames_total` | ratios (dimensionless) |

**Detrending:** low-pass / moving-average on `(x(t), y(t))` with window ≈ one expected nutation period. Window expressed in frames (integer) via a constructor kwarg `detrend_window_frames: int`, default `max(3, n_frames // 4)`. Residuals `(dx, dy)` are the detrended oscillation. Required because nutation is oscillation *around* the growth axis, not absolute position — without detrending the net growth drift swamps the oscillation signal.

**Edge cases** (deliberate behavior, tested):

- Short tracks (`n_frames_tracked < 3`) — all derived scalars = NaN, no crash.
- Monotonic trajectory (no zero-crossings) — `nutation_period_mean = NaN`, `rotation_direction = 0`, `n_nutation_cycles = 0`.
- Partial tracking (gaps in frames) — compute on available window; `tracking_coverage < 1.0`.
- Single-frame track — NaN everywhere except `sample_uid` + `track_id`.
- Uneven frame spacing — use `timepoint` if column present, else `frame_rate` kwarg, else assume 1 unit per frame and document.

**Why one pipeline, not two:** circumnutation and growth kinematics both operate on the same per-track `(t, x, y)` trajectory. Separating them would mean two pipelines consuming identical inputs and producing overlapping outputs. No modularity benefit.

### Workstream 3 — `TimeDiffPipeline` (new)

Wrapper `Pipeline` class (D2 style from the brainstorm) that takes an inner pipeline and appends a between-timepoint delta table.

**API:**

```python
from sleap_roots.diff_pipelines import TimeDiffPipeline

# Plate intra-series (one .slp, many frames)
diff = TimeDiffPipeline(
    inner=MultipleDicotPlatePipeline(),
    identity_cols=["series", "plant_id"],
    time_col="frame",
    mode="consecutive",
)
result = diff.compute_plate_traits(series)
# result["plants"] — unchanged from inner pipeline
# result["deltas"] — new parallel delta table

# Cylinder inter-series (N series across timepoints)
diff_cyl = TimeDiffPipeline(
    inner=MultipleDicotPipeline(),
    identity_cols=["sample_uid"],
    time_col="timepoint",
    mode="consecutive",
)
df = diff_cyl.compute_batch_plate_traits(all_series)
# df contains raw rows + delta rows; distinguished by row_type column
```

**Algorithm:**

1. Call `inner.<method>(...)` on the input.
2. Flatten the inner result to a DataFrame (the way the inner pipeline does for CSV output).
3. Verify `identity_cols` and `time_col` all exist as columns. Raise `ValueError` with the list of available columns if any are missing.
4. `df.groupby(identity_cols)` then within each group sort by `time_col` and compute pairwise deltas for every numeric column.
5. Return both the inner DataFrame and the deltas DataFrame.

**Modes:**

- `"consecutive"` — `Δ_{i, i+1}` for each ordered pair in the group.
- `"vs_baseline"` — `Δ_{0, i}` for every `i > 0` in the group (baseline = earliest `time_col` value).

**Output structure:**

- Per-series dict gains a `deltas` key alongside existing `plants` (or equivalent inner result).
- Non-numeric columns are skipped from delta computation but `identity_cols` + `time_col_from` + `time_col_to` + `dt` are repeated so the delta table is self-contained.
- CSV output: default is two files (`*.plate_traits.csv` + `*.plate_traits.deltas.csv`). Constructor kwarg `combined_csv: bool = False` emits a single file with a `row_type in {"raw", "delta"}` discriminator column.
- JSON output: single dict with both tables under distinct keys.

**Supported inner pipelines:**

Any pipeline that emits per-row-over-time output. Specifically:

- `MultipleDicotPlatePipeline` — emits per-plant-per-frame rows; `frame` is the time axis.
- `DicotPipeline` — per-plant per frame (if the user loops it over a timelapse `.slp`).
- `MultipleDicotPipeline` — per-series aggregate row; `timepoint` from CSV is the time axis; per-plant diffs NOT supported until #159 lands.
- `TrackedTipPipeline` (Workstream 2) — per-track summary rows; `track_id` is the identity, `timepoint` (inter-series) or `frame` (intra-series) is the time axis.

**Why D2 (wrapper pipeline) and not D1 (utility module):**

D2's cost is small: the wrapper delegates method dispatch to the inner pipeline, intercepts the result, and appends delta computation. Inner pipelines require no modification. The user writes one pipeline instantiation and gets raw + deltas in one call — the UX requested during brainstorming.

D1 (pure utility module) would have required the user to chain two calls (`inner.compute()` then `diffs.compute()`) and manually flatten the inner output to a DataFrame before passing it in. D2 hides that plumbing and centralizes it in one class.

Estimated size: ~150 lines including tests.

## Decision: kill #112

`PrimaryRootTimelapsePipeline` as originally scoped is redundant. The plate pipeline's frame loop already handles multi-frame per-plant trait output; `TimeDiffPipeline` handles the between-timepoint delta computation. A new pipeline class adds no capability.

**Issue #112 action:** close with a comment pointing to this design doc, the plate pipeline (PR #165), and the new `TimeDiffPipeline` issue.

## Issue updates summary

| Issue | Action | Summary |
|---|---|---|
| #112 | Close as obsolete | Plate pipeline's frame loop + `TimeDiffPipeline` cover this scope. |
| #129 | Rewrite | Expand to include circumnutation traits. Single pipeline (tip-kinematics + circumnutation). Root-agnostic. Requires tracked `.slp`. |
| #159 | Keep (related) | Multi-plant cylinder per-plant diffs blocked on this. |
| #163 | Broaden | Add `sample_uid` column convention (fallback to `plant_qr_code`); add `timepoint` column convention. |
| NEW | File | `TimeDiffPipeline` wrapper class. |
| NEW | File | `sample_uid` + `timepoint` metadata layer + `Series.get_metadata()` generalized accessor. |
| NEW | File | `sleap_roots/circumnutation.py` trait functions + `sleap_roots/metadata.py` CSV builder helpers. |

## Identity table (reference)

Decision-cheatsheet for users choosing `identity_cols` and `time_col`:

| Shape | Identity | Time col | Inner pipeline |
|---|---|---|---|
| Plate intra-series timelapse, untracked | `(series, plant_id)` | `frame` | `MultipleDicotPlatePipeline` |
| Plate intra-series timelapse, SLEAP-tracked | `(series, track_id)` | `frame` | `MultipleDicotPlatePipeline` (after track_id emission lands) |
| Plate inter-series (per-plant stable) | `(sample_uid, plant_id)` | `timepoint` | `MultipleDicotPlatePipeline` or run-per-series then batch |
| Cylinder inter-series, multi-plant aggregate | `(sample_uid,)` | `timepoint` | `MultipleDicotPipeline` |
| Cylinder inter-series, single-plant | `(sample_uid,)` | `timepoint` | `DicotPipeline` or `MultipleDicotPipeline` |
| TrackedTipPipeline output (intra-series) | `(series, track_id)` | `frame` | `TrackedTipPipeline` |
| TrackedTipPipeline output (inter-series) | `(sample_uid, track_id)` | `timepoint` | `TrackedTipPipeline` |

## Out of scope (deferred)

- **Spatial matching of plant IDs across scans** — D3 in the brainstorm. Fragile; file follow-up only if user demand materializes.
- **Date-string parsing into numeric timepoints** (`"2024-03-15"` → day number) — too domain-specific. User responsibility.
- **Multi-plant cylinder per-plant diffs** — blocked on #159.
- **Image-pixel data in plate JSON for the viewer** — tracked in #161 / #128. Viewer must resolve frame-image paths externally.
- **Inner pipelines that don't emit per-row-over-time output** — `TimeDiffPipeline` rejects with a helpful message.
- **Streaming write for large batch JSON** — tracked in #167.
- **Shared helper to deduplicate `_assign_laterals_to_primaries_by_distance` vs `associate_lateral_to_primary`** — tracked in design D7 + possibly #166.

## Dependencies between workstreams

```
Workstream 1 (metadata layer)
    ├──> Workstream 2 (TrackedTipPipeline — needs `timepoint` / `sample_uid`)
    └──> Workstream 3 (TimeDiffPipeline — needs pipelines to emit `timepoint` / `sample_uid`)
```

Workstream 1 is prerequisite for the other two but can land in its own small PR. Workstream 2 and Workstream 3 are independent of each other and can land in parallel once Workstream 1 ships.

## Test plan

Same discipline as PR #165 — synthetic `.slp` round-trip for integration tests; pure-function unit tests for trait modules; openspec proposal per workstream with full scenario coverage.

Each workstream gets its own OpenSpec change:

- `change-id: add-sample-uid-timepoint-metadata` (Workstream 1)
- `change-id: add-tracked-tip-pipeline` (Workstream 2)
- `change-id: add-time-diff-pipeline` (Workstream 3)

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `sample_uid` rename confuses users on existing workflows | Backward-compat — kwarg defaults to `series_name`; CSV column stays `plant_qr_code` until #163. |
| Circumnutation detrending window is hard to tune automatically | User-configurable kwarg; default = `n_frames / 4`; document with a "choose window ≈ expected period" note. |
| `TimeDiffPipeline` diverges from inner pipeline method signatures | Delegate via `getattr(inner, method)(*args, **kwargs)` and let Python's arg-binding enforce compatibility. Test against each supported inner pipeline. |
| SLEAP tracking is expensive / not commonly enabled | Document sleap tracking workflow in `TrackedTipPipeline`'s docstring; raise with a link to sleap docs on untracked input. |
| Users have existing `plant_qr_code`-based workflows that expect per-scan rows to have unique values | No change for them — the kwarg default is `series_name`, which already produces unique per-scan rows. |
