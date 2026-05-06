# Design: Timelapse diffs, tracked-tip kinematics, and metadata generalization

**Date**: 2026-04-23 (updated 2026-05-06 — Workstream 2 brainstorm complete)
**Related issues**: #112 (close as obsolete), #129 (refresh), #163 (broaden), #159 (related), #169 (Workstream 1 metadata layer — implemented in PR #171), #170 (Workstream 3 TimeDiffPipeline), #186 (per-frame metadata accessor — Workstream 2 follow-up), #187 (preprocessing helpers — Workstream 2 follow-up), #188 (generic META→CSV converter — Workstream 2 follow-up), #189 (split `trait_pipelines.py` megafile — triggered by #129)
**Depends on**: PR #165 (MultipleDicotPlatePipeline) — merged 2026-04-21; PR #171 (Workstream 1 metadata layer) — merged 2026-05-06
**Status**: Workstream 1 shipped (PR #171). Workstream 2 brainstorm complete (2026-05-06); ready for OpenSpec proposal + TDD implementation. Circumnutation trait set deferred to a follow-up design once the maintainer has assembled a literature reference for method selection.

## Summary

Three coordinated workstreams that replace the obsolete `PrimaryRootTimelapsePipeline` idea (#112) and refresh the existing `TrackedTipPipeline` scope (#129) with:

1. A generalized metadata layer on `Series` that reads arbitrary CSV columns — enables `sample_uid` (cross-scan stable identity) and `timepoint` (time-axis value) as first-class concepts.
2. A refreshed `TrackedTipPipeline` that consumes tracked `.slp` predictions and emits per-track tip-kinematics (trajectory + growth-kinematic scalars). Circumnutation traits are deferred to a follow-up PR that will be designed against published literature.
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

**Pipeline output changes** (plate first; cylinder + others as a follow-up issue):

- Emit `sample_uid` and `timepoint` on every output row (CSV columns 1 and 2, right after `series`) and in the top-level dict (JSON). NaN/None when not set. This is what makes `TimeDiffPipeline`'s `identity_cols` / `time_col` work without extra plumbing.
- Plate pipeline `schema_version` bumps from `1` → `2` because the column shift at positions 1 and 2 is non-additive. `_PLATE_UNITS` gains a `"time"` key, set to `"unspecified"` until a future follow-up adds a `time_unit` pipeline kwarg or `timepoint_unit` CSV column. While `units["time"] == "unspecified"` and a non-NaN `timepoint` is emitted, the plate pipeline logs a one-shot WARNING per call to surface the missing-unit hazard at runtime.

**Why this ordering matters:** `sample_uid` is the rename for a role that was previously implicit and entangled with `plant_qr_code`. Naming the role separately makes the other two workstreams cleaner — the diff pipeline doesn't need to care what row-granularity the user is diffing (plant, cylinder aggregate, tracked tip).

### Workstream 2 — `TrackedTipPipeline` (refresh of #129)

Root-agnostic pipeline consuming a **tracked** `.slp` file. Emits per-track tip trajectories + a **minimum-viable substrate** of per-track geometric scalars. **TrackedTipPipeline's scope is the substrate, period.** Velocity, curvature, smoothness, tortuosity, direction-changes, circumnutation traits — none of these belong in TrackedTipPipeline now or in the future. They live in **separate downstream pipelines** (e.g. `CircumnutationPipeline`, future growth-rate / gravitropism pipelines) that REUSE this pipeline's `Series.get_tracked_tips` accessor and trajectory-row output as their input substrate. Keeping TrackedTipPipeline minimal makes it reusable across many downstream pipelines without coupling to any single analysis's opinions.

**Why minimum-viable substrate, not full kinematics:** every downstream tip-aware pipeline needs (a) the raw `(t, x, y)` per track and (b) track-existence metadata (`tracking_coverage`) for filtering. Beyond those two, this pipeline emits only the unambiguous geometric scalars (`tip_trajectory_length`, `tip_displacement_net`) that have no detrending opinion. Derivative-based scalars (velocity, curvature) force a `frame_rate` decision and a "raw vs detrended" decision — those decisions belong in the downstream pipeline that owns the analysis (e.g. circumnutation analyses redefine velocity post-detrending). The substrate stays opinion-free so it serves every downstream consumer equally well.

**Module placement:**

- `sleap_roots/tracked_tip_pipeline.py` (NEW FILE) — `TrackedTipPipeline` class. Starts the per-pipeline-module pattern. Existing `trait_pipelines.py` megafile (3763 lines, 8 pipelines) split tracked in #189.
- `sleap_roots/tip_kinematics.py` (NEW FILE) — pure trait functions parallel to `sleap_roots/lengths.py` and `sleap_roots/angle.py`.
- `sleap_roots/series.py` (EXTEND) — new `Series.get_tracked_tips(root_type=None)` method + module-level validation helpers (`validate_tracked_slp`, `validate_series_for_tracked_tip`).

**Inputs:**

- `Series` with ONE of `primary_path` / `lateral_path` / `crown_path` populated. Pipeline is root-agnostic — works on whichever root type the SLEAP model tracked. If multiple paths are populated, caller must specify `root_type` explicitly; otherwise `ValueError`.
- Predictions in the `.slp` MUST carry SLEAP track identities (`instance.track is not None` for every instance). Pipeline raises `ValueError("TrackedTipPipeline requires tracked .slp predictions; see sleap.ai/tracking")` on the first untracked instance.
- Optional CSV metadata via `Series.load(csv_path=...)` — when present, `sample_uid` and `timepoint` are emitted in output rows. (`sample_uid` defaults to `series.series_name` per Workstream 1; `timepoint` is `np.nan` when no CSV.)
- **Single-node and multi-node skeletons both supported.** Tip is the LAST node by skeleton convention (`pts[-1]`), already used by `sleap_roots.tips.get_tips`. Verified on a single-node circumnutation skeleton (`['r0']`) — the `[:, -1, :]` slice degenerates to `[:, 0, :]`, which IS the only point. A future user training a multi-node tip skeleton (e.g. `[base, midpoint, tip]`) gets correct extraction for free.

**Output (two tables, emitted together):**

1. **Trajectory rows** (per-track-per-frame, raw positions):
   ```
   series, sample_uid, timepoint, track_id, frame, tip_x, tip_y
   ```
   One row per `(track_id, frame)`. Self-contained tip-position timeline suitable as input to downstream circumnutation analysis once that workstream lands. `frame` is the integer frame index in the .slp; real-time per-frame is **deferred to #186** (companion CSV via `Series.get_per_frame_metadata`). `timepoint` here is the per-series scalar (constant within a series), enabling uniform `TimeDiffPipeline` (Workstream 3) consumption across both tables.

2. **Track summary rows** (per-track scalars — substrate only):
   ```
   series, sample_uid, timepoint, track_id, n_frames_tracked, n_frames_total,
   tracking_coverage, tip_trajectory_length, tip_displacement_net
   ```
   One row per `track_id`. Substrate scalars only; **no derivative-based metrics in v1**.

Both tables emit in one call. Output written to:

- **Per-series**: `<series>.tracked_tip_traits.csv` (summary), `<series>.tracked_tip_trajectories.csv` (trajectory), `<series>.tracked_tip_traits.json` (both tables under `{tracks: [...], trajectories: [...]}` plus top-level scalars).
- **Batch** (`compute_batch_tracked_tip_traits(all_series)`): `tracked_tip_batch_traits.csv` (concatenated summary), `tracked_tip_batch_trajectories.csv` (concatenated trajectory), `tracked_tip_batch_traits.json` (list of per-series dicts). Mirrors PR #165's `compute_batch_plate_traits` pattern (`pd.concat` of per-series DataFrames; per-series result-dict list for JSON).

`emit_trajectories: bool = True` kwarg on both methods — when `False`, skip writing the trajectory CSV and omit the `trajectories` array from JSON. Lets users running large batches skip the bulky table when only summary scalars are needed. At realistic experiment scale this is rarely needed (4 plates × 6 tracks × 311 frames ≈ 7,500 rows, ~750 KB concatenated trajectory CSV — fine), but the kwarg costs nothing.

**CSV/JSON contract:**

- `schema_version: 1` at top-level JSON (fresh pipeline, starts at 1).
- Structured `units` dict — `lengths: "pixels"`, `ratios: "dimensionless"`, `counts: "dimensionless"`, `time: "unspecified"` (Workstream 1's convention; future `time_unit` plumbing upgrades in place).
- NaN → JSON `null` via `_json_sanitize` (existing helper from PR #165).
- Top-level scalars (`series`, `sample_uid`, `timepoint`) in JSON do NOT repeat inside `tracks` / `trajectories` rows; CSV does repeat them on every row (PR #165's CSV/JSON asymmetry preserved).

**`Series.get_tracked_tips(root_type=None)` (NEW ACCESSOR):**

Returns a long-format `pd.DataFrame` with columns `track_id, frame, tip_x, tip_y`. One row per `(track_id, frame)` where the track has an instance.

- Auto-detects `root_type` from whichever of `primary_path` / `lateral_path` / `crown_path` is populated; raises `ValueError` if zero or >1 are populated and `root_type` is `None`.
- **Iterates per-instance, NOT per-frame-stack** — tracker output does NOT preserve positional ordering across frames. Verified on the fixture: frame 0 instances are ordered `[track_0, 1, 2, 3, 4, 5]` but frame 1 is `[track_0, 3, 4, 2, 1, 5]`. Implementation reads `inst.track.name` per instance and `inst.numpy()[-1]` for the tip coordinate.
- Untracked instances (`inst.track is None` or `inst.track.name` falsy) trip `ValueError` with the sleap.ai/tracking pointer.

**Tip-kinematics trait functions** (`sleap_roots/tip_kinematics.py`, NEW MODULE — thin wrappers around existing trait functions):

Pure functions on `(N, 2)` xy arrays. The pipeline calls these in its `TraitDef` DAG. **DRY-driven: the actual numpy computation lives in `sleap_roots/lengths.py` and `sleap_roots/bases.py` already.** This new module exists to (a) document tip-trajectory-specific edge cases and (b) provide a stable import path for future tip-trajectory-aware pipelines.

| Function | Computation source | Single-frame behavior | Units family |
|---|---|---|---|
| `tip_trajectory_length(xy)` | Delegates to [`lengths.get_root_lengths`](sleap_roots/lengths.py#L56) (cumulative arclength via `np.diff` + `np.linalg.norm` + `np.nansum`). Wrapper special-cases single-frame to return `0.0` instead of NaN (existing function's vacuous-truth NaN behavior on empty segments). | `0.0` (no segments) | lengths |
| `tip_displacement_net(xy)` | Delegates to [`bases.get_base_tip_dist`](sleap_roots/bases.py#L34) (Euclidean via `np.linalg.norm`). Pass `xy[0]` as base, `xy[-1]` as tip. | `0.0` (xy[0] == xy[-1] — natural, no special-case) | lengths |
| `tracking_coverage(n_tracked, n_total)` | Trivial division — no existing utility. ~3 lines. | well-defined for `n_tracked >= 1` | ratios |

Single-frame tracks emit `length=0.0, displacement=0.0` — mathematically correct (no segments → no length); no NaN special-casing in the pipeline output. Downstream filtering on `n_frames_tracked == 1` is the user's option.

**Implementation sketch** (~30 lines total — nearly all delegation):

```python
from sleap_roots.lengths import get_root_lengths
from sleap_roots.bases import get_base_tip_dist

def tip_trajectory_length(xy: np.ndarray) -> float:
    if xy.shape[0] == 0: return np.nan
    if xy.shape[0] == 1: return 0.0
    return float(get_root_lengths(xy))

def tip_displacement_net(xy: np.ndarray) -> float:
    if xy.shape[0] == 0: return np.nan
    return float(get_base_tip_dist(xy[0], xy[-1]))

def tracking_coverage(n_frames_tracked: int, n_frames_total: int) -> float:
    if n_frames_total == 0: return np.nan
    return n_frames_tracked / n_frames_total
```

Existing functions in `lengths.py` / `bases.py` have full test coverage and known NaN semantics. The wrappers' tests just cover the tip-trajectory-specific edge cases (single-frame zeros, empty input).

**DAG-A — TraitDef topology:**

Pipeline uses the existing `TraitDef` graph plumbing (per `DicotPipeline` and friends), but the input is the long-format `tracked_tips_df` and the iteration unit is `track_id`, not `frame`:

```
tracked_tips_df  (input from series.get_tracked_tips())
        │
        │ groupby(track_id) → track_xy (Nx2)
        ▼
        ├─→ tip_trajectory_length(track_xy)
        ├─→ tip_displacement_net(track_xy)
        ├─→ tracking_coverage(n_frames_tracked, n_frames_total)
        ├─→ n_frames_tracked
        └─→ n_frames_total
                │
                ▼
        per-track summary row
```

**Circumnutation and other downstream pipelines REUSE this substrate; they DO NOT extend `TrackedTipPipeline`.** A future `CircumnutationPipeline` is a separate class that consumes `series.get_tracked_tips(...)` (or this pipeline's trajectory CSV / JSON output) as its input substrate, runs its own `TraitDef` DAG with circumnutation-specific nodes, and emits its own per-track summary table. Conceptually:

```
                                    ┌──────────────────────────────────┐
                                    │  Series.get_tracked_tips(...)    │
                                    │  + TrackedTipPipeline trajectory │
                                    │     CSV / JSON                   │
                                    │  (THIS PR — frozen substrate)    │
                                    └────────────────┬─────────────────┘
                                                     │ reused as input
                       ┌─────────────────────────────┼──────────────────────────────┐
                       ▼                             ▼                              ▼
              ┌──────────────────┐        ┌────────────────────┐         ┌──────────────────┐
              │ CircumnutationPipe│        │ (future) GrowthRate│         │ (future) Gravi-  │
              │ (NEXT PR — own   │        │ Pipeline           │         │ tropismPipeline  │
              │  TraitDef DAG)    │        │ (own DAG, own CSV) │         │ (own DAG, own CSV)│
              │  Period/amplitude/│        │ Velocity/accel etc.│         │ Angle vs gravity │
              │  rotation etc.    │        │                    │         │                  │
              └──────────────────┘        └────────────────────┘         └──────────────────┘
```

`TrackedTipPipeline`'s 5 substrate scalars stay frozen forever. New downstream pipelines come into existence as siblings, not as extensions of this one.

**Validation helpers** (in scope for #129):

- `sleap_roots.series.validate_tracked_slp(slp_path) -> None` — opens the .slp, asserts every instance has `inst.track is not None` AND `inst.track.name` is non-empty, raises `ValueError` listing offending frame indices if not. Module-level, paired with existing `find_all_slp_paths`.
- `sleap_roots.series.validate_series_for_tracked_tip(series, root_type=None) -> None` — composite check: validates the relevant `<root_type>_path`, asserts skeleton has ≥1 node, calls `validate_tracked_slp`. Raises with actionable messages.

**Edge cases** (deliberate behavior, tested):

- **Zero tracked instances anywhere** — pipeline emits empty `tracks` and `trajectories` arrays / empty CSVs, no crash.
- **Single-frame track** — `length=0.0, displacement=0.0`, `n_frames_tracked=1`, `tracking_coverage > 0`, no NaN.
- **Partial tracking (gaps)** — compute on available window; `tracking_coverage < 1.0`.
- **Track-fragment / re-ID** — trust the tracker. If one physical tip becomes two `track_id`s in the source, two summary rows appear. **Documented assumption: input `.slp` is proofread.**
- **Multiple root paths populated** — `ValueError` requesting explicit `root_type`.
- **`track.name` is `None` or empty string** — treated as untracked; trips the "requires tracked .slp" `ValueError`.
- **Single-node skeleton** (only the tip node) — supported via the `pts[-1]` convention; verified on the circumnutation fixture (`r0`-only skeleton).
- **Multi-node skeleton** (e.g. `[base, midpoint, tip]`) — supported via the same `pts[-1]` convention.

### Deferred: circumnutation trait set

Circumnutation-specific traits (period, amplitude, angular velocity, rotation direction, nutation-cycle count, etc.) are intentionally **out of scope for this design** and will be spec'd in a separate follow-up once the maintainer has assembled a literature reference on method selection. Questions to answer with the literature:

- Which detrending method is the field standard (moving average, low-pass Butterworth, polynomial fit, empirical mode decomposition)?
- Which period-estimation approach: zero-crossing, peak-to-peak, FFT, autocorrelation, wavelet?
- Is "amplitude" defined as radial distance from detrended mean, half peak-to-peak, or RMS of residuals?
- How to define rotation direction when trajectory is noisy / partially tracked?

The TrackedTipPipeline's trajectory-row output (and `Series.get_tracked_tips` accessor) is the input substrate for those analyses. **The circumnutation trait set lives in a separate `CircumnutationPipeline` class** — it does NOT extend `TrackedTipPipeline`. Keeping `TrackedTipPipeline` minimal and opinion-free is what makes it reusable across multiple downstream pipelines (circumnutation today; growth-rate, gravitropism, and other tip-aware analyses tomorrow).

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
result = diff_cyl.compute_batch_multiple_dicots_traits(all_series)
# result["plants"] — unchanged from inner pipeline (per-series aggregate rows)
# result["deltas"] — new parallel delta table
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
- `DicotPipeline` — emits per-plant-per-frame rows for a multi-frame `Series`; a single `compute_plant_traits(series)` call already iterates over `range(len(series))` and returns one row per frame, with `frame` as the time axis (no external loop required).
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
| #129 | Refresh (now scoped to **substrate**, not full kinematics) | Trajectory rows + 5 substrate scalars (`tip_trajectory_length`, `tip_displacement_net`, `tracking_coverage`, `n_frames_tracked`, `n_frames_total`). Root-agnostic. Requires tracked `.slp`. **Scope frozen** — velocity / curvature / circumnutation traits do NOT belong in this pipeline; they live in separate downstream pipeline classes that REUSE the substrate. Output contract aligns with PR #165 (`schema_version=1`, structured units, `_json_sanitize`). |
| #159 | Keep (related) | Multi-plant cylinder per-plant diffs blocked on this. |
| #163 | Broaden | Add `sample_uid` column convention (fallback to `plant_qr_code`); add `timepoint` column convention; rename legacy `plant_qr_code` column. |
| #169 | Implemented (PR #171) | Workstream 1 metadata layer shipped 2026-05-06. |
| #170 | Keep (Workstream 3) | `TimeDiffPipeline` wrapper class. |
| #186 | Filed 2026-05-06 (Workstream 2 follow-up) | Per-frame metadata accessor on `Series` (`get_per_frame_metadata`, companion CSV). Required for any pipeline that needs real-time-per-frame; not blocking #129 (substrate uses integer `frame` only). |
| #187 | Filed 2026-05-06 (Workstream 2 follow-up) | Preprocessing helpers: image folder → `.h5` + per-frame metadata CSV. Captures the upstream of the data-prep workflow. |
| #188 | Filed 2026-05-06 (Workstream 2 follow-up) | Generic source-META → sleap-roots-CSV converter (`convert_meta_to_qr_csv`). Captures the experiment-metadata side. |
| #189 | Filed 2026-05-06 (triggered by #129) | Refactor: split `trait_pipelines.py` (3763 lines, 8 pipelines) into per-pipeline modules. #129 starts the per-pipeline-file pattern by living in `sleap_roots/tracked_tip_pipeline.py`. |
| NEW (deferred) | File once literature is assembled | Circumnutation trait set (period, amplitude, angular velocity, rotation direction, etc.) — depends on Workstream 2 shipping first so the trajectory-row output exists. |

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

- **Velocity / curvature / smoothness / tortuosity / direction-changes traits** — these are NEVER added to `TrackedTipPipeline`. They live in separate downstream pipeline classes (e.g. a future `CircumnutationPipeline`, `GrowthRatePipeline`, etc.) that REUSE this pipeline's `Series.get_tracked_tips` accessor and trajectory output as their input substrate. `TrackedTipPipeline`'s scope is frozen at the substrate so it remains opinion-free and reusable across many downstream pipelines.
- **Per-frame real-time timestamps** in `TrackedTipPipeline` output — deferred to **#186** (per-frame metadata accessor on `Series`). Substrate uses integer `frame` column only.
- **Image folder → `.h5` + per-frame metadata CSV preprocessing** — deferred to **#187**. Captures the upstream of the data-prep workflow (raw imaging output → sleap-roots-readable inputs).
- **Generic source-META → sleap-roots-CSV conversion helpers** — deferred to **#188**. Captures the experiment-metadata side of the data-prep workflow.
- **Splitting `trait_pipelines.py` into per-pipeline modules** — deferred to **#189**. `TrackedTipPipeline` lands in its own new file (`sleap_roots/tracked_tip_pipeline.py`), starting the pattern.
- **Circumnutation trait set** (period, amplitude, angular velocity, rotation direction, n_nutation_cycles, detrending method) — deferred to a separate design once the maintainer has assembled a literature reference. Workstream 2's trajectory-row output is the natural input.
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

- `change-id: add-sample-uid-timepoint-metadata` (Workstream 1 — shipped via PR #171, archived 2026-05-06)
- `change-id: add-tracked-tip-pipeline` (Workstream 2 — substrate scope: trajectory + 5 per-track scalars; no velocity / curvature / circumnutation)
- `change-id: add-time-diff-pipeline` (Workstream 3)

When the circumnutation literature is assembled, a fourth change (`add-circumnutation-traits` or similar) picks up that scope on top of Workstream 2's trajectory-row output.

## Test data strategy

Per-workstream fixture plan. PR #165 shipped synthetic-only with real fixtures deferred to #162; this design is more aggressive on real fixtures for Workstream 2 because the source data is available.

### Workstream 1 — metadata layer

**Synthetic-only.** All behavior is CSV plumbing + Series kwarg lookup:

- Synthetic CSV strings written to `tmp_path` (same pattern as PR #165's `_plate_csv` helper).
- Optional: reuse existing `multiple_arabidopsis_11do_csv` (cylinder-multi-plant dataset in Git LFS) for one integration-level test that exercises a real CSV file with real column values.
- No new SLEAP data needed.

### Workstream 2 — `TrackedTipPipeline`

**Synthetic + real, from day one.**

Synthetic coverage (primary test surface):

- Pure trait functions in `sleap_roots/tip_kinematics.py` against known `(N, 2)` arrays — exact-value assertions on multi-segment trajectories, single-frame degenerate case (`length=0.0, displacement=0.0`), known geometric shapes (square, straight line).
- `Series.get_tracked_tips` accessor on synthetic `.slp` files with `sio.Track` objects attached to `sio.Instance` via `from_numpy(..., track=...)` — exercises root-type auto-detection, multiple-paths-populated `ValueError`, untracked-instance `ValueError`, single-node and multi-node skeleton paths.
- Pipeline DAG, CSV/JSON emission, batch concat, `emit_trajectories=False` skip path, empty-input / single-frame / short-track edge cases — all via synthetic `.slp` files.

Real-fixture coverage (committed in the PR):

- **Source**: `Z:\users\eberrigan\circumnutation\20250819_Suyash_Patil_CMTN_Kitx_vs_Hk1-3_07-30-25\run_20250827_091833\plate_001_greyscale.tracked.slp` (verified 2026-05-06) — 311 frames, 6 tracks, single-node skeleton (`['r0']`), HDF5 video backend.
- **Fixture file**: `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp` — **184 KB**. Take the WHOLE 311-frame plate as-is (no subsetting needed; size is comfortably under git-LFS thresholds). Commit via Git LFS.
- **Fixture metadata CSV**: `tests/data/circumnutation_plate/fixture_metadata.csv` — synthesized BY HAND (one-time, NOT a script) from `CMTN_KITXvsHK1-3_META.csv`'s plate-1 row. Schema follows the existing `plant_qr_code`-keyed convention (legacy column name; rename tracked in #163):
  ```
  plant_qr_code,genotype,treatment,number_of_plants_cylinder,timepoint
  plate_001,KitaakeX,MOCK,6,0
  ```
  README documents the column-name caveat (`plant_qr_code` value `"plate_001"` is a plate identifier, not a plant identifier; `number_of_plants_cylinder` is a misnomer here — no cylinder).
- **Per-frame metadata** (`plate_001_metadata.csv` from the source) is **deliberately NOT shipped** — consumed only after #186 lands. README points to #186 as the path forward for real-time-per-frame consumption.
- **Fixture README required**: `tests/data/circumnutation_plate/README.md` following the standard template from issue #168 (purpose, imaging geometry, acquisition context, contents, known limitations, related issues, conversion provenance). The new fixture lands with documentation from day one so Workstream 2's PR does not add to the existing test-data documentation debt.
- **Integration tests** (two paths exercise both CSV-plumbing and no-CSV branches):
  1. `Series.load(primary_path=..., csv_path=fixture_metadata.csv, sample_uid="plate_001")` → `TrackedTipPipeline.compute_tracked_tip_traits(series)` → assert: every output column present, `tracking_coverage ∈ [0, 1]` for every track, summary row count equals 6 (tracks), trajectory row count equals total tracked instances (311 frames × 6 tracks where every frame has 6 tracked instances → 1866 rows). No exact trait-value assertions (those belong in synthetic tests).
  2. `Series.load(primary_path=...)` (NO `csv_path`) → same pipeline call → assert `sample_uid` defaults to `series_name`, `timepoint` is NaN, all other columns identical to (1).
- **Purpose**: catch sleap-io track-representation drift, real tracker-output edge cases (track-order non-determinism across frames — verified during brainstorm), ensure single-node skeleton path works on real tracked predictions.

**Gate on PR merge**: real fixture lands WITH the PR (source data is available; size is small). No follow-up deferral.

### Workstream 3 — `TimeDiffPipeline`

**Synthetic-only.**

- Unit tests with stub inner pipelines returning known DataFrames.
- Integration tests: run real `MultipleDicotPlatePipeline` (already in the repo) on synthetic plate `.slp` with 3+ frames → `TimeDiffPipeline` wrapping it → verify delta rows match expected `Δprimary_length`, etc.
- No new real data required; the plate pipeline's own test fixtures are sufficient realism.

### Deferred real fixtures (follow-ups)

- **#162** — real plate `.slp` fixture tests for `MultipleDicotPlatePipeline` (PR #165). Blocked on SLEAP predictions for `Z:\users\eberrigan\20260401_Kappes_Medicago_MK22_Plates`. Not in this design's scope.
- **#168** — backfill test-data README documentation (top-level `tests/data/README.md` + per-directory READMEs for the 9 existing fixture directories). Out of scope for Workstreams 1–3 but tracked so the gap closes. Workstream 2's new fixture ships with its README in-PR (per the "Fixture README required" requirement above).
- **Future** — once circumnutation workstream lands, add dedicated nutation-specific test fixtures (e.g., a known-period plant trajectory) from the same `circumnutation` source folder.

### Source-data layout

The source folders under `Z:\users\eberrigan\` are the maintainer's working copies, not yet organized for direct test-fixture use. Fixtures committed to `tests/data/` under this design are subsets chosen for size and representativeness. Reorganizing the source-folder layout is out of scope here; follow-up refactor can consolidate once we have multiple pipelines consuming the data.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| `sample_uid` rename confuses users on existing workflows | Backward-compat — kwarg defaults to `series_name`; CSV column stays `plant_qr_code` until #163. |
| `TimeDiffPipeline` diverges from inner pipeline method signatures | Delegate via `getattr(inner, method)(*args, **kwargs)` and let Python's arg-binding enforce compatibility. Test against each supported inner pipeline. |
| SLEAP tracking is expensive / not commonly enabled | Document sleap tracking workflow in `TrackedTipPipeline`'s docstring; raise with a link to sleap docs on untracked input. |
| Users have existing `plant_qr_code`-based workflows that expect per-scan rows to have unique values | No change for them — the kwarg default is `series_name`, which already produces unique per-scan rows. |
| Users expect velocity / curvature / kinematic scalars in `TrackedTipPipeline` from the original #129 scope | Document explicitly in the pipeline docstring + README that `TrackedTipPipeline` is the minimum-viable substrate, scope frozen. Velocity / curvature etc. live in separate downstream pipelines (e.g. future `CircumnutationPipeline`) — never added to `TrackedTipPipeline`. The trajectory-row output + `Series.get_tracked_tips` accessor are the input substrate for those analyses. |
| Track-order non-determinism across frames could be assumed away accidentally during refactors | `Series.get_tracked_tips` iterates per-instance and reads `inst.track.name` per instance — never relies on positional ordering. Documented in the accessor's docstring with the verified example from the fixture brainstorm (frame 0: `[0,1,2,3,4,5]`; frame 1: `[0,3,4,2,1,5]`). Test asserts that two adjacent frames with shuffled positional order still produce identically-keyed track rows. |
| Single-node skeleton support breaks if the `[-1]` convention is changed | `[-1]` slice is the established convention in `sleap_roots.tips.get_tips`. Test asserts the accessor works on both single-node (`['r0']`) and multi-node skeletons. |
