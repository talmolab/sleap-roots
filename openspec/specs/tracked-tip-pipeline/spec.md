# tracked-tip-pipeline Specification

## Purpose
TBD - created by archiving change add-tracked-tip-pipeline. Update Purpose after archive.
## Requirements
### Requirement: `TrackedTipPipeline` SHALL live in its own module

A new module `sleap_roots/tracked_tip_pipeline.py` MUST exist and MUST contain a `TrackedTipPipeline(Pipeline)` class. The class MUST be re-exported from `sleap_roots.__init__` at the top level. Implementation MUST NOT be appended to `sleap_roots/trait_pipelines.py`. The module starts the per-pipeline-file pattern; the existing megafile split for the 8 pre-existing pipelines is tracked separately in #189.

#### Scenario: Top-level import works

- **Given** a fresh Python session
- **When** `from sleap_roots import TrackedTipPipeline` is executed
- **Then** the import succeeds and `TrackedTipPipeline` is the class defined in `sleap_roots/tracked_tip_pipeline.py`

#### Scenario: Module-level import works

- **Given** a fresh Python session
- **When** `from sleap_roots.tracked_tip_pipeline import TrackedTipPipeline` is executed
- **Then** the import succeeds and resolves to the same class as the top-level import

### Requirement: `Series.get_tracked_tips` SHALL extract long-format tracked-tip rows

A new method `Series.get_tracked_tips(self, root_type: Optional[Literal["primary", "lateral", "crown"]] = None) -> pd.DataFrame` MUST be added to the `Series` class. Behavior:

- The `root_type` argument selects which of `primary_path` / `lateral_path` / `crown_path` to read. When `root_type` is `None`, MUST auto-detect from whichever single path is populated; MUST raise `ValueError` mentioning the missing or ambiguous paths if zero or more than one is populated.
- The method MUST return a `pd.DataFrame` with exactly the columns `["track_id", "frame", "tip_x", "tip_y"]` in that order.
- The DataFrame MUST be sorted lexicographically by `(track_id, frame)`. Output index MUST be a clean range index (`reset_index(drop=True)`).
- The `tip_x` and `tip_y` values MUST come from `inst.numpy()[-1]` for each instance — the LAST node of the skeleton, by SLEAP convention. This applies to single-node skeletons (e.g. `['r0']`) and multi-node skeletons (e.g. `['base', 'mid', 'tip']`) identically.
- The method MUST iterate per-instance (not by stacking `inst.numpy()` arrays per frame). Tracker output does NOT preserve positional ordering across frames; the implementation MUST read `inst.track.name` per instance individually.
- Untracked instances (`inst.track is None` or `inst.track.name` empty/None/falsy) MUST cause the method to raise `ValueError` whose message contains the offending frame index AND the URL `https://sleap.ai/tutorials/tracking.html` (or the closest supported sleap.ai documentation URL — whichever is canonical at the time).
- **(Added in PR #2)** On proofread `.slp` files, where SLEAP retains both the `PredictedInstance` (the tracker's prediction; `score` is a float) and the user-corrected `Instance` (`score` is `None`) for the same `(frame_idx, track.name)`, the method MUST keep ONLY the user-corrected `Instance` and drop the `PredictedInstance`. This honors the prelim §3.1 convention: *"Where an `instance_type=0` (user-corrected) and `instance_type=1` (predicted) instance exist for the same frame and track, the user-corrected value takes precedence."* The dedup MUST use `isinstance(inst, Instance) and not isinstance(inst, PredictedInstance)` as the test for user-corrected (`PredictedInstance` subclasses `Instance` in `sleap_io`, so the negative `isinstance(_, PredictedInstance)` is required to distinguish them).
- **(Added in PR #2)** Other duplicate patterns are NOT collapsed: two `PredictedInstance` instances for the same `(track_id, frame_idx)` (rare buggy tracker output) remain as two rows in the returned DataFrame; two user-corrected `Instance` instances for the same `(track_id, frame_idx)` (rare proofreader pathology) remain as two rows. Only the predicted-with-user-correction pattern is deduplicated. This preserves the existing per-track-summary dedup contract documented by `tests/test_tracked_tip_pipeline.py` §6.14 (where `TrackedTipPipeline` exercises the case of two user-corrected instances at the same frame and expects `len(trajectories) == 2` while `n_frames_tracked == 1`).

#### Scenario: Returns long-format DataFrame with expected columns

- **Given** a `Series` loaded from a tracked .slp with 3 frames and 2 tracks (every instance tracked)
- **When** `series.get_tracked_tips()` is called
- **Then** the return value is a `pd.DataFrame` with columns `["track_id", "frame", "tip_x", "tip_y"]`
- **And** `len(df) == 6` (one row per `(track_id, frame)` where the track has an instance)

#### Scenario: Output is sorted by (track_id, frame)

- **Given** a tracked .slp where instance positional order varies across frames (e.g. frame 0 is `[track_a, track_b]`, frame 1 is `[track_b, track_a]`)
- **When** `series.get_tracked_tips()` is called
- **Then** within each `track_id` group of the returned DataFrame, the `frame` column is monotonically non-decreasing
- **And** `df.equals(df.sort_values(["track_id", "frame"]).reset_index(drop=True))` is `True`

#### Scenario: Auto-detects root type when only primary_path is set

- **Given** a `Series.load(primary_path=tracked_slp_path)` call (lateral_path and crown_path None)
- **When** `series.get_tracked_tips()` is called WITHOUT a `root_type` argument
- **Then** the method succeeds and reads from `primary_path`

#### Scenario: Raises when multiple paths populated and no root_type

- **Given** a `Series.load(primary_path=..., lateral_path=...)` call (both populated)
- **When** `series.get_tracked_tips()` is called WITHOUT a `root_type` argument
- **Then** `ValueError` is raised with a message mentioning `root_type` and listing the populated paths

#### Scenario: Raises when zero paths populated

- **Given** a `Series` with `primary_path=None`, `lateral_path=None`, `crown_path=None`
- **When** `series.get_tracked_tips()` is called
- **Then** `ValueError` is raised with a message mentioning `root_type` and the missing path kwargs

#### Scenario: Raises on instance with `track is None`

- **Given** a tracked .slp where one instance at frame 5 has `inst.track is None`
- **When** `series.get_tracked_tips()` is called
- **Then** `ValueError` is raised with a message containing `"5"` (the frame index) and `"sleap.ai/tutorials/tracking"`

#### Scenario: Raises on instance with empty track name

- **Given** a tracked .slp where one instance has `inst.track = sio.Track(name="")` (or `name=None`)
- **When** `series.get_tracked_tips()` is called
- **Then** `ValueError` is raised with the same message format as the `track is None` case (empty/None track name treated as untracked)

#### Scenario: Single-node skeleton — tip is the only node

- **Given** a tracked .slp whose skeleton has a single node (e.g. `['r0']`) and one tracked instance per frame
- **When** `series.get_tracked_tips()` is called
- **Then** the returned `tip_x` / `tip_y` values equal that single node's coordinates (`inst.numpy()[0]` and `inst.numpy()[-1]` are identical for a 1-node skeleton)

#### Scenario: Multi-node skeleton — tip is the last node

- **Given** a tracked .slp whose skeleton has 3 nodes named `['base', 'mid', 'tip']` and one tracked instance per frame
- **When** `series.get_tracked_tips()` is called
- **Then** the returned `tip_x` / `tip_y` values equal the LAST-node coordinates (`inst.numpy()[-1]`), NOT the base or mid coordinates

#### Scenario: Proofread `.slp` — PredictedInstance with user-corrected Instance for same (track, frame) collapses to one row

- **Given** a proofread `.slp` (e.g. the Nipponbare plate 001 fixture at `tests/data/circumnutation_nipponbare_plate_001/plate_001_greyscale.tracked_proofread.slp`) where every frame the user manually corrected retains BOTH a `PredictedInstance` (tracker's prediction; `score`=float) AND a user-corrected `Instance` (`score`=None) for the same `(track.name, frame_idx)`
- **When** `series.get_tracked_tips()` is called
- **Then** the returned DataFrame contains exactly ONE row per `(track_id, frame_idx)` 2-tuple (no duplicates)
- **And** for each `(track_id, frame_idx)` where both instance types coexisted, the `tip_x`/`tip_y` values equal the user-corrected `Instance`'s `inst.numpy()[-1]`, NOT the `PredictedInstance`'s
- **And** for the Nipponbare fixture specifically: `len(df) == 3450` (6 tracks × 575 frames, no duplicates) and `df.duplicated(subset=["track_id", "frame"]).sum() == 0`

#### Scenario: Non-proofread `.slp` — dedup is a no-op

- **Given** a tracked `.slp` where ALL instances are `PredictedInstance` (no user corrections; e.g. the existing KitaakeX fixture at `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp`)
- **When** `series.get_tracked_tips()` is called
- **Then** the returned row count equals the row count under the pre-dedup behavior (the dedup logic finds no PredictedInstance-with-user-correction pairs to collapse)
- **And** `df.duplicated(subset=["track_id", "frame"]).sum() == 0` (vacuously satisfied)
- **And** for the KitaakeX fixture specifically: `len(df) == 1866` (311 frames × 6 tracks)

#### Scenario: Two PredictedInstance for same (track, frame) — NOT collapsed

- **Given** a tracked `.slp` containing pathological duplicate `PredictedInstance` rows for the same `(track.name, frame_idx)` (rare buggy-tracker output)
- **When** `series.get_tracked_tips()` is called
- **Then** both rows appear in the returned DataFrame — the dedup logic only addresses the predicted-with-user-correction pattern; same-type duplicates are preserved for downstream debugging visibility

#### Scenario: Two user-corrected Instance for same (track, frame) — NOT collapsed

- **Given** a tracked `.slp` containing pathological duplicate user-corrected `Instance` rows for the same `(track.name, frame_idx)` (rare proofreader pathology)
- **When** `series.get_tracked_tips()` is called
- **Then** both rows appear in the returned DataFrame
- **And** the existing `TrackedTipPipeline` per-track summary contract (tested at `tests/test_tracked_tip_pipeline.py` §6.14: `len(trajectories) == 2` while `n_frames_tracked == 1`) remains intact

### Requirement: `validate_tracked_slp` SHALL validate a .slp path

A new module-level function `validate_tracked_slp(slp_path: Union[str, Path]) -> None` MUST be added to `sleap_roots/series.py` and re-exported from `sleap_roots.__init__`. The function MUST:

- Open the .slp via `sio.load_slp`.
- Walk every labeled frame and every instance.
- Collect the frame indices of any frame containing an instance with `inst.track is None` or empty/None `inst.track.name`.
- Return `None` if all instances are tracked.
- Raise `ValueError` listing ALL offending frame indices in the message if any are untracked.

#### Scenario: Returns None on fully-tracked .slp

- **Given** a .slp where every instance has a non-empty `inst.track.name`
- **When** `validate_tracked_slp(path)` is called
- **Then** the return value is `None` and no exception is raised

#### Scenario: Raises listing all offending frame indices

- **Given** a .slp where instances at frame 2, frame 7, and frame 15 are untracked (others are tracked)
- **When** `validate_tracked_slp(path)` is called
- **Then** `ValueError` is raised
- **And** the error message contains the strings `"2"`, `"7"`, and `"15"` (or a sorted/comma-separated representation thereof)

### Requirement: `validate_series_for_tracked_tip` SHALL validate a Series instance

A new module-level function `validate_series_for_tracked_tip(series: Series, root_type: Optional[str] = None) -> None` MUST be added to `sleap_roots/series.py` and re-exported. Behavior:

- Resolve `root_type` (auto-detect from populated `<root_type>_path` when `None`); MUST raise `ValueError` on zero or >1 populated paths with the same wording as `Series.get_tracked_tips`.
- Verify the resolved `<root_type>_labels` is loadable and the skeleton has at least one node; raise `ValueError` otherwise.
- Call `validate_tracked_slp` on the resolved path; propagate any `ValueError` from there.

#### Scenario: Resolves root_type and validates

- **Given** a `Series.load(primary_path=tracked_slp_path)` (lateral and crown unset)
- **When** `validate_series_for_tracked_tip(series)` is called WITHOUT `root_type`
- **Then** the function returns `None` and no exception is raised

#### Scenario: Explicit root_type validates the chosen path

- **Given** a `Series.load(primary_path=tracked_slp_a, lateral_path=tracked_slp_b)` where BOTH .slp files are validly tracked
- **When** `validate_series_for_tracked_tip(series, root_type="lateral")` is called
- **Then** the function validates `tracked_slp_b` (the lateral path) and returns `None`
- **And** calling with `root_type="primary"` validates `tracked_slp_a` and returns `None`

### Requirement: `TrackedTipPipeline` SHALL define traits via DAG-A composition reusing existing trait functions

The `TrackedTipPipeline.traits` list MUST contain TraitDef nodes that reuse existing trait functions DIRECTLY as `fn` values (no wrapper module). The DAG MUST provide per-track input slicing so existing functions plug in unchanged. Specifically:

- A `track_first_xy` TraitDef with `fn=lambda xy: xy[0]` and `input_traits=["track_xy"]`.
- A `track_last_xy` TraitDef with `fn=lambda xy: xy[-1]` and `input_traits=["track_xy"]`.
- A `tip_displacement_net` TraitDef with `fn=sleap_roots.bases.get_base_tip_dist` (used directly, no wrapper) and `input_traits=["track_first_xy", "track_last_xy"]`.
- A `tip_trajectory_length` TraitDef with `fn=sleap_roots.lengths.get_root_lengths` (used directly, no wrapper) and `input_traits=["track_xy"]`.
- A `tracking_coverage` TraitDef with `fn=lambda nt, ntot: nt/ntot if ntot else np.nan` and `input_traits=["n_frames_tracked", "n_frames_total"]`.

No new trait module (e.g. `tip_kinematics.py`) SHALL be created. The pipeline file `sleap_roots/tracked_tip_pipeline.py` itself holds the small lambdas inline in TraitDef definitions.

#### Scenario: tip_displacement_net delegates to get_base_tip_dist

- **Given** a `TrackedTipPipeline` instance and a single track with a 3-4-5-triangle trajectory `[(0,0), (3,0), (3,4)]`
- **When** `compute_tracked_tip_traits(series)` is called
- **Then** the per-track summary row's `tip_displacement_net == 5.0` (Euclidean distance from `(0,0)` to `(3,4)`, computed via `get_base_tip_dist`)
- **And** `tip_trajectory_length == 7.0` (total path length 3 + 4)

#### Scenario: tracking_coverage uses inline lambda

- **Given** a track present in 3 of 5 total frames
- **When** the pipeline computes the per-track summary
- **Then** `tracking_coverage == 0.6`

#### Scenario: No `tip_kinematics` module is created

- **Given** the merged PR for this change
- **When** `find sleap_roots -name 'tip_kinematics*'` is executed
- **Then** no file matches (the module deliberately does not exist)

### Requirement: `compute_tracked_tip_traits` SHALL emit a structured per-series result dict

The `TrackedTipPipeline.compute_tracked_tip_traits(self, series, *, write_csv=False, write_json=False, output_dir=".", emit_trajectories=True, csv_summary_suffix=".tracked_tip_traits.csv", csv_trajectory_suffix=".tracked_tip_trajectories.csv", json_suffix=".tracked_tip_traits.json") -> Dict[str, Any]` method MUST return a dict with exactly these top-level keys:

- `schema_version: int` — value `1` (initial schema version).
- `pipeline: str` — value `"TrackedTipPipeline"`.
- `units: Dict[str, str]` — value `{"lengths": "pixels", "ratios": "dimensionless", "counts": "dimensionless", "time": "unspecified"}`.
- `series: str` — value of `series.series_name`.
- `sample_uid: str` — value of `series.sample_uid` (defaults to `series_name` when no kwarg/CSV per Workstream 1).
- `timepoint: Union[float, None]` — value of `series.timepoint` (NaN when no CSV; coerced to `null` in JSON via `_json_sanitize`).
- `tracks: List[Dict[str, Any]]` — one entry per `track_id` (see next requirement).
- `trajectories: List[Dict[str, Any]]` — one entry per `(track_id, frame)` (see next requirement). When `emit_trajectories=False`, this list MUST be `[]` (empty list — present for shape stability, NOT omitted from the dict).

#### Scenario: Result dict has all top-level keys

- **Given** a `TrackedTipPipeline` and a fully-tracked `Series` with 5 frames and 2 tracks
- **When** `compute_tracked_tip_traits(series)` is called
- **Then** the returned dict's `set(keys)` equals `{"schema_version", "pipeline", "units", "series", "sample_uid", "timepoint", "tracks", "trajectories"}`

#### Scenario: Schema version is 1 for the initial release

- **Given** the initial release of `TrackedTipPipeline`
- **When** any `compute_tracked_tip_traits` call is made
- **Then** `result["schema_version"] == 1`

#### Scenario: Units dict has exactly the expected keys

- **Given** any `compute_tracked_tip_traits` call
- **Then** `result["units"] == {"lengths": "pixels", "ratios": "dimensionless", "counts": "dimensionless", "time": "unspecified"}`

#### Scenario: emit_trajectories=False yields empty trajectories list

- **Given** any compute call with `emit_trajectories=False`
- **When** the result is inspected
- **Then** `result["trajectories"] == []` (empty list, not missing key)
- **And** `result["tracks"]` is unchanged from the `emit_trajectories=True` call

### Requirement: `tracks` and `trajectories` rows SHALL have specified column sets

Each entry in `result["tracks"]` MUST be a dict with exactly these keys: `track_id` (str), `n_frames_tracked` (int), `n_frames_total` (int), `tracking_coverage` (float), `tip_trajectory_length` (float), `tip_displacement_net` (float).

Each entry in `result["trajectories"]` MUST be a dict with exactly these keys: `track_id` (str), `frame` (int), `tip_x` (float), `tip_y` (float).

Top-level scalars (`series`, `sample_uid`, `timepoint`) MUST NOT be repeated in either per-row dict in JSON output. They MUST be repeated on every row in CSV output (PR #165's CSV/JSON asymmetry).

#### Scenario: Per-track summary row has exact key set

- **Given** any `compute_tracked_tip_traits` result
- **And** a track with `track_id="track_0"`
- **When** the row is selected from `result["tracks"]`
- **Then** `set(row.keys()) == {"track_id", "n_frames_tracked", "n_frames_total", "tracking_coverage", "tip_trajectory_length", "tip_displacement_net"}`
- **And** the row does NOT contain `series`, `sample_uid`, or `timepoint` keys

#### Scenario: Per-frame trajectory row has exact key set

- **Given** any `compute_tracked_tip_traits` result with `emit_trajectories=True`
- **When** a row is selected from `result["trajectories"]`
- **Then** `set(row.keys()) == {"track_id", "frame", "tip_x", "tip_y"}`
- **And** the row does NOT contain `series`, `sample_uid`, or `timepoint` keys

### Requirement: Single-frame edge case behavior is documented and asymmetric

For a track present in exactly one frame, the per-track summary row MUST satisfy:

- `n_frames_tracked == 1`.
- `tracking_coverage > 0` (specifically `1 / n_frames_total`).
- `tip_displacement_net == 0.0` — natural geometric result (xy[0] equals xy[-1]; `get_base_tip_dist` returns 0.0).
- `tip_trajectory_length` is `NaN` (numpy `np.nan`) — `get_root_lengths`'s vacuous-truth NaN guard on empty-segment arrays. This asymmetry with `tip_displacement_net` is **intentional** — accepted as the trade-off for using the existing function directly via DAG composition without a wrapper. Users who want either filtered or coerced apply `fillna(0.0)` post-hoc; users who want only "real trajectories" filter on `n_frames_tracked > 1`.

In JSON output, the `NaN` value MUST be serialized as `null` via `_json_sanitize` (existing helper from `trait_pipelines.py`).

#### Scenario: Single-frame track has zero displacement

- **Given** a synthetic .slp with one track that appears in exactly 1 of 10 frames
- **When** `compute_tracked_tip_traits(series)` is called
- **Then** the per-track summary row has `tip_displacement_net == 0.0`

#### Scenario: Single-frame track has NaN trajectory length

- **Given** the same .slp
- **When** the result is inspected
- **Then** `np.isnan(row["tip_trajectory_length"])` is `True`

#### Scenario: Single-frame trajectory length serializes to JSON null

- **Given** the same compute call with `write_json=True`
- **When** the JSON file is parsed
- **Then** the corresponding `tracks[i]["tip_trajectory_length"]` is JSON `null`

### Requirement: `TrackedTipPipeline` TraitDef `fn` values SHALL be picklable

Every `TraitDef` returned by `TrackedTipPipeline.define_traits()` MUST have a `fn` value that is picklable via the standard library `pickle` module. Inline lambdas (which are not picklable by default) MUST NOT be used — TraitDef functions MUST be either references to module-level functions or references to imported callables (e.g. `bases.get_base_tip_dist`).

Rationale: this is a precondition for any future `multiprocessing`-based parallelization of the trait DAG. The DAG is currently executed serially; a future change that distributes trait computation across processes (or uses `joblib.Parallel` with the loky backend) requires every trait function to round-trip through pickle. Inline lambdas in TraitDef definitions silently block that future refactor.

#### Scenario: TrackedTipPipeline traits list pickles cleanly

- **Given** a `TrackedTipPipeline()` instance
- **When** `pickle.dumps(pipeline.traits)` is called
- **Then** the call succeeds and returns a non-empty `bytes` object
- **And** `pickle.loads(...)` round-trips the trait list back to an equal-shape list of `TraitDef` objects with the same names and (callable) `fn` references

#### Scenario: Each TraitDef.fn value individually pickles cleanly

- **Given** a `TrackedTipPipeline()` instance
- **When** for each `trait_def` in `pipeline.traits`, `pickle.dumps(trait_def.fn)` is called
- **Then** every call succeeds and returns a non-empty `bytes` object

### Requirement: `tracking_coverage` SHALL be bounded to `[0.0, 1.0]` regardless of duplicate `(track_id, frame)` rows

`tracking_coverage` MUST be a float in `[0.0, 1.0]` for every track in the result. To enforce this bound, `n_frames_tracked` MUST equal the number of UNIQUE frame indices in which the track has an instance — NOT the raw count of instances. Pathological tracker output (e.g. merged tracks producing two instances with the same `track_id` in the same frame) MUST NOT inflate `tracking_coverage` above `1.0`.

Implementation: in `compute_tracked_tip_traits`'s per-track groupby, `n_frames_tracked = int(group["frame"].nunique())` (NOT `len(group)`). This deduplicates duplicate `(track_id, frame)` rows from the trajectory DataFrame before computing the coverage ratio.

Note: this requirement specifies the OUTPUT contract on `n_frames_tracked` and `tracking_coverage`. The trajectory CSV still records every row from `Series.get_tracked_tips` (one per tracked instance), so duplicate `(track_id, frame)` instances remain visible in the per-frame trajectory table for downstream debugging. Only the per-track summary row's `n_frames_tracked` and the derived `tracking_coverage` are deduplicated.

#### Scenario: Duplicate `(track_id, frame)` does not inflate tracking_coverage above 1.0

- **Given** a synthetic .slp with 1 frame containing 2 instances of the same `track_id="t"` (a buggy-tracker pathology — over-eager merger producing two coincident detections under one track id)
- **When** `compute_tracked_tip_traits(series)` is called
- **Then** the per-track summary row for `track_id="t"` has `tracking_coverage == 1.0` (`1 unique frame / 1 total frame`), NOT `2.0`
- **And** `n_frames_tracked == 1` (the unique-frame count)
- **And** `n_frames_total == 1`

#### Scenario: Duplicate instances appear in trajectory rows but not inflate summary

- **Given** the same .slp
- **When** the result is inspected
- **Then** `len(result["trajectories"])` reflects every tracked-instance row (including the duplicate — so `2` for a frame with two same-track instances)
- **And** `n_frames_tracked` in the summary row is the deduplicated count (`1`, not `2`)

### Requirement: Zero-track and zero-frame edge cases SHALL not crash

When the input `Series` has no tracked instances anywhere (e.g. all instances were untracked and validation was skipped, or the .slp is empty), `compute_tracked_tip_traits` MUST return a result dict with `result["tracks"] == []` and `result["trajectories"] == []`. No exception is raised, no division-by-zero on `tracking_coverage` (the lambda's `if ntot else np.nan` guard handles `n_frames_total == 0`).

This requirement is paired with the validate-on-input-required behavior: in the normal flow, `validate_series_for_tracked_tip` raises BEFORE the pipeline runs. The zero-track edge case is reachable only when the user explicitly bypasses validation OR when a .slp is empty (e.g. `sio.Labels(labeled_frames=[])`).

#### Scenario: Empty .slp produces empty result

- **Given** an empty `sio.Labels` saved to a .slp
- **When** the user constructs `Series` and calls `compute_tracked_tip_traits` (bypassing validation)
- **Then** `result["tracks"] == []` and `result["trajectories"] == []`
- **And** no exception is raised

### Requirement: CSV output SHALL emit two files per series

When `compute_tracked_tip_traits(series, write_csv=True, output_dir=...)` is called, the method MUST write:

- `<output_dir>/<series_name>.tracked_tip_traits.csv` — summary CSV. One row per `track_id`. Columns in order: `series, sample_uid, timepoint, track_id, n_frames_tracked, n_frames_total, tracking_coverage, tip_trajectory_length, tip_displacement_net`.
- `<output_dir>/<series_name>.tracked_tip_trajectories.csv` — trajectory CSV. One row per `(track_id, frame)`. Columns in order: `series, sample_uid, timepoint, track_id, frame, tip_x, tip_y`. **NOT written when `emit_trajectories=False`.**

The `series`, `sample_uid`, and `timepoint` values MUST be repeated on every row of both CSVs.

#### Scenario: CSV files are written with expected names

- **Given** a `Series` with `series_name="plate_001"` and a tmp output directory
- **When** `compute_tracked_tip_traits(series, write_csv=True, output_dir=tmp_path)` is called
- **Then** `tmp_path / "plate_001.tracked_tip_traits.csv"` exists
- **And** `tmp_path / "plate_001.tracked_tip_trajectories.csv"` exists

#### Scenario: Summary CSV column order is exact

- **Given** the summary CSV from a successful compute call
- **When** parsed via `pd.read_csv`
- **Then** `list(df.columns) == ["series", "sample_uid", "timepoint", "track_id", "n_frames_tracked", "n_frames_total", "tracking_coverage", "tip_trajectory_length", "tip_displacement_net"]`

#### Scenario: Trajectory CSV column order is exact

- **Given** the trajectory CSV from a successful compute call
- **When** parsed via `pd.read_csv`
- **Then** `list(df.columns) == ["series", "sample_uid", "timepoint", "track_id", "frame", "tip_x", "tip_y"]`

#### Scenario: emit_trajectories=False suppresses trajectory CSV

- **Given** a compute call with `emit_trajectories=False, write_csv=True`
- **When** the output directory is inspected
- **Then** the summary CSV exists
- **And** no trajectory CSV exists in `output_dir` for this series

#### Scenario: CSV repeats top-level scalars on every row

- **Given** a summary CSV from a 6-track series with `sample_uid="plate_001"` and `timepoint=0.0`
- **When** parsed
- **Then** every row has `series == "plate_001"` (or whatever `series_name` is), `sample_uid == "plate_001"`, `timepoint == 0.0`

### Requirement: JSON output SHALL emit a single file per series with both tables

When `compute_tracked_tip_traits(series, write_json=True, output_dir=...)` is called, the method MUST write `<output_dir>/<series_name>.tracked_tip_traits.json`. The JSON content MUST:

- Be valid RFC-8259 JSON.
- Have exactly the top-level keys `{"schema_version", "pipeline", "units", "series", "sample_uid", "timepoint", "tracks", "trajectories"}`.
- Carry NaN values as JSON `null` (via `_json_sanitize`).
- NOT repeat `series`, `sample_uid`, or `timepoint` inside `tracks[i]` or `trajectories[i]` rows.

#### Scenario: JSON file is written with expected name

- **Given** a `Series` with `series_name="plate_001"`
- **When** `compute_tracked_tip_traits(series, write_json=True, output_dir=tmp_path)` is called
- **Then** `tmp_path / "plate_001.tracked_tip_traits.json"` exists and is valid JSON

#### Scenario: JSON top-level scalars are present once

- **Given** the JSON from a 6-track series
- **When** parsed
- **Then** the top-level dict has `series`, `sample_uid`, `timepoint` keys
- **And** none of the `tracks[i]` or `trajectories[i]` row dicts contain those keys

#### Scenario: JSON validates as RFC-8259 with NaN sanitized

- **Given** any compute call where some scalars are `NaN`
- **When** the JSON file is read with `json.load(...)` (Python's stdlib parser using default settings)
- **Then** the call succeeds (no `JSONDecodeError`)
- **And** the NaN values appear as Python `None` (JSON `null`)

### Requirement: `compute_batch_tracked_tip_traits` SHALL concatenate per-series results

`TrackedTipPipeline.compute_batch_tracked_tip_traits(self, all_series, *, write_csv=False, write_json=False, output_dir=".", csv_summary_name="tracked_tip_batch_traits.csv", csv_trajectory_name="tracked_tip_batch_trajectories.csv", json_name="tracked_tip_batch_traits.json", emit_trajectories=True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[Dict[str, Any]]]` MUST:

- Walk `all_series` and call `compute_tracked_tip_traits(series)` per series.
- Concatenate per-series summary DataFrames into one DataFrame written to `<output_dir>/tracked_tip_batch_traits.csv` when `write_csv=True`.
- Concatenate per-series trajectory DataFrames into one DataFrame written to `<output_dir>/tracked_tip_batch_trajectories.csv` when `write_csv=True` AND `emit_trajectories=True`.
- Write the per-series result-dict list to `<output_dir>/tracked_tip_batch_traits.json` when `write_json=True` (the JSON content is a JSON array of per-series dicts, NOT a single combined dict).
- Return the concatenated summary DataFrame, the concatenated trajectory DataFrame (or `None` when `emit_trajectories=False`), and the list of per-series result dicts.

This MUST mirror the existing pattern in [trait_pipelines.py:3327](sleap_roots/trait_pipelines.py#L3327) (`compute_batch_plate_traits`).

#### Scenario: Batch summary CSV concatenates rows across series

- **Given** 3 synthetic `Series`, each with 2 tracked tracks
- **When** `compute_batch_tracked_tip_traits(all_series, write_csv=True)` is called
- **Then** `tracked_tip_batch_traits.csv` has 6 rows total (2 tracks × 3 series)

#### Scenario: Batch trajectory CSV concatenates rows across series

- **Given** the same 3 series, each with 5 frames × 2 fully-tracked tracks
- **When** the same call is made
- **Then** `tracked_tip_batch_trajectories.csv` has 30 rows total (10 per series × 3 series)

#### Scenario: Batch JSON is a list of per-series dicts

- **Given** the same call with `write_json=True`
- **When** the JSON is parsed
- **Then** the parsed value is a Python `list` of length 3
- **And** each list element is shaped like a single `compute_tracked_tip_traits` output dict (same key set)

#### Scenario: Empty input list yields empty outputs

- **Given** `compute_batch_tracked_tip_traits([])` with all write flags `True`
- **When** the call returns
- **Then** the returned summary DataFrame is empty
- **And** the JSON file (if written) parses to `[]` (empty list)
- **And** no exception is raised

#### Scenario: emit_trajectories=False suppresses batch trajectory CSV

- **Given** a batch call with `emit_trajectories=False, write_csv=True`
- **When** the output directory is inspected
- **Then** `tracked_tip_batch_traits.csv` exists
- **And** `tracked_tip_batch_trajectories.csv` does NOT exist

### Requirement: Real-fixture integration tests SHALL ship with the change

A real-data fixture MUST be committed under `tests/data/circumnutation_plate/` containing:

- `plate_001_greyscale.tracked.slp` — 184 KB tracked predictions, sliced from the source data on `Z:\users\eberrigan\circumnutation\...\run_20250827_091833\plate_001_greyscale.tracked.slp` (provenance documented in README). Committed via Git LFS.
- `fixture_metadata.csv` — synthesized plate-level metadata, single row, `plant_qr_code="plate_001"`, columns `[plant_qr_code, genotype, treatment, number_of_plants_cylinder, timepoint]`.
- `README.md` — documentation per the standard template (issue #168), covering provenance, conversion steps, the `plant_qr_code` legacy-name caveat, and the deferral of per-frame metadata to #186.

Two integration tests MUST exist:

1. WITH `csv_path`: asserts `result["sample_uid"] == "plate_001"`, `result["timepoint"] == 0.0`, `len(result["tracks"]) == 6`, `len(result["trajectories"]) == 1866`, every `tracking_coverage ∈ [0, 1]`.
2. WITHOUT `csv_path`: same series, same pipeline call, asserts `result["sample_uid"]` defaults to `series_name`, `np.isnan(result["timepoint"])`, all numeric trait values match path 1.

#### Scenario: Real fixture loads and produces expected track count

- **Given** the committed fixture .slp at `tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp`
- **When** loaded via `Series.load(primary_path=..., csv_path=fixture_metadata_path, sample_uid="plate_001")` and processed by `TrackedTipPipeline().compute_tracked_tip_traits(series)`
- **Then** `len(result["tracks"]) == 6`
- **And** every `track["tracking_coverage"]` is in `[0.0, 1.0]`
- **And** `len(result["trajectories"]) == 1866`

#### Scenario: No-CSV path defaults sample_uid and timepoint

- **Given** the same fixture .slp loaded WITHOUT `csv_path`
- **When** `compute_tracked_tip_traits(series)` is called
- **Then** `result["sample_uid"] == series.series_name`
- **And** `np.isnan(result["timepoint"])`
- **And** every numeric trait value matches the corresponding row in the WITH-CSV scenario (only the metadata changes, not the geometry)

#### Scenario: Track names are the brainstorm-verified track_0 .. track_5

- **Given** the real fixture
- **When** `compute_tracked_tip_traits(series)` is called
- **Then** `set(row["track_id"] for row in result["tracks"]) == {"track_0", "track_1", "track_2", "track_3", "track_4", "track_5"}`

