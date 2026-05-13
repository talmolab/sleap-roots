# Spec delta — `tracked-tip-pipeline` (PR #2 scope expansion)

This delta extends the canonical `tracked-tip-pipeline` capability spec with proofread-`.slp` dedup behavior on `Series.get_tracked_tips`. The fix is needed by Tier 0 (PR #2) because the original `get_tracked_tips` (PR #190) returns BOTH the `PredictedInstance` and the user-corrected `Instance` for every frame the user proofread, producing duplicate `(track_id, frame_idx)` rows that propagate `Δframe = 0` divides into every downstream velocity-bearing trait in `sleap_roots.circumnutation.kinematics.compute`. The dedup convention follows `docs/circumnutation/preliminary_results_2026-05-07.md` §3.1: *"Where an instance_type=0 (user-corrected) and instance_type=1 (predicted) instance exist for the same frame and track, the user-corrected value takes precedence."*

## MODIFIED Requirements

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
