# Proposal: Make `expected_count` Optional

**Change ID:** `make-expected-count-optional`
**Status:** PROPOSED
**Issue:** https://github.com/talmolab/sleap-roots/issues/125
**Related (downstream):** https://github.com/talmolab/sleap-roots/issues/126 (`MultipleDicotPlatePipeline`)

## Why

`filter_plants_with_unexpected_ct()` in `sleap_roots/points.py` rejects `None` today. Its type check at [points.py:392](../../../sleap_roots/points.py#L392) — `if not np.issubdtype(type(expected_count), np.number)` — raises `ValueError` when given `None`. Callers must pass `np.nan` as a sentinel to mean "no expected count". This works today because `Series.expected_count` happens to return `np.nan` when the CSV is missing ([series.py:170](../../../sleap_roots/series.py#L170)), but:

1. It is not Pythonic. A truly optional numeric parameter should accept `None`.
2. Issue #126 (`MultipleDicotPlatePipeline`) will call this function from contexts where `None` is the natural "unknown" value. Blocking that with `ValueError` forces every future caller to shim `NaN`.
3. The current signature advertises `expected_count: float` with no default — the NaN sentinel convention is undocumented at the signature level.

Making `expected_count` a true `Optional[float] = None` motivates a cleaner API contract for issue #126 without disturbing the `MultipleDicotPipeline` cylinder workflow's filter-on-mismatch behavior, which is intentionally strict (~72-frame cylinder series can afford to drop mismatched frames; plates cannot, but that is #126's concern).

### Scope narrowing relative to the current issue #125 body

**Authorization.** The current text of issue #125 (titled *"Updated Spec: Optional Expected Count for Plates"*) lists `count_mismatch: True` / `count_validated` flagging and "include all plants on mismatch" as Required Changes and Acceptance Criteria. During the session that produced this proposal, the maintainer (@eberrigan) and the implementing agent reached an explicit verbal decision to **narrow the scope of issue #125 to the signature change only**, and to migrate the flagging work to issue #126 (`MultipleDicotPlatePipeline`). The reasoning recorded in that discussion:

- **Cylinders and plates have different mismatch semantics.** `MultipleDicotPipeline` serves ~72-frame cylinder series where the "right" response to a count mismatch on a single frame is to drop that frame from per-series aggregation — there are plenty of other frames and the resulting summary stats stay clean. Plates have one image per timepoint; dropping a mismatched frame means losing the entire series. "Include all plants with a `count_mismatch` flag" is therefore load-bearing for plates but harmful for cylinders.
- **`MultipleDicotPlatePipeline` (issue #126) is a new pipeline class with its own `define_traits()`.** It is the natural home for plate-specific "keep all plants and flag mismatch" semantics. Issue #126's current body already includes an *"Expected Count Behavior (#125)"* section that picks up the `count_mismatch` flag and `expected_count` optional-path requirements.
- **Trying to bake both semantics into `filter_plants_with_unexpected_ct` would either break cylinder tests or require a `strict` kwarg that couples two pipelines through one function.** Keeping the cylinder function strict and letting the plate pipeline choose its own strategy (either skip the count-filter TraitDef entirely, or call a new flagging function) is cleaner.

**Follow-up commitment.** Before this proposal is merged, a comment will be posted on issue #125 summarizing this scope split, explicitly transferring the `count_mismatch`/`count_validated` flagging and "include all plants on mismatch" acceptance criteria to issue #126. This records the out-of-band decision in GitHub so future readers see the scope split in the issue tracker, not only in this proposal.

## What Changes

1. **`sleap_roots/points.py` — `filter_plants_with_unexpected_ct`**
   - Change signature from `expected_count: float` to `expected_count: Optional[float] = None`.
   - Add an early-return branch: when `expected_count is None`, return both arrays unchanged — after the `np.ndarray` argument validation but before the numeric subdtype check (which would reject `type(None)`).
   - Leave the existing NaN-skip branch (`if not np.isnan(expected_count):`) unchanged. `None` and `NaN` are now equivalent "skip filtering" signals.
   - Leave the match branch (pass-through) unchanged.
   - Leave the mismatch branch (return empty `(0, n_nodes, 2)` arrays) unchanged — `MultipleDicotPipeline` relies on this to drop frames from per-series aggregation.
   - Update the docstring: document `expected_count` as `Optional[float] = None`, state explicitly that `None` and `NaN` both mean "skip filter", and describe the three numeric paths (match, mismatch, NaN).
   - Non-numeric, non-`None` values (e.g. strings) continue to raise `ValueError`.

2. **`sleap_roots/trait_pipelines.py` — `MultipleDicotPipeline`**
   - **No code change.** The `expected_plant_ct` initial trait at [trait_pipelines.py:2695](../../../sleap_roots/trait_pipelines.py#L2695) is still sourced from `Series.expected_count`, which returns `np.nan` on missing CSV, and the broadened filter still treats NaN as "skip". This change only widens the function contract so that issue #126 can pass `None`.

3. **Tests — `tests/test_points.py`**
   - Add `test_filter_plants_with_unexpected_ct_none_expected_count`: passing `None` skips filter.
   - Add `test_filter_plants_with_unexpected_ct_none_with_mismatched_shapes`: passing `None` when primary/lateral array instance counts differ still returns the full arrays (verifies `None` is not a stealth "match-required" path).
   - Add `test_filter_plants_with_unexpected_ct_default_expected_count`: calling with only two positional arguments (relying on the new default) passes through.
   - Keep existing tests (matching count, mismatching count, NaN, invalid string type) unchanged.

4. **Tests — `tests/test_trait_pipelines.py`**
   - Add `test_multiple_dicot_pipeline_without_csv`: use the existing `multiple_arabidopsis_11do_primary_slp` and `multiple_arabidopsis_11do_lateral_slp` fixtures from [tests/fixtures/data.py:179-187](../../../tests/fixtures/data.py#L179-L187) (do **not** pass the `multiple_arabidopsis_11do_csv` fixture). Call `Series.load(series_name="997_1", primary_path=..., lateral_path=...)` with no `csv_path` kwarg so `Series.expected_count` returns `np.nan`. Instantiate `pipeline = MultipleDicotPipeline()`. Then assert in order:
     1. `np.isnan(series.expected_count)` is `True` (documents the precondition).
     2. `result = pipeline.compute_multiple_dicots_traits(series)` completes without raising.
     3. `set(result.keys()) == {"series", "group", "qc_fail", "traits", "summary_stats"}` — note this includes `qc_fail`, which the pipeline populates from `series.qc_fail` at [trait_pipelines.py:417](../../../sleap_roots/trait_pipelines.py#L417).
     4. **Because `primary_pts_expected_plant_ct` is an `include_in_csv=False` per-frame TraitDef** at [trait_pipelines.py:2647-2654](../../../sleap_roots/trait_pipelines.py#L2647-L2654) and is discarded after each frame inside `compute_multiple_dicots_traits`, the per-frame non-empty assertion must invoke the pipeline directly frame-by-frame: `for frame_idx in range(len(series)): frame_traits = pipeline.compute_frame_traits(pipeline.get_initial_frame_traits(series, frame_idx))`. Collect the `frame_traits["primary_pts_expected_plant_ct"].shape[0]` values across all frames and assert `max(...) >= 1`. Looping across all frames (rather than hard-coding `frame_idx == 0`) avoids flakiness from per-sample labeling drift where a specific frame has all-NaN predictions.

## What is explicitly NOT in scope

- **No `count_mismatch: True` flagging.** See the "Scope narrowing" section in `## Why` above — this was the result of an explicit maintainer decision in the session that produced this proposal, with the work transferred to issue #126.
- **No `count_validated` flag** either (also part of issue #125's original acceptance criteria) — same reasoning, same transfer target.
- **No warning log on mismatch** (also part of issue #125's original acceptance criteria) — same reasoning, same transfer target.
- **No change to `Series.expected_count`.** It still returns `np.nan` for missing CSV and for series not found in the CSV. Promoting it to `Optional[int]` or distinguishing "CSV absent" from "CSV present but series missing" is a separate concern with wider blast radius (Series tests, batch loaders, downstream CSV readers). If issue #126 needs that distinction, it will propose it separately.
- **No rename of `filter_plants_with_unexpected_ct`.** The name still reflects its cylinder-pipeline role (drop mismatched frames from aggregation).
- **No changes to `filter_primary_roots_with_unexpected_count`** — a sibling function at [points.py:408](../../../sleap_roots/points.py#L408) that is not called by `MultipleDicotPipeline` and is not in scope for issue #125. **Contract divergence note:** after this change, the two functions' type contracts diverge — `filter_plants_with_unexpected_ct` accepts `None`, `filter_primary_roots_with_unexpected_count` still rejects it. Documented here so a future reader doesn't assume symmetry.
- **Spec `Requirement 2` is a regression baseline, not new behavior.** Because the `multiple-dicot-pipeline` capability is being created fresh (there was no prior spec file to MODIFY), pre-existing numeric semantics are expressed as ADDED requirements alongside the new `None` requirement. They are verbatim captures of today's behavior; no code change is proposed for them. This is called out inside the spec itself.

## Impact

- **Affected specs:** `multiple-dicot-pipeline` (new capability).
- **Affected code:** `sleap_roots/points.py` (one signature change + one new branch + docstring), `tests/test_points.py` (3 new unit tests), `tests/test_trait_pipelines.py` (1 new integration test).
- **Risk:** Very low. The only behavioral change is broadening the accepted input domain to include `None`; all existing inputs (numeric, NaN, invalid strings) behave identically. No existing caller passes `None`, so no regression surface.
- **Backward compatibility:** Fully backward compatible.
- **Reproducibility:** No effect on trait values for any existing workflow. The new `None` path produces the same output arrays as the existing NaN path.
- **Enables #126 API cleanliness:** Issue #126 can call `filter_plants_with_unexpected_ct(..., expected_count=None)` without working around a `ValueError`, or construct its own pipeline that skips the count-filter TraitDef entirely. This is a cleanup motivated by #126, not a hard dependency — #126 could proceed today by passing `np.nan` — so this is not framed as "unblocking" #126.
