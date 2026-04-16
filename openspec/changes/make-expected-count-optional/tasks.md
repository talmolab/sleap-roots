# Tasks: Make `expected_count` Optional

## Pre-existing test coverage map (DO NOT add duplicate tests)

Requirement 2 in the spec captures existing numeric semantics as a regression baseline. Four of its scenarios are already covered by existing tests in [tests/test_points.py](../../../tests/test_points.py):

| Spec scenario | Existing test | Lines |
|---|---|---|
| `np.nan` skips filtering | `test_filter_plants_with_unexpected_ct_nan_expected_count` | 738-747 |
| Matching numeric passes through | `test_filter_plants_with_unexpected_ct_valid_input_matching_count` | 714-723 |
| Mismatching numeric produces empty arrays | `test_filter_plants_with_unexpected_ct_valid_input_non_matching_count` | 726-735 |
| Non-numeric, non-None raises ValueError | `test_filter_plants_with_unexpected_ct_incorrect_input_types` | 750-767 |

These tests MUST continue to pass unchanged after the implementation in section 2. Task 3.1 runs the full `test_points.py` file to verify this explicitly.

## 1. Write failing tests first (TDD — `None` contract)

- [x] 1.1 Add `test_filter_plants_with_unexpected_ct_none_expected_count` in `tests/test_points.py`. Create primary_pts (5, 10, 2) and lateral_pts (5, 10, 2) via `np.random.rand`, call `filter_plants_with_unexpected_ct(primary_pts, lateral_pts, expected_count=None)`, and assert both output arrays equal the inputs via `np.array_equal`. **Expected failure before implementation**: `ValueError("expected_count must be a numeric type.")` raised by the existing numeric subdtype check on `type(None)` at [points.py:392](../../../sleap_roots/points.py#L392).
- [x] 1.2 Add `test_filter_plants_with_unexpected_ct_none_with_mismatched_shapes` in `tests/test_points.py`. Create primary_pts shape (5, 10, 2) and lateral_pts shape (3, 10, 2), call with `expected_count=None`, and assert the returned primary has shape (5, 10, 2) and the returned lateral has shape (3, 10, 2). Verifies `None` is a genuine skip (not a "match required against primary count" shortcut). **Expected failure before implementation**: same `ValueError` as 1.1.
- [x] 1.3 Add `test_filter_plants_with_unexpected_ct_default_expected_count` in `tests/test_points.py`. Call `filter_plants_with_unexpected_ct(primary_pts, lateral_pts)` with exactly two positional args, relying on the new default, and assert pass-through. Verifies the default value binding of `None`. **Expected failure before implementation**: `TypeError` for missing positional argument `expected_count`.

## 1b. Edge-case tests that lock existing numeric semantics (pin Req 2 scenarios that have no pre-existing test coverage)

- [x] 1.4 Add `test_filter_plants_with_unexpected_ct_zero_count_empty_primary` in `tests/test_points.py`. Create `primary_pts = np.empty((0, 10, 2))`, `lateral_pts = np.empty((0, 10, 2))`, call `filter_plants_with_unexpected_ct(primary_pts, lateral_pts, 0)`, and assert both returned shapes are `(0, 10, 2)` (pass-through, because `round(0) == 0 == len(primary_pts)`). This pins the `expected_count=0` scenario added to Req 2. **Expected state before implementation**: test passes today (preexisting behavior). Written as a regression guard, not TDD.
- [x] 1.5 Add `test_filter_plants_with_unexpected_ct_zero_count_nonempty_primary` in `tests/test_points.py`. Create `primary_pts = np.random.rand(3, 10, 2)`, `lateral_pts = np.random.rand(3, 10, 2)`, call with `expected_count=0`, and assert both returned arrays are shape `(0, 10, 2)` (mismatch branch). **Expected state before implementation**: passes today.
- [x] 1.6 Add `test_filter_plants_with_unexpected_ct_half_integer_expected_count` in `tests/test_points.py`. Two assertions in one test:
  - With `primary_pts = np.random.rand(2, 10, 2)`, `lateral_pts = np.random.rand(2, 10, 2)`, `expected_count=2.5`, assert pass-through (because `round(2.5) == 2 == len(primary_pts)` under Python banker's rounding).
  - With `primary_pts = np.random.rand(4, 10, 2)`, `lateral_pts = np.random.rand(4, 10, 2)`, `expected_count=3.5`, assert pass-through (because `round(3.5) == 4`).

  Pins the banker's-rounding rule against a future refactor that swaps `round()` for `int()` truncation or `math.floor(x + 0.5)`. **Expected state before implementation**: passes today.

## 1c. Regression guard for the NaN integration path (pipeline behavior)

- [x] 1.7 Add `test_multiple_dicot_pipeline_without_csv` in `tests/test_trait_pipelines.py`. Use the existing fixtures `multiple_arabidopsis_11do_primary_slp` and `multiple_arabidopsis_11do_lateral_slp` from [tests/fixtures/data.py:179-187](../../../tests/fixtures/data.py#L179-L187) — do **NOT** use the `multiple_arabidopsis_11do_csv` fixture. Call `Series.load(series_name="997_1", primary_path=..., lateral_path=...)` without `csv_path`. Instantiate `pipeline = MultipleDicotPipeline()`. Then:
  1. Assert `np.isnan(series.expected_count)` is `True` (documents the precondition; fails loudly if `Series.expected_count` is later changed to return `None`).
  2. Call `result = pipeline.compute_multiple_dicots_traits(series)`. Assert the call does not raise.
  3. Assert `set(result.keys()) == {"series", "group", "qc_fail", "traits", "summary_stats"}` — the full five-key set from [trait_pipelines.py:414-420](../../../sleap_roots/trait_pipelines.py#L414-L420). Include a one-line comment in the test: `# Strict equality is intentional: new keys require an explicit spec update.` Do NOT assert the *value* of `result["group"]` — it will be the literal string `"nan"` when CSV is absent because `series.group` returns `np.nan` and [trait_pipelines.py:416](../../../sleap_roots/trait_pipelines.py#L416) coerces via `str(...)`. In contrast, `result["qc_fail"]` is stored as a raw float (no `str()` coercion) at [trait_pipelines.py:417](../../../sleap_roots/trait_pipelines.py#L417), so it will be a Python `float('nan')`, not the string `"nan"` — if you need to assert its value use `math.isnan(result["qc_fail"])`, not `result["qc_fail"] == "nan"`. Just checking key membership is the safest option and is what the spec requires.
  4. **Because `primary_pts_expected_plant_ct` is declared `include_in_csv=False`** at [trait_pipelines.py:2647-2654](../../../sleap_roots/trait_pipelines.py#L2647-L2654) and is discarded after each frame inside `compute_multiple_dicots_traits`, the per-frame "at least one frame has plants" assertion must call the pipeline directly frame-by-frame. Write a loop:

     ```python
     max_n_instances = 0
     for frame_idx in range(len(series)):
         frame_traits = pipeline.compute_frame_traits(
             pipeline.get_initial_frame_traits(series, frame_idx)
         )
         max_n_instances = max(
             max_n_instances,
             int(frame_traits["primary_pts_expected_plant_ct"].shape[0]),
         )
     assert max_n_instances >= 1, (
         "Pipeline with no CSV should not strip all plants from all frames"
     )
     ```

     Iterating over all frames (instead of hard-coding frame 0) guarantees robustness if the chosen sample's frame 0 happens to have all-NaN predictions.

  **Expected state before implementation**: this test should already pass today because the NaN skip path is already wired. It is a regression guard — not TDD — for the downstream dependency that issue #126 will rely on. If it unexpectedly fails today, stop and triage before continuing.

- [x] 1.8 Run `uv run pytest tests/test_points.py::test_filter_plants_with_unexpected_ct_none_expected_count tests/test_points.py::test_filter_plants_with_unexpected_ct_none_with_mismatched_shapes tests/test_points.py::test_filter_plants_with_unexpected_ct_default_expected_count -x -v`. Required outcome: **all three fail** with `ValueError`/`TypeError` as documented above. This is the TDD discipline gate for Section 2.
- [x] 1.9 Run `uv run pytest tests/test_points.py::test_filter_plants_with_unexpected_ct_zero_count_empty_primary tests/test_points.py::test_filter_plants_with_unexpected_ct_zero_count_nonempty_primary tests/test_points.py::test_filter_plants_with_unexpected_ct_half_integer_expected_count tests/test_trait_pipelines.py::test_multiple_dicot_pipeline_without_csv -x -v`. Required outcome: **all four pass** today (these are regression guards, not failing-first TDD). If any fail, stop and investigate.
- [x] 1.10 **File pre-existing bug as a new GitHub issue**: `tests/test_trait_pipelines.py` currently defines `test_multiple_dicot_pipeline` twice (lines 1389 and 1439). Python's module-level name collision means pytest only runs the second definition; the first is dead code. This is unrelated to issue #125 but was discovered during openspec review. File a new issue (`gh issue create`) titled "Duplicate `test_multiple_dicot_pipeline` in tests/test_trait_pipelines.py — first definition is dead code" before committing, and reference the new issue number in the commit message for task 1.7. **Recorded new issue: #154** (https://github.com/talmolab/sleap-roots/issues/154) Do **not** fix this in this PR — out of scope.

## 2. Implement the minimal signature change

- [x] 2.1 Edit `sleap_roots/points.py::filter_plants_with_unexpected_ct`:
  - Ensure `Optional` is imported from `typing` (add to the existing `from typing import ...` import if missing).
  - Change the signature to `expected_count: Optional[float] = None`.
  - Keep the `primary_pts`/`lateral_pts` `np.ndarray` type check as the first guard.
  - Add an early-return branch immediately after the array check: `if expected_count is None: return primary_pts, lateral_pts`. This must run **before** the `np.issubdtype(type(expected_count), np.number)` check (which would otherwise reject `None`).
  - Leave the numeric subdtype check, the NaN skip, the round + length comparison, and the empty-array mismatch branch unchanged.
  - Rewrite the docstring `Args:` entry for `expected_count` to read **verbatim** as the text below (prose wording may be tightened for line length by black, but the semantic content must match):

    > `expected_count: Optional expected number of primary roots. If `None` or `np.nan`, no count-based filtering is applied and both input arrays are returned unchanged. `None` conceptually indicates "no expected count was configured"; `np.nan` conceptually indicates "configured but missing from metadata" (e.g. CSV entry absent). The cylinder pipeline treats both identically because it has no way to distinguish them downstream — the distinction is a deferred concern for future plate-pipeline work (issue #126). If a finite number, it is rounded via Python's built-in `round()` (banker's rounding, half-to-even: `round(2.5) == 2`, `round(3.5) == 4`); if `len(primary_pts)` does not match the rounded value, both arrays are replaced with empty `(0, n_nodes, 2)` placeholders so `MultipleDicotPipeline`'s per-series aggregation drops the frame.

  - Update the `Raises:` entry to state that a `ValueError` is raised when `expected_count` is neither `None` nor a numeric type, and when `primary_pts`/`lateral_pts` are not numpy arrays.
- [x] 2.2 Re-run `uv run pytest tests/test_points.py::test_filter_plants_with_unexpected_ct_none_expected_count tests/test_points.py::test_filter_plants_with_unexpected_ct_none_with_mismatched_shapes tests/test_points.py::test_filter_plants_with_unexpected_ct_default_expected_count -x -v`. All three new TDD tests (1.1–1.3) must now pass. Task 1.9's regression tests (1.4–1.7) must still pass.

## 3. Regression, style, and validation

- [x] 3.1 Run `uv run pytest tests/test_points.py -v`. Verify all tests pass, and **explicitly confirm** that the four pre-existing tests from the coverage map above still pass unchanged:
  - `test_filter_plants_with_unexpected_ct_valid_input_matching_count`
  - `test_filter_plants_with_unexpected_ct_valid_input_non_matching_count`
  - `test_filter_plants_with_unexpected_ct_nan_expected_count`
  - `test_filter_plants_with_unexpected_ct_incorrect_input_types` (specifically the string-input case at lines 763-767, which must continue to raise `ValueError` because the string still fails the numeric subdtype check).
- [x] 3.2 Run `uv run pytest tests/test_trait_pipelines.py -v` and verify no regression in the existing `MultipleDicotPipeline` tests.
- [x] 3.3 Run `uv run pytest tests/ -x` and verify the full suite passes.
- [x] 3.4 Run `uv run black --check sleap_roots/points.py tests/test_points.py tests/test_trait_pipelines.py` and verify formatting is clean.
- [x] 3.5 Run `uv run pydocstyle --convention=google sleap_roots/points.py` and verify the updated docstring passes.
- [x] 3.6 Run `openspec validate make-expected-count-optional --strict` and verify clean.
- [x] 3.7 **Post the scope-narrowing comment on issue #125** — this is the mechanical gate for the maintainer-authorized scope split documented in proposal.md §Why. Run:

  ```bash
  gh issue comment 125 --body "$(cat <<'EOF'
  ## Scope split (session decision, 2026-04-15)

  Per discussion with @eberrigan, this issue is being narrowed to cover **only** the `Optional[float] = None` signature change for `filter_plants_with_unexpected_ct`. The remaining acceptance criteria are transferred to #126 (`MultipleDicotPlatePipeline`):

  - `count_mismatch: True` flag on mismatch → transferred to #126
  - `count_validated: False` on the None/skip path → transferred to #126 (already covered in #126's "Expected Count Behavior (#125)" section)
  - `count_validated: True` on the match path → transferred to #126 — **please extend #126's "Expected Count Behavior" section to cover this**
  - `count_validated: True` on the mismatch path → transferred to #126 — **please extend #126's "Expected Count Behavior" section to cover this**
  - Warning log on mismatch → transferred to #126
  - "Include all detected plants on mismatch" → transferred to #126

  **Reasoning**: `MultipleDicotPipeline` serves ~72-frame cylinder series where dropping a count-mismatched frame from aggregation is the right response — there are plenty of other frames and summary stats stay clean. Plates have one frame per timepoint, so "keep all plants + flag mismatch" semantics are load-bearing for plates and harmful for cylinders. Keeping the cylinder function strict and letting the plate pipeline (issue #126) choose its own strategy is cleaner than coupling two pipelines through one function's behavior.

  OpenSpec proposal: `openspec/changes/make-expected-count-optional/` (PR link TBD).
  EOF
  )"
  ```

  Then verify the comment was posted successfully:

  ```bash
  gh issue view 125 --comments | grep -q "Scope split (session decision, 2026-04-15)" && echo "Comment posted" || { echo "COMMENT MISSING — fix before merge"; exit 1; }
  ```

  This task MUST be completed before the PR is marked ready for review. Without it, the maintainer-authorized scope narrowing has no record in the GitHub issue tracker and future readers cannot audit the decision trail.

## Dependencies and sequencing

- **Section 2 depends on Section 1** (TDD + regression guards): all tests in 1.1–1.7 must be written before any edit in Section 2.
- **Task 1.8** is the TDD discipline gate for the `None` contract tests (1.1–1.3). They MUST fail before implementation with the documented `ValueError`/`TypeError` patterns; anything else means the tests don't exercise the correct pre-change contract and must be fixed before proceeding.
- **Task 1.9** verifies the regression-guard tests (1.4–1.7) pass today. If any fail today, stop and triage — they are expressing pre-existing behavior that the implementation must not break.
- **Task 1.10** (file the duplicate-function-name issue) must complete before any commit in the `feature/optional-expected-count-125` branch so the commit message can reference the new issue number.
- Section 3 depends on Section 2.
- Within each subsection, individual tasks are order-independent.

### Commit safety note

Sections 1 and 2 leave the test suite **red** between the `None`-contract tests landing (1.1–1.3) and the implementation landing (2.1). This is tolerable on a local feature branch but will show red CI on every intermediate push. Recommended workflow: develop sections 1 and 2 locally, then push them as a single squashed commit OR as consecutive commits within a single push to minimize the sustained red CI window. The repo's `.github/workflows/ci.yml` does not enforce green-on-every-commit, so this is cosmetic, not blocking.

## Notes

- The `MultipleDicotPipeline` TraitDef wiring at [trait_pipelines.py:2633-2645](../../../sleap_roots/trait_pipelines.py#L2633-L2645) is **not** touched by this change. No TraitDef rename, no new TraitDef, no output-shape change.
- No changes to `Series.expected_count` — it continues to return `np.nan` for missing CSV. If issue #126 later needs `Series.expected_count` to return `None`, that will be a separate proposal.
- No changes to existing tests' assertions about the `(0, n_nodes, 2)` mismatch output — cylinder pipeline semantics remain strict-drop.
- The duplicate `test_multiple_dicot_pipeline` function name at [tests/test_trait_pipelines.py:1389](../../../tests/test_trait_pipelines.py#L1389) and [:1439](../../../tests/test_trait_pipelines.py#L1439) is a pre-existing bug surfaced by openspec review. It is filed as a separate issue under task 1.10 and NOT fixed in this PR.
