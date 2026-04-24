# Metadata Layer (Workstream 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `Series.sample_uid` + `Series.timepoint` + `Series.get_metadata()` generic CSV accessor + `sleap_roots/metadata.py` CSV builder helpers, and emit `sample_uid` / `timepoint` in `MultipleDicotPlatePipeline` output. Prerequisite for Workstreams 2 and 3 of the 2026-04-23 timelapse design.

**Architecture:** Extend the existing CSV-metadata pattern on `Series` (today: `expected_count`, `group`, `qc_fail`) by (a) adding two new role-specific properties, (b) centralizing the lookup logic in a generic `get_metadata(column, plant_id=None)` method that existing properties become thin wrappers around, (c) adding a small `metadata.py` module with CSV-builder utilities, and (d) threading the two new properties into pipeline output rows. Backward compatible: CSV lookup key stays `plant_qr_code`; rename of CSV column names is deferred to #163.

**Tech Stack:** Python 3.11, `attrs` dataclasses, `pandas` for CSV I/O, `pytest` for tests, `openspec` for spec proposal, `uv run` for all test/lint invocations.

**Issue:** #169
**Source design doc:** `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` (Workstream 1 section). Design PR #171.

---

## Phase 0: Setup

### Task 0.1: Verify branch and working tree

- [ ] **Step 1: Confirm you are NOT on main**

Run: `git branch --show-current`
Expected: any branch except `main`. If on `main`, create a feature branch:

```bash
git checkout -b feature/metadata-layer-workstream-1
```

- [ ] **Step 2: Confirm clean working tree**

Run: `git status --short`
Expected: empty (or only `.claude/settings.local.json` which is fine).

- [ ] **Step 3: Confirm design-docs branch is present locally for reference**

Run: `git log --all --oneline | grep "timelapse-design-docs\|add design spec for timelapse" | head -3`
Expected: at least one commit referencing the Workstream 1 design.

## File structure

Files this plan creates or modifies:

| Path | Action | Responsibility |
|---|---|---|
| `openspec/changes/add-sample-uid-timepoint-metadata/proposal.md` | Create | Why/what/impact for the metadata layer |
| `openspec/changes/add-sample-uid-timepoint-metadata/design.md` | Create | OpenSpec-specific design notes; references the source design doc |
| `openspec/changes/add-sample-uid-timepoint-metadata/tasks.md` | Create | TDD task breakdown + follow-up-issue discipline |
| `openspec/changes/add-sample-uid-timepoint-metadata/specs/series-metadata/spec.md` | Create | ADDED Requirements for the new capability |
| `sleap_roots/series.py` | Modify | New `sample_uid` attrs field + `Series.load(sample_uid=...)` kwarg + `get_metadata()` method + `timepoint` property + refactor existing properties to use `get_metadata()` |
| `sleap_roots/metadata.py` | Create | `build_metadata_csv()` + `infer_timepoints_from_filenames()` helpers |
| `sleap_roots/__init__.py` | Modify | Re-export the two helper functions from the top-level namespace |
| `sleap_roots/trait_pipelines.py` | Modify | `MultipleDicotPlatePipeline` output emits `sample_uid` + `timepoint` at top level and in per-plant rows / CSV columns |
| `tests/test_series.py` | Modify | Tests for `get_metadata`, `sample_uid`, `timepoint` + regression coverage for existing properties (now thin wrappers) |
| `tests/test_metadata.py` | Create | Tests for `build_metadata_csv` and `infer_timepoints_from_filenames` |
| `tests/test_multiple_dicot_plate_pipeline.py` | Modify | Update `_plate_csv` helper if needed; add new tests for `sample_uid` / `timepoint` emission; update column-order assertions in existing tests |

---

## Phase 1: OpenSpec proposal scaffolding

Follow the pattern from PR #165 (`openspec/changes/add-multiple-dicot-plate-pipeline`).

### Task 1.1: Create the OpenSpec change directory

**Files:**
- Create: `openspec/changes/add-sample-uid-timepoint-metadata/proposal.md`
- Create: `openspec/changes/add-sample-uid-timepoint-metadata/design.md`
- Create: `openspec/changes/add-sample-uid-timepoint-metadata/tasks.md`
- Create: `openspec/changes/add-sample-uid-timepoint-metadata/specs/series-metadata/spec.md`

- [ ] **Step 1: Create the directory structure**

Run:
```bash
mkdir -p openspec/changes/add-sample-uid-timepoint-metadata/specs/series-metadata
```

- [ ] **Step 2: Write `proposal.md`**

Content:

```markdown
## Why

The existing CSV-metadata pattern on `Series` (`expected_count`, `group`, `qc_fail`) hardcodes three property-specific lookups against the `plant_qr_code` column. Three new concerns from the timelapse design (issue #169, Workstream 1 of the 2026-04-23 design) need the same lookup mechanism:

1. `sample_uid` — a cross-scan stable identity (the semantic rename of the role `plant_qr_code` has been implicitly filling). Not every workflow needs this; when unset it defaults to `series_name`.
2. `timepoint` — a numeric time-axis value needed for cross-series diffs in `TimeDiffPipeline` (Workstream 3 of the same design).
3. A generic `get_metadata(column, plant_id=None)` accessor so future properties don't each need their own lookup boilerplate.

Without this layer, Workstream 2 (TrackedTipPipeline, issue #129) and Workstream 3 (TimeDiffPipeline, issue #170) have no clean way to consume per-timepoint metadata.

## What Changes

- **NEW** `Series.load(..., sample_uid: Optional[str] = None)` kwarg. Defaults to `series_name` when unset — preserves today's behavior for every existing workflow.
- **NEW** `Series.sample_uid: str` attribute (attrs field), initialized from the kwarg or `series_name`.
- **NEW** `Series.get_metadata(column: str, plant_id: Optional[int] = None) -> Any` method. Looks up `df[df["plant_qr_code"] == self.sample_uid]`. If `plant_id` is given AND the CSV has a `plant_id` column, composite lookup returns the `(sample_uid, plant_id)` row. If `plant_id` is given but the CSV has no `plant_id` column, the argument is silently ignored and sample-uid-only lookup is used. Returns `np.nan` when the column is missing or no row matches.
- **NEW** `Series.timepoint: Union[float, int]` property — thin wrapper `return self.get_metadata("timepoint")`.
- **REFACTORED** `Series.expected_count`, `Series.group`, `Series.qc_fail` become thin wrappers around `get_metadata()`. Observable behavior unchanged.
- **NEW** `sleap_roots/metadata.py` module with two pure-function helpers:
  - `build_metadata_csv(rows: List[Dict], path: Union[str, Path]) -> Path`
  - `infer_timepoints_from_filenames(slp_paths: List[Path], pattern: str) -> Dict[str, float]`
- **NEW** top-level re-exports of both helpers from `sleap_roots.__init__`.
- **MODIFIED** `MultipleDicotPlatePipeline.compute_plate_traits` output:
  - Top-level dict gains `sample_uid` and `timepoint` keys.
  - Every per-plant row gains `sample_uid` and `timepoint` keys.
  - CSV emits `sample_uid` and `timepoint` columns right after `series` (positions 1 and 2), shifting the remaining metadata columns by 2.

**Not breaking.** Existing `Series` API is preserved (`expected_count`, `group`, `qc_fail` keep their signatures and return values). Existing `MultipleDicotPlatePipeline` consumers who read CSV by column NAME continue to work; consumers who read CSV by column POSITION will see columns shifted — the plate pipeline is new enough (PR #165) that no published downstream scripts exist. CSV column rename to `sample_uid` / `timepoint` (from `plant_qr_code` / n/a) is explicitly deferred to #163.

## Impact

- **Affected specs**: new capability `series-metadata` (ADDED requirements for `sample_uid`, `timepoint`, `get_metadata`, CSV-builder helpers, pipeline output emission). No modifications to existing specs.
- **Affected code**:
  - `sleap_roots/series.py` — new kwarg, new method, new property, refactor of three existing properties.
  - `sleap_roots/metadata.py` — new module.
  - `sleap_roots/__init__.py` — re-export additions.
  - `sleap_roots/trait_pipelines.py` — plate pipeline output additions (~10 lines).
  - `tests/test_series.py` — new test coverage; regression coverage for refactored properties.
  - `tests/test_metadata.py` — new test file.
  - `tests/test_multiple_dicot_plate_pipeline.py` — new emission tests; update existing CSV column-order assertions.
- **Source design doc**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1. PR #171.
- **Dependencies**: Unblocks #129 (TrackedTipPipeline) and #170 (TimeDiffPipeline).
```

- [ ] **Step 3: Write `design.md`**

Content:

```markdown
# Design: `add-sample-uid-timepoint-metadata`

**Architectural source of truth**: [`docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md`](../../../docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md) § Workstream 1.

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
```

- [ ] **Step 4: Write `specs/series-metadata/spec.md`**

Content:

````markdown
## ADDED Requirements

### Requirement: `Series.sample_uid` SHALL provide a cross-scan stable identity

A new attrs field `sample_uid: Optional[str]` MUST be added to the `Series` class. `Series.load` MUST accept a `sample_uid: Optional[str] = None` kwarg. When the kwarg is `None` (not passed, or explicitly `None`), `Series.sample_uid` MUST equal `Series.series_name`. When the kwarg is a non-empty string, `Series.sample_uid` MUST equal that string.

#### Scenario: `sample_uid` defaults to `series_name` when the kwarg is omitted

- **Given** a `Series.load(series_name="plant1_day0", primary_path=..., csv_path=None)` call without `sample_uid`
- **When** `series.sample_uid` is read
- **Then** the value equals `"plant1_day0"`

#### Scenario: `sample_uid` kwarg sets the attribute

- **Given** a `Series.load(series_name="plant1_day0", sample_uid="plant1", primary_path=...)` call
- **When** `series.sample_uid` is read
- **Then** the value equals `"plant1"`

#### Scenario: Two Series can share a `sample_uid` while having distinct `series_name` values

- **Given** `series_a = Series.load(series_name="plant1_day0", sample_uid="plant1", ...)` and `series_b = Series.load(series_name="plant1_day1", sample_uid="plant1", ...)`
- **Then** `series_a.series_name != series_b.series_name`
- **And** `series_a.sample_uid == series_b.sample_uid == "plant1"`

### Requirement: `Series.get_metadata(column, plant_id=None)` SHALL provide a generic CSV-column accessor

A new method `Series.get_metadata(self, column: str, plant_id: Optional[int] = None) -> Any` MUST be added. Behavior:

- If `self.csv_path` is None or the file does not exist, MUST return `np.nan`.
- If the CSV does not contain the requested `column`, MUST return `np.nan`.
- Lookup: `df[df["plant_qr_code"] == self.sample_uid]`.
- If `plant_id` is given AND the CSV contains a `plant_id` column, the lookup MUST be the composite `(plant_qr_code, plant_id)` match.
- If `plant_id` is given but the CSV has no `plant_id` column, the `plant_id` argument MUST be silently ignored and the sample-uid-only lookup used.
- If no row matches, MUST return `np.nan`.
- If one or more rows match, MUST return the value in the first matching row's `column` field.

#### Scenario: No CSV path returns NaN

- **Given** a `Series` loaded without `csv_path`
- **When** `series.get_metadata("number_of_plants_cylinder")` is called
- **Then** the return value is `np.nan`

#### Scenario: CSV column missing returns NaN

- **Given** a `Series` loaded with `csv_path` pointing to a CSV that has columns `{plant_qr_code, genotype}` but no `timepoint` column
- **When** `series.get_metadata("timepoint")` is called
- **Then** the return value is `np.nan`

#### Scenario: No matching row returns NaN

- **Given** a CSV with rows keyed by `plant_qr_code` values `{"a", "b"}`, and a Series with `sample_uid="c"`
- **When** `series.get_metadata("genotype")` is called
- **Then** the return value is `np.nan`

#### Scenario: Matching row returns the column value

- **Given** a CSV with a row `plant_qr_code="plant1", genotype="MK22", timepoint=3` and a Series with `sample_uid="plant1"`
- **When** `series.get_metadata("genotype")` is called
- **Then** the return value is `"MK22"`
- **And** `series.get_metadata("timepoint")` returns `3`

#### Scenario: `plant_id` composite lookup

- **Given** a CSV with two rows both having `plant_qr_code="plant1"` but distinct `plant_id` values (`0` and `1`) and distinct `genotype` values (`"A"` and `"B"`)
- **And** a Series with `sample_uid="plant1"`
- **When** `series.get_metadata("genotype", plant_id=1)` is called
- **Then** the return value is `"B"`

#### Scenario: `plant_id` argument ignored when CSV has no `plant_id` column

- **Given** a CSV with columns `{plant_qr_code, genotype, timepoint}` (no `plant_id` column) and a matching row for `sample_uid="plant1"`
- **When** `series.get_metadata("genotype", plant_id=99)` is called
- **Then** the return value is the row's `genotype` value (not NaN)

### Requirement: `Series.timepoint` SHALL wrap `get_metadata("timepoint")`

The property `Series.timepoint: Union[float, int]` MUST exist and MUST return `self.get_metadata("timepoint")`.

#### Scenario: `timepoint` returns the CSV value

- **Given** a CSV with `plant_qr_code="plant1", timepoint=2` and a Series with `sample_uid="plant1"`
- **When** `series.timepoint` is read
- **Then** the return value equals `2`

#### Scenario: `timepoint` returns NaN when no CSV

- **Given** a Series loaded without `csv_path`
- **When** `series.timepoint` is read
- **Then** the return value is `np.nan`

### Requirement: Existing properties SHALL become thin wrappers around `get_metadata`

`Series.expected_count`, `Series.group`, and `Series.qc_fail` MUST be refactored to call `self.get_metadata(...)` with the appropriate column name. Observable behavior (return value on every input that worked before) MUST be preserved byte-for-byte.

#### Scenario: `expected_count` still returns the correct value from an existing CSV

- **Given** the existing `multiple_arabidopsis_11do_csv` fixture and a Series whose `series_name` matches a row in that CSV
- **When** `series.expected_count` is read
- **Then** the return value equals the `number_of_plants_cylinder` value in that row

#### Scenario: `group` still returns the genotype value

- **Given** the same fixture
- **When** `series.group` is read
- **Then** the return value equals the `genotype` value in that row

#### Scenario: `qc_fail` still returns the qc_cylinder value

- **Given** the same fixture
- **When** `series.qc_fail` is read
- **Then** the return value equals the `qc_cylinder` value in that row

### Requirement: `sleap_roots/metadata.py` SHALL provide CSV builder helpers

A new module `sleap_roots/metadata.py` MUST exist and MUST expose:

- `build_metadata_csv(rows: List[Dict[str, Any]], path: Union[str, Path]) -> Path` — writes a CSV from row dicts. Validates that every row has a `plant_qr_code` key and raises `ValueError` otherwise. Returns the Path the file was written to. Column order: `plant_qr_code, genotype, number_of_plants_cylinder, qc_cylinder, qc_code, timepoint, <extras in sorted order>`. Columns not present in any row MUST be omitted.
- `infer_timepoints_from_filenames(slp_paths: List[Path], pattern: str) -> Dict[str, float]` — regex-parses `(series_name, timepoint)` from the stem of each path using named groups. Raises `ValueError` if the pattern lacks `series_name` or `timepoint` named groups. Skips paths whose stems don't match (does not raise). `timepoint` values are cast to `float`; values that can't be cast are skipped.

Both functions MUST be re-exported from `sleap_roots.__init__` at the top level.

#### Scenario: `build_metadata_csv` writes canonical column order

- **Given** `rows = [{"plant_qr_code": "a", "genotype": "X", "timepoint": 0}, {"plant_qr_code": "b", "genotype": "Y", "timepoint": 1}]`
- **When** `build_metadata_csv(rows, tmp_path / "out.csv")` is called
- **And** the result is parsed via `pd.read_csv`
- **Then** `list(df.columns) == ["plant_qr_code", "genotype", "timepoint"]`
- **And** `df["plant_qr_code"].tolist() == ["a", "b"]`

#### Scenario: `build_metadata_csv` raises on missing `plant_qr_code`

- **Given** `rows = [{"genotype": "X"}]` (no `plant_qr_code`)
- **When** `build_metadata_csv(rows, tmp_path / "out.csv")` is called
- **Then** `ValueError` is raised with a message mentioning "plant_qr_code"

#### Scenario: `build_metadata_csv` returns the written Path

- **Given** valid rows and a `tmp_path / "x.csv"` target
- **When** `build_metadata_csv(...)` is called
- **Then** the return value equals `tmp_path / "x.csv"` and the file exists

#### Scenario: `infer_timepoints_from_filenames` parses named groups

- **Given** `slp_paths = [Path("plant1_0.slp"), Path("plant1_5.slp"), Path("plant1_10.slp")]` and `pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"`
- **When** `infer_timepoints_from_filenames(slp_paths, pattern)` is called
- **Then** the return value is `{"plant1_0": 0.0, "plant1_5": 5.0, "plant1_10": 10.0}`

#### Scenario: `infer_timepoints_from_filenames` raises when pattern lacks named groups

- **Given** `pattern = r".+_\d+"` (no named groups)
- **When** `infer_timepoints_from_filenames([Path("plant1_0.slp")], pattern)` is called
- **Then** `ValueError` is raised with a message mentioning `series_name` and `timepoint`

#### Scenario: Non-matching stems are skipped without raising

- **Given** `slp_paths = [Path("plant1_0.slp"), Path("garbage.slp")]` and a valid pattern
- **When** `infer_timepoints_from_filenames(...)` is called
- **Then** the return value contains only `{"plant1_0": 0.0}`

### Requirement: `MultipleDicotPlatePipeline` output SHALL include `sample_uid` and `timepoint`

The per-series result dict returned by `compute_plate_traits(series)` MUST contain top-level keys `sample_uid` (str) and `timepoint` (numeric or `np.nan`). Each entry in `result["plants"]` MUST also contain `sample_uid` and `timepoint` keys with the same values. When the inner pipeline writes CSV (`write_csv=True`), the output CSV MUST include `sample_uid` and `timepoint` columns at positions 1 and 2 (directly after `series`). The remaining columns MUST retain their relative ordering.

#### Scenario: Top-level dict has `sample_uid` and `timepoint`

- **Given** a synthetic Series with `sample_uid="plate_abc"` and a CSV row with `timepoint=3`
- **When** `compute_plate_traits(series)` is called
- **Then** `result["sample_uid"] == "plate_abc"`
- **And** `result["timepoint"] == 3`

#### Scenario: Every per-plant row has `sample_uid` and `timepoint`

- **Given** the same series with 2 plants
- **When** `compute_plate_traits(series)` is called
- **Then** every entry in `result["plants"]` has `sample_uid == "plate_abc"`
- **And** every entry has `timepoint == 3`

#### Scenario: CSV column order puts `sample_uid` at position 1 and `timepoint` at position 2

- **Given** a synthetic Series and `write_csv=True`
- **When** the CSV is read back via `pd.read_csv`
- **Then** `list(df.columns)[0] == "series"`
- **And** `list(df.columns)[1] == "sample_uid"`
- **And** `list(df.columns)[2] == "timepoint"`
- **And** `list(df.columns)[3] == "frame"`

#### Scenario: `timepoint` is NaN in output when no CSV is attached

- **Given** a Series loaded without `csv_path`
- **When** `compute_plate_traits(series)` is called
- **Then** `pd.isna(result["timepoint"])` is True
- **And** for every plant, `pd.isna(plant["timepoint"])` is True
- **And** `result["sample_uid"]` equals `series.series_name` (defaulted)
````

- [ ] **Step 5: Write `tasks.md`**

Content (this is the developer-facing TDD checklist; references this plan for detail):

```markdown
# Tasks: Add Series metadata layer (sample_uid, timepoint, get_metadata)

**Source of truth for architecture**: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1. Implementation plan with task-level detail: `docs/superpowers/plans/2026-04-24-metadata-layer.md`.

**Workflow guarantees**:

- Strict TDD: each subsection writes failing tests BEFORE implementation.
- All commands use `uv run` (`uv run pytest`, `uv run black`, `uv run pydocstyle`).
- No implementation is merged until every follow-up issue in section 5 is filed and linked in the PR body.

## 1. Write failing tests for `Series.get_metadata`

- [ ] 1.1 Add `test_series_get_metadata_no_csv` — Series without `csv_path` → `get_metadata("anything")` returns `np.nan`.
- [ ] 1.2 Add `test_series_get_metadata_missing_column` — CSV exists but lacks the requested column → NaN.
- [ ] 1.3 Add `test_series_get_metadata_no_matching_row` — CSV has the column but no row matches sample_uid → NaN.
- [ ] 1.4 Add `test_series_get_metadata_matches_row` — CSV has matching row → returns the column value.
- [ ] 1.5 Add `test_series_get_metadata_plant_id_composite_lookup` — CSV has a `plant_id` column; composite lookup returns the correct row.
- [ ] 1.6 Add `test_series_get_metadata_plant_id_ignored_when_no_column` — plant_id argument silently ignored when CSV lacks `plant_id` column.
- [ ] 1.7 Run `uv run pytest tests/test_series.py -k get_metadata -x` — confirm all 6 tests FAIL with `AttributeError: 'Series' object has no attribute 'get_metadata'`.

## 2. Implement `Series.get_metadata`

- [ ] 2.1 Add `get_metadata(column, plant_id=None)` to `Series`. Refactor `expected_count`, `group`, `qc_fail` as thin wrappers.
- [ ] 2.2 Run `uv run pytest tests/test_series.py -k get_metadata -v` — all 6 pass.
- [ ] 2.3 Run `uv run pytest tests/test_series.py -x` — existing tests still pass (regression check for refactored properties).

## 3. Write failing tests for `Series.sample_uid` + `Series.timepoint`

- [ ] 3.1 Add `test_series_sample_uid_defaults_to_series_name` — no kwarg → `sample_uid == series_name`.
- [ ] 3.2 Add `test_series_sample_uid_explicit_kwarg` — explicit kwarg → `sample_uid == kwarg value`.
- [ ] 3.3 Add `test_series_sample_uid_shared_across_series` — two Series with different `series_name` but same `sample_uid`.
- [ ] 3.4 Add `test_series_timepoint_from_csv` — CSV has timepoint column → property returns value.
- [ ] 3.5 Add `test_series_timepoint_no_csv` → NaN.
- [ ] 3.6 Run `uv run pytest tests/test_series.py -k "sample_uid or timepoint" -x` — all FAIL with `AttributeError` / `TypeError`.

## 4. Implement `Series.sample_uid` + `Series.timepoint`

- [ ] 4.1 Add `sample_uid: Optional[str] = None` attrs field to `Series`.
- [ ] 4.2 Add `sample_uid: Optional[str] = None` kwarg to `Series.load`; set `sample_uid = sample_uid or series_name` before `cls(...)`.
- [ ] 4.3 Add `timepoint` property wrapping `get_metadata("timepoint")`.
- [ ] 4.4 Run `uv run pytest tests/test_series.py -k "sample_uid or timepoint" -v` — all pass.
- [ ] 4.5 Run `uv run pytest tests/test_series.py -x` — full file still passes.

## 5. Write failing tests for `sleap_roots/metadata.py`

- [ ] 5.1 Add `tests/test_metadata.py` with tests 5.2-5.7 below.
- [ ] 5.2 Add `test_build_metadata_csv_canonical_column_order` — verify canonical order with subset of standard columns.
- [ ] 5.3 Add `test_build_metadata_csv_omits_unused_columns` — columns not present in any row are omitted.
- [ ] 5.4 Add `test_build_metadata_csv_raises_on_missing_plant_qr_code` — `ValueError` with "plant_qr_code" in message.
- [ ] 5.5 Add `test_build_metadata_csv_returns_path` — function returns the Path of the written file.
- [ ] 5.6 Add `test_infer_timepoints_from_filenames_named_groups` — correct dict output.
- [ ] 5.7 Add `test_infer_timepoints_from_filenames_missing_named_groups` — `ValueError` with `series_name` / `timepoint` in message.
- [ ] 5.8 Add `test_infer_timepoints_from_filenames_skips_non_matches` — garbage stems skipped silently.
- [ ] 5.9 Run `uv run pytest tests/test_metadata.py -x` — all FAIL with `ImportError`.

## 6. Implement `sleap_roots/metadata.py`

- [ ] 6.1 Create `sleap_roots/metadata.py` with `build_metadata_csv` and `infer_timepoints_from_filenames`.
- [ ] 6.2 Add re-exports to `sleap_roots/__init__.py`.
- [ ] 6.3 Run `uv run pytest tests/test_metadata.py -v` — all pass.

## 7. Write failing tests for plate pipeline `sample_uid`/`timepoint` emission

- [ ] 7.1 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_top_level` — result dict has both keys at top level.
- [ ] 7.2 Add `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_per_plant` — every plant row has both keys.
- [ ] 7.3 Add `test_multiple_dicot_plate_pipeline_csv_column_positions` — CSV column 1 = `sample_uid`, column 2 = `timepoint`.
- [ ] 7.4 Add `test_multiple_dicot_plate_pipeline_timepoint_nan_without_csv` — no CSV → NaN at top-level + every plant row.
- [ ] 7.5 Update `test_multiple_dicot_plate_pipeline_csv_output` to assert the new column order (positions 0-7 metadata, 8+ traits).
- [ ] 7.6 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -k "sample_uid or timepoint or csv" -x` — 4 new tests FAIL; updated test 7.5 FAILS.

## 8. Implement plate pipeline emission

- [ ] 8.1 Modify `compute_plate_traits` in `trait_pipelines.py` to add `sample_uid` + `timepoint` to the top-level result dict.
- [ ] 8.2 Modify `_build_plant_row` to add `sample_uid` + `timepoint` to every plant-row dict.
- [ ] 8.3 Modify `_build_plate_dataframe` `meta_cols` list to insert `sample_uid` at position 1 and `timepoint` at position 2.
- [ ] 8.4 Run `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -x` — all 20+ tests pass (18 existing + new additions).
- [ ] 8.5 Run `uv run pytest tests/ -x` — full suite green (~445+ tests).

## 9. File follow-up issues BEFORE PR merges

- [ ] 9.1 File issue: "Apply `sample_uid`/`timepoint` emission to cylinder pipelines (MultipleDicotPipeline, MultiplePrimaryRootPipeline, DicotPipeline)". Related to #159 and #170.
- [ ] 9.2 Post comment on issue #169 linking the OpenSpec change + PR.
- [ ] 9.3 Verification gate: `gh issue list --state open --search "#169 in:body follow-up" --limit 20 --json number,title` — confirm the new issue is present.
- [ ] 9.4 Copy follow-up issue number into PR body under `## Follow-up issues filed during this PR`.

## 10. Pre-merge validation

- [ ] 10.1 Run `openspec validate add-sample-uid-timepoint-metadata --strict` — passes.
- [ ] 10.2 Run `uv run black --check sleap_roots/ tests/` — clean.
- [ ] 10.3 Run `uv run pydocstyle --convention=google sleap_roots/` — clean.
- [ ] 10.4 Run `uv run pytest tests/ -x` — full suite passes.
- [ ] 10.5 Invoke `/review-pr` on the branch before opening the PR.
- [ ] 10.6 Open PR; body references issue #169, links the follow-up from 9.1, includes `## TDD evidence` from 1.7 + 3.6 + 5.9 + 7.6.
```

- [ ] **Step 6: Validate the OpenSpec change**

Run: `openspec validate add-sample-uid-timepoint-metadata --strict`
Expected: `Change 'add-sample-uid-timepoint-metadata' is valid`

- [ ] **Step 7: Commit the proposal**

```bash
git add openspec/changes/add-sample-uid-timepoint-metadata/
git commit -m "openspec: Scaffold add-sample-uid-timepoint-metadata proposal

Workstream 1 of docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md.

Adds OpenSpec proposal for Series.sample_uid + Series.timepoint +
Series.get_metadata() generic CSV accessor + sleap_roots/metadata.py
CSV builder helpers + MultipleDicotPlatePipeline sample_uid/timepoint
emission. Backward-compatible: plant_qr_code remains the CSV lookup key;
column rename deferred to #163.

Related: #169, #171

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 2: OpenSpec review loop

### Task 2.1: Run openspec-review

- [ ] **Step 1: Invoke the openspec-review skill**

Dispatch 5 parallel subagents (code quality, testing, scientific rigor, perf/cross-platform, behavioural correctness) on `openspec/changes/add-sample-uid-timepoint-metadata`. Reference the design doc at `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md`.

- [ ] **Step 2: Reconcile every BLOCKING finding**

For each BLOCKING finding: quote the reviewer verbatim, name the exact mechanism, cite the revised location in the proposal. No silent swaps.

- [ ] **Step 3: Re-validate after reconciliation**

Run: `openspec validate add-sample-uid-timepoint-metadata --strict`
Expected: `valid`.

- [ ] **Step 4: Commit reconciliation if any**

If the reconciliation produced file changes, commit with message explaining the BLOCKING findings and how each was addressed (pattern from PR #165).

---

## Phase 3: TDD implementation

### Task 3.1: Write failing tests for `Series.get_metadata`

**Files:**
- Modify: `tests/test_series.py` (append to end)
- Test: `tests/test_series.py::test_series_get_metadata_*`

- [ ] **Step 1: Read current end of `tests/test_series.py`**

Run: `wc -l tests/test_series.py` — note line count. Open the last ~20 lines to understand existing fixture patterns.

- [ ] **Step 2: Add the 6 get_metadata tests at the end of `tests/test_series.py`**

Append this code:

```python
# ---------------------------------------------------------------------------
# Series.get_metadata (issue #169 — metadata layer)
# ---------------------------------------------------------------------------


def _write_csv(path, rows):
    """Helper: write a CSV from row dicts for get_metadata tests."""
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, index=False)


def test_series_get_metadata_no_csv(tmp_path):
    """When csv_path is None, get_metadata returns np.nan for any column."""
    series = Series.load(series_name="plant1")
    assert np.isnan(series.get_metadata("anything"))


def test_series_get_metadata_missing_column(tmp_path):
    """When the CSV exists but lacks the column, returns np.nan."""
    csv = tmp_path / "metadata.csv"
    _write_csv(csv, [{"plant_qr_code": "plant1", "genotype": "X"}])
    series = Series.load(series_name="plant1", csv_path=csv.as_posix())
    assert np.isnan(series.get_metadata("timepoint"))


def test_series_get_metadata_no_matching_row(tmp_path):
    """When the CSV has the column but no row matches sample_uid, returns np.nan."""
    csv = tmp_path / "metadata.csv"
    _write_csv(csv, [{"plant_qr_code": "a", "genotype": "X"}])
    series = Series.load(series_name="c", csv_path=csv.as_posix())
    assert np.isnan(series.get_metadata("genotype"))


def test_series_get_metadata_matches_row(tmp_path):
    """Matching row returns the column value."""
    csv = tmp_path / "metadata.csv"
    _write_csv(csv, [
        {"plant_qr_code": "plant1", "genotype": "MK22", "timepoint": 3}
    ])
    series = Series.load(series_name="plant1", csv_path=csv.as_posix())
    assert series.get_metadata("genotype") == "MK22"
    assert series.get_metadata("timepoint") == 3


def test_series_get_metadata_plant_id_composite_lookup(tmp_path):
    """plant_id given AND CSV has plant_id column -> composite lookup."""
    csv = tmp_path / "metadata.csv"
    _write_csv(csv, [
        {"plant_qr_code": "plant1", "plant_id": 0, "genotype": "A"},
        {"plant_qr_code": "plant1", "plant_id": 1, "genotype": "B"},
    ])
    series = Series.load(series_name="plant1", csv_path=csv.as_posix())
    assert series.get_metadata("genotype", plant_id=0) == "A"
    assert series.get_metadata("genotype", plant_id=1) == "B"


def test_series_get_metadata_plant_id_ignored_when_no_column(tmp_path):
    """plant_id argument silently ignored when CSV has no plant_id column."""
    csv = tmp_path / "metadata.csv"
    _write_csv(csv, [{"plant_qr_code": "plant1", "genotype": "X"}])
    series = Series.load(series_name="plant1", csv_path=csv.as_posix())
    assert series.get_metadata("genotype", plant_id=99) == "X"
```

- [ ] **Step 3: Run tests — confirm they fail**

Run: `uv run pytest tests/test_series.py -k get_metadata -x`
Expected: all 6 FAIL with `AttributeError: 'Series' object has no attribute 'get_metadata'`.

- [ ] **Step 4: Commit the failing tests**

```bash
git add tests/test_series.py
git commit -m "test(series): Add failing tests for get_metadata (TDD red phase)

Red phase for the generic CSV accessor introduced in #169. Six tests
cover: no csv_path -> NaN, missing column -> NaN, no matching row -> NaN,
matching row returns value, plant_id composite lookup, plant_id silently
ignored when CSV has no plant_id column.

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.2: Implement `Series.get_metadata`

**Files:**
- Modify: `sleap_roots/series.py:165-208`

- [ ] **Step 1: Read the current `expected_count`, `group`, `qc_fail` properties**

Open `sleap_roots/series.py` lines 165-208. The three properties follow a shared pattern: check csv_path, read CSV, lookup `plant_qr_code == series_name`, catch IndexError.

- [ ] **Step 2: Add `get_metadata` method and refactor the three existing properties**

Replace the three existing properties with:

```python
    def get_metadata(
        self, column: str, plant_id: Optional[int] = None
    ) -> Any:
        """Look up a value in the metadata CSV by sample_uid (and optional plant_id).

        Args:
            column: Name of the column whose value to return.
            plant_id: If given AND the CSV has a ``plant_id`` column, restrict
                the lookup to the row matching ``(sample_uid, plant_id)``. If
                given but the CSV has no ``plant_id`` column, the argument is
                silently ignored and sample-uid-only lookup is used.

        Returns:
            The column value from the first matching row. Returns ``np.nan``
            when the CSV is missing, the column is absent, or no row matches.
        """
        if not self.csv_path or not Path(self.csv_path).exists():
            return np.nan
        df = pd.read_csv(self.csv_path)
        if column not in df.columns:
            return np.nan
        mask = df["plant_qr_code"] == self.sample_uid
        if plant_id is not None and "plant_id" in df.columns:
            mask = mask & (df["plant_id"] == plant_id)
        matched = df[mask]
        if len(matched) == 0:
            return np.nan
        return matched[column].iloc[0]

    @property
    def expected_count(self) -> Union[float, int]:
        """Fetch the expected plant count for this series from the CSV."""
        return self.get_metadata("number_of_plants_cylinder")

    @property
    def group(self) -> str:
        """Group name for the series from the CSV."""
        return self.get_metadata("genotype")

    @property
    def qc_fail(self) -> Union[int, float]:
        """Flag to indicate if the series failed QC from the CSV."""
        return self.get_metadata("qc_cylinder")
```

**Note**: `self.sample_uid` is referenced but not yet defined as an attrs field — that lands in Task 3.4. The tests in Task 3.1 don't pass a `sample_uid` kwarg; they rely on a default. To unblock these tests now, add a minimal fallback: a `sample_uid` property that returns `series_name` if the attrs field is absent. But that's fragile. Cleaner: reorder Tasks — add the `sample_uid` attrs field NOW (as a no-op default) even though its tests come later.

Actually the cleanest sequencing is: in this step, ALSO add the `sample_uid: Optional[str] = None` attrs field and default it in `Series.load` so `get_metadata` has a valid attribute to read. Do NOT add `timepoint` yet.

- [ ] **Step 3: Also add `sample_uid` attrs field + `Series.load` default (prerequisite for get_metadata)**

At the Series class field block (around line 60), add:

```python
    sample_uid: Optional[str] = None
```

In `Series.load`, add the kwarg and default before the `return cls(...)`:

```python
    @classmethod
    def load(
        cls,
        series_name: str,
        h5_path: Optional[str] = None,
        primary_path: Optional[str] = None,
        lateral_path: Optional[str] = None,
        crown_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        sample_uid: Optional[str] = None,   # NEW
    ) -> "Series":
        ...
        # Default sample_uid to series_name when not explicitly set.
        if sample_uid is None:
            sample_uid = series_name
        ...
        return cls(
            series_name=series_name,
            h5_path=h5_path,
            primary_path=primary_path,
            lateral_path=lateral_path,
            crown_path=crown_path,
            primary_labels=primary_labels,
            lateral_labels=lateral_labels,
            crown_labels=crown_labels,
            video=video,
            csv_path=csv_path,
            sample_uid=sample_uid,   # NEW
        )
```

- [ ] **Step 4: Add `Any` to the imports if not already present**

Check the imports at the top of `sleap_roots/series.py`:

```python
from typing import Any, Dict, List, Optional, Tuple, Union
```

If `Any` is missing, add it.

- [ ] **Step 5: Run the get_metadata tests — confirm they pass**

Run: `uv run pytest tests/test_series.py -k get_metadata -v`
Expected: all 6 pass.

- [ ] **Step 6: Run the full Series test file — confirm no regressions**

Run: `uv run pytest tests/test_series.py -x`
Expected: all tests pass, including the existing `expected_count` / `group` / `qc_fail` tests (regression check for the refactor).

- [ ] **Step 7: Commit**

```bash
git add sleap_roots/series.py
git commit -m "feat(series): Add get_metadata() + sample_uid attrs field (#169)

Generic CSV-column accessor with optional plant_id composite lookup.
Refactors expected_count, group, qc_fail into thin wrappers.

The sample_uid attrs field + Series.load kwarg land here too because
get_metadata's lookup references self.sample_uid — they're the same
PR's prerequisite. Tests for sample_uid specifically land in Task 3.3.

Backward compatible: kwarg defaults to series_name, so existing
workflows are unchanged. CSV lookup key remains plant_qr_code
(column rename deferred to #163).

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.3: Write failing tests for `sample_uid` + `timepoint`

**Files:**
- Modify: `tests/test_series.py` (append)

- [ ] **Step 1: Append these 5 tests to `tests/test_series.py`**

```python
# ---------------------------------------------------------------------------
# Series.sample_uid and Series.timepoint (issue #169 — metadata layer)
# ---------------------------------------------------------------------------


def test_series_sample_uid_defaults_to_series_name():
    """No kwarg -> sample_uid == series_name."""
    series = Series.load(series_name="plant1_day0")
    assert series.sample_uid == "plant1_day0"


def test_series_sample_uid_explicit_kwarg():
    """Explicit kwarg -> sample_uid == kwarg value."""
    series = Series.load(series_name="plant1_day0", sample_uid="plant1")
    assert series.sample_uid == "plant1"


def test_series_sample_uid_shared_across_series():
    """Two Series with distinct series_name can share a sample_uid."""
    a = Series.load(series_name="plant1_day0", sample_uid="plant1")
    b = Series.load(series_name="plant1_day1", sample_uid="plant1")
    assert a.series_name != b.series_name
    assert a.sample_uid == b.sample_uid == "plant1"


def test_series_timepoint_from_csv(tmp_path):
    """CSV has timepoint column -> property returns value."""
    csv = tmp_path / "metadata.csv"
    _write_csv(csv, [{"plant_qr_code": "plant1", "timepoint": 3}])
    series = Series.load(
        series_name="plant1_day3",
        sample_uid="plant1",
        csv_path=csv.as_posix(),
    )
    assert series.timepoint == 3


def test_series_timepoint_no_csv():
    """No csv_path -> timepoint returns np.nan."""
    series = Series.load(series_name="plant1")
    assert np.isnan(series.timepoint)
```

- [ ] **Step 2: Run the tests — confirm they fail**

Run: `uv run pytest tests/test_series.py -k "sample_uid or timepoint" -v`
Expected: tests 3.1-3.3 PASS (sample_uid field was added in Task 3.2). Tests 3.4 + 3.5 FAIL with `AttributeError: 'Series' object has no attribute 'timepoint'`.

If the pattern is: 3 pass + 2 fail → correct. Commit the tests even though some already pass.

- [ ] **Step 3: Commit the tests**

```bash
git add tests/test_series.py
git commit -m "test(series): Add failing tests for sample_uid + timepoint

TDD red phase for timepoint property. sample_uid tests pass now because
the attrs field landed with Task 3.2 (it was a prerequisite for
get_metadata). timepoint tests fail with AttributeError until Task 3.4.

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.4: Implement `Series.timepoint`

**Files:**
- Modify: `sleap_roots/series.py` (add one property)

- [ ] **Step 1: Add the `timepoint` property to `Series`**

After the `qc_fail` property, add:

```python
    @property
    def timepoint(self) -> Union[float, int]:
        """Timepoint value for this series from the CSV.

        Generic numeric value — user decides what unit (days, hours,
        minutes, frame-index-in-timelapse). Used by ``TimeDiffPipeline``
        to order series in a cross-scan timelapse. Returns ``np.nan``
        when the CSV is missing, has no ``timepoint`` column, or has no
        matching row.
        """
        return self.get_metadata("timepoint")
```

- [ ] **Step 2: Run the tests — confirm they pass**

Run: `uv run pytest tests/test_series.py -k "sample_uid or timepoint" -v`
Expected: all 5 pass.

- [ ] **Step 3: Run the full Series test file**

Run: `uv run pytest tests/test_series.py -x`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add sleap_roots/series.py
git commit -m "feat(series): Add Series.timepoint property (#169)

Thin wrapper around get_metadata('timepoint'). Returns np.nan when no
CSV attached or no timepoint column.

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.5: Write failing tests for `sleap_roots/metadata.py`

**Files:**
- Create: `tests/test_metadata.py`

- [ ] **Step 1: Create `tests/test_metadata.py`**

Content:

```python
"""Tests for sleap_roots/metadata.py (issue #169)."""

from pathlib import Path

import pandas as pd
import pytest

from sleap_roots.metadata import (
    build_metadata_csv,
    infer_timepoints_from_filenames,
)


# ---------------------------------------------------------------------------
# build_metadata_csv
# ---------------------------------------------------------------------------


def test_build_metadata_csv_canonical_column_order(tmp_path):
    """Columns appear in canonical order regardless of row dict key order."""
    rows = [
        {"timepoint": 0, "plant_qr_code": "a", "genotype": "X"},
        {"plant_qr_code": "b", "genotype": "Y", "timepoint": 1},
    ]
    path = build_metadata_csv(rows, tmp_path / "out.csv")
    df = pd.read_csv(path)
    # Canonical order: plant_qr_code, genotype, timepoint (no others used here).
    assert list(df.columns) == ["plant_qr_code", "genotype", "timepoint"]
    assert df["plant_qr_code"].tolist() == ["a", "b"]


def test_build_metadata_csv_omits_unused_columns(tmp_path):
    """Canonical columns not present in any row are omitted from output."""
    rows = [{"plant_qr_code": "a", "genotype": "X"}]
    path = build_metadata_csv(rows, tmp_path / "out.csv")
    df = pd.read_csv(path)
    # Canonical order has number_of_plants_cylinder / qc_cylinder / qc_code /
    # timepoint but none are in the rows; all are omitted.
    assert list(df.columns) == ["plant_qr_code", "genotype"]


def test_build_metadata_csv_raises_on_missing_plant_qr_code(tmp_path):
    """ValueError if any row lacks plant_qr_code."""
    rows = [{"genotype": "X"}]
    with pytest.raises(ValueError, match="plant_qr_code"):
        build_metadata_csv(rows, tmp_path / "out.csv")


def test_build_metadata_csv_returns_path(tmp_path):
    """Return value is the Path of the written file and the file exists."""
    rows = [{"plant_qr_code": "a"}]
    out = tmp_path / "x.csv"
    returned = build_metadata_csv(rows, out)
    assert Path(returned) == out
    assert out.exists()


def test_build_metadata_csv_accepts_str_path(tmp_path):
    """Accepts str path (not just Path)."""
    rows = [{"plant_qr_code": "a"}]
    out = tmp_path / "x.csv"
    returned = build_metadata_csv(rows, str(out))
    assert Path(returned) == out
    assert out.exists()


def test_build_metadata_csv_extras_sorted(tmp_path):
    """Non-canonical columns appear at the end, sorted alphabetically."""
    rows = [
        {
            "plant_qr_code": "a",
            "genotype": "X",
            "zzz_extra": 1,
            "aaa_extra": 2,
        }
    ]
    path = build_metadata_csv(rows, tmp_path / "out.csv")
    df = pd.read_csv(path)
    # Canonical: plant_qr_code, genotype. Extras sorted: aaa_extra, zzz_extra.
    assert list(df.columns) == [
        "plant_qr_code",
        "genotype",
        "aaa_extra",
        "zzz_extra",
    ]


# ---------------------------------------------------------------------------
# infer_timepoints_from_filenames
# ---------------------------------------------------------------------------


def test_infer_timepoints_from_filenames_named_groups():
    """Basic case: underscore-separated series_name + integer timepoint."""
    paths = [
        Path("plant1_0.slp"),
        Path("plant1_5.slp"),
        Path("plant1_10.slp"),
    ]
    pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"
    result = infer_timepoints_from_filenames(paths, pattern)
    assert result == {"plant1_0": 0.0, "plant1_5": 5.0, "plant1_10": 10.0}


def test_infer_timepoints_from_filenames_missing_named_groups():
    """ValueError if the pattern lacks series_name or timepoint named groups."""
    with pytest.raises(ValueError, match="series_name"):
        infer_timepoints_from_filenames([Path("plant1_0.slp")], r".+_\d+")


def test_infer_timepoints_from_filenames_skips_non_matches():
    """Non-matching stems are skipped silently."""
    paths = [Path("plant1_0.slp"), Path("garbage.slp")]
    pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"
    result = infer_timepoints_from_filenames(paths, pattern)
    assert result == {"plant1_0": 0.0}


def test_infer_timepoints_from_filenames_casts_to_float():
    """Timepoint values are cast to float even if pattern matches an integer."""
    paths = [Path("plant1_3.slp")]
    pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"
    result = infer_timepoints_from_filenames(paths, pattern)
    assert result["plant1_3"] == 3.0
    assert isinstance(result["plant1_3"], float)
```

- [ ] **Step 2: Run the tests — confirm they fail**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: all 10 FAIL at collection with `ImportError: No module named 'sleap_roots.metadata'`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_metadata.py
git commit -m "test(metadata): Add failing tests for build_metadata_csv + infer_timepoints_from_filenames

TDD red phase for sleap_roots/metadata.py module. 10 tests: 6 for
build_metadata_csv (canonical column order, omit unused, raise on
missing plant_qr_code, return path, str path accepted, extras sorted),
4 for infer_timepoints_from_filenames (named groups, missing groups
raise, non-matches skipped, float cast).

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.6: Implement `sleap_roots/metadata.py`

**Files:**
- Create: `sleap_roots/metadata.py`
- Modify: `sleap_roots/__init__.py`

- [ ] **Step 1: Create `sleap_roots/metadata.py`**

```python
"""Metadata CSV utilities for sleap-roots pipelines.

Pure-function helpers for constructing and introspecting the metadata CSVs
that Series uses to populate `sample_uid`, `timepoint`, `expected_count`,
`group`, and `qc_fail`. Companion to `sleap_roots/series.py`'s
`Series.get_metadata()` accessor.

See issue #169 and
``docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md``
§ Workstream 1.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


# Canonical column order. Columns not present in any input row are omitted
# from the written CSV. Non-canonical columns appear at the end in sorted order.
_CANONICAL_COLUMNS: List[str] = [
    "plant_qr_code",
    "genotype",
    "number_of_plants_cylinder",
    "qc_cylinder",
    "qc_code",
    "timepoint",
]


def build_metadata_csv(
    rows: List[Dict[str, Any]],
    path: Union[str, Path],
) -> Path:
    """Write a metadata CSV from row dicts.

    Every row MUST contain a ``plant_qr_code`` key. Columns in the output CSV
    are ordered as ``_CANONICAL_COLUMNS`` (omitting columns not present in any
    row), followed by any non-canonical columns in sorted order.

    Args:
        rows: List of row dicts. Each row MUST contain ``plant_qr_code``.
        path: Output path for the CSV. Accepts ``str`` or ``Path``.

    Returns:
        The ``Path`` the file was written to.

    Raises:
        ValueError: If any row lacks the ``plant_qr_code`` key.
    """
    for i, row in enumerate(rows):
        if "plant_qr_code" not in row:
            raise ValueError(
                f"Row {i} is missing required 'plant_qr_code' key: {row!r}"
            )

    present = set()
    for row in rows:
        present.update(row.keys())

    canonical_present = [c for c in _CANONICAL_COLUMNS if c in present]
    extras = sorted(present - set(_CANONICAL_COLUMNS))
    columns = canonical_present + extras

    df = pd.DataFrame(rows)
    df = df.reindex(columns=columns)

    out_path = Path(path)
    df.to_csv(out_path.as_posix(), index=False)
    return out_path


def infer_timepoints_from_filenames(
    slp_paths: List[Path],
    pattern: str,
) -> Dict[str, float]:
    """Parse ``(series_name, timepoint)`` from path stems using a regex.

    The pattern MUST contain two named groups: ``series_name`` and
    ``timepoint``. Path stems that don't match the pattern are skipped
    silently. Timepoint values that can't be cast to ``float`` are also
    skipped silently.

    Args:
        slp_paths: List of paths whose stems carry the series_name + timepoint
            encoding.
        pattern: Regex with ``(?P<series_name>...)`` and ``(?P<timepoint>...)``
            named groups.

    Returns:
        Dict mapping ``series_name`` string to ``timepoint`` float.

    Raises:
        ValueError: If ``pattern`` does not contain both ``series_name`` and
            ``timepoint`` named groups.
    """
    compiled = re.compile(pattern)
    groupnames = compiled.groupindex
    missing = {"series_name", "timepoint"} - set(groupnames)
    if missing:
        raise ValueError(
            "pattern must contain named groups 'series_name' and 'timepoint'; "
            f"missing: {sorted(missing)}"
        )

    result: Dict[str, float] = {}
    for p in slp_paths:
        m = compiled.match(Path(p).stem)
        if not m:
            continue
        try:
            tp = float(m.group("timepoint"))
        except (TypeError, ValueError):
            continue
        result[m.group("series_name") + "_" + m.group("timepoint")] = tp
    return result
```

**Note on `series_name` key in the return dict**: the test expects `{"plant1_0": 0.0, ...}` — the full stem, not just the `series_name` group. The regex matches a prefix; the returned key is the full stem that was matched. Let me re-check the test...

Looking at test 5.6: `result == {"plant1_0": 0.0, "plant1_5": 5.0, "plant1_10": 10.0}`. The stem of `Path("plant1_0.slp")` is `plant1_0`. So the key is the full stem. With the pattern `(?P<series_name>.+?)_(?P<timepoint>\d+)` applied to stem `plant1_0`, `series_name` = `plant1` and `timepoint` = `0`. Concatenating: `plant1_0` matches the stem. Good.

BUT: if the stem is `plant_1_0` (underscore in series name), `series_name` = `plant` (lazy match) and `timepoint` = `1`, so concatenation gives `plant_1` which is NOT the full stem. This is a subtle bug. Let me use `m.group(0)` (the whole match) as the key instead:

```python
        result[m.group(0)] = tp
```

That's cleaner — the key is always exactly the matched portion.

- [ ] **Step 2: Fix the return-dict key to use `m.group(0)`**

In `metadata.py`, replace:

```python
        result[m.group("series_name") + "_" + m.group("timepoint")] = tp
```

with:

```python
        result[m.group(0)] = tp
```

- [ ] **Step 3: Add re-exports to `sleap_roots/__init__.py`**

Find the existing imports block. Add:

```python
from sleap_roots.metadata import (
    build_metadata_csv,
    infer_timepoints_from_filenames,
)
```

- [ ] **Step 4: Run the metadata tests — confirm they pass**

Run: `uv run pytest tests/test_metadata.py -v`
Expected: all 10 pass.

- [ ] **Step 5: Run the full suite — confirm no regressions**

Run: `uv run pytest tests/ -x`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add sleap_roots/metadata.py sleap_roots/__init__.py
git commit -m "feat(metadata): Add build_metadata_csv + infer_timepoints_from_filenames (#169)

New module sleap_roots/metadata.py with pure-function CSV-building
utilities. Re-exported at top level.

- build_metadata_csv(rows, path): writes CSV with canonical column order,
  validates plant_qr_code presence, returns path.
- infer_timepoints_from_filenames(paths, pattern): regex-parses
  (series_name, timepoint) from stems using named groups; skips
  non-matching stems silently; raises on malformed patterns.

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.7: Write failing tests for plate pipeline emission

**Files:**
- Modify: `tests/test_multiple_dicot_plate_pipeline.py`

- [ ] **Step 1: Read the existing `_plate_csv` helper**

In `tests/test_multiple_dicot_plate_pipeline.py`, locate the `_plate_csv` helper (used by expected-count tests). Note its column structure for reference.

- [ ] **Step 2: Add 4 new tests after the existing batch test**

Append this code:

```python
# ---------------------------------------------------------------------------
# sample_uid + timepoint emission (issue #169, Workstream 1)
# ---------------------------------------------------------------------------


def _plate_csv_with_timepoint(series_name, expected_count, genotype, timepoint):
    """Helper: plate CSV row with timepoint column."""
    lines = [
        "plant_qr_code,genotype,number_of_plants_cylinder,qc_cylinder,timepoint",
        f"{series_name},{genotype},{expected_count},0,{timepoint}",
    ]
    return "\n".join(lines) + "\n"


def test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_top_level(
    tmp_path,
):
    """Top-level result dict has sample_uid + timepoint keys."""
    primary_pts_list, lateral_pts_list = _synthetic_plate_data(
        frames=[{"primary_xs": [100.0], "lateral_xs": [105.0]}]
    )
    csv_text = _plate_csv_with_timepoint(
        series_name="plate_abc_day3",
        expected_count=1,
        genotype="MK22",
        timepoint=3,
    )
    series = _build_synthetic_series(
        tmp_path=tmp_path,
        series_name="plate_abc_day3",
        sample_uid="plate_abc",
        primary_pts_list=primary_pts_list,
        lateral_pts_list=lateral_pts_list,
        csv_text=csv_text,
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    assert result["sample_uid"] == "plate_abc"
    assert result["timepoint"] == 3


def test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_per_plant(
    tmp_path,
):
    """Every per-plant row has sample_uid + timepoint keys."""
    primary_pts_list, lateral_pts_list = _synthetic_plate_data(
        frames=[{"primary_xs": [100.0, 200.0], "lateral_xs": [105.0, 205.0]}]
    )
    csv_text = _plate_csv_with_timepoint(
        series_name="plate_abc_day3",
        expected_count=2,
        genotype="MK22",
        timepoint=3,
    )
    series = _build_synthetic_series(
        tmp_path=tmp_path,
        series_name="plate_abc_day3",
        sample_uid="plate_abc",
        primary_pts_list=primary_pts_list,
        lateral_pts_list=lateral_pts_list,
        csv_text=csv_text,
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    assert len(result["plants"]) == 2
    for plant in result["plants"]:
        assert plant["sample_uid"] == "plate_abc"
        assert plant["timepoint"] == 3


def test_multiple_dicot_plate_pipeline_csv_column_positions(tmp_path):
    """CSV column order: series, sample_uid, timepoint, frame, plant_id, ..."""
    primary_pts_list, lateral_pts_list = _synthetic_plate_data(
        frames=[{"primary_xs": [100.0], "lateral_xs": [105.0]}]
    )
    series = _build_synthetic_series(
        tmp_path=tmp_path,
        series_name="plate_xyz",
        sample_uid="plate_xyz",
        primary_pts_list=primary_pts_list,
        lateral_pts_list=lateral_pts_list,
        csv_text=None,
    )
    MultipleDicotPlatePipeline().compute_plate_traits(
        series, write_csv=True, output_dir=tmp_path.as_posix()
    )
    import pandas as pd
    df = pd.read_csv((tmp_path / "plate_xyz.plate_traits.csv").as_posix())
    assert list(df.columns)[0] == "series"
    assert list(df.columns)[1] == "sample_uid"
    assert list(df.columns)[2] == "timepoint"
    assert list(df.columns)[3] == "frame"
    assert list(df.columns)[4] == "plant_id"


def test_multiple_dicot_plate_pipeline_timepoint_nan_without_csv(tmp_path):
    """No CSV attached -> top-level + every plant row has NaN timepoint."""
    import math
    primary_pts_list, lateral_pts_list = _synthetic_plate_data(
        frames=[{"primary_xs": [100.0], "lateral_xs": [105.0]}]
    )
    series = _build_synthetic_series(
        tmp_path=tmp_path,
        series_name="plate_nocsv",
        sample_uid=None,  # default to series_name
        primary_pts_list=primary_pts_list,
        lateral_pts_list=lateral_pts_list,
        csv_text=None,
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    assert math.isnan(result["timepoint"])
    assert result["sample_uid"] == "plate_nocsv"  # defaulted to series_name
    for plant in result["plants"]:
        assert math.isnan(plant["timepoint"])
        assert plant["sample_uid"] == "plate_nocsv"
```

**Note on helpers**: the existing test file has `_build_synthetic_series` and `_synthetic_plate_data` helpers (used by the 18 existing tests). Verify their exact names — the test file may have a slightly different naming convention. Adjust the new tests to use the existing helper names. If the existing helpers don't accept a `sample_uid` kwarg, extend them in Task 3.8.

- [ ] **Step 3: Update the existing CSV column-order test**

Locate `test_multiple_dicot_plate_pipeline_csv_output`. The existing assertion is:

```python
assert list(df.columns)[0:6] == ["series", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]
```

Update to:

```python
assert list(df.columns)[0:8] == ["series", "sample_uid", "timepoint", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]
```

- [ ] **Step 4: Run the tests — confirm they fail**

Run: `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -k "sample_uid or timepoint or csv_output" -v`
Expected: 4 new tests FAIL with `KeyError: 'sample_uid'` or similar. Updated existing test FAILS with assertion error on column order.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/test_multiple_dicot_plate_pipeline.py
git commit -m "test(plate): Add failing tests for sample_uid/timepoint emission

TDD red phase. 4 new tests assert plate pipeline emits sample_uid +
timepoint at top level and per plant. 1 updated test captures the new
CSV column order (series, sample_uid, timepoint, frame, ...).

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 3.8: Implement plate pipeline emission

**Files:**
- Modify: `sleap_roots/trait_pipelines.py`

- [ ] **Step 1: Locate `compute_plate_traits` and its result dict initialization**

In `sleap_roots/trait_pipelines.py`, find the method (~line 3050-3090). It starts with:

```python
result: Dict[str, Any] = {
    "schema_version": 1,
    "units": dict(_PLATE_UNITS),
    "series": str(series.series_name),
    "group": series.group,
    "qc_fail": series.qc_fail,
    "expected_count": series.expected_count,
    "plants": [],
}
```

- [ ] **Step 2: Add `sample_uid` and `timepoint` to the top-level dict**

Modify to:

```python
result: Dict[str, Any] = {
    "schema_version": 1,
    "units": dict(_PLATE_UNITS),
    "series": str(series.series_name),
    "sample_uid": str(series.sample_uid),
    "timepoint": series.timepoint,
    "group": series.group,
    "qc_fail": series.qc_fail,
    "expected_count": series.expected_count,
    "plants": [],
}
```

- [ ] **Step 3: Thread `sample_uid` and `timepoint` into `_build_plant_row`**

Locate `_build_plant_row` in the same file (~line 2960). Add two parameters:

```python
    def _build_plant_row(
        self,
        series: Series,
        frame_idx: int,
        plant_id: int,
        primary_sleap_idx: int,
        lateral_sleap_idxs_for_plant: List[int],
        assoc: Dict[str, np.ndarray],
        expected_count: Any,
        detected_count: int,
        dicot_pipeline: "DicotPipeline",
    ) -> Dict[str, Any]:
```

The `series` parameter is already there. Inside the method, when building the output dict, add:

```python
        return {
            "frame": frame_idx,
            "sample_uid": str(series.sample_uid),
            "timepoint": series.timepoint,
            "plant_id": plant_id,
            ...
        }
```

Inserting `sample_uid` and `timepoint` early in the dict (after `frame`) for readability — dict ordering doesn't affect the output CSV, which is controlled by `_build_plate_dataframe`'s `meta_cols` list.

- [ ] **Step 4: Update `_build_plate_dataframe` meta_cols to insert sample_uid + timepoint**

Locate `_build_plate_dataframe` and its `meta_cols` list. Update:

```python
        meta_cols = [
            "series",
            "sample_uid",
            "timepoint",
            "frame",
            "plant_id",
            "primary_sleap_idx",
            "expected_count",
            "detected_count",
        ]
```

Also make sure the `row` dict inside the loop pulls `sample_uid` and `timepoint` — since they're keys on the plant dict built by `_build_plant_row`, they should flow through automatically when `pd.DataFrame(rows)` is called. Verify.

- [ ] **Step 5: Run the 4 new tests + updated test**

Run: `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -k "sample_uid or timepoint or csv_output" -v`
Expected: all 5 pass.

- [ ] **Step 6: Run full plate pipeline test file**

Run: `uv run pytest tests/test_multiple_dicot_plate_pipeline.py -x`
Expected: all 24 tests pass (20 existing + 4 new).

- [ ] **Step 7: Run full test suite**

Run: `uv run pytest tests/ -x`
Expected: all ~449 tests pass (445 previous + 4 new + existing tests still green).

- [ ] **Step 8: Commit**

```bash
git add sleap_roots/trait_pipelines.py tests/test_multiple_dicot_plate_pipeline.py
git commit -m "feat(plate): Emit sample_uid + timepoint in pipeline output (#169)

- compute_plate_traits result dict gains sample_uid (str) and timepoint
  (numeric | NaN) at the top level.
- Every per-plant row in result['plants'] also has sample_uid + timepoint.
- CSV output inserts sample_uid at column position 1 and timepoint at
  position 2 (after 'series'); other columns retain relative order.
- Updated existing column-order test to reflect the new ordering.

Backward compatible for consumers who read CSV by column NAME.
Consumers who read by column POSITION will see a 2-column shift (the
plate pipeline is new in PR #165; no known downstream scripts exist).

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 4: Other pipelines assessment

### Task 4.1: Decide whether to propagate emission to other pipelines

Workstream 1's scope is "plate + trivial extensions". Cylinder and single-plant pipelines have different output shapes (series-level aggregate vs per-plant-per-frame), so threading `sample_uid` + `timepoint` through them is a separate concern.

- [ ] **Step 1: Check if `MultipleDicotPipeline.compute_multiple_dicots_traits` already emits `series.group` / `series.qc_fail`**

Run:

```bash
grep -n "series.group\|series.qc_fail\|series.expected_count" sleap_roots/trait_pipelines.py | head -20
```

Expected: `compute_multiple_dicots_traits` uses `series.group` and `series.qc_fail` already. If so, adding `sample_uid` and `timepoint` is trivial (one-line each in the result dict).

- [ ] **Step 2: If trivial (< 5 lines of code), add the emission to `MultipleDicotPipeline.compute_multiple_dicots_traits`**

Find the result-dict initialization in that method and add:

```python
result = {
    "series": str(series.series_name),
    "sample_uid": str(series.sample_uid),
    "timepoint": series.timepoint,
    "group": str(series.group),
    ...
}
```

No per-plant row changes needed (cylinder aggregate has no per-plant rows today).

Add a test in `tests/test_trait_pipelines.py` (or the cylinder test file) mirroring `test_multiple_dicot_plate_pipeline_emits_sample_uid_and_timepoint_top_level`. If this requires more than a one-liner test + one-liner impl, defer to Step 3.

- [ ] **Step 3: If non-trivial OR if step 2 is complete, file a follow-up issue**

Follow-up issue body:

```
Apply sample_uid + timepoint emission to cylinder + single-plant pipelines.

Workstream 1 of #169 added sample_uid + timepoint emission to
MultipleDicotPlatePipeline [AND MultipleDicotPipeline, if Step 2 completed].
Remaining pipelines that should emit these fields for TimeDiffPipeline (#170)
compatibility:

- DicotPipeline
- MultiplePrimaryRootPipeline
- PrimaryRootPipeline
- LateralRootPipeline
- YoungerMonocotPipeline
- OlderMonocotPipeline

Each pipeline's output shape is different — the addition is additive
(new keys in top-level dict; new columns in CSV at consistent positions).

Related: #169, #170, #159
```

Run: `gh issue create --title "Emit sample_uid + timepoint in remaining pipelines (follow-up of #169)" --body "..."`
Record the issue number.

- [ ] **Step 4: Commit any Step 2 changes**

```bash
git add -A
git commit -m "feat(cylinder): Emit sample_uid + timepoint in MultipleDicotPipeline (#169)

Minor additive change — two new keys in the per-series result dict,
matching the plate pipeline's contract. No CSV shape change (cylinder
output is already per-series aggregate).

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Skip commit if step 2 was skipped.

---

## Phase 5: Follow-up issues

### Task 5.1: File follow-up issues

- [ ] **Step 1: File the "emit in remaining pipelines" issue (if not already filed in Task 4.1 Step 3)**

See Task 4.1 Step 3 for the body.

- [ ] **Step 2: File a "CSV column-name rename" follow-up if not already in #163**

Check if #163 (already broadened in Phase 0) covers the `plant_qr_code` → `sample_uid` column rename. If yes, no action. If no, add it as a comment on #163.

- [ ] **Step 3: Post a comment on #169 linking the landed work**

```bash
gh issue comment 169 --body "PR [opened]($PR_URL) adds sample_uid + timepoint + get_metadata + metadata.py helpers + plate pipeline emission. Follow-ups: #<issue-from-Task-4.1> for remaining pipelines."
```

### Task 5.2: Verification gate

- [ ] **Step 1: Run the follow-up verification command**

```bash
gh issue list --state open --search "#169 in:body" --limit 20 --json number,title
```

Expected: at least one follow-up issue plus any existing cross-references.

- [ ] **Step 2: Copy follow-up issue numbers into PR body**

When creating the PR, include under `## Follow-up issues filed during this PR` the issue number(s) from Task 5.1.

---

## Phase 6: Pre-merge validation

### Task 6.1: Final local CI checks

- [ ] **Step 1: OpenSpec validate**

Run: `openspec validate add-sample-uid-timepoint-metadata --strict`
Expected: `valid`.

- [ ] **Step 2: Black**

Run: `uv run black --check sleap_roots/ tests/`
Expected: clean. If not, run `uv run black sleap_roots/ tests/` and commit the reformat.

- [ ] **Step 3: Pydocstyle**

Run: `uv run pydocstyle --convention=google sleap_roots/`
Expected: exit 0.

- [ ] **Step 4: Full pytest**

Run: `uv run pytest tests/ -x`
Expected: all ~449 tests pass.

### Task 6.2: Pre-PR self-review

- [ ] **Step 1: Invoke `/review-pr` on the local branch**

Dispatch 5 parallel subagents (code quality, testing, scientific rigor, perf/cross-platform, behavioural correctness) on the local diff vs main.

- [ ] **Step 2: Reconcile any BLOCKING + high-value IMPORTANT findings**

Same discipline as PR #165: fix the easy wins inline, defer genuine follow-ups with filed issues.

- [ ] **Step 3: Commit any review fixes**

```bash
git add -A
git commit -m "chore: Self-review polish for metadata-layer PR

Addresses findings from the 5-subagent pre-PR review.

Related: #169

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

### Task 6.3: Open the PR

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feature/metadata-layer-workstream-1
```

- [ ] **Step 2: Create the PR**

```bash
cat > /tmp/metadata-pr-body.md <<'EOF'
## Summary

Workstream 1 of the timelapse/tip-kinematics design (#171, merged). Adds the metadata layer that Workstreams 2 (#129) and 3 (#170) depend on.

Addresses #169.

## What changes

- **NEW** `Series.sample_uid` attrs field + `Series.load(sample_uid=...)` kwarg. Defaults to `series_name` when unset — existing workflows unchanged.
- **NEW** `Series.get_metadata(column, plant_id=None)` generic CSV accessor.
- **NEW** `Series.timepoint` property (thin wrapper).
- **REFACTORED** `Series.expected_count` / `.group` / `.qc_fail` as thin wrappers around `get_metadata()`. Observable behavior preserved.
- **NEW** `sleap_roots/metadata.py` module with `build_metadata_csv()` + `infer_timepoints_from_filenames()` CSV builder helpers; re-exported at top level.
- **MODIFIED** `MultipleDicotPlatePipeline.compute_plate_traits` output — emits `sample_uid` + `timepoint` at top level and in every per-plant row. CSV gains two columns at positions 1 and 2 (after `series`).

## Scope decisions

- Backward compatible: CSV lookup key stays `plant_qr_code`. Column rename to `sample_uid` / `qc` / `expected_count` is deferred to #163.
- Emission of `sample_uid` + `timepoint` to other pipelines (cylinder, single-plant) is tracked as a follow-up issue filed during this PR.

## Test plan

- [x] `uv run pytest tests/`: **~449 passed** (445 prior + 4 new; updated one existing column-order test).
- [x] `uv run black --check`: clean.
- [x] `uv run pydocstyle --convention=google sleap_roots/`: clean.
- [x] `openspec validate add-sample-uid-timepoint-metadata --strict`: passes.

### TDD evidence

Red phase confirmed for each section before green:

- Section 1 (`get_metadata`): 6 tests FAILED with `AttributeError: 'Series' object has no attribute 'get_metadata'`. Then implementation + 6 pass.
- Section 3 (`sample_uid`, `timepoint`): 5 tests; 2 FAILED with `AttributeError`. Then `timepoint` property + 5 pass.
- Section 5 (`metadata.py`): 10 tests FAILED at collection with `ImportError`. Then implementation + 10 pass.
- Section 7 (plate pipeline emission): 4 tests + 1 updated test FAILED with `KeyError`/assertion error. Then implementation + all 5 pass.

## Follow-up issues filed during this PR

- #<N> — emit `sample_uid` + `timepoint` in remaining pipelines (cylinder, single-plant).

## Related

- Design doc: `docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md` § Workstream 1 (PR #171, merged).
- Unblocks #129 (TrackedTipPipeline, Workstream 2) and #170 (TimeDiffPipeline, Workstream 3).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF

gh pr create --title "feat: Add Series metadata layer (sample_uid, timepoint, get_metadata) — Workstream 1 of #169" --body-file /tmp/metadata-pr-body.md
rm /tmp/metadata-pr-body.md
```

- [ ] **Step 3: Monitor CI**

Run: `gh pr checks <PR_NUMBER>`
Expected: Lint, Build Docs, PR Benchmark, Test (ubuntu-22.04), Test (macos-14), Test (windows-2022), codecov/patch, codecov/project all pass. PR Benchmark has no regressions.

- [ ] **Step 4: Address Copilot review feedback if any**

When Copilot posts inline review comments, address them in a follow-up commit (pattern from PR #165).

---

## Self-review checklist

Reviewing this plan against the Workstream 1 spec:

**Spec coverage**:

- ✅ `Series.sample_uid` kwarg + attrs field + property — Task 3.2 Step 3.
- ✅ `Series.timepoint` property — Task 3.4.
- ✅ `Series.get_metadata(column, plant_id=None)` — Tasks 3.1-3.2.
- ✅ Existing properties refactored — Task 3.2 Step 2.
- ✅ `sleap_roots/metadata.py` with `build_metadata_csv` + `infer_timepoints_from_filenames` — Tasks 3.5-3.6.
- ✅ Top-level re-exports — Task 3.6 Step 3.
- ✅ Plate pipeline emission (top-level dict + per-plant row + CSV column positions) — Tasks 3.7-3.8.
- ✅ Backward compatibility (kwarg defaults, column name fallback) — baked into every task.
- ✅ Follow-up discipline — Phase 4 + 5.
- ✅ TDD discipline — each implementation task is preceded by a failing-tests task with a red-phase run step.

**Placeholder scan**: no TBDs, TODOs, or "similar to above" found.

**Type consistency**: `get_metadata` signature is consistent across the plan (`column: str, plant_id: Optional[int] = None -> Any`). `sample_uid` consistently `Optional[str]` in the attrs field and `str` in the property / post-load. `timepoint` consistently `Union[float, int]`.

**Bite-sized**: every task has explicit steps; code blocks shown where code is written.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-24-metadata-layer.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
