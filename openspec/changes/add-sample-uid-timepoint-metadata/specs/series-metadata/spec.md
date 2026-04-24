## ADDED Requirements

### Requirement: `Series.sample_uid` SHALL provide a cross-scan stable identity

A new attrs field `sample_uid: Optional[str]` MUST be added to the `Series` class. `Series.load` MUST accept a `sample_uid: Optional[str] = None` kwarg. When the kwarg is `None` (not passed, or explicitly `None`) OR an empty string `""`, `Series.sample_uid` MUST equal `Series.series_name` (defaulting covers both falsy cases so users who pass `""` don't silently get no-match CSV lookups). When the kwarg is a non-empty string, `Series.sample_uid` MUST equal that string.

The defaulting MUST also apply when a `Series` is constructed directly via `Series(...)` (bypassing `Series.load`): the class's `__attrs_post_init__` MUST set `self.sample_uid = self.series_name` when `self.sample_uid` is `None` or empty. Without this, direct construction leaves `sample_uid` as `None`, breaking every downstream CSV lookup (all existing test fixtures construct Series directly).

#### Scenario: `sample_uid` defaults to `series_name` when the kwarg is omitted

- **Given** a `Series.load(series_name="plant1_day0", primary_path=..., csv_path=None)` call without `sample_uid`
- **When** `series.sample_uid` is read
- **Then** the value equals `"plant1_day0"`

#### Scenario: `sample_uid` kwarg sets the attribute

- **Given** a `Series.load(series_name="plant1_day0", sample_uid="plant1", primary_path=...)` call
- **When** `series.sample_uid` is read
- **Then** the value equals `"plant1"`

#### Scenario: Empty-string `sample_uid` falls through to `series_name`

- **Given** a `Series.load(series_name="plant1_day0", sample_uid="", primary_path=...)` call (empty-string kwarg)
- **When** `series.sample_uid` is read
- **Then** the value equals `"plant1_day0"` (empty string is treated as "not set", same as `None`)

#### Scenario: Direct `Series(...)` construction defaults `sample_uid` via `__attrs_post_init__`

- **Given** a direct `Series(series_name="test_video")` call (bypassing `Series.load`, no `sample_uid` kwarg)
- **When** `series.sample_uid` is read
- **Then** the value equals `"test_video"` (populated by `__attrs_post_init__`, not by `Series.load`)

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
- If `plant_id` is given but the CSV has no `plant_id` column, the `plant_id` argument MUST be silently ignored and the sample-uid-only lookup used. The method MUST additionally log a `logger.warning` (one-shot per Series instance, not per-call — guarded by `self._warned_missing_plant_id_column`) listing the CSV path and explaining that `plant_id` is being ignored. Silent-ignore without signalling hides user bookkeeping errors.
- If no row matches, MUST return `np.nan`.
- If one or more rows match, MUST return the value in the first matching row's `column` field (deterministic first-row semantics via `.iloc[0]`).
- `plant_id=None` (explicit) MUST be treated identically to omitting the `plant_id` argument — both fall into the sample-uid-only branch.

**Known pre-existing limitation (not fixed here; tracked as follow-up)**: If `plant_qr_code` values in the CSV are inferred by pandas as an integer dtype (pure-numeric values like `1002`), and `self.sample_uid` is a string (`"1002"`), the `==` comparison silently matches no rows. This behavior existed in the original `expected_count` / `group` / `qc_fail` properties and is preserved. Users should ensure `plant_qr_code` CSV values are string-like (include at least one non-numeric character, e.g. `1002_1`) or use `dtype={"plant_qr_code": str}` in custom CSV writers. Tracked as a separate follow-up issue.

**NaN-in-cell vs no-matching-row**: both paths return `np.nan`. They are semantically distinct (unrecorded value vs unknown sample) but the current contract collapses them. This is intentional for PR 1 simplicity; a future `strict=True` mode raising on missing rows is deferred as a follow-up.

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

#### Scenario: `plant_id` argument ignored when CSV has no `plant_id` column emits a warning

- **Given** a CSV with columns `{plant_qr_code, genotype, timepoint}` (no `plant_id` column) and a matching row for `sample_uid="plant1"`
- **And** `caplog` capturing log records at WARNING level
- **When** `series.get_metadata("genotype", plant_id=99)` is called
- **Then** the return value is the row's `genotype` value (not NaN)
- **And** `caplog.records` contains at least one WARNING record from logger `sleap_roots.series` whose message mentions the CSV path and "plant_id" being ignored
- **And** a second call to `series.get_metadata("timepoint", plant_id=99)` on the SAME Series instance MUST NOT emit a second warning (one-shot per Series)

#### Scenario: Explicit `plant_id=None` is treated identically to omitting the argument

- **Given** a CSV with columns `{plant_qr_code, genotype}` and a matching row for `sample_uid="plant1"`
- **When** `series.get_metadata("genotype", plant_id=None)` is called
- **And** `series.get_metadata("genotype")` is also called
- **Then** both return values are identical (the row's genotype value)

#### Scenario: Multiple matching rows (no plant_id given) return the first row's value

- **Given** a CSV with two rows both having `plant_qr_code="plant1"` (no `plant_id` column at all) and distinct `genotype` values (`"A"` on the first row, `"B"` on the second)
- **And** a Series with `sample_uid="plant1"`
- **When** `series.get_metadata("genotype")` is called (no `plant_id` kwarg)
- **Then** the return value is `"A"` (first row's value; deterministic `.iloc[0]` semantics)

### Requirement: `Series.timepoint` SHALL return a numeric value coerced from the CSV

The property `Series.timepoint: float` MUST exist. Its implementation MUST call `self.get_metadata("timepoint")` and coerce the result to `float` via `float(...)`. Coercion semantics:

- If `get_metadata` returns `np.nan` (no CSV, missing column, no matching row), the property MUST return `np.nan` (a float).
- If `get_metadata` returns a numeric value (int, numpy int, float, numpy float), the property MUST return `float(value)`.
- If `get_metadata` returns a string that parses as a float (`"2"`, `"2.5"`, `"-1.0"`), the property MUST return the parsed float.
- If `get_metadata` returns a string that does NOT parse as a float (`"2024-03-15"`, `"abc"`), the property MUST raise `ValueError` with a clear message naming the series_name, the column name (`timepoint`), and the raw value. Rationale: downstream `TimeDiffPipeline` arithmetic on non-numeric timepoints would silently produce wrong results or deep stack-trace errors; failing loudly at the metadata layer is correct.

#### Scenario: `timepoint` returns the CSV value coerced to float

- **Given** a CSV with `plant_qr_code="plant1", timepoint=2` and a Series with `sample_uid="plant1"`
- **When** `series.timepoint` is read
- **Then** the return value equals `2.0` (float)
- **And** `isinstance(series.timepoint, float)` is `True`

#### Scenario: `timepoint` returns NaN when no CSV

- **Given** a Series loaded without `csv_path`
- **When** `series.timepoint` is read
- **Then** the return value is `np.nan`

#### Scenario: `timepoint` parses string values that are float-castable

- **Given** a CSV with `plant_qr_code="plant1", timepoint="3.5"` (stored as string) and a Series with `sample_uid="plant1"`
- **When** `series.timepoint` is read
- **Then** the return value equals `3.5`

#### Scenario: `timepoint` raises `ValueError` on non-numeric string values

- **Given** a CSV with `plant_qr_code="plant1", timepoint="2024-03-15"` (a date string) and a Series with `sample_uid="plant1"`
- **When** `series.timepoint` is read
- **Then** `ValueError` is raised with a message containing `"plant1"`, `"timepoint"`, and `"2024-03-15"`

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
- `infer_timepoints_from_filenames(slp_paths: List[Path], pattern: str) -> Dict[str, float]` — regex-parses `(series_name, timepoint)` from the stem of each path using named groups. Raises `ValueError` if the pattern lacks `series_name` or `timepoint` named groups. Skips paths whose stems don't match (does not raise). `timepoint` values are cast to `float`; values that can't be cast are skipped. The returned dict keys on the full matched portion of the stem (regex group 0), not on the `series_name` group alone — this keeps keys uniquely identifying the source path even when `series_name` is a lazy match.

  **Silent-skip logging**: for EVERY skipped path (either because the stem didn't match, OR because `timepoint` couldn't be cast to float), the function MUST emit a `logger.warning` naming the skipped path and the reason. Rationale: silent-skip from 100 `.slp` files is a reproducibility hazard — the user never knows 3 were missed. With logging, `pytest caplog` / CLI output surfaces the skip.

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

#### Scenario: `build_metadata_csv` extras are sorted

- **Given** `rows = [{"plant_qr_code": "a", "genotype": "X", "zzz_extra": 1, "aaa_extra": 2}]`
- **When** `build_metadata_csv(rows, tmp_path / "out.csv")` is called
- **Then** `list(df.columns) == ["plant_qr_code", "genotype", "aaa_extra", "zzz_extra"]`

#### Scenario: `infer_timepoints_from_filenames` parses named groups

- **Given** `slp_paths = [Path("plant1_0.slp"), Path("plant1_5.slp"), Path("plant1_10.slp")]` and `pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"`
- **When** `infer_timepoints_from_filenames(slp_paths, pattern)` is called
- **Then** the return value is `{"plant1_0": 0.0, "plant1_5": 5.0, "plant1_10": 10.0}`

#### Scenario: `infer_timepoints_from_filenames` raises when pattern lacks named groups

- **Given** `pattern = r".+_\d+"` (no named groups)
- **When** `infer_timepoints_from_filenames([Path("plant1_0.slp")], pattern)` is called
- **Then** `ValueError` is raised with a message mentioning `series_name` and `timepoint`

#### Scenario: Non-matching stems are skipped with a warning

- **Given** `slp_paths = [Path("plant1_0.slp"), Path("garbage.slp")]` and a valid pattern
- **And** `caplog` capturing log records at WARNING level for logger `sleap_roots.metadata`
- **When** `infer_timepoints_from_filenames(...)` is called
- **Then** the return value contains only `{"plant1_0": 0.0}`
- **And** `caplog.records` contains at least one WARNING record from `sleap_roots.metadata` whose message mentions `"garbage"` and a skip reason (e.g. "pattern did not match")

#### Scenario: `build_metadata_csv` overwrites an existing file

- **Given** an existing file at `tmp_path / "x.csv"` with arbitrary prior content
- **When** `build_metadata_csv(rows, tmp_path / "x.csv")` is called with valid rows
- **Then** the file is overwritten (pandas `to_csv` default behavior) and the new content matches the written rows
- **And** no exception is raised (no "file exists" error)

### Requirement: Pipeline output emission of `sample_uid` and `timepoint` is specified per-pipeline

The emission of `sample_uid` and `timepoint` into pipeline output (result dicts and CSV/JSON files) is NOT covered by the `series-metadata` capability. Each pipeline's spec MUST specify its own emission behavior via a `MODIFIED Requirement` when the pipeline is updated to consume `Series.sample_uid` / `Series.timepoint`.

This change (add-sample-uid-timepoint-metadata) MODIFIES the `multiple-dicot-plate-pipeline` capability to emit both values (see `specs/multiple-dicot-plate-pipeline/spec.md` in this change). Other pipelines (`MultipleDicotPipeline`, `DicotPipeline`, etc.) are NOT modified by this change — they will receive emission in a separate follow-up issue so the per-pipeline contract is explicit in each capability's spec.

#### Scenario: `series-metadata` capability has no requirement about pipeline output

- **Given** this capability's spec (`series-metadata`)
- **When** a reader searches for requirements about per-pipeline CSV/JSON output
- **Then** no such requirement exists in this capability
- **And** the reader is directed to `multiple-dicot-plate-pipeline` (for the plate pipeline's emission) or future follow-up issues (for other pipelines)
