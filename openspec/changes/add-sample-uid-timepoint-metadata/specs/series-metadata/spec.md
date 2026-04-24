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
- `infer_timepoints_from_filenames(slp_paths: List[Path], pattern: str) -> Dict[str, float]` — regex-parses `(series_name, timepoint)` from the stem of each path using named groups. Raises `ValueError` if the pattern lacks `series_name` or `timepoint` named groups. Skips paths whose stems don't match (does not raise). `timepoint` values are cast to `float`; values that can't be cast are skipped. The returned dict keys on the full matched portion of the stem (regex group 0), not on the `series_name` group alone — this keeps keys uniquely identifying the source path even when `series_name` is a lazy match.

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
