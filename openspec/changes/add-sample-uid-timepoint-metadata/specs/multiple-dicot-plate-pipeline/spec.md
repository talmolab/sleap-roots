## MODIFIED Requirements

### Requirement: `compute_plate_traits(series)` SHALL emit a per-series dict with a flat per-plant-per-frame list

`MultipleDicotPlatePipeline.compute_plate_traits(series, write_csv=False, write_json=False, output_dir=".", csv_suffix=".plate_traits.csv", json_suffix=".plate_traits.json")` MUST return a dict with top-level keys `schema_version`, `units`, `series`, `sample_uid`, `timepoint`, `group`, `qc_fail`, `expected_count`, and `plants`. The same keys appear in the written JSON after `json.dump` (see Requirement on JSON output). The `plants` field MUST be a list in which each entry represents exactly one (frame, plant) pair. For every frame `frame_idx` in `range(len(series))`, the method MUST append one entry to `plants` for each plant detected in that frame, in `plant_id` order (left-to-right by primary base x).

The top-level `sample_uid` value MUST equal `str(series.sample_uid)`. The top-level `timepoint` value MUST equal `series.timepoint` (a float coerced from the CSV `timepoint` column, or `np.nan` when the CSV is absent, the column is absent, or no row matches).

Each `plants[i]` entry MUST contain the keys `frame`, `sample_uid`, `timepoint`, `plant_id`, `primary_sleap_idx`, `lateral_sleap_idxs`, `primary_points`, `lateral_points`, `expected_count`, `detected_count`, and `traits`. Per-plant `sample_uid` and `timepoint` MUST equal the top-level values (same series → same identity/time for every plant). The `traits` dict MUST contain every trait from `DicotPipeline().compute_frame_traits(...)` that is flagged `include_in_csv=True` (i.e. the exact set `DicotPipeline().csv_traits_multiple_plants`). Trait names are emitted unchanged — no renaming. Intermediate helper traits flagged `include_in_csv=False` (raw-point ndarrays like `primary_max_length_pts`, node-index arrays, Shapely `Point` and `ConvexHull` geometry primitives) are EXCLUDED because they are not JSON-serializable and are internal DAG plumbing rather than analysis-ready outputs. The `primary_sleap_idx` value MUST be the original SLEAP instance index (into `series.get_primary_points(frame_idx)` pre-`filter_roots_with_nans`); the `lateral_sleap_idxs` value MUST be the list of original SLEAP lateral instance indices associated to this primary.

#### Scenario: Three-plant synthetic frame yields three plants in left-to-right order

- **Given** a synthetic `.slp` file written via `sio.save_slp` containing one frame with 3 primary-root instances whose node-0 x-coordinates are `[200, 50, 100]` (arbitrary y), plus 0 laterals
- **And** a `Series` loaded from that `.slp` via `Series.load`
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** the returned dict has key `plants` with length 3
- **And** `plants[0]["plant_id"] == 0` with `primary_sleap_idx == 1` (the primary at x=50)
- **And** `plants[1]["plant_id"] == 1` with `primary_sleap_idx == 2` (the primary at x=100)
- **And** `plants[2]["plant_id"] == 2` with `primary_sleap_idx == 0` (the primary at x=200)
- **And** `plants[0]["frame"] == plants[1]["frame"] == plants[2]["frame"] == 0`
- **And** every entry has `detected_count == 3`
- **And** every entry's `traits` dict contains at least `primary_length`, `lateral_count`, `network_length`, `primary_base_tip_dist` (i.e. the DicotPipeline trait names, unchanged)

#### Scenario: Top-level `sample_uid` and `timepoint` are populated from the Series

- **Given** a synthetic `.slp` loaded with `Series.load(..., sample_uid="plate_abc")` and a CSV containing a row `plant_qr_code=plate_abc, timepoint=3`
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** `result["sample_uid"] == "plate_abc"`
- **And** `result["timepoint"] == 3.0`
- **And** every entry in `result["plants"]` has `sample_uid == "plate_abc"` and `timepoint == 3.0`

#### Scenario: Missing CSV yields defaulted `sample_uid` and NaN `timepoint`

- **Given** a Series loaded without `csv_path` and without an explicit `sample_uid` kwarg (series_name = "plate_nocsv")
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** `result["sample_uid"] == "plate_nocsv"` (defaulted to series_name)
- **And** `math.isnan(result["timepoint"])` is True
- **And** every entry in `result["plants"]` has `sample_uid == "plate_nocsv"` and `math.isnan(plant["timepoint"])` is True

### Requirement: Per-plant CSV output SHALL emit metadata columns first, then the full DicotPipeline CSV trait set with unchanged names

When `compute_plate_traits(series, write_csv=True, output_dir=...)` is invoked, the method MUST write a CSV file whose **first eight columns** are, in order: `series`, `sample_uid`, `timepoint`, `frame`, `plant_id`, `primary_sleap_idx`, `expected_count`, `detected_count`. The remaining columns MUST be exactly the set produced by `DicotPipeline().csv_traits` (the property at `sleap_roots/trait_pipelines.py:262-274`), in the order defined by that property — **no renaming**, **no new synonyms**. Scalar DicotPipeline traits emit one column; non-scalar traits emit the `{name}_{min,max,mean,median,std,p5,p25,p75,p95}` expansion. Empty cells MUST be used for `np.nan` or `None` values (pandas default `na_rep=""`). `lateral_sleap_idxs` MUST NOT appear in CSV (variable-length list; JSON only). The `sample_uid` column defaults to `series_name` when `Series.load(sample_uid=...)` is not explicitly set — callers who do not need cross-scan identity will see `sample_uid == series` on every row (intentional: schema stability over column-count minimization).

**Rationale for columns 1 and 2** (`sample_uid`, `timepoint` inserted after `series`, shifting the other metadata columns by 2): these are identity-metadata columns belonging with `series`, and grouping them keeps the metadata block contiguous. Existing positional readers of the plate CSV are affected; see the `schema_version` requirement below for the breakage-handling policy.

#### Scenario: CSV column order and trait-name preservation

- **Given** a synthetic `.slp` loaded as a Series with 1 plant in frame 0
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series, write_csv=True, output_dir=tmp_path)` is called
- **And** the resulting CSV is read back via `pandas.read_csv`
- **Then** `list(df.columns)[0:8]` equals `["series", "sample_uid", "timepoint", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`
- **And** the remaining columns equal `DicotPipeline().csv_traits` (same set, same order)
- **And** column names like `primary_length`, `lateral_count`, `network_length`, `primary_base_tip_dist`, `lateral_lengths_mean`, `lateral_lengths_max` are present
- **And** no column contains the substring `_root_` (e.g. `primary_root_length`, `lateral_root_count` MUST NOT exist)

#### Scenario: Missing expected_count renders as empty cell

- **Given** a Series loaded without a CSV (so `series.expected_count` is `np.nan`) and 1 detected plant
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series, write_csv=True, output_dir=tmp_path)` is called
- **And** the resulting CSV is loaded via `df = pd.read_csv(path)`
- **Then** `pd.isna(df.loc[0, "expected_count"])` is `True` (the cell was empty and pandas parsed it as NaN)
- **And** `df.loc[0, "detected_count"] == 1` (populated, non-null; matches the 1 detected plant in the Given)
- **And** `"count_validated"` is NOT in `df.columns`
- **And** `"count_mismatch"` is NOT in `df.columns`
- **And** `pd.isna(df.loc[0, "timepoint"])` is `True` (no CSV attached → NaN)
- **And** `df.loc[0, "sample_uid"] == df.loc[0, "series"]` (defaulted to series_name)

### Requirement: Per-plant JSON output SHALL include raw points, schema metadata, and RFC-8259-valid encoding

When `compute_plate_traits(series, write_json=True, output_dir=...)` is invoked, the method MUST write a JSON file whose top-level dict mirrors the in-memory return value. The top-level dict MUST include `"schema_version": 2` (int; **bumped from 1** because this change shifts CSV column positions and adds top-level keys, which is not a purely additive change — see Requirement "schema_version bump" below) and a `"units"` object that identifies the unit for each trait family, structured as:

```json
"units": {
  "lengths": "pixels",
  "areas": "pixels^2",
  "inverse_lengths": "1/pixels",
  "angles": "degrees",
  "counts": "unitless",
  "ratios": "dimensionless",
  "indices": "unitless",
  "time": "unspecified"
}
```

The `"time"` family covers the `timepoint` value. Its value SHOULD be populated from a `time_unit` kwarg on the pipeline (e.g. `"days"`, `"hours"`, `"frames"`, `"seconds"`) or from a `timepoint_unit` column in the metadata CSV. When neither is provided, the unit is `"unspecified"` and downstream consumers MUST treat `timepoint` values as unit-less numeric coordinates with unknown conversion to physical time. This keeps plate JSON self-contained: a consumer reading the JSON alone can tell whether the `timepoint` values are meaningful physical durations or just ordering indices.

Within each `plants[i]` entry, `primary_points` MUST be a nested list representing the full `(n_nodes, 2)` primary-root points for that plant, and `lateral_points` MUST be the `(n_laterals, n_nodes, 2)` laterals associated to that primary (nested lists). The `traits` dict MUST carry exactly the set `DicotPipeline().csv_traits_multiple_plants` (all traits flagged `include_in_csv=True`). Scalar traits serialize as numbers; non-scalar arrays serialize as nested lists. Intermediate helpers flagged `include_in_csv=False` (Shapely `Point`, scipy `ConvexHull`, raw-point ndarrays) are NOT serialized. Round-tripping the JSON via `json.load` MUST yield structurally equivalent nested-list content for `primary_points`, `lateral_points`, `primary_sleap_idx` (int), and `lateral_sleap_idxs` (list of ints; `[]` when the plant has zero laterals, NOT `[null]` or missing key).

The JSON writer MUST emit Python `float('nan')` and numpy NaN values as JSON `null`, NOT the non-standard bare `NaN` literal that Python's default `json.dumps` produces (which is invalid per RFC 8259 and rejected by JavaScript `JSON.parse`, Jackson strict mode, Go `encoding/json`, and `jq`). **Implementation mechanism**: the result dict MUST be passed through a recursive pre-serialization sanitizer (a helper walking dicts, lists, and converting ndarrays to lists via `.tolist()`) that replaces all NaN floats with `None` **before** `json.dump` is called. A `JSONEncoder.default()` subclass hook alone is insufficient — CPython's fast path for native `float` (including `np.float64`, which subclasses `float`) bypasses `default()`. After sanitizing, call `json.dump(sanitized, f, cls=NumpyArrayEncoder, allow_nan=False, ...)` — `allow_nan=False` is defense-in-depth: if any NaN survives the walker, `json.dump` raises with a clear error rather than producing invalid output.

#### Scenario: Written JSON is self-contained and round-trips

- **Given** a synthetic `.slp` loaded as a Series with 2 plants in frame 0, with `Series.load(sample_uid="plate_abc", ...)` and a CSV row `plant_qr_code=plate_abc, timepoint=3`
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series, write_json=True, output_dir=tmp_path)` is called
- **And** the resulting JSON file is read back via `json.load`
- **Then** the parsed top-level dict contains keys `schema_version`, `units`, `series`, `sample_uid`, `timepoint`, `group`, `qc_fail`, `expected_count`, `plants`
- **And** `schema_version == 2`
- **And** `units` is a structured dict containing at least keys `{"lengths", "areas", "inverse_lengths", "angles", "counts", "ratios", "indices", "time"}` with the values matching the Requirement text
- **And** `sample_uid == "plate_abc"` and `timepoint == 3.0`
- **And** each plant entry contains `primary_points` as a list of `[x, y]` pairs
- **And** each plant entry contains `lateral_points` as a list of lists of `[x, y]` pairs
- **And** each plant entry contains `sample_uid == "plate_abc"`, `timepoint == 3.0`, `primary_sleap_idx` as an integer, and `lateral_sleap_idxs` as a list of integers
- **And** each plant entry contains `count_validated` (bool) and `count_mismatch` (bool) as JSON-native booleans (verified via `isinstance(plants[i]["count_validated"], bool)`)
- **And** each plant entry's `traits` dict contains the DicotPipeline trait names unchanged (no `_root_` infix, no plate-specific renames)

## ADDED Requirements

### Requirement: `schema_version` SHALL bump to 2 when the plate pipeline output shape changes non-additively

When a change to `MultipleDicotPlatePipeline`'s output shape shifts CSV column positions, renames keys, or removes fields, the `schema_version` field in the top-level JSON dict MUST bump by 1 (`1 → 2`, `2 → 3`, etc.). Additions that do not shift positions or rename (e.g. a new scalar trait added to `DicotPipeline.csv_traits`) remain at the current version.

This change (add-sample-uid-timepoint-metadata) inserts `sample_uid` and `timepoint` at CSV column positions 1 and 2, shifting every other column — non-additive — therefore `schema_version` bumps to `2`.

#### Scenario: Written JSON has `schema_version: 2` after this change

- **Given** any synthetic plate Series
- **When** `compute_plate_traits(series, write_json=True, output_dir=tmp_path)` is called
- **And** the written JSON is read back via `json.load`
- **Then** `parsed["schema_version"] == 2`
