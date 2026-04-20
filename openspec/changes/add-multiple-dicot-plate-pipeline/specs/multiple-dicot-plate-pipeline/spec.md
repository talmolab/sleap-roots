## ADDED Requirements

### Requirement: `MultipleDicotPlatePipeline` class SHALL exist and extend `Pipeline`

A new class `MultipleDicotPlatePipeline` MUST be added to `sleap_roots/trait_pipelines.py`. It MUST extend the `Pipeline` base class via the `@attrs.define` decorator, matching the pattern used by the sibling `MultipleDicotPipeline` class. The class MUST implement `define_traits()` (returning the plate-specific TraitDef DAG) and `get_initial_frame_traits(plant, frame_idx)` (returning the per-frame input dict). The class MUST NOT modify the `Pipeline` base class, `DicotPipeline`, `MultipleDicotPipeline`, or any other existing pipeline.

#### Scenario: Class is importable and instantiable

- **Given** the `sleap_roots` package is installed
- **When** `from sleap_roots.trait_pipelines import MultipleDicotPlatePipeline` is imported and `MultipleDicotPlatePipeline()` is instantiated with no arguments
- **Then** an instance is returned without raising
- **And** `isinstance(instance, Pipeline)` is `True`

#### Scenario: `define_traits` returns the plate DAG

- **Given** an instance `pipeline = MultipleDicotPlatePipeline()`
- **When** `pipeline.define_traits()` is called
- **Then** the returned list contains at least these TraitDef names in dependency order: `primary_pts_no_nans`, `lateral_pts_no_nans`, `detected_count`, `plant_associations_dict`, `plant_id_order`
- **And** `primary_pts_no_nans` uses `filter_roots_with_nans` with input `["primary_pts"]`
- **And** `lateral_pts_no_nans` uses `filter_roots_with_nans` with input `["lateral_pts"]`
- **And** `detected_count` uses `get_count` with input `["primary_pts_no_nans"]`, scalar=True, include_in_csv=True
- **And** `plant_associations_dict` uses `associate_lateral_to_primary` with inputs `["primary_pts_no_nans", "lateral_pts_no_nans"]`
- **And** `plant_id_order` uses `argsort_primaries_by_base_x` with input `["plant_associations_dict"]`
- **And** the list does NOT contain any count-filter TraitDef (no `filter_plants_with_unexpected_ct`), because plates keep all detected plants regardless of expected count

#### Scenario: `get_initial_frame_traits` preserves SLEAP instance indices through NaN filtering

- **Given** a `Series` whose `get_primary_points(frame_idx)` returns an array of shape `(4, 6, 2)` in which instance 1 contains NaN coordinates and instances 0, 2, 3 are valid
- **And** an instance `pipeline = MultipleDicotPlatePipeline()`
- **When** `pipeline.get_initial_frame_traits(series, frame_idx)` is called
- **Then** the returned dict contains keys `primary_pts`, `lateral_pts`, `primary_sleap_idxs`, `lateral_sleap_idxs`, `expected_count`
- **And** `primary_pts` is the raw unfiltered array of shape `(4, 6, 2)` as returned by `series.get_primary_points(frame_idx)` (pre-filter)
- **And** `primary_sleap_idxs` is `[0, 2, 3]` (the original SLEAP indices of primaries that will survive `filter_roots_with_nans`)
- **And** `expected_count` is `series.expected_count` unchanged (pass-through of `None`, `np.nan`, or a numeric value)

### Requirement: `argsort_primaries_by_base_x` helper SHALL exist in `sleap_roots/points.py`

A new helper `argsort_primaries_by_base_x(plant_associations_dict: dict) -> List[int]` MUST be added to `sleap_roots/points.py`. It MUST return a list of primary instance indices (the keys of `plant_associations_dict`) sorted left-to-right by the x-coordinate of each primary's base node (node index 0). Sorting MUST be stable (identical x-values preserve original key order). Empty input MUST return an empty list.

#### Scenario: Three primaries sorted left-to-right

- **Given** a `plant_associations_dict` with three entries whose primary-points arrays have base x-coordinates `{0: 100.0, 1: 50.0, 2: 200.0}` (i.e. primary 0's node-0 x is 100.0, etc.)
- **When** `argsort_primaries_by_base_x(plant_associations_dict)` is called
- **Then** the returned list is `[1, 0, 2]` (primary 1 is leftmost at x=50, then primary 0 at x=100, then primary 2 at x=200)

#### Scenario: Single-plant dict returns single-element list

- **Given** a `plant_associations_dict` with one entry `{5: {"primary_points": ..., "lateral_points": ...}}`
- **When** `argsort_primaries_by_base_x(plant_associations_dict)` is called
- **Then** the returned list is `[5]`

#### Scenario: Empty dict returns empty list

- **Given** an empty dict `{}`
- **When** `argsort_primaries_by_base_x({})` is called
- **Then** the returned list is `[]`
- **And** no exception is raised

#### Scenario: Identical x-values break ties by original key order

- **Given** a `plant_associations_dict` with three entries whose primary-points arrays share the same base x-coordinate `{0: 100.0, 1: 100.0, 2: 100.0}` (inserted in key order 0, 1, 2)
- **When** `argsort_primaries_by_base_x(plant_associations_dict)` is called
- **Then** the returned list is `[0, 1, 2]` (stable sort preserves insertion order on ties)

### Requirement: `compute_plate_traits(series)` SHALL emit a per-series dict with a flat per-plant-per-frame list

`MultipleDicotPlatePipeline.compute_plate_traits(series, write_csv=False, write_json=False, output_dir=".", csv_suffix=".plate_traits.csv", json_suffix=".plate_traits.json")` MUST return a dict with top-level keys `series`, `group`, `qc_fail`, `expected_count`, and `plants`. The `plants` field MUST be a list in which each entry represents exactly one (frame, plant) pair. For every frame `frame_idx` in `range(len(series))`, the method MUST append one entry to `plants` for each plant detected in that frame, in `plant_id` order (left-to-right by primary base x).

Each `plants[i]` entry MUST contain the keys `frame`, `plant_id`, `primary_sleap_idx`, `lateral_sleap_idxs`, `primary_points`, `lateral_points`, `expected_count`, `detected_count`, and `traits`. The `traits` dict MUST be the full output of `DicotPipeline().compute_frame_traits(...)` invoked on that plant's isolated primary and lateral points (no trait renaming, no curation). The `primary_sleap_idx` value MUST be the original SLEAP instance index (into `series.get_primary_points(frame_idx)` pre-`filter_roots_with_nans`); the `lateral_sleap_idxs` value MUST be the list of original SLEAP lateral instance indices associated to this primary.

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

#### Scenario: SLEAP instance indices survive NaN filtering

- **Given** a synthetic `.slp` with 3 primaries of which SLEAP index 1 has all-NaN coordinates (partial prediction) and SLEAP indices 0, 2 are valid
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** `plants` has length 2 (one per valid primary)
- **And** the set `{plants[0]["primary_sleap_idx"], plants[1]["primary_sleap_idx"]}` equals `{0, 2}` (NOT `{0, 1}` — the collapsed post-filter indices)
- **And** `plant_id` values are `0` and `1` in left-to-right order of primary base x

#### Scenario: `expected_count` mismatch does NOT drop any plants

- **Given** a synthetic `.slp` with 3 detected primaries AND a Series loaded with a CSV that specifies `expected_count=2`
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** `plants` has length 3 (all detected primaries retained)
- **And** every `plants[i]["detected_count"] == 3`
- **And** every `plants[i]["expected_count"] == 2`
- **And** the top-level `expected_count` is `2`

#### Scenario: `expected_count` missing is tolerated without error

- **Given** a Series loaded without a `csv_path` so `series.expected_count` is `np.nan`
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** the call completes without raising
- **And** `plants[i]["expected_count"]` is `np.nan` (or `None`) for every plant row
- **And** `plants[i]["detected_count"]` is a non-negative integer for every plant row

#### Scenario: All-NaN primary predictions yield empty `plants` list

- **Given** a synthetic `.slp` where every primary instance has all-NaN node coordinates in frame 0
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** the returned dict has `plants == []`
- **And** no exception is raised

#### Scenario: Multi-frame series produces N_plants × N_frames rows

- **Given** a synthetic `.slp` with 2 frames, each containing the same 3 primaries plus matching laterals
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series)` is called
- **Then** `len(plants) == 6` (3 plants × 2 frames)
- **And** `plants[0..2]["frame"] == 0` and `plants[3..5]["frame"] == 1` (or any ordering whose frame counts sum correctly — order within a frame must still be left-to-right by primary base x)

### Requirement: Per-plant JSON output SHALL include raw points so the file is a self-contained analysis artifact

When `compute_plate_traits(series, write_json=True, output_dir=...)` is invoked, the method MUST write a JSON file whose top-level dict mirrors the in-memory return value. Within each `plants[i]` entry, `primary_points` MUST be a nested list (serialized via the existing `NumpyArrayEncoder` at `sleap_roots/trait_pipelines.py:123`) representing the full `(n_nodes, 2)` primary-root points for that plant, and `lateral_points` MUST be the `(n_laterals, n_nodes, 2)` laterals associated to that primary (nested lists). The `traits` dict MUST carry every trait emitted by `DicotPipeline.compute_frame_traits`, preserving both scalar and non-scalar entries (non-scalars serialize as nested lists). Round-tripping the JSON via `json.load` MUST yield structurally equivalent nested-list content for `primary_points`, `lateral_points`, `primary_sleap_idx` (int), and `lateral_sleap_idxs` (list of ints).

#### Scenario: Written JSON is self-contained and round-trips

- **Given** a synthetic `.slp` loaded as a Series with 2 plants in frame 0
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series, write_json=True, output_dir=tmp_path)` is called
- **And** the resulting JSON file is read back via `json.load`
- **Then** the parsed top-level dict contains keys `series`, `group`, `qc_fail`, `expected_count`, `plants`
- **And** each plant entry contains `primary_points` as a list of `[x, y]` pairs
- **And** each plant entry contains `lateral_points` as a list of lists of `[x, y]` pairs
- **And** each plant entry contains `primary_sleap_idx` as an integer and `lateral_sleap_idxs` as a list of integers
- **And** each plant entry's `traits` dict contains the DicotPipeline trait names unchanged (no `_root_` infix, no plate-specific renames)

### Requirement: Per-plant CSV output SHALL emit metadata columns first, then the full DicotPipeline CSV trait set with unchanged names

When `compute_plate_traits(series, write_csv=True, output_dir=...)` is invoked, the method MUST write a CSV file whose first six columns are, in order: `series`, `frame`, `plant_id`, `primary_sleap_idx`, `expected_count`, `detected_count`. The remaining columns MUST be exactly the set produced by `DicotPipeline().csv_traits` (the property at `sleap_roots/trait_pipelines.py:262-274`), in the order defined by that property — **no renaming**, **no new synonyms**. Scalar DicotPipeline traits emit one column; non-scalar traits emit the `{name}_{min,max,mean,median,std,p5,p25,p75,p95}` expansion. Empty cells MUST be used for `np.nan` or `None` values (pandas default `na_rep=""`). `lateral_sleap_idxs` MUST NOT appear in CSV (variable-length list; JSON only).

#### Scenario: CSV column order and trait-name preservation

- **Given** a synthetic `.slp` loaded as a Series with 1 plant in frame 0
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series, write_csv=True, output_dir=tmp_path)` is called
- **And** the resulting CSV is read back via `pandas.read_csv`
- **Then** `list(df.columns)[0:6]` equals `["series", "frame", "plant_id", "primary_sleap_idx", "expected_count", "detected_count"]`
- **And** the remaining columns equal `DicotPipeline().csv_traits` (same set, same order)
- **And** column names like `primary_length`, `lateral_count`, `network_length`, `primary_base_tip_dist`, `lateral_lengths_mean`, `lateral_lengths_max` are present
- **And** no column contains the substring `_root_` (e.g. `primary_root_length`, `lateral_root_count` MUST NOT exist)

#### Scenario: Missing expected_count renders as empty cell

- **Given** a Series loaded without a CSV (so `series.expected_count` is `np.nan`) and 1 detected plant
- **When** `MultipleDicotPlatePipeline().compute_plate_traits(series, write_csv=True, output_dir=tmp_path)` is called
- **And** the resulting CSV is read back as raw text
- **Then** the `expected_count` cell for that plant row is empty (e.g. `"...,0,0,,3,..."` — two commas surrounding no value)
- **And** the `detected_count` cell is `"3"` (populated)

### Requirement: `compute_batch_plate_traits(all_series)` SHALL concatenate per-series CSV rows

`MultipleDicotPlatePipeline.compute_batch_plate_traits(all_series: List[Series], write_csv=False, write_json=False, output_dir=".", csv_name="plate_batch_traits.csv", json_name="plate_batch_traits.json")` MUST return a `pandas.DataFrame` whose rows are the concatenation of every per-plant-per-frame row produced by `compute_plate_traits` across all input Series, preserving per-series row order. When `write_csv=True`, the DataFrame MUST be written to `output_dir/csv_name`. When `write_json=True`, a JSON file at `output_dir/json_name` MUST contain a list whose elements are the per-series dicts returned by `compute_plate_traits` (same self-contained format as the single-series JSON).

#### Scenario: Two synthetic series concatenate into one DataFrame

- **Given** two synthetic `.slp` Series `seriesA` (2 plants, 1 frame) and `seriesB` (3 plants, 1 frame)
- **When** `MultipleDicotPlatePipeline().compute_batch_plate_traits([seriesA, seriesB])` is called
- **Then** the returned DataFrame has exactly 5 rows (2 + 3)
- **And** rows 0-1 have `df.iloc[:, 0] == seriesA.series_name`
- **And** rows 2-4 have `df.iloc[:, 0] == seriesB.series_name`
- **And** the column order matches the single-series CSV (metadata columns first, then `DicotPipeline().csv_traits`)

#### Scenario: Batch JSON emits list of per-series dicts

- **Given** two synthetic `.slp` Series
- **When** `compute_batch_plate_traits([seriesA, seriesB], write_json=True, output_dir=tmp_path, json_name="batch.json")` is called
- **And** the resulting `tmp_path/batch.json` is read back via `json.load`
- **Then** the parsed value is a list of length 2
- **And** each element is a dict with keys `series`, `group`, `qc_fail`, `expected_count`, `plants`
