# pixel-unit-invariance Specification

## Purpose
TBD - created by archiving change add-pixel-unit-regression-test. Update Purpose after archive.
## Requirements
### Requirement: Pipeline MUST return pixel-unit trait values regardless of source image DPI metadata, end-to-end through .slp serialization

The full data path — image I/O, `.slp` serialization, `sio.load_slp()`, `Series.load()`, pipeline orchestration, and trait computation — MUST produce trait values in pixel units. No layer SHALL apply DPI-based unit conversion.

#### Scenario: Primary root length through full load+pipeline round-trip is in pixels, not DPI-converted mm

- **Given** a synthetic TIFF image of size 200x400 with 1200 DPI metadata created via Pillow
- **And** the TIFF is opened via `sio.Video.from_filename()`, forcing the backend to read file metadata
- **And** a synthetic sleap-io Labels object with a 6-node Skeleton
- **And** an Instance with node coordinates spanning (100, 50) to (100, 150), 100px vertically
- **And** the Labels are serialized to a `.slp` file via `sio.save_slp()`
- **And** the `.slp` file is reloaded via `Series.load(primary_path=...)`
- **When** `PrimaryRootPipeline().compute_plant_traits(series)` is called
- **Then** the `primary_length` column value is 100.0 (pixels)
- **And** the `primary_base_tip_dist` column value is 100.0 (pixels)
- **And** neither value is approximately 2.117 (which would be 100px / 1200 DPI * 25.4 mm/inch)

#### Scenario: Root length computation function returns pixel values for straight segments

- **Given** a synthetic numpy array of shape (1, 2, 2) with nodes at (100, 50) and (100, 150)
- **When** `get_root_lengths()` is called with this array
- **Then** the result is the scalar 100.0 (pixels)

#### Scenario: Root length computation function returns pixel values for polyline roots

- **Given** a synthetic numpy array with 3 nodes forming an L-shape: (0, 0), (0, 60), (80, 60)
- **When** `get_root_lengths()` is called with this array
- **Then** the result is 140.0 (pixels), the sum of segment lengths 60 + 80

#### Scenario: Base-to-tip distance is computed in pixels

- **Given** a synthetic base point array at (100, 50) and tip point array at (100, 150)
- **When** `get_base_tip_dist()` is called with these separate arrays
- **Then** the result is 100.0 (pixels)

#### Scenario: Multi-instance root lengths are all in pixel units

- **Given** a synthetic numpy array of shape (3, 2, 2) with 3 root instances: nodes [(0,0),(0,100)], [(0,0),(0,50)], [(0,0),(0,75)]
- **When** `get_root_lengths()` is called with this array
- **Then** the returned array is [100.0, 50.0, 75.0] in pixels

