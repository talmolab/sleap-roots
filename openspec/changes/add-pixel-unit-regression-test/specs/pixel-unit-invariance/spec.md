# Capability: Pixel Unit Invariance

Trait calculations MUST produce results in pixel units regardless of image DPI metadata.

## ADDED Requirements

### Requirement: Pipeline MUST return pixel-unit trait values regardless of source image DPI metadata

The full pipeline — from data loading through trait computation — MUST produce trait values in pixel units. No layer (image loading, coordinate extraction, pipeline orchestration, or trait computation) SHALL apply DPI-based unit conversion.

#### Scenario: Primary root length through PrimaryRootPipeline is in pixels, not DPI-converted mm

- **Given** a synthetic TIFF image of size 200x400 with 1200 DPI metadata created via Pillow
- **And** a synthetic sleap-io Labels object with a Skeleton containing two nodes
- **And** an Instance with node coordinates at (100, 50) and (100, 150), representing a 100px vertical root
- **And** a Series object constructed from these synthetic predictions
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
