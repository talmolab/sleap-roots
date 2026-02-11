## MODIFIED Requirements

### Requirement: Support Multiple Video Sources

The viewer SHALL support loading images from both HDF5 files and image directories.

#### Scenario: HDF5 video source
- **WHEN** predictions directory contains `.h5` files alongside `.slp` files
- **THEN** images are loaded from the HDF5 files

#### Scenario: Pipeline output with image directories
- **WHEN** `.slp` files contain embedded paths to image directories that don't exist locally
- **AND** the image directory exists in the predictions directory (by matching directory path)
- **THEN** video paths are remapped to local image directory
- **AND** images are loaded from the local directory

#### Scenario: Multi-timepoint directory matching
- **WHEN** the same plant name exists in multiple timepoint directories (e.g., `Day0/Fado_1` and `Day3/Fado_1`)
- **AND** the embedded path contains distinguishing path components (e.g., `Wave1/Day0/Fado_1`)
- **THEN** video paths are remapped to the correct timepoint directory
- **AND** predictions are displayed on the correct day's images

## ADDED Requirements

### Requirement: Display Plant Name

The viewer SHALL display the plant name (QR code) extracted from the image directory path instead of scan IDs.

#### Scenario: Plant name in gallery
- **WHEN** user views the scan overview gallery
- **THEN** each scan card displays the plant name prominently (e.g., `Fado_1`)
- **AND** the scan ID is shown as secondary information

#### Scenario: Plant name in frame view
- **WHEN** user views a frame
- **THEN** the header displays the plant name prominently
- **AND** the scan ID is available for reference

### Requirement: Timepoint Grouping

The viewer SHALL organize scans by timepoint (parent directory) in the gallery view.

#### Scenario: Grouped gallery view
- **WHEN** user views the scan overview
- **AND** scans come from multiple timepoint directories (e.g., Day0, Day3, Day5)
- **THEN** scans are grouped by timepoint directory
- **AND** each group has a collapsible header showing the timepoint name and scan count

#### Scenario: Single timepoint
- **WHEN** all scans come from the same parent directory
- **THEN** no grouping headers are shown
- **AND** gallery displays as a flat grid

### Requirement: Timepoint Filtering

The viewer SHALL support filtering scans by timepoint pattern via CLI argument.

#### Scenario: Filter by timepoint pattern
- **WHEN** user specifies `--timepoint "Day0*"`
- **THEN** only scans from directories matching the pattern are included
- **AND** other timepoints are excluded from the viewer

#### Scenario: Multiple timepoint patterns
- **WHEN** user specifies multiple `--timepoint` arguments (e.g., `--timepoint "Day0*" --timepoint "Day3*"`)
- **THEN** scans matching any of the patterns are included