# HTML Prediction Viewer

Self-contained HTML viewer for visualizing SLEAP root predictions.

## Requirements

### Requirement: Generate Self-Contained HTML

The viewer SHALL generate a single HTML file containing all images, styles, and scripts with no external dependencies.

#### Scenario: Basic generation
- **WHEN** user runs `sleap-roots viewer <predictions_dir> -o output.html`
- **THEN** a self-contained HTML file is created at `output.html`

### Requirement: Support Multiple Video Sources

The viewer SHALL support loading images from both HDF5 files and image directories.

#### Scenario: HDF5 video source
- **WHEN** predictions directory contains `.h5` files alongside `.slp` files
- **THEN** images are loaded from the HDF5 files

#### Scenario: Pipeline output with image directories
- **WHEN** `.slp` files contain embedded paths to image directories that don't exist locally
- **AND** the image directory exists in the predictions directory (by matching directory name)
- **THEN** video paths are remapped to local image directory
- **AND** images are loaded from the local directory

### Requirement: Dual Visualization Modes

The viewer SHALL provide two visualization modes: root type overlay and confidence colormap.

#### Scenario: Root type view
- **WHEN** user views a frame in root type mode
- **THEN** roots are colored by type (primary, lateral, crown)

#### Scenario: Confidence view
- **WHEN** user views a frame in confidence mode
- **THEN** roots are colored by normalized confidence score using a colormap

#### Scenario: Confidence badge on overview
- **WHEN** user views the scan overview
- **THEN** each scan card shows a normalized prediction score (0-1 range)
- **AND** the badge uses the viridis colormap (same as confidence overlay)
- **AND** the score is labeled "Score:" with a tooltip explaining its meaning

#### Scenario: Confidence in frame stats
- **WHEN** user views a frame
- **THEN** the stats bar shows a normalized prediction score (0-1 range)
- **AND** the score is labeled "Score:" with a tooltip explaining its meaning

### Requirement: Keyboard Navigation

The viewer SHALL support keyboard navigation for efficient review.

#### Scenario: Frame navigation
- **WHEN** user presses left/right arrow keys in frame view
- **THEN** the previous/next frame is displayed

#### Scenario: Scan selection
- **WHEN** user presses arrow keys in overview
- **THEN** scan selection moves accordingly
- **AND** pressing Enter opens the selected scan

#### Scenario: View toggle
- **WHEN** user presses 'C' key
- **THEN** visualization mode toggles between root type and confidence

### Requirement: Frame Sampling for Performance

The viewer SHALL sample frames by default to ensure reasonable file sizes and browser performance.

#### Scenario: Default frame sampling
- **WHEN** user runs viewer without `--max-frames` option
- **THEN** 10 frames are sampled per scan (evenly distributed, including first and last)

#### Scenario: Custom frame sampling
- **WHEN** user specifies `--max-frames N`
- **THEN** N frames are sampled per scan
- **AND** if N is 0, all frames are included

#### Scenario: Warning for large output
- **WHEN** total frames across all scans exceeds 100
- **THEN** a warning message is displayed

#### Scenario: Hard limit protection
- **WHEN** total frames across all scans exceeds 1000
- **AND** `--no-limit` is not specified
- **THEN** generation fails with an error message
- **AND** user is instructed to use `--no-limit` or reduce `--max-frames`

#### Scenario: Override hard limit
- **WHEN** `--no-limit` is specified
- **THEN** no frame limit is enforced

### Requirement: Progress Feedback

The viewer SHALL provide progress feedback during generation.

#### Scenario: Progress display
- **WHEN** viewer generation is running
- **THEN** a progress indicator shows current scan name and frame progress
- **AND** the indicator updates as each frame is rendered