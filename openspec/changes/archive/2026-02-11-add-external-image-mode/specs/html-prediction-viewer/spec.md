## MODIFIED Requirements

### Requirement: Generate Self-Contained HTML

The viewer SHALL support three output modes: client-render (default), pre-rendered, and embedded.

#### Scenario: Client-render mode (default)
- **WHEN** user runs `sleap-roots viewer <predictions_dir> -o output.html`
- **THEN** prediction data (skeleton points, edges, scores) is serialized as JSON
- **AND** source images are referenced via relative paths from their filesystem location
- **AND** JavaScript Canvas draws prediction overlays when viewing each frame
- **AND** generation completes in seconds regardless of dataset size

#### Scenario: Client-render with h5 source
- **WHEN** client-render mode is used with h5 video source
- **THEN** frames are extracted as JPEG to `output_images/` directory
- **AND** HTML references the extracted images

#### Scenario: Pre-rendered mode
- **WHEN** user runs `sleap-roots viewer <predictions_dir> -o output.html --render`
- **THEN** matplotlib renders prediction overlays onto each frame
- **AND** rendered images are saved as JPEG to `output_images/` directory
- **AND** HTML references images via relative paths
- **AND** both root-type and confidence views are pre-rendered

#### Scenario: Embedded mode
- **WHEN** user runs `sleap-roots viewer <predictions_dir> -o output.html --embed`
- **THEN** a single self-contained HTML file is created with all images as base64 data URIs
- **AND** no external image directory is created
- **AND** this matches current behavior for backwards compatibility

#### Scenario: Mutually exclusive flags
- **WHEN** user specifies both `--render` and `--embed`
- **THEN** an error is raised indicating the flags are mutually exclusive

#### Scenario: Default image format
- **WHEN** user runs viewer with `--render` without `--format` option
- **THEN** rendered images are saved as JPEG with quality 85

#### Scenario: PNG image format
- **WHEN** user specifies `--format png` with `--render`
- **THEN** rendered images are saved as lossless PNG

#### Scenario: Custom JPEG quality
- **WHEN** user specifies `--quality N` with `--render`
- **THEN** JPEG images are saved with quality N (1-100)

### Requirement: Overlay Toggle

The viewer SHALL support toggling prediction overlays on and off.

#### Scenario: Toggle overlay visibility in client-render mode
- **WHEN** user is in client-render mode viewing a frame
- **THEN** a checkbox allows hiding/showing prediction overlays
- **AND** the underlying source image is visible when overlays are hidden

#### Scenario: Toggle unavailable in pre-rendered mode
- **WHEN** user is in pre-rendered mode viewing a frame
- **THEN** overlay toggle is not available (overlays baked into images)

### Requirement: ZIP Archive for Sharing

The viewer SHALL support creating a zip archive for sharing.

#### Scenario: Zip archive with client-render mode
- **WHEN** user specifies `--zip` with client-render mode
- **THEN** source images are copied into the zip archive
- **AND** HTML paths are rewritten to reference images within the archive
- **AND** the archive is self-contained and portable

#### Scenario: Zip archive with pre-rendered mode
- **WHEN** user specifies `--zip` with `--render`
- **THEN** a zip archive is created containing the HTML file and images directory

#### Scenario: Zip archive with embedded mode
- **WHEN** user specifies `--zip` with `--embed`
- **THEN** a zip archive is created containing only the single HTML file

#### Scenario: Zip file naming
- **WHEN** output path is `path/to/viewer.html`
- **THEN** zip archive is named `path/to/viewer.zip`

### Requirement: Image Directory Naming

The viewer SHALL use consistent naming for output image directories.

#### Scenario: Pre-rendered image directory
- **WHEN** output path is `path/to/viewer.html` with `--render`
- **THEN** images are saved to `path/to/viewer_images/`
- **AND** images are organized as `{scan_name}/frame_{idx}_{mode}.{ext}`

#### Scenario: Extracted h5 frames directory
- **WHEN** output path is `path/to/viewer.html` with h5 source
- **THEN** extracted frames are saved to `path/to/viewer_images/`
- **AND** frames are organized as `{scan_name}/frame_{idx}.{ext}`