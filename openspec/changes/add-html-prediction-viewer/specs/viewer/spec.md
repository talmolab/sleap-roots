## ADDED Requirements

### Requirement: HTML Viewer Generation

The system SHALL generate a self-contained HTML file for visualizing SLEAP prediction overlays on root images.

#### Scenario: Generate viewer from predictions directory
- **GIVEN** a directory containing .slp prediction files from a pipeline run
- **WHEN** the user runs `sleap-roots viewer <predictions_dir> --output viewer.html`
- **THEN** the system generates a valid HTML5 file containing:
  - Scan overview with thumbnail grid
  - Frame drill-down for each scan
  - Prediction overlays rendered on images
  - Embedded images as base64 data URIs

#### Scenario: Generate viewer with external images
- **GIVEN** a predictions directory and a separate images directory
- **WHEN** the user runs `sleap-roots viewer <predictions_dir> --images <images_dir> --output viewer.html`
- **THEN** the system loads images from the specified directory for overlay rendering

#### Scenario: Handle missing images gracefully
- **GIVEN** a predictions directory without corresponding images
- **WHEN** the user runs `sleap-roots viewer <predictions_dir> --output viewer.html`
- **THEN** the system generates the viewer with placeholder images or skeleton-only visualization
- **AND** displays a warning message about missing images

---

### Requirement: Scan Overview Navigation

The system SHALL provide a thumbnail grid view for quick overview of all scans in an experiment.

#### Scenario: Display scan thumbnails
- **GIVEN** a generated HTML viewer with multiple scans
- **WHEN** the user opens the HTML file in a browser
- **THEN** the system displays a grid of scan thumbnails
- **AND** each thumbnail shows a representative frame (first frame with predictions)
- **AND** each thumbnail displays the scan name

#### Scenario: Navigate to scan details
- **GIVEN** the scan overview grid is displayed
- **WHEN** the user clicks on a scan thumbnail
- **THEN** the system navigates to the frame drill-down view for that scan

#### Scenario: Keyboard navigation in overview
- **GIVEN** the scan overview grid is displayed
- **WHEN** the user presses arrow keys
- **THEN** the selection moves to adjacent thumbnails
- **AND** pressing Enter opens the selected scan's drill-down view

---

### Requirement: Frame Drill-Down Navigation

The system SHALL provide frame-by-frame navigation within a single scan.

#### Scenario: Display frame with overlay
- **GIVEN** the frame drill-down view for a scan
- **WHEN** the view is displayed
- **THEN** the system shows the current frame image with prediction overlays
- **AND** displays the frame index (e.g., "Frame 15 of N" where N is the total frames in that scan)
- **AND** displays navigation controls

#### Scenario: Navigate between frames with keyboard
- **GIVEN** the frame drill-down view is displayed
- **WHEN** the user presses the right arrow key
- **THEN** the system displays the next frame
- **AND** when the user presses the left arrow key
- **THEN** the system displays the previous frame

#### Scenario: Return to overview
- **GIVEN** the frame drill-down view is displayed
- **WHEN** the user presses Escape key
- **THEN** the system returns to the scan overview grid

#### Scenario: Navigate past boundaries
- **GIVEN** the frame drill-down view showing the last frame
- **WHEN** the user presses the right arrow key
- **THEN** the system wraps to the first frame (or stays on last frame)

---

### Requirement: Prediction Quality Indicators

The system SHALL display prediction quality metrics to help scientists assess model performance.

#### Scenario: Display instance counts
- **GIVEN** a frame with predicted root instances
- **WHEN** the frame is displayed
- **THEN** the system shows the number of detected instances per root type
- **AND** displays counts for primary, lateral, and/or crown roots as applicable

#### Scenario: Display mean confidence score
- **GIVEN** a frame with predicted instances
- **WHEN** the frame is displayed in drill-down view
- **THEN** the system shows the mean confidence score for the frame

---

### Requirement: Toggle Visualization Mode

The system SHALL provide a toggle to switch between root type coloring and confidence coloring.

#### Scenario: Root type view (default)
- **GIVEN** the frame drill-down view is displayed
- **WHEN** the root type view is active (default)
- **THEN** the overlay colors represent root types (e.g., primary, lateral, crown)
- **AND** each root type is displayed with a distinct color

#### Scenario: Confidence view
- **GIVEN** the frame drill-down view is displayed
- **WHEN** the user toggles to confidence view
- **THEN** the overlay colors represent confidence scores
- **AND** colors use a continuous colormap (e.g., viridis) mapped to the confidence score range
- **AND** a color legend is displayed showing the score-to-color mapping

#### Scenario: Toggle via keyboard or button
- **GIVEN** the frame drill-down view is displayed
- **WHEN** the user presses a toggle key (e.g., 'C') or clicks a toggle button
- **THEN** the view switches between root type and confidence modes
- **AND** the current mode is indicated in the UI

---

### Requirement: Composite Root Type Rendering

The system SHALL render all available root types as a composite overlay on each frame image.

#### Scenario: Primary root only predictions
- **GIVEN** predictions containing only primary root skeletons
- **WHEN** the viewer is generated
- **THEN** the system renders primary root overlays on the image

#### Scenario: Dicot predictions (primary + lateral)
- **GIVEN** predictions containing primary and lateral root skeletons
- **WHEN** the viewer is generated
- **THEN** the system renders both root types together on the same image
- **AND** each root type is displayed with a distinct color

#### Scenario: Monocot predictions (primary + crown or crown only)
- **GIVEN** predictions containing crown root skeletons
- **WHEN** the viewer is generated
- **THEN** the system renders crown root overlays on the image
- **AND** renders primary roots on the same image if present
- **AND** each root type is displayed with a distinct color

---

### Requirement: CLI Interface

The system SHALL provide a command-line interface for generating HTML viewers.

#### Scenario: Basic CLI usage
- **GIVEN** a valid predictions directory
- **WHEN** the user runs `sleap-roots viewer <predictions_dir>`
- **THEN** the system generates `viewer.html` in the current directory

#### Scenario: Custom output path
- **GIVEN** a valid predictions directory
- **WHEN** the user runs `sleap-roots viewer <predictions_dir> --output /path/to/output.html`
- **THEN** the system generates the viewer at the specified path

#### Scenario: Progress feedback
- **GIVEN** a large experiment with many scans
- **WHEN** the viewer generation is running
- **THEN** the system displays a progress bar showing generation status

#### Scenario: Invalid input handling
- **GIVEN** an invalid or non-existent predictions directory
- **WHEN** the user runs `sleap-roots viewer <invalid_path>`
- **THEN** the system displays a clear error message
- **AND** exits with a non-zero status code

---

### Requirement: Offline Functionality

The system SHALL generate viewers that work without network connectivity.

#### Scenario: Standalone HTML file
- **GIVEN** a generated HTML viewer file
- **WHEN** the file is opened in a browser without internet connection
- **THEN** all functionality works correctly
- **AND** no external resources are required

#### Scenario: Shareable output
- **GIVEN** a generated HTML viewer file
- **WHEN** the file is shared with another user via email or file transfer
- **THEN** the recipient can open and use the viewer without additional setup