## MODIFIED Requirements

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