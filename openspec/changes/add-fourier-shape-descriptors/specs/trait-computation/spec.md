## ADDED Requirements

### Requirement: Fourier Shape Descriptors

The system SHALL provide Fourier-based shape descriptors that encode root polyline morphology into fixed-length vectors suitable for machine learning applications.

The descriptors SHALL be computed using tangent angle parameterization, where the root skeleton is represented as a function θ(s) of normalized arc length s ∈ [0, 1].

The implementation SHALL:
- Accept root skeleton points as input arrays of shape (n_nodes, 2)
- Resample polylines to uniform arc-length spacing before analysis
- Compute Fourier coefficients of the tangent angle function
- Return fixed-length descriptor arrays regardless of input polyline length
- Handle missing nodes (NaN values) gracefully by filtering before computation
- Require no external dependencies beyond NumPy

#### Scenario: Compute Fourier descriptors for a single root

- **GIVEN** a root skeleton with points array of shape (6, 2)
- **WHEN** `get_fourier_descriptors(points, n_harmonics=10)` is called
- **THEN** the function returns a 1D array of shape (20,) containing Fourier coefficients

#### Scenario: Handle roots with NaN nodes

- **GIVEN** a root skeleton with some NaN coordinate values
- **WHEN** `get_fourier_descriptors(points)` is called
- **THEN** NaN nodes are filtered before computation
- **AND** valid descriptors are returned if at least 2 non-NaN nodes exist

#### Scenario: Handle degenerate roots

- **GIVEN** a root skeleton with fewer than 2 valid (non-NaN) nodes
- **WHEN** `get_fourier_descriptors(points)` is called
- **THEN** the function returns an array filled with NaN values

### Requirement: Arc-Length Parameterization

The system SHALL provide arc-length parameterization to resample polylines to uniform spacing along the curve length.

#### Scenario: Resample polyline to uniform arc length

- **GIVEN** a polyline with non-uniform point spacing
- **WHEN** `get_arc_length_parameterization(points, n_samples=100)` is called
- **THEN** the function returns an array of shape (100, 2)
- **AND** consecutive points are approximately equidistant along the curve

#### Scenario: Preserve polyline endpoints

- **GIVEN** a polyline with defined start and end points
- **WHEN** `get_arc_length_parameterization(points)` is called
- **THEN** the first output point matches the original start point
- **AND** the last output point matches the original end point

### Requirement: Tangent Angle Computation

The system SHALL compute tangent angles along polylines for shape analysis.

The tangent angle θ at each point SHALL represent the direction of the curve relative to the positive x-axis, measured in radians.

#### Scenario: Compute tangent angles for a straight horizontal line

- **GIVEN** a horizontal line from (0, 0) to (10, 0)
- **WHEN** `get_tangent_angles(points, normalize=False)` is called
- **THEN** all returned angles are approximately 0

#### Scenario: Compute tangent angles for a straight vertical line

- **GIVEN** a vertical line from (0, 0) to (0, 10)
- **WHEN** `get_tangent_angles(points, normalize=False)` is called
- **THEN** all returned angles are approximately π/2

#### Scenario: Rotation-invariant tangent angles

- **GIVEN** any polyline
- **WHEN** `get_tangent_angles(points, normalize=True)` is called
- **THEN** the mean of returned angles is approximately 0

### Requirement: Curvature Computation

The system SHALL compute curvature values along polylines as the rate of change of tangent angle with respect to arc length.

#### Scenario: Compute curvature for a straight line

- **GIVEN** a straight line polyline
- **WHEN** `get_curvature(points)` is called
- **THEN** all returned curvature values are approximately 0

#### Scenario: Compute curvature for a circular arc

- **GIVEN** a semicircular arc with radius R
- **WHEN** `get_curvature(points)` is called
- **THEN** all returned curvature values are approximately 1/R

### Requirement: Fourier Coefficient Extraction

The system SHALL compute Fourier coefficients from 1D signals using the Fast Fourier Transform.

#### Scenario: Extract specified number of harmonics

- **GIVEN** a 1D signal of arbitrary length
- **WHEN** `get_fourier_coefficients(signal, n_harmonics=5)` is called
- **THEN** the function returns an array of length 10 (2 values per harmonic)

#### Scenario: Fourier coefficients capture shape information

- **GIVEN** two polylines with visually different shapes
- **WHEN** Fourier descriptors are computed for both
- **THEN** the resulting descriptor vectors have different values
