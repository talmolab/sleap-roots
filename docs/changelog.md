# Changelog

All notable changes to sleap-roots will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive MkDocs documentation site with Material theme
- Auto-generated API reference from docstrings
- User guides for all pipeline types (7 pipelines)
- Trait reference documentation
- Developer guides and contributing documentation
- 8 tutorial pages for all pipelines
- Cookbook with code recipes (filtering, custom traits, batch optimization, exporting)
- Troubleshooting guide with common issues and solutions
- uv package manager support with PEP 735 dependency groups

### Changed
- Updated installation documentation with uv best practices
- Enhanced developer setup guide with modern workflows
- Migrated to uv for development dependency management

## [0.1.4] - 2024-11-10

### Added
- `MultiplePrimaryRootPipeline` for analyzing plants with multiple primary roots
- `MultipleDicotPipeline` tests for multi-plant batch analysis
- Comprehensive Claude Code slash command suite for developer workflows
- OpenSpec project documentation and change management

### Changed
- Improved test coverage across pipeline classes
- Updated README with latest pipeline examples

## [0.1.3] - 2024-10-29

### Added
- `LateralRootPipeline` for lateral-root-only analysis

## [0.1.2] - 2024-08-26

### Changed
- Version bump and maintenance release

## [0.1.1] - 2024-08-26

### Fixed
- Corrected `crown-curve-indices` definition in trait pipeline
- Applied Black formatting to test files

## [0.1.0] - 2024-05-13

### Added
- `Series.load()` method for loading SLEAP predictions directly
- High-level imports: `find_all_h5_paths`, `find_all_slp_paths`, `load_series_from_h5s`, `load_series_from_slps`
- Increased test coverage across modules

### Changed
- **Breaking**: `Series` class now takes SLEAP predictions directly using `Series.load()`
- **Breaking**: H5 paths are now optional (but required for plotting)
- **Breaking**: `series_name` is now an attribute instead of a property
- **Breaking**: `find_all_series` removed (use `find_all_h5_paths` or `find_all_slp_paths`)
- Upgraded Python requirement to 3.11
- Improved geometry intersection helper functions

## [0.0.9] - 2024-04-23

### Added
- Quality control property for batch processing over genotypes

### Fixed
- Edge cases in older monocot pipeline traits

## [0.0.8] - 2024-04-12

### Added
- Jupyter notebooks for code instruction
- Enhanced documentation
- JupyterLab to development environment

### Changed
- Excluded Jupyter notebooks from language statistics

### Fixed
- Tips calculation functions

## [0.0.7] - 2024-03-31

### Added
- `MultipleDicotPipeline` for analyzing multiple dicot plants simultaneously

## [0.0.6] - 2024-03-11

### Added
- `OlderMonocotPipeline` for mature monocot analysis

### Changed
- Updated README with pipeline examples

## [0.0.5] - 2023-10-08

### Added
- `YoungerMonocotPipeline` for younger monocot plants

### Changed
- Renamed `grav_index` to `curve_index` for clarity

### Fixed
- `get_network_distribution` function

## [0.0.4] - 2023-09-13

### Changed
- Version bump

## [0.0.3] - 2023-09-13

### Changed
- Updated sleap-io minimum version to 0.0.11

## [0.0.2] - 2023-09-12

### Added
- Python 3.7 compatibility
- Checks and tests for ellipse fitter

### Fixed
- Node index calculation
- Dicot pipeline edge cases
- Ellipse fitting robustness

## [0.0.1] - 2023-09-03

Initial release of sleap-roots package.

### Added
- Core `Series` class for SLEAP prediction data
- `DicotPipeline` for dicot root analysis
- Trait computation modules:
  - `bases` - Root base detection and analysis
  - `tips` - Root tip identification
  - `angle` - Root angle measurements
  - `convhull` - Convex hull calculations
  - `lengths` - Root length measurements
  - `networklength` - Network-level metrics
  - `scanline` - Scan line analysis
  - `ellipse` - Ellipse fitting
  - `points` - Point extraction utilities
  - `summary` - Summary statistics
- Test suite with fixtures for rice and soy
- Basic plotting functionality
- sleap-io integration for loading predictions

### New Contributors
- @talmo - Project lead and core architecture
- @eberrigan - Primary developer, pipelines and traits
- @linwang9926 - Trait modules and testing
- @emdavis02 - Test coverage improvements

---

[Unreleased]: https://github.com/talmolab/sleap-roots/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.4
[0.1.3]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.3
[0.1.2]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.2
[0.1.1]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.1
[0.1.0]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.0
[0.0.9]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.9
[0.0.8]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.8
[0.0.7]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.7
[0.0.6]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.6
[0.0.5]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.5
[0.0.4]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.4
[0.0.3]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.3
[0.0.2]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.2
[0.0.1]: https://github.com/talmolab/sleap-roots/releases/tag/v0.0.1