# Update Changelog

Maintain CHANGELOG.md following [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

## Quick Commands

```bash
# View recent changes
git log --oneline --decorate -10

# View changes since last tag
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# View all tags
git tag -l | sort -V

# View changes by author
git log --author="<name>" --oneline

# View changes to specific module
git log --oneline -- sleap_roots/trait_pipelines.py

# View current version
python -c "import sleap_roots; print(sleap_roots.__version__)"
```

## Changelog Format

The CHANGELOG.md follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) principles:

- **Guiding Principle**: Changelogs are for humans, not machines
- **Latest First**: Most recent version at the top
- **One Version Per Release**: Each release gets a section
- **Same Date Format**: YYYY-MM-DD
- **Semantic Versioning**: Version numbers follow [SemVer](https://semver.org/)

### Change Categories

- **Added**: New features (new pipelines, new traits, new functionality)
- **Changed**: Changes to existing functionality (algorithm improvements, refactors)
- **Deprecated**: Soon-to-be removed features (warn users)
- **Removed**: Removed features (breaking change)
- **Fixed**: Bug fixes
- **Security**: Security vulnerability fixes

## Template

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- New feature description

### Changed

- Change description

### Fixed

- Bug fix description

## [0.2.0] - 2024-11-15

### Added

- `LateralRootPipeline` for analyzing lateral roots only
- `PrimaryRootPipeline` for analyzing primary roots only
- Support for multi-plant analysis with `MultipleDicotPipeline`

### Changed

- Improved angle calculation to handle collinear points
- Updated test data to use Git LFS

### Fixed

- Fixed NaN in angle calculation for collinear points (#142)

## [0.1.4] - 2024-06-27

### Added

- Initial release with core trait pipelines
- `DicotPipeline`, `YoungerMonocotPipeline`, `OlderMonocotPipeline`
- Trait calculations: lengths, angles, tips, bases, convex hull

[Unreleased]: https://github.com/talmolab/sleap-roots/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/talmolab/sleap-roots/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.4
```

## Workflow: Adding Changes to Changelog

### Step 1: Identify Changes Since Last Release

```bash
# Find last version tag
git tag -l | sort -V | tail -1

# List commits since last tag (e.g., v0.1.4)
git log v0.1.4..HEAD --oneline

# Or view detailed diff
git log v0.1.4..HEAD --pretty=format:"%h %s" --reverse
```

### Step 2: Categorize Each Change

Group commits by category:

- **Added**: New pipelines, new traits, new modules
- **Changed**: Refactors, performance improvements, algorithm updates
- **Fixed**: Bug fixes, error handling improvements
- **Security**: Security patches, data handling fixes

### Step 3: Update CHANGELOG.md

Add changes to the `[Unreleased]` section:

```markdown
## [Unreleased]

### Added

- `LateralRootPipeline` for analyzing lateral roots only (#121)
- `MultiplePrimaryRootPipeline` for batch primary root analysis (#117)
- Support for analyzing crown roots without primary roots

### Fixed

- Fixed NaN in angle calculation for collinear points (#142)
- Fixed SLEAP file loading on Windows paths (#145)
```

### Step 4: When Releasing a Version

Move `[Unreleased]` to a versioned section:

```markdown
## [Unreleased]

## [0.2.0] - 2024-11-15

### Added

- `LateralRootPipeline` for analyzing lateral roots only (#121)
- `MultiplePrimaryRootPipeline` for batch primary root analysis (#117)
```

Update the version in `sleap_roots/__init__.py`:

```python
__version__ = "0.2.0"
```

Update the links at the bottom:

```markdown
[Unreleased]: https://github.com/talmolab/sleap-roots/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/talmolab/sleap-roots/compare/v0.1.4...v0.2.0
[0.1.4]: https://github.com/talmolab/sleap-roots/releases/tag/v0.1.4
```

## Writing Good Changelog Entries

### Good Examples

```markdown
### Added

- `LateralRootPipeline` for analyzing plants with only lateral root predictions
- Convex hull area calculation for root system spread analysis
- Support for batch processing multiple series with `compute_batch_traits()`

### Fixed

- Angle calculation no longer returns NaN for collinear points
- SLEAP file loading now works correctly on Windows paths
- Empty array handling in length calculations
```

### Bad Examples

```markdown
### Added

- New stuff ❌ (too vague)
- Fix bug ❌ (belongs in "Fixed", not "Added")
- Updated dependencies ❌ (unless breaking change, don't include routine updates)
```

## Tips

1. **Update continuously**: Add to `[Unreleased]` as you merge PRs, don't batch at release time
2. **Link to PRs**: Include `(#42)` references for traceability
3. **Be user-focused**: Write for users, not developers
   - Good: "Added lateral root pipeline for analyzing lateral roots only"
   - Bad: "Implemented LateralRootPipeline class with compute_traits method"
4. **Note breaking changes**: Clearly mark with `**BREAKING:**`
5. **Skip internal changes**: Don't include CI config, test refactors, or minor internal changes
6. **Group related changes**: If a feature required multiple commits, summarize as one entry

## Breaking Changes

If a change is breaking, mark it clearly:

```markdown
### Changed

- **BREAKING**: `compute_traits()` now requires `frame_index` parameter (previously defaulted to 0)
  - Migration: Pass explicit `frame_index=0` to maintain previous behavior
  - Example: `pipeline.compute_traits(series, frame_index=0)`
```

## Release Checklist

Before cutting a release:

- [ ] All changes moved from `[Unreleased]` to versioned section
- [ ] Version number follows SemVer (major.minor.patch)
- [ ] Version updated in `sleap_roots/__init__.py`
- [ ] Date is today's date in YYYY-MM-DD format
- [ ] Links at bottom are updated
- [ ] Breaking changes are clearly marked
- [ ] Notable changes are user-friendly and descriptive

## Semantic Versioning Quick Reference

Given a version number `MAJOR.MINOR.PATCH`:

- **MAJOR**: Breaking changes (0.x.x → 1.0.0)
  - Changed trait calculation algorithms
  - Removed deprecated pipelines
  - Changed CSV output format
- **MINOR**: New features, backwards-compatible (0.1.x → 0.2.0)
  - New pipeline classes
  - New trait calculations
  - New batch processing features
- **PATCH**: Bug fixes, backwards-compatible (0.1.1 → 0.1.2)
  - Fixed NaN in calculations
  - Fixed file loading bugs
  - Fixed edge case handling

## Examples for sleap-roots

### Version 0.1.4 (Initial Release)

```markdown
## [0.1.4] - 2024-06-27

### Added

- Initial release with trait analysis pipelines
- `DicotPipeline` for dicot plants (primary + lateral roots)
- `YoungerMonocotPipeline` for young monocots (primary + crown roots)
- `OlderMonocotPipeline` for older monocots (crown roots only)
- Core trait calculations: lengths, angles, tips, bases, convex hull, network length, scanline
- Batch processing support with CSV export
- Cross-platform support (Ubuntu, Windows, macOS)
```

### Version 0.2.0 (New Features)

```markdown
## [0.2.0] - 2024-11-15

### Added

- `LateralRootPipeline` for analyzing lateral roots only (#121)
- `PrimaryRootPipeline` for analyzing primary roots only
- `MultipleDicotPipeline` for multi-plant batch processing (#104)
- `MultiplePrimaryRootPipeline` for multi-plant primary root analysis (#117)
- Support for expected plant count CSV metadata

### Changed

- Improved angle calculation performance with vectorized operations
- Enhanced test data coverage with Git LFS

### Fixed

- Angle calculation no longer returns NaN for collinear points (#142)
- SLEAP file loading now handles Windows paths correctly
- Empty array edge cases in all trait calculations
```

### Version 0.2.1 (Bug Fix)

```markdown
## [0.2.1] - 2024-11-20

### Fixed

- Fixed convex hull calculation failing on single-point arrays
- Fixed CSV export error when series name contains special characters
- Corrected trait names in `MultipleDicotPipeline` output
```

### Version 1.0.0 (Breaking Change)

```markdown
## [1.0.0] - 2025-01-15

### Changed

- **BREAKING**: Switched angle units from radians to degrees for consistency
  - All angle traits now return values in degrees (0-360)
  - Migration: If you need radians, use `np.radians(angle_degrees)`
  - This affects: `get_root_angle()`, `get_base_angle()`, all pipeline angle traits

### Added

- Comprehensive trait validation against published data
- Detailed documentation for all trait calculations

### Fixed

- Corrected lateral root length calculation for branching patterns
```

## Project-Specific Notes

### Version Location

Version is defined in `sleap_roots/__init__.py`:

```python
__version__ = "0.1.4"
```

This is read by `setuptools` in `pyproject.toml`:

```toml
[tool.setuptools.dynamic]
version = {attr = "sleap_roots.__version__"}
```

### Published Results

Be extra careful with changes to trait calculations:

- Note if calculations change in any way
- Provide validation data showing accuracy
- Consider impact on reproducibility of published papers
- Document any algorithm changes thoroughly

### PyPI Releases

When releasing:

```bash
# Update version in sleap_roots/__init__.py
# Update CHANGELOG.md
# Commit changes
git add sleap_roots/__init__.py CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"

# Tag release
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin main --tags

# Build and publish to PyPI
python -m build
twine upload dist/*
```