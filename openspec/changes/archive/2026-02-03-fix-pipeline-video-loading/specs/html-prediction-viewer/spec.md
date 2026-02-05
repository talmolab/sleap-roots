## ADDED Requirements

### Requirement: Support Multiple Video Sources

The viewer SHALL support loading images from both HDF5 files and image directories.

#### Scenario: Pipeline output with image directories
- **WHEN** `.slp` files contain embedded paths to image directories that don't exist locally
- **AND** the image directory exists in the predictions directory (by matching directory name)
- **THEN** video paths are remapped to local image directory
- **AND** images are loaded from the local directory