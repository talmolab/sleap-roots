## MODIFIED Requirements

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

#### Scenario: Case-insensitive image extensions
- **WHEN** image directory contains files with uppercase extensions (e.g., `.JPG`, `.PNG`, `.TIFF`)
- **THEN** images are discovered and loaded correctly

#### Scenario: Non-numeric image filenames
- **WHEN** image directory contains files with non-numeric names
- **THEN** files are sorted alphabetically as a fallback after numeric sorting

#### Scenario: Video remapping failure
- **WHEN** video path remapping encounters an error checking video existence
- **THEN** a warning is emitted and remapping continues