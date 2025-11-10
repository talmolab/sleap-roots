# Project Context

## Purpose
sleap-roots is an analysis tool for SLEAP-based plant root phenotyping. It provides automated trait extraction pipelines for analyzing root system architecture from pose estimation data. The project supports various root system types (dicot, monocot) and developmental stages, enabling researchers to quantify root traits for plant phenotyping research at the Salk Institute's Harnessing Plants Initiative.

## Tech Stack
- **Language**: Python 3.7+ (recommended: 3.11)
- **Core Dependencies**:
  - `numpy` - Numerical computing
  - `pandas` - Data manipulation and CSV output
  - `h5py` - HDF5 file handling for image series
  - `sleap-io>=0.0.11` - SLEAP predictions file loading
  - `attrs` - Class definitions with attributes
  - `scikit-image` - Image processing
  - `shapely` - Geometric operations
  - `matplotlib` & `seaborn` - Visualization
- **Package Management**: setuptools with pyproject.toml
- **Environment**: conda (recommended) or pip
- **Testing**: pytest with pytest-cov
- **CI/CD**: GitHub Actions (cross-platform: Ubuntu, Windows, macOS)

## Project Conventions

### Code Style
- **Formatter**: Black (line length: 88 characters)
- **Docstrings**: Google-style convention (enforced by pydocstyle)
- **Module Structure**: Each module focuses on specific trait computations (e.g., `lengths.py`, `angles.py`, `tips.py`)
- **Imports**: High-level imports exposed through `__init__.py`
- **Naming**:
  - Classes: PascalCase (e.g., `DicotPipeline`, `Series`)
  - Functions: snake_case (e.g., `get_root_lengths`, `compute_traits`)
  - Private functions: Prefixed with underscore
- **Type Hints**: Used extensively with `typing` module (Dict, Optional, Tuple, List, Union)

### Architecture Patterns
- **Pipeline Architecture**: Trait computation organized through pipeline classes that inherit from base `Pipeline` class
  - `DicotPipeline` - Primary + lateral roots
  - `YoungerMonocotPipeline` - Primary + crown roots
  - `OlderMonocotPipeline` - Crown roots only
  - `MultipleDicotPipeline` - Multi-plant batch processing
  - `PrimaryRootPipeline` - Primary root only
  - `LateralRootPipeline` - Lateral roots only
- **Data Model**: Uses `attrs` for declarative class definitions (e.g., `Series`, `TraitDef`)
- **Modular Traits**: Individual trait computations in dedicated modules:
  - `lengths.py` - Root length measurements
  - `angle.py` - Angular measurements
  - `tips.py` - Root tip detection
  - `bases.py` - Root base detection
  - `convhull.py` - Convex hull analysis
  - `ellipse.py` - Ellipse fitting
  - `networklength.py` - Network path lengths
  - `scanline.py` - Scanline-based measurements
- **Series-Centric Design**: `Series` class encapsulates all data for a single plant/timepoint
- **Output**: Structured pandas DataFrames with CSV export capabilities

### Testing Strategy
- **Framework**: pytest with pytest-cov for coverage reporting
- **Test Organization**: 
  - One test file per module (e.g., `test_lengths.py` for `lengths.py`)
  - Test fixtures in `tests/fixtures/`
  - Test data in `tests/data/` (stored with Git LFS)
- **Coverage**: Target full coverage with reports uploaded to Codecov
- **CI Requirements**: 
  - All tests must pass on Ubuntu 22.04, Windows 2022, and macOS latest
  - Python 3.11 is the primary test version
  - Coverage measured on Ubuntu + Python 3.11
- **Test Data**: Uses real SLEAP prediction files (.slp) and HDF5 image series
  - `tests/data/canola_7do/` - Canola dicot samples
  - `tests/data/soy_6do/` - Soybean samples
- **Fixtures**: Defined in `tests/conftest.py` for reusable test setup

### Git Workflow
- **Main Branch**: `main` (protected)
- **CI Triggers**: 
  - Pull requests (opened, reopened, synchronized)
  - Pushes to main
  - Only runs when relevant paths change (`sleap_roots/**`, `tests/**`, CI config, environment.yml)
- **Pre-merge Requirements**:
  - Black formatting check must pass
  - Pydocstyle docstring checks must pass
  - All pytest tests must pass on all platforms
  - Code coverage must be maintained
- **Lint Jobs**:
  - `black --check` for code formatting
  - `pydocstyle --convention=google` for docstring style
- **Release Process**: Version defined in `sleap_roots/__version__` and published to PyPI
- **Git LFS**: Used for large test data files

## Domain Context
**Plant Root Phenotyping**:
- Project focuses on analyzing root system architecture from time-series images
- Uses SLEAP (Social LEAP Estimates Animal Poses) adapted for plant roots
- **Root Types**:
  - **Primary roots**: Main tap root growing downward
  - **Lateral roots**: Secondary roots branching from primary
  - **Crown roots**: Adventitious roots in monocots (rice, wheat)
- **Plant Categories**:
  - **Dicots**: Two cotyledons (e.g., soy, canola) - have primary + lateral roots
  - **Monocots**: One cotyledon (e.g., rice) - have crown roots ± primary
- **Developmental Stages**: Younger monocots have primary + crown; older have crown only
- **Traits Measured**: Root lengths, angles, tip counts, bases, convex hull area, network metrics, scanline intersections
- **Data Format**: 
  - Predictions stored as `.slp` files (SLEAP format)
  - Images stored as HDF5 (`.h5`) files
  - Output as CSV with trait measurements per frame/plant
- **Research Context**: Part of Harnessing Plants Initiative at Salk Institute (Talmo Lab & Busch Lab)
- **Citation**: Berrigan et al., "Fast and Efficient Root Phenotyping via Pose Estimation", Plant Phenomics, DOI: 10.34133/plantphenomics.0175

## Important Constraints
- **Python Version**: Must support Python 3.7+ (declared in classifiers), recommend 3.11
- **Cross-Platform**: Must work on Ubuntu, Windows, and macOS
- **Data Requirements**: 
  - Requires SLEAP predictions files (`.slp` format) as input
  - Optional HDF5 image series for visualization
  - At least one root type (primary, lateral, or crown) must be provided
- **Performance**: Should handle batch processing of multiple plants/series efficiently
- **Backward Compatibility**: Changes to trait computation may affect reproducibility of published results
- **Scientific Accuracy**: Trait measurements must be biologically meaningful and validated
- **Dependency Constraints**: 
  - `sleap-io>=0.0.11` for predictions loading
  - Compatible with scientific Python stack (numpy, pandas, scikit-image, etc.)
- **Documentation**: All public functions require Google-style docstrings
- **Code Quality**: Must pass Black formatting and pydocstyle checks before merge

## External Dependencies
- **SLEAP (sleap.ai)**: Pose estimation framework that generates the `.slp` prediction files
  - This project consumes SLEAP outputs but doesn't run SLEAP itself
  - Uses `sleap-io` library for reading SLEAP files
- **GitHub Actions**: CI/CD infrastructure
  - Uses `mamba-org/setup-micromamba` for conda environment setup
  - Uses `codecov/codecov-action` for coverage reporting
- **Codecov**: Code coverage reporting and tracking
- **PyPI**: Package distribution platform
- **Git LFS**: Large File Storage for test data
- **HackMD**: External trait documentation maintained at https://hackmd.io/DMiXO2kXQhKH8AIIcIy--g
- **Conda/Micromamba**: Package and environment management (recommended installation method)
- **Research Dependencies**: 
  - Results may be referenced in scientific publications
  - API changes should consider impact on reproducibility
