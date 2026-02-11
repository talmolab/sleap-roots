# sleap-roots

[![CI](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml/badge.svg)](https://github.com/talmolab/sleap-roots/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/talmolab/sleap-roots/branch/main/graph/badge.svg)](https://codecov.io/gh/talmolab/sleap-roots)
[![Release](https://img.shields.io/github/v/release/talmolab/sleap-roots?label=Latest)](https://github.com/talmolab/sleap-roots/releases/)
[![PyPI](https://img.shields.io/pypi/v/sleap-roots?label=PyPI)](https://pypi.org/project/sleap-roots)

Analysis tools for [SLEAP](https://sleap.ai)-based plant root phenotyping.

📖 **[Documentation](https://roots.sleap.ai)** | 📊 **[HackMD Trait Reference](https://hackmd.io/DMiXO2kXQhKH8AIIcIy--g)** | 📝 **[Paper](https://doi.org/10.34133/plantphenomics.0175)**

---

## 📦 Installation

### For End Users

```bash
pip install sleap-roots
```

### For Development (Recommended: uv)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager (10-100x faster than conda):

```bash
# Install uv (one-time setup)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/talmolab/sleap-roots && cd sleap-roots
uv sync

# Run commands
uv run pytest tests/
uv run black sleap_roots tests
```

**Why uv?** Environment setup in ~30 seconds (vs 10-15 minutes with conda) with reproducible builds via lockfile.

### Alternative: conda

```bash
conda env create -f environment.yml
conda activate sleap-roots
```

---

## 🌱 Usage

Trait pipelines supported:

- `DicotPipeline` – Primary + lateral roots (e.g. soy, canola)
- `YoungerMonocotPipeline` – Primary + crown roots (e.g. early rice)
- `OlderMonocotPipeline` – Crown roots only (e.g. later rice)
- `PrimaryRootPipeline` – Primary root only
- `LateralRootPipeline` – Lateral roots only
- `MultipleDicotPipeline` – Multi-plant dicot setup (batch from a single image)

### 🔁 Example: Dicot Pipeline

**1. Compute traits for a single plant**

```python
import sleap_roots as sr

series = sr.Series.load(
    series_name="919QDUH",
    h5_path="tests/data/canola_7do/919QDUH.h5",
    primary_path="tests/data/canola_7do/919QDUH.primary.slp",
    lateral_path="tests/data/canola_7do/919QDUH.lateral.slp"
)

pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)
```

**2. Compute traits for a batch**

```python
paths = sr.find_all_slp_paths("tests/data/soy_6do")
plants = sr.load_series_from_slps(paths, h5s=True)

pipeline = sr.DicotPipeline()
batch_df = pipeline.compute_batch_traits(plants, write_csv=True)
```

**3. Use a single trait utility**

```python
from sleap_roots.lengths import get_root_lengths

pts = series.get_primary_points(frame_idx=0)
lengths = get_root_lengths(pts)
```

### 👁️ Prediction Viewer

Validate SLEAP predictions before computing traits with an interactive HTML viewer:

```bash
sleap-roots viewer predictions/ --output viewer.html
```

Features keyboard navigation, confidence scores, and multiple output modes. See the [Prediction Viewer Guide](https://roots.sleap.ai/guides/prediction-viewer/) for details.

---

## 📓 Notebooks & Tutorials

Explore tutorials under `notebooks/`:

```bash
cd notebooks
jupyter lab
```

You can use the test data in `tests/data` or replace it with your own.

---

## 🧪 Development

### Using uv (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/talmolab/sleap-roots && cd sleap-roots

# 2. Install dependencies (creates .venv automatically)
uv sync

# 3. Run tests
uv run pytest tests/

# 4. Format code
uv run black sleap_roots tests

# 5. Check docstrings
uv run pydocstyle sleap_roots/
```

**Adding dependencies:**
```bash
uv add numpy  # Add runtime dependency
uv add --group dev pytest  # Add dev dependency
```

### Using conda (Alternative)

```bash
# 1. Clone the repository
git clone https://github.com/talmolab/sleap-roots && cd sleap-roots

# 2. Create environment
conda env create -f environment.yml
conda activate sleap-roots

# 3. Run tests
pytest tests/
```

---

## 📖 Trait Reference

See the latest trait documentation here:

👉 [HackMD: sleap-roots Trait Docs](https://hackmd.io/DMiXO2kXQhKH8AIIcIy--g)

---

## 🤝 Acknowledgments

Created by the [Talmo Lab](https://talmolab.org) and [Busch Lab](https://busch.salk.edu) at the Salk Institute, as part of the [Harnessing Plants Initiative](https://www.salk.edu/harnessing-plants-initiative/).

### Contributors

- Elizabeth Berrigan
- Lin Wang
- Andrew O'Connor
- Talmo Pereira

### Citation

E.M. Berrigan et al., *"Fast and Efficient Root Phenotyping via Pose Estimation"*, Plant Phenomics.  
DOI: [10.34133/plantphenomics.0175](https://doi.org/10.34133/plantphenomics.0175)