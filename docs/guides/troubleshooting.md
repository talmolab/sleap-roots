# Troubleshooting

This guide addresses common issues when using sleap-roots and provides solutions.

## Installation Issues

### Git LFS Data Not Downloaded

**Problem**: Example data files are missing or show as text pointers

```
Error: cannot load file 'test_data.h5'
```

**Solution**:
```bash
# Install Git LFS
git lfs install

# Pull LFS data
git lfs pull

# Verify files are binary (not text pointers)
file tests/data/*.h5  # Should show "HDF5 data file"
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'sleap_roots'`

**Solution**:
```bash
# Install in editable mode for development
pip install -e .

# Or install normally
pip install sleap-roots

# Verify installation
python -c "import sleap_roots; print(sleap_roots.__version__)"
```

### Dependency Conflicts

**Problem**: Version conflicts with other packages

**Solution**:
```bash
# Create fresh environment
conda create -n sleap-roots-env python=3.11
conda activate sleap-roots-env

# Install sleap-roots
pip install sleap-roots

# Or use uv (faster)
pip install uv
uv pip install sleap-roots
```

## Data Loading Issues

### Cannot Load H5 File

**Problem**: `OSError: Unable to open file`

**Causes and solutions**:

1. **File doesn't exist**:
```python
from pathlib import Path
assert Path("predictions.h5").exists(), "H5 file not found"
```

2. **File is corrupted**:
```bash
# Check file integrity
h5ls predictions.h5  # Should list datasets

# Try opening in Python
import h5py
with h5py.File("predictions.h5", "r") as f:
    print(list(f.keys()))
```

3. **Wrong file format**:
Ensure file is exported from SLEAP, not a raw labels file:
```bash
# Export from SLEAP
sleap-convert predictions.slp --format analysis --output predictions.h5
```

### Cannot Load SLP File

**Problem**: `ValueError: Cannot load SLEAP file`

**Solution**:
```python
# Verify SLP file is valid
import sleap_io as sio

labels = sio.load_slp("predictions.slp")
print(f"Loaded {len(labels)} frames")
print(f"Skeleton: {labels.skeleton.node_names}")
```

### Missing Predictions

**Problem**: Empty or sparse predictions in H5/SLP files

**Solution**:
1. Check SLEAP tracking quality
2. Lower confidence threshold in SLEAP
3. Review frames with missing tracks in SLEAP GUI

## Trait Computation Issues

### NaN Values in Traits

**Problem**: Computed traits contain `NaN` values

**Common causes**:

1. **Insufficient tracking points**:
```python
# Check point count
for i, pts in enumerate(series.primary_pts):
    if len(pts) < 5:
        print(f"Frame {i}: only {len(pts)} points (need at least 5)")
```

2. **Empty root arrays**:
```python
# Check for empty roots
if len(series.primary_pts) == 0:
    print("No primary root points found")
if len(series.lateral_pts) == 0:
    print("No lateral root points found")
```

3. **Division by zero**:
Handle edge cases in custom computations:
```python
def safe_ratio(numerator, denominator):
    """Compute ratio with NaN for zero denominator."""
    return numerator / denominator if denominator != 0 else np.nan
```

### Incorrect Trait Values

**Problem**: Trait values seem wrong or unexpected

**Debugging steps**:

1. **Visualize the data**:
```python
import matplotlib.pyplot as plt

# Plot root points
pts = series.primary_pts[0]
plt.plot(pts[:, 0], pts[:, 1], 'o-')
plt.title("Primary Root Points")
plt.gca().invert_yaxis()  # Match image coordinates
plt.show()
```

2. **Check intermediate values**:
```python
# Debug trait computation
pts = series.primary_pts[0]
length = sr.lengths.get_root_lengths([pts])[0]
print(f"Computed length: {length:.2f} pixels")
print(f"Expected range: ~100-500 pixels")
```

3. **Verify pixel scaling**:
If physical units are needed:
```python
# Convert pixels to mm
pixels_per_mm = 10  # Calibrate from your setup
length_mm = length_pixels / pixels_per_mm
```

### Missing Lateral Roots

**Problem**: Lateral root count is lower than expected

**Solutions**:

1. **Check SLEAP tracking**:
   - Open `lateral.slp` in SLEAP GUI
   - Verify all laterals are tracked
   - Check for broken tracks

2. **Verify node names**:
```python
# Check skeleton structure
labels = sio.load_slp("lateral.slp")
print("Lateral root nodes:", labels.skeleton.node_names)
# Should include "base" or "lateral" node
```

3. **Lower confidence threshold**:
```python
# When loading series, use lower threshold
series = sr.Series.load(
    "plant",
    h5_path="pred.h5",
    lateral_path="lateral.slp",
    confidence_threshold=0.3  # Lower threshold
)
```

## Pipeline-Specific Issues

### DicotPipeline

**Problem**: Primary-lateral connections incorrect

**Solution**:
```python
# Verify primary and lateral bases align
primary_pts = series.primary_pts[0]
lateral_pts_list = series.lateral_pts[0]

# Primary base should be near (0,0) in root coordinates
print(f"Primary base: {primary_pts[0]}")

# Each lateral should start near primary
for i, lateral_pts in enumerate(lateral_pts_list):
    print(f"Lateral {i} base: {lateral_pts[0]}")
```

### MonocotPipeline

**Problem**: Crown roots not detected

**Solution**:
1. Ensure crown roots are in separate file from primary
2. Check that crown root bases are labeled correctly
3. Verify SLEAP model detects all crown roots

```python
# Check crown root data
crown_pts_list = series.crown_pts[0]  # For monocot pipelines
print(f"Crown root count: {len(crown_pts_list)}")
```

### MultiplePipeline

**Problem**: Plants not separated correctly

**Solution**:
```python
# Check track IDs
labels = sio.load_slp("predictions.slp")
track_ids = set()
for frame in labels:
    for instance in frame:
        if instance.track is not None:
            track_ids.add(instance.track.name)

print(f"Found {len(track_ids)} unique tracks: {track_ids}")
# Should match expected plant count
```

## Performance Issues

### Slow Computation

**Problem**: Trait computation takes too long

**Solutions**:

1. **Use vectorized operations**:
```python
# Fast: vectorized
lengths = sr.lengths.get_root_lengths(all_roots)

# Slow: iterative
lengths = [sr.lengths.get_root_lengths([root])[0] for root in all_roots]
```

2. **Process in parallel**:
```python
from concurrent.futures import ProcessPoolExecutor

def process_plant(series):
    pipeline = sr.DicotPipeline()
    return pipeline.compute_plant_traits(series)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_plant, series_list))
```

3. **Reduce temporal resolution**:
```python
# Sample every Nth frame instead of all frames
series_subsampled = series[::5]  # Every 5th frame
```

### High Memory Usage

**Problem**: Out of memory errors

**Solutions**:

1. **Process in batches**:
```python
# Process plants in batches
batch_size = 10
for i in range(0, len(series_list), batch_size):
    batch = series_list[i:i+batch_size]
    traits = pipeline.compute_multi_plant_traits(batch)
    traits.to_csv(f"traits_batch_{i}.csv")
```

2. **Clear unused data**:
```python
import gc

for series in series_list:
    traits = pipeline.compute_plant_traits(series)
    # Process traits...
    del series  # Free memory
    gc.collect()
```

## Visualization Issues

### Plots Not Showing

**Problem**: `plt.show()` doesn't display plots

**Solutions**:

1. **Use interactive backend**:
```python
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
```

2. **For Jupyter notebooks**:
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

3. **Save to file instead**:
```python
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
```

### Coordinate System Issues

**Problem**: Plots appear flipped or rotated

**Solution**:
```python
# Image coordinates: origin at top-left
plt.plot(pts[:, 0], pts[:, 1], 'o-')
plt.gca().invert_yaxis()  # Flip y-axis to match image
plt.axis('equal')  # Equal aspect ratio
```

## Export Issues

### CSV Not Created

**Problem**: `write_csv=True` but no file appears

**Solutions**:

1. **Check file permissions**:
```python
import os
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)
traits = pipeline.compute_plant_traits(series, write_csv=True,
                                      csv_path=f"{output_dir}/traits.csv")
```

2. **Verify write succeeded**:
```python
from pathlib import Path

csv_path = "traits.csv"
traits = pipeline.compute_plant_traits(series, write_csv=True, csv_path=csv_path)

if Path(csv_path).exists():
    print(f"CSV written successfully: {Path(csv_path).absolute()}")
else:
    print("CSV write failed!")
```

### CSV Encoding Issues

**Problem**: Special characters garbled in CSV

**Solution**:
```python
# Specify UTF-8 encoding
traits.to_csv("traits.csv", index=False, encoding='utf-8')
```

## Testing Issues

### Tests Fail Locally

**Problem**: Tests pass in CI but fail locally

**Solutions**:

1. **Check Git LFS data**:
```bash
git lfs pull
pytest tests/ -v
```

2. **Verify environment**:
```bash
python --version  # Should match CI (3.11+)
pip list | grep sleap  # Check versions
```

3. **Run specific test**:
```bash
pytest tests/test_pipelines.py::test_dicot_pipeline -v
```

### Import Errors in Tests

**Problem**: `ModuleNotFoundError` when running tests

**Solution**:
```bash
# Install package in editable mode
pip install -e .

# Run tests from repo root
pytest tests/
```

## Platform-Specific Issues

### Windows

**Path separators**:
```python
from pathlib import Path

# Good: platform-independent
path = Path("data") / "predictions.h5"

# Avoid: hardcoded separators
# path = "data/predictions.h5"  # May fail on Windows
```

**Long paths**:
Enable long path support in Windows or use shorter paths

### macOS

**File permissions**:
```bash
# Fix permissions if needed
chmod +x scripts/process_data.sh
```

### Linux

**Display issues**:
```bash
# Set display for headless systems
export DISPLAY=:0
```

## Common Error Messages

### `KeyError: 'primary_root'`

**Cause**: Node name mismatch

**Solution**:
```python
# Check actual node names
labels = sio.load_slp("primary.slp")
print("Available nodes:", labels.skeleton.node_names)

# Pass correct names to pipeline
pipeline = sr.DicotPipeline(primary_name="actual_node_name")
```

### `IndexError: list index out of range`

**Cause**: Empty tracking results

**Solution**:
```python
# Check for empty frames
for i, frame_pts in enumerate(series.primary_pts):
    if len(frame_pts) == 0:
        print(f"Frame {i} has no points")
```

### `ValueError: array size must be at least 2`

**Cause**: Too few points for computation

**Solution**:
```python
# Filter frames with sufficient points
def has_enough_points(pts, min_points=5):
    return len(pts) >= min_points

valid_frames = [i for i, pts in enumerate(series.primary_pts)
                if has_enough_points(pts)]
```

## Getting Help

If you're still stuck:

1. **Check existing issues**: [GitHub Issues](https://github.com/talmolab/sleap-roots/issues)

2. **Search documentation**: Use search bar (top of page)

3. **Create a minimal example**:
```python
import sleap_roots as sr

# Minimal code that reproduces the issue
series = sr.Series.load(...)
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series)
# Error occurs here
```

4. **Open an issue**: Include:
   - Error message and full traceback
   - Minimal reproducible example
   - sleap-roots version: `import sleap_roots; print(sleap_roots.__version__)`
   - Python version: `python --version`
   - Operating system

5. **Community support**:
   - [GitHub Discussions](https://github.com/talmolab/sleap-roots/discussions)
   - SLEAP Slack workspace

## Best Practices for Avoiding Issues

### 1. Validate Data Early

```python
def validate_series(series):
    """Check series has required data."""
    assert len(series.primary_pts) > 0, "No primary root data"
    assert all(len(pts) >= 2 for pts in series.primary_pts), "Insufficient points"
    print("Series validation passed!")

validate_series(series)
```

### 2. Use Try-Except Blocks

```python
try:
    traits = pipeline.compute_plant_traits(series)
except Exception as e:
    print(f"Error processing {series.series_name}: {e}")
    # Log error and continue with next plant
    continue
```

### 3. Log Intermediate Results

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"Processing {series.series_name}")
traits = pipeline.compute_plant_traits(series)
logger.info(f"Computed {len(traits.columns)} traits")
```

### 4. Test on Small Subset First

```python
# Test on first 3 plants before processing all 100
test_series = series_list[:3]
for series in test_series:
    traits = pipeline.compute_plant_traits(series)
    print(f"{series.series_name}: OK")

# If tests pass, process all
all_traits = pipeline.compute_multi_plant_traits(series_list)
```

## Next Steps

- Review [Quick Start](../getting-started/quickstart.md) for basic usage
- Read [Pipeline Guides](pipelines/dicot.md) for pipeline-specific details
- Check [Trait Reference](trait-reference.md) for trait definitions
- See [Batch Processing](batch-processing.md) for large-scale workflows