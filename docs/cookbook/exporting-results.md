# Exporting Results

This recipe shows various methods for exporting trait data for statistical analysis, visualization, and sharing.

## Problem

You've computed traits and need to export them in formats suitable for:
- Statistical software (R, SPSS, SAS)
- Spreadsheet analysis (Excel, Google Sheets)
- Data visualization tools
- Collaborator sharing
- Long-term archival

## Solution Overview

Export strategies for different use cases:
1. **CSV**: Universal format for most tools
2. **Excel**: Multi-sheet workbooks with metadata
3. **HDF5**: Large datasets with hierarchical organization
4. **JSON**: Web integration and APIs
5. **Parquet**: Efficient storage for big data

## Basic CSV Export

### Standard Export

```python
import sleap_roots as sr
import pandas as pd

# Compute traits
series = sr.Series.load(...)
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series)

# Export to CSV
traits.to_csv("traits.csv", index=False)
```

### With Metadata

```python
def export_with_metadata(traits, output_path, metadata=None):
    """
    Export traits with metadata header.

    Args:
        traits: DataFrame with trait data
        output_path: Output CSV path
        metadata: Dictionary of metadata to include
    """
    with open(output_path, 'w') as f:
        # Write metadata as comments
        if metadata:
            f.write("# Trait Data Export\n")
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("#\n")

        # Write traits data
        traits.to_csv(f, index=False)

# Usage
metadata = {
    'experiment': 'Drought stress 2024',
    'pipeline': 'DicotPipeline',
    'date': '2024-01-15',
    'researcher': 'J. Smith'
}

export_with_metadata(traits, "traits_with_metadata.csv", metadata)
```

### Multiple Plants

```python
# Process multiple plants
all_traits = []

for h5_file in Path("data/").glob("*.h5"):
    series = sr.Series.load(
        series_name=h5_file.stem,
        h5_path=h5_file,
        primary_path=h5_file.with_suffix(".primary.slp"),
        lateral_path=h5_file.with_suffix(".lateral.slp")
    )

    traits = pipeline.compute_plant_traits(series)
    traits['plant_id'] = h5_file.stem
    traits['genotype'] = get_genotype_from_filename(h5_file.stem)
    traits['treatment'] = get_treatment_from_filename(h5_file.stem)
    all_traits.append(traits)

# Combine and export
combined = pd.concat(all_traits, ignore_index=True)
combined.to_csv("all_plants_traits.csv", index=False)
```

## Excel Export

### Single Sheet

```python
# Simple Excel export
traits.to_excel("traits.xlsx", index=False, sheet_name="Traits")
```

### Multi-Sheet Workbook

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils.dataframe import dataframe_to_rows

def export_to_excel_multisheet(
    traits,
    output_path,
    metadata=None,
    summary_stats=True
):
    """
    Export traits to multi-sheet Excel workbook.

    Args:
        traits: DataFrame with trait data
        output_path: Output Excel path
        metadata: Dictionary of metadata
        summary_stats: Whether to include summary statistics
    """
    # Create workbook
    wb = Workbook()

    # Sheet 1: Metadata
    if metadata:
        ws_meta = wb.active
        ws_meta.title = "Metadata"

        ws_meta['A1'] = "Parameter"
        ws_meta['B1'] = "Value"
        ws_meta['A1'].font = Font(bold=True)
        ws_meta['B1'].font = Font(bold=True)

        for i, (key, value) in enumerate(metadata.items(), start=2):
            ws_meta[f'A{i}'] = key
            ws_meta[f'B{i}'] = value

    # Sheet 2: Raw Traits
    ws_traits = wb.create_sheet("Traits")

    for r in dataframe_to_rows(traits, index=False, header=True):
        ws_traits.append(r)

    # Style header row
    for cell in ws_traits[1]:
        cell.font = Font(bold=True)
        cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

    # Sheet 3: Summary Statistics
    if summary_stats:
        ws_summary = wb.create_sheet("Summary")

        # Compute summary statistics
        summary = traits.describe()
        for r in dataframe_to_rows(summary, index=True, header=True):
            ws_summary.append(r)

        # Style header
        for cell in ws_summary[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")

    # Sheet 4: Per-Plant Summary
    if 'plant_id' in traits.columns:
        ws_plant_summary = wb.create_sheet("Per-Plant Summary")

        plant_summary = traits.groupby('plant_id').agg(['mean', 'std', 'count'])
        for r in dataframe_to_rows(plant_summary, index=True, header=True):
            ws_plant_summary.append(r)

    # Save workbook
    wb.save(output_path)

# Usage
metadata = {
    'Experiment': 'Root phenotyping 2024',
    'Pipeline': 'DicotPipeline',
    'Date': '2024-01-15',
    'Number of Plants': len(traits['plant_id'].unique()) if 'plant_id' in traits.columns else 1,
    'Traits Computed': len(traits.columns)
}

export_to_excel_multisheet(
    traits,
    "comprehensive_traits.xlsx",
    metadata=metadata,
    summary_stats=True
)
```

### Excel with Plots

```python
from openpyxl.chart import LineChart, Reference

def export_excel_with_plots(traits, output_path):
    """Export with embedded plots."""

    # Create workbook with traits
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        traits.to_excel(writer, sheet_name="Traits", index=False)

        # Access workbook
        wb = writer.book
        ws = wb["Traits"]

        # Add line chart for growth over time
        if 'frame' in traits.columns and 'primary_length' in traits.columns:
            chart = LineChart()
            chart.title = "Primary Root Growth"
            chart.x_axis.title = "Frame"
            chart.y_axis.title = "Length (pixels)"

            data = Reference(ws, min_col=traits.columns.get_loc('primary_length') + 1,
                           min_row=1, max_row=len(traits) + 1)
            cats = Reference(ws, min_col=traits.columns.get_loc('frame') + 1,
                           min_row=2, max_row=len(traits) + 1)

            chart.add_data(data, titles_from_data=True)
            chart.set_categories(cats)

            ws.add_chart(chart, "N2")  # Position chart

# Usage
export_excel_with_plots(traits, "traits_with_plots.xlsx")
```

## HDF5 Export

### Basic HDF5

```python
import h5py
import numpy as np

def export_to_hdf5(traits, output_path, include_arrays=True):
    """
    Export traits to HDF5 format.

    Args:
        traits: DataFrame with traits
        output_path: Output HDF5 path
        include_arrays: Whether to save array-valued traits
    """
    with h5py.File(output_path, 'w') as f:
        # Create groups
        grp_traits = f.create_group('traits')
        grp_metadata = f.create_group('metadata')

        # Save scalar traits
        for col in traits.columns:
            if traits[col].dtype == 'object':
                # Skip arrays or convert to strings
                if not include_arrays:
                    continue

                # Try to save as array
                try:
                    data = np.array([np.array(x) for x in traits[col]])
                    grp_traits.create_dataset(col, data=data)
                except:
                    # Save as strings
                    str_data = traits[col].astype(str).values
                    grp_traits.create_dataset(
                        col,
                        data=str_data,
                        dtype=h5py.special_dtype(vlen=str)
                    )
            else:
                # Save numeric data
                grp_traits.create_dataset(col, data=traits[col].values)

        # Save metadata
        grp_metadata.attrs['export_date'] = str(pd.Timestamp.now())
        grp_metadata.attrs['n_plants'] = len(traits)
        grp_metadata.attrs['n_traits'] = len(traits.columns)

# Usage
export_to_hdf5(traits, "traits.h5")
```

### Hierarchical HDF5

```python
def export_hierarchical_hdf5(all_traits_dict, output_path):
    """
    Export multiple plants to hierarchical HDF5.

    Args:
        all_traits_dict: Dict mapping plant_id -> traits DataFrame
        output_path: Output HDF5 path
    """
    with h5py.File(output_path, 'w') as f:
        # Create group for each plant
        for plant_id, traits in all_traits_dict.items():
            plant_grp = f.create_group(f'plant_{plant_id}')

            # Save traits for this plant
            for col in traits.columns:
                if traits[col].dtype != 'object':
                    plant_grp.create_dataset(col, data=traits[col].values)

            # Add plant metadata
            plant_grp.attrs['plant_id'] = plant_id
            plant_grp.attrs['n_frames'] = len(traits)

        # Global metadata
        f.attrs['n_plants'] = len(all_traits_dict)
        f.attrs['export_date'] = str(pd.Timestamp.now())

# Usage
traits_by_plant = {
    'plant_001': traits_001,
    'plant_002': traits_002,
    # ...
}

export_hierarchical_hdf5(traits_by_plant, "all_plants.h5")
```

## JSON Export

### Standard JSON

```python
# Simple JSON export
traits.to_json("traits.json", orient="records", indent=2)
```

### Custom JSON with Arrays

```python
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def export_to_json(traits, output_path, metadata=None):
    """
    Export traits to JSON with proper array handling.

    Args:
        traits: DataFrame with traits
        output_path: Output JSON path
        metadata: Optional metadata dictionary
    """
    data = {
        'metadata': metadata or {},
        'traits': traits.to_dict(orient='records')
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)

# Usage
metadata = {
    'experiment': 'Drought stress 2024',
    'pipeline': 'DicotPipeline',
    'date': '2024-01-15'
}

export_to_json(traits, "traits.json", metadata)
```

## Parquet Export

### Efficient Parquet

```python
# Parquet is efficient for large datasets
traits.to_parquet("traits.parquet", compression='snappy', index=False)

# Read back
traits_loaded = pd.read_parquet("traits.parquet")
```

### With Partitioning

```python
def export_partitioned_parquet(traits, output_dir, partition_cols=['genotype', 'treatment']):
    """
    Export to partitioned Parquet for efficient filtering.

    Args:
        traits: DataFrame with traits
        output_dir: Output directory
        partition_cols: Columns to partition by
    """
    traits.to_parquet(
        output_dir,
        partition_cols=partition_cols,
        compression='snappy',
        index=False
    )

# Usage
export_partitioned_parquet(
    all_traits,
    "traits_partitioned/",
    partition_cols=['genotype', 'treatment']
)

# Read specific partition
specific_traits = pd.read_parquet(
    "traits_partitioned/",
    filters=[('genotype', '==', 'WT'), ('treatment', '==', 'drought')]
)
```

## Wide vs. Long Format

### Convert to Long Format

```python
def convert_to_long_format(traits, id_vars=['plant_id', 'frame']):
    """
    Convert traits from wide to long format for easier plotting.

    Args:
        traits: Wide-format DataFrame
        id_vars: Columns that identify each observation

    Returns:
        Long-format DataFrame
    """
    return pd.melt(
        traits,
        id_vars=id_vars,
        var_name='trait',
        value_name='value'
    )

# Usage
traits_long = convert_to_long_format(traits, id_vars=['plant_id', 'frame'])
traits_long.to_csv("traits_long.csv", index=False)

# Now easy to plot with seaborn/ggplot
import seaborn as sns

sns.lineplot(data=traits_long, x='frame', y='value', hue='trait')
```

### Convert to Wide Format

```python
# If you have long format, convert to wide
traits_wide = traits_long.pivot_table(
    index=['plant_id', 'frame'],
    columns='trait',
    values='value'
).reset_index()

traits_wide.to_csv("traits_wide.csv", index=False)
```

## Export for Statistical Software

### R-Compatible CSV

```python
# Export for R (handles special characters differently)
traits.to_csv(
    "traits_for_R.csv",
    index=False,
    na_rep='NA',  # R's missing value representation
    quoting=1  # Quote all text fields
)
```

### SPSS/SAS Format

```python
# Export for SPSS
traits.to_stata("traits.dta", write_index=False)

# Or use CSV with specific formatting
traits.to_csv(
    "traits_for_spss.csv",
    index=False,
    na_rep='.',  # SPSS uses '.' for missing
    decimal=','  # Some locales use comma for decimal
)
```

## Export with Experimental Design

### Include Design Matrix

```python
def export_with_design(traits, design_matrix, output_path):
    """
    Export traits merged with experimental design.

    Args:
        traits: DataFrame with traits
        design_matrix: DataFrame with experimental design
            (must have matching ID column)
        output_path: Output path
    """
    # Merge traits with design
    merged = traits.merge(design_matrix, on='plant_id', how='left')

    # Export
    merged.to_csv(output_path, index=False)

    # Also save design separately for reference
    design_matrix.to_csv(
        output_path.replace('.csv', '_design.csv'),
        index=False
    )

# Example design matrix
design = pd.DataFrame({
    'plant_id': ['plant_001', 'plant_002', 'plant_003'],
    'genotype': ['WT', 'WT', 'mutant'],
    'treatment': ['control', 'drought', 'control'],
    'replicate': [1, 2, 1],
    'plate': ['A', 'A', 'B']
})

export_with_design(traits, design, "traits_with_design.csv")
```

## Data Validation Before Export

### Quality Checks

```python
def validate_and_export(traits, output_path):
    """
    Validate traits before export.

    Args:
        traits: DataFrame with traits
        output_path: Output path

    Returns:
        Dictionary with validation results
    """
    validation = {
        'valid': True,
        'issues': []
    }

    # Check for missing values
    missing = traits.isnull().sum()
    if missing.any():
        validation['issues'].append(f"Missing values: {missing[missing > 0].to_dict()}")

    # Check for infinite values
    numeric_cols = traits.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if np.isinf(traits[col]).any():
            validation['issues'].append(f"Infinite values in {col}")

    # Check for unexpected ranges
    if 'primary_length' in traits.columns:
        if (traits['primary_length'] < 0).any():
            validation['issues'].append("Negative primary_length values")
            validation['valid'] = False

    # Export if valid, otherwise raise warning
    if validation['valid']:
        traits.to_csv(output_path, index=False)
        print(f"Exported to {output_path}")
    else:
        print("Validation failed:")
        for issue in validation['issues']:
            print(f"  - {issue}")
        print("Fix issues before exporting")

    return validation

# Usage
validation_result = validate_and_export(traits, "validated_traits.csv")
```

## Archival and Documentation

### Create Analysis Package

```python
import zipfile
import shutil
from datetime import datetime

def create_analysis_package(
    traits,
    metadata,
    output_dir="analysis_package"
):
    """
    Create complete analysis package with all files.

    Args:
        traits: DataFrame with traits
        metadata: Dictionary with metadata
        output_dir: Output directory name

    Returns:
        Path to created package
    """
    # Create output directory
    pkg_dir = Path(output_dir)
    pkg_dir.mkdir(exist_ok=True)

    # Export traits in multiple formats
    traits.to_csv(pkg_dir / "traits.csv", index=False)
    traits.to_excel(pkg_dir / "traits.xlsx", index=False)
    traits.to_json(pkg_dir / "traits.json", orient="records", indent=2)

    # Export summary statistics
    summary = traits.describe()
    summary.to_csv(pkg_dir / "summary_statistics.csv")

    # Create README
    readme_content = f"""
# Root Trait Analysis Package

## Experiment Information
- Experiment: {metadata.get('experiment', 'N/A')}
- Date: {metadata.get('date', datetime.now().strftime('%Y-%m-%d'))}
- Researcher: {metadata.get('researcher', 'N/A')}
- Pipeline: {metadata.get('pipeline', 'N/A')}

## Files Included
- `traits.csv`: Raw trait data (universal format)
- `traits.xlsx`: Trait data in Excel format
- `traits.json`: Trait data in JSON format
- `summary_statistics.csv`: Descriptive statistics
- `metadata.json`: Detailed metadata
- `README.md`: This file

## Data Dictionary
{_generate_data_dictionary(traits)}

## Citation
If you use this data, please cite:
[Add citation information]

## Contact
[Add contact information]
"""

    with open(pkg_dir / "README.md", 'w') as f:
        f.write(readme_content)

    # Save metadata as JSON
    with open(pkg_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Create zip archive
    zip_path = f"{output_dir}_{datetime.now().strftime('%Y%m%d')}.zip"
    shutil.make_archive(
        zip_path.replace('.zip', ''),
        'zip',
        pkg_dir
    )

    print(f"Analysis package created: {zip_path}")
    return zip_path

def _generate_data_dictionary(traits):
    """Generate data dictionary from DataFrame."""
    lines = []
    for col in traits.columns:
        dtype = traits[col].dtype
        lines.append(f"- `{col}`: {dtype}")
    return '\n'.join(lines)

# Usage
metadata = {
    'experiment': 'Drought stress response 2024',
    'date': '2024-01-15',
    'researcher': 'J. Smith',
    'pipeline': 'DicotPipeline',
    'n_plants': len(traits['plant_id'].unique()) if 'plant_id' in traits.columns else 1
}

package_path = create_analysis_package(traits, metadata)
```

## Best Practices

### 1. Always Include Metadata

```python
# Good: metadata included
export_with_metadata(traits, "traits.csv", metadata={...})

# Less useful: just data
traits.to_csv("traits.csv")
```

### 2. Use Multiple Formats

```python
# Export in multiple formats for different use cases
traits.to_csv("traits.csv", index=False)  # Universal
traits.to_excel("traits.xlsx", index=False)  # Spreadsheets
traits.to_parquet("traits.parquet")  # Efficient storage
```

### 3. Validate Before Sharing

```python
# Always validate before sharing
validation = validate_and_export(traits, "traits.csv")
if not validation['valid']:
    print("Fix issues before sharing!")
```

## Next Steps

- See [Batch Processing](batch-optimization.md) for exporting large datasets
- Read [Custom Traits](custom-traits.md) for specialized export needs
- Check [Filtering Data](filtering-data.md) for data quality before export