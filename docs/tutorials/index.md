# Tutorials

Welcome to the sleap-roots tutorials! These interactive Jupyter notebooks provide hands-on walkthroughs of analyzing plant root systems with different pipeline types.

## Overview

Each tutorial demonstrates a complete analysis workflow from loading SLEAP predictions to computing and exporting morphological traits. The notebooks include:

- Step-by-step code examples
- Visualization of intermediate results
- Trait computation and interpretation
- Export to CSV for downstream analysis

## Available Tutorials

### Primary Pipelines

<div class="grid cards" markdown>

-   :seedling:{ .lg .middle } **Dicot Pipeline**

    ---

    Analyze dicot root systems with primary and lateral roots (soy, canola, arabidopsis).

    [:octicons-arrow-right-24: Tutorial](dicot-pipeline.md)

-   :ear_of_rice:{ .lg .middle } **Younger Monocot Pipeline**

    ---

    Process early-stage monocots with primary and crown roots (rice, maize).

    [:octicons-arrow-right-24: Tutorial](younger-monocot-pipeline.md)

-   :corn:{ .lg .middle } **Older Monocot Pipeline**

    ---

    Analyze mature monocots with crown roots only (rice, maize).

    [:octicons-arrow-right-24: Tutorial](older-monocot-pipeline.md)

</div>

### Multi-Plant Pipelines

<div class="grid cards" markdown>

-   :potted_plant:{ .lg .middle } **Multiple Dicot Pipeline**

    ---

    Batch process multiple dicot plants in a single image.

    [:octicons-arrow-right-24: Tutorial](multiple-dicot-pipeline.md)

-   :herb:{ .lg .middle } **Multiple Primary Root Pipeline**

    ---

    Analyze multiple plants with primary roots.

    [:octicons-arrow-right-24: Tutorial](multiple-primary-root-pipeline.md)

</div>

### Specialized Pipelines

<div class="grid cards" markdown>

-   :deciduous_tree:{ .lg .middle } **Primary Root Pipeline**

    ---

    Focus on primary root traits only.

    [:octicons-arrow-right-24: Tutorial](primary-root-pipeline.md)

-   :fallen_leaf:{ .lg .middle } **Lateral Root Pipeline**

    ---

    Specialized analysis for lateral root systems.

    [:octicons-arrow-right-24: Tutorial](lateral-root-pipeline.md)

</div>

## How to Use These Tutorials

### Option 1: View Online

All tutorials are rendered directly in the documentation. You can read through them and copy code snippets as needed.

### Option 2: Run Interactively

To run the notebooks yourself:

1. Clone the sleap-roots repository:
   ```bash
   git clone https://github.com/talmolab/sleap-roots.git
   cd sleap-roots
   ```

2. Install with dev dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Launch Jupyter:
   ```bash
   jupyter notebook notebooks/
   ```

4. Open any tutorial notebook and run the cells.

### Data Requirements

The tutorials use example data included in the repository's test fixtures. To use your own data:

- Replace file paths with your SLEAP prediction files (`.h5`, `.slp`)
- Ensure your SLEAP models match the expected node structure for each pipeline
- See [Data Formats](../guides/data-formats/sleap-files.md) for details

## Troubleshooting

**Notebook won't run:**
- Ensure you've installed sleap-roots: `pip install sleap-roots`
- Check that Git LFS data is downloaded: `git lfs pull`

**Missing data files:**
- The example data is stored with Git LFS
- Run `git lfs install && git lfs pull` to download

**Trait values look wrong:**
- Verify your SLEAP predictions use the correct skeleton structure
- Check pixel-to-mm conversion if needed (see [Trait Reference](../guides/trait-reference.md))

## Next Steps

After completing a tutorial:

- Read the [Pipeline Guide](../guides/index.md) for in-depth explanations
- Explore the [Trait Reference](../guides/trait-reference.md) to understand computed traits
- Check the [Cookbook](../cookbook/index.md) for advanced recipes
- See [Batch Processing](../guides/batch-processing.md) for high-throughput analysis