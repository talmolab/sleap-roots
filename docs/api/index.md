# API Reference

Welcome to the sleap-roots API reference documentation. This page provides an overview of the entire API organized by functional area.

## Quick Links

- **New to sleap-roots?** Start with the [Quick Start Guide](../getting-started/quickstart.md)
- **Ready to analyze?** Check out [Common Workflows](examples/common-workflows.md)
- **Need a specific function?** Browse by category below

---

## Core Modules

The foundation of sleap-roots: loading data and running analysis pipelines.

### [Series](core/series.md)
**Load and manage SLEAP predictions**

The `Series` class is your primary interface for working with SLEAP root tracking data. Load predictions from .slp files and access point arrays for analysis.

**Key methods**:
- `Series.load()` - Load single plant
- `get_primary_points()` - Extract primary root coordinates
- `get_lateral_points()` - Extract lateral root coordinates
- `get_crown_points()` - Extract crown root coordinates

[View Series documentation →](core/series.md)

---

### [Pipelines](core/pipelines.md)
**Pre-built trait extraction workflows**

Choose from 7 specialized pipelines for different root system architectures. Each pipeline automatically computes dozens of relevant traits.

**Available pipelines**:
- `DicotPipeline` - Primary + lateral roots (canola, Arabidopsis)
- `YoungerMonocotPipeline` - Primary + crown roots (young rice, wheat)
- `OlderMonocotPipeline` - Crown roots only (mature monocots)
- `MultipleDicotPipeline` - Multiple dicot plants per image
- `PrimaryRootPipeline` - Single primary root analysis
- `MultiplePrimaryRootPipeline` - Multiple primary roots per image
- `LateralRootPipeline` - Individual lateral root analysis

[View Pipelines documentation →](core/pipelines.md)

---

## Trait Computation Modules

Individual functions for computing specific root traits. Use these for custom analysis workflows.

### [Lengths](traits/lengths.md)
Measure root lengths and curvature.

**Functions**: `get_root_lengths`, `get_curve_index`, `get_max_length_pts`

[View documentation →](traits/lengths.md)

---

### [Angles](traits/angles.md)
Analyze root growth angles and gravitropism.

**Functions**: `get_root_angle`, `get_vector_angle_from_gravity`, `get_node_ind`

[View documentation →](traits/angles.md)

---

### [Tips](traits/tips.md)
Detect and analyze root tips for growth tracking.

**Functions**: `get_tips`, `get_tip_xs`, `get_tip_ys`

[View documentation →](traits/tips.md)

---

### [Bases](traits/bases.md)
Analyze lateral root emergence patterns and density.

**Functions**: `get_bases`, `get_base_length`, `get_base_ct_density`, `get_root_widths`

[View documentation →](traits/bases.md)

---

### [Convex Hull](traits/convhull.md)
Compute spatial coverage and distribution metrics.

**Functions**: `get_convhull`, `get_convhull_features`, `get_chull_area`, `get_chull_perimeter`

[View documentation →](traits/convhull.md)

---

### [Ellipse](traits/ellipse.md)
Fit ellipses to root point distributions.

**Functions**: `get_ellipse`, `fit_ellipse`

[View documentation →](traits/ellipse.md)

---

### [Network Length](traits/networklength.md)
Analyze whole-plant network-level metrics.

**Functions**: `get_network_length`, `get_network_width_depth_ratio`, `get_network_distribution`, `get_bbox`

[View documentation →](traits/networklength.md)

---

### [Scanline](traits/scanline.md)
Count root intersections with horizontal scan lines.

**Functions**: `count_scanline_intersections`, `get_scanline_first_ind`, `get_scanline_last_ind`

[View documentation →](traits/scanline.md)

---

### [Points](traits/points.md)
Utility functions for manipulating root point arrays.

**Functions**: `join_pts`, `get_all_pts_array`, `associate_lateral_to_primary`, `filter_roots_with_nans`

[View documentation →](traits/points.md)

---

## Utilities

### [Summary Statistics](utilities/summary.md)
Compute comprehensive summary statistics for trait distributions.

**Function**: `get_summary` - Calculate min, max, mean, median, std, and percentiles

[View documentation →](utilities/summary.md)

---

## Examples and Workflows

### [Common Workflows](examples/common-workflows.md)
Complete end-to-end examples for typical analysis tasks.

**Workflows included**:
1. Quick pipeline analysis
2. Custom trait computation
3. Lateral root analysis
4. Temporal growth tracking
5. Network-level spatial analysis
6. Batch processing multiple plants
7. Quality control and filtering
8. Multiple dicot plants

[View workflows →](examples/common-workflows.md)

---

## API Organization

The sleap-roots API is organized hierarchically:

```
sleap_roots/
├── Series                    # Data loading
├── DicotPipeline            # Pre-built pipelines
├── YoungerMonocotPipeline
├── OlderMonocotPipeline
├── MultipleDicotPipeline
├── PrimaryRootPipeline
├── MultiplePrimaryRootPipeline
├── LateralRootPipeline
│
├── get_root_lengths()       # Individual trait functions
├── get_root_angle()
├── get_tips()
├── get_bases()
├── get_convhull()
├── get_ellipse()
├── get_network_length()
├── count_scanline_intersections()
│
├── join_pts()               # Utility functions
├── get_all_pts_array()
├── filter_roots_with_nans()
└── get_summary()
```

---

## Finding What You Need

**I want to...**

- **Load SLEAP predictions** → [Series](core/series.md)
- **Extract traits automatically** → [Pipelines](core/pipelines.md)
- **Measure root lengths** → [Lengths](traits/lengths.md)
- **Analyze growth angles** → [Angles](traits/angles.md)
- **Track tip movement** → [Tips](traits/tips.md)
- **Study lateral emergence** → [Bases](traits/bases.md)
- **Measure spatial coverage** → [Convex Hull](traits/convhull.md)
- **Analyze network architecture** → [Network Length](traits/networklength.md)
- **Process multiple plants** → [Common Workflows](examples/common-workflows.md)
- **Get summary statistics** → [Summary Statistics](utilities/summary.md)

---

## Complete Function Reference

For auto-generated documentation from source code, see the [reference/](reference/) section.

## See Also

- **[Getting Started Guide](../getting-started/quickstart.md)** - Your first sleap-roots analysis
- **[Tutorials](../tutorials/index.md)** - Step-by-step pipeline guides
- **[User Guide](../guides/index.md)** - In-depth explanations
- **[Cookbook](../cookbook/index.md)** - Recipes for common tasks