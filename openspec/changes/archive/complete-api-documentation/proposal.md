# Proposal: Complete API Documentation (Phase 2)

## Executive Summary

Complete the API documentation for sleap-roots by creating dedicated pages for all 13 modules, enhancing docstrings, adding practical code examples, and establishing cross-references. This builds on Phase 1's infrastructure to provide comprehensive, user-friendly API documentation.

## Problem Statement

### Current State

**Infrastructure**: ✅ Complete
- MkDocs Material theme configured
- mkdocstrings auto-generation working
- `docs/api/index.md` exists (456 lines)
- Navigation structure in place

**Content**: ⚠️ Incomplete
- Only `docs/api/index.md` exists with inline API references
- No dedicated pages for individual modules
- Docstrings exist but vary in completeness
- Missing practical examples for many functions
- No systematic cross-referencing between related modules

### User Impact

Users currently face:
1. **Single-page overload**: All API docs crammed into one 456-line file
2. **Hard to navigate**: No dedicated module pages for deep diving
3. **Missing context**: Lack of examples showing real-world usage
4. **Unclear relationships**: No explicit links between related functions
5. **Inconsistent depth**: Some modules well-documented, others sparse

### Why This Matters

sleap-roots is a **scientific library** where:
- Users need to understand **exactly** what each trait computes
- Researchers need **reproducible** documentation of methods
- Contributors need clear **contracts** for each function
- Integration requires understanding **data structures** and **return types**

Poor API docs = users struggle to use the library correctly = reduced adoption.

## Proposed Solution

### Goals

1. **Create dedicated module pages** for all 13 modules
2. **Enhance docstrings** where clarity is needed
3. **Add practical code examples** to key functions
4. **Establish cross-references** between related modules
5. **Organize by functional area** for easy discovery

### Architecture

#### 1. Documentation Structure

```
docs/api/
├── index.md                    # Overview + navigation (keep current)
├── core/
│   ├── series.md              # Series class (primary data structure)
│   └── pipelines.md           # All pipeline classes
├── traits/
│   ├── lengths.md             # Length calculations
│   ├── angles.md              # Angle calculations
│   ├── tips.md                # Tip detection
│   ├── bases.md               # Base detection
│   ├── convhull.md            # Convex hull analysis
│   ├── ellipse.md             # Ellipse fitting
│   ├── networklength.md       # Network-level metrics
│   ├── scanline.md            # Scanline intersection
│   └── points.md              # Point manipulation utilities
├── utilities/
│   └── summary.md             # Summary statistics
└── examples/
    └── common-workflows.md    # Cross-cutting examples
```

**Rationale**:
- **Organized by purpose**: Core → Traits → Utilities
- **Discoverability**: Users can find what they need by function
- **Scalability**: Easy to add new modules in the future
- **Separation**: Examples page for multi-module workflows

#### 2. Module Page Template

Each module page follows this structure:

```markdown
# module_name

## Overview
[1-2 paragraph description of what this module does and when to use it]

## Key Functions

### function_name

::: sleap_roots.module_name.function_name
    options:
      show_source: true

**Example**:
```python
# Practical example with real data
import sleap_roots as sr
import numpy as np

# Setup
pts = np.array([[10, 20], [15, 25], [20, 30]])

# Usage
result = sr.module_name.function_name(pts)
print(result)  # Expected output: ...
```

**See also**: [related_function](#related_function), [other_module](../other_module.md)

---

[Repeat for each key function]

## Advanced Usage

[Optional section for complex scenarios]

## Related Modules

- [module1](../module1.md) - Brief description
- [module2](../module2.md) - Brief description
```

**Benefits**:
- **Consistent structure**: Users know what to expect
- **Practical examples**: Real code they can copy/paste
- **Cross-references**: Easy navigation to related functionality
- **Auto-generated**: mkdocstrings pulls from docstrings

#### 3. Docstring Enhancement Strategy

**Audit all public functions** for:

✅ **Must have**:
- Summary line (one sentence)
- Args section with types
- Returns section with type and description
- Brief description of what the function does

🎯 **Should have**:
- Raises section (if applicable)
- Example in docstring (for complex functions)
- Notes about edge cases

📚 **Nice to have**:
- References to papers/methods
- Performance notes
- Cross-references to related functions

**Focus areas** (based on user pain points):
1. **Pipeline classes** - Most user-facing, needs examples
2. **Core trait functions** - Scientific accuracy crucial
3. **Series class** - Primary entry point, must be crystal clear

**Approach**:
- Read existing docstrings module by module
- Identify gaps (missing Args, unclear Returns, etc.)
- Enhance in same PR as creating module page
- Don't break existing docstrings that are already good

#### 4. Example Strategy

**Three levels of examples**:

1. **Inline (in docstrings)**: Minimal, single-function usage
   ```python
   >>> get_root_angle(pts, n_points=5)
   45.0
   ```

2. **Module pages**: Practical, realistic usage
   ```python
   # Working with real data
   series = sr.Series.load(...)
   pts = series.get_primary_root_points()
   angle = sr.get_root_angle(pts, n_points=5)
   ```

3. **Examples page**: Complete workflows
   ```python
   # Complete pipeline: load → compute → export
   # [Full end-to-end example]
   ```

**Example selection criteria**:
- Uses realistic data (not just `[[1,2],[3,4]]`)
- Shows common parameters
- Demonstrates expected output
- Can be copy/pasted and run

### Implementation Plan

#### Phase 2A: Core Module Pages (Priority 1)
**Estimated: 4 hours**

Create pages for core modules that users interact with most:

1. **docs/api/core/series.md** - Series class (2 hours)
   - Load methods (load, load_multi)
   - Point extraction (get_primary_root_points, get_lateral_root_points, get_crown_root_points)
   - Properties (series_name, h5_path, primary_path, etc.)
   - Complete workflow example

2. **docs/api/core/pipelines.md** - All pipeline classes (2 hours)
   - DicotPipeline
   - YoungerMonocotPipeline
   - OlderMonocotPipeline
   - MultipleDicotPipeline
   - MultiplePrimaryRootPipeline
   - PrimaryRootPipeline
   - LateralRootPipeline
   - When to use each (decision tree)
   - Complete pipeline example for each

**Deliverable**: Users can understand and use core classes

---

#### Phase 2B: Trait Module Pages (Priority 2)
**Estimated: 6 hours**

Create pages for trait computation modules (9 modules @ ~40 min each):

1. **docs/api/traits/lengths.md** (40 min)
   - get_root_lengths
   - get_curve_index
   - get_max_length_pts
   - Examples with root point arrays

2. **docs/api/traits/angles.md** (40 min)
   - get_root_angle
   - get_vector_angle_from_gravity
   - get_node_ind
   - Examples showing angle calculation

3. **docs/api/traits/tips.md** (40 min)
   - get_tips
   - get_tip_xs, get_tip_ys
   - Examples with NaN handling

4. **docs/api/traits/bases.md** (40 min)
   - get_bases
   - get_base_xs, get_base_ys
   - get_base_length
   - get_base_ct_density
   - get_root_widths
   - Examples for lateral roots

5. **docs/api/traits/convhull.md** (40 min)
   - get_convhull
   - get_convhull_features
   - get_chull_perimeter
   - get_chull_area
   - Examples with shapely geometries

6. **docs/api/traits/ellipse.md** (40 min)
   - get_ellipse
   - fit_ellipse
   - Examples with fitting parameters

7. **docs/api/traits/networklength.md** (40 min)
   - get_network_length
   - get_network_width_depth_ratio
   - get_network_distribution
   - get_network_solidity
   - get_bbox
   - Examples for whole-plant metrics

8. **docs/api/traits/scanline.md** (40 min)
   - count_scanline_intersections
   - get_scanline_first_ind
   - get_scanline_last_ind
   - Examples with horizontal lines

9. **docs/api/traits/points.md** (40 min)
   - join_pts
   - get_all_pts_array
   - associate_lateral_to_primary
   - flatten_associated_points
   - filter_roots_with_nans
   - get_node
   - Examples for point manipulation

**Deliverable**: Complete reference for all trait computations

---

#### Phase 2C: Utilities & Cross-Cutting (Priority 3)
**Estimated: 2 hours**

1. **docs/api/utilities/summary.md** (30 min)
   - get_summary
   - Example exporting to CSV

2. **docs/api/examples/common-workflows.md** (1.5 hours)
   - Complete pipeline workflow
   - Batch processing example
   - Custom trait computation
   - Filtering and cleaning data
   - Multi-plant analysis

**Deliverable**: End-to-end workflow examples

---

### Total Estimated Time: 12 hours

**Breakdown**:
- Module page creation: 10 hours
- Docstring enhancement: 1.5 hours (as-needed during page creation)
- Navigation updates: 0.5 hours

## Success Criteria

### Measurable Goals

1. ✅ **All 13 modules have dedicated pages**
2. ✅ **Every public function has at least one example**
3. ✅ **All pages follow consistent template structure**
4. ✅ **Navigation links work correctly**
5. ✅ **mkdocs build --strict succeeds**
6. ✅ **Cross-references between related modules**

### Quality Checks

- [ ] Can a new user understand what each module does?
- [ ] Are examples copy/paste-able and runnable?
- [ ] Are all Args/Returns documented?
- [ ] Is it clear when to use each function?
- [ ] Are edge cases (NaN, empty arrays) documented?
- [ ] Do cross-references help users discover related functionality?

## Non-Goals

**Out of scope for Phase 2**:

- ❌ **Complete docstring rewrite** - Only enhance where needed
- ❌ **Adding new functionality** - Documentation only
- ❌ **Jupyter notebook examples** - That's Phase 6
- ❌ **Video tutorials** - Not in current plan
- ❌ **Translating docs** - English only for now
- ❌ **PDF export** - Web-only for now

## Risks & Mitigation

### Risk 1: Time Estimation Too Optimistic
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Break into Phase 2A/2B/2C for incremental progress
- 2A (core) provides immediate value even if 2B/2C delayed
- Can merge pages incrementally

### Risk 2: Docstrings Need Major Rewrite
**Likelihood**: Low
**Impact**: High (would balloon timeline)
**Mitigation**:
- Audit docstrings first (30 min per module)
- Only enhance where truly needed
- Existing docstrings are generally good (Google style)

### Risk 3: Examples Don't Run
**Likelihood**: Medium
**Impact**: Medium
**Mitigation**:
- Test all examples locally before committing
- Use real test data from tests/fixtures
- Document which data files are needed

### Risk 4: Auto-generation Fails
**Likelihood**: Low
**Impact**: High
**Mitigation**:
- mkdocstrings already working in Phase 1
- Test with one module first before scaling

## Alternatives Considered

### Alternative 1: Keep Single-Page API Docs
**Pros**:
- Less work
- Single searchable page

**Cons**:
- Already 456 lines, hard to navigate
- Doesn't scale as library grows
- Poor UX for deep diving into specific modules

**Decision**: ❌ Rejected - UX too poor

---

### Alternative 2: Auto-Generate Everything (No Manual Pages)
**Pros**:
- Minimal manual work
- Always stays in sync with code

**Cons**:
- No practical examples
- No cross-references
- No "when to use" guidance
- Sterile, hard to learn from

**Decision**: ❌ Rejected - Need human-curated examples

---

### Alternative 3: Organize by Alphabetical (Not Functional)
**Pros**:
- Easy to find if you know the name
- Simple structure

**Cons**:
- Doesn't match how users think
- New users don't know what modules exist
- No logical grouping

**Decision**: ❌ Rejected - Functional grouping better for discovery

---

### Alternative 4: Use Sphinx Instead of MkDocs
**Pros**:
- Common in scientific Python
- Strong autodoc capabilities

**Cons**:
- Already invested in MkDocs (Phase 1)
- MkDocs Material has better UX
- Would require throwing away working infrastructure

**Decision**: ❌ Rejected - MkDocs is working great

## Dependencies

### Required Infrastructure (Already Complete ✅)
- MkDocs configured
- mkdocstrings plugin installed
- Material theme
- GitHub Pages deployment
- CI/CD workflow

### Required for Implementation
- Access to test data (for realistic examples)
- Understanding of each module's purpose
- Knowledge of common use cases

### Blocking Issues
- None - can start immediately

## Timeline

**Optimistic** (focused work): 2-3 days
**Realistic** (part-time): 1 week
**Conservative** (with reviews/iteration): 2 weeks

**Recommended Approach**: Break into 2A → 2B → 2C increments

Week 1:
- Days 1-2: Phase 2A (Core modules)
- Days 3-5: Phase 2B (Trait modules, 3-4 per day)

Week 2:
- Day 6: Phase 2B completion
- Day 7: Phase 2C (Utilities + examples)
- Review and polish

## Future Work (Not in Phase 2)

**Phase 3**: User Guides
- Data preparation workflows
- Batch processing patterns
- Visualization tutorials

**Phase 4**: Trait Reference
- Auto-generated trait tables
- HackMD content migration
- Validation data

**Phase 5**: Developer Docs
- Architecture documentation
- Adding new pipelines
- Testing guide

**Phase 6**: Cookbook + Notebooks
- Jupyter notebook examples
- Advanced recipes
- Integration patterns

## Approval & Next Steps

### Questions for Stakeholders

1. Should we include performance notes in API docs? (e.g., "O(n²) complexity")
2. Preferred example data source: test fixtures or synthetic minimal examples?
3. Should we add "Common mistakes" sections to frequently misused functions?
4. Any specific modules that need priority attention?

### After Approval

1. Create `openspec/changes/complete-api-documentation/tasks.md`
2. Create feature branch: `docs/complete-api-documentation`
3. Start with Phase 2A (core modules)
4. Iterate based on feedback

---

**Status**: 🟡 Awaiting approval
**Estimated Effort**: 12 hours
**Value**: High - Essential for library usability
**Risk**: Low - Builds on working infrastructure