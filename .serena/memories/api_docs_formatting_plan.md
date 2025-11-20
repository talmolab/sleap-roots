# API Documentation Formatting Issues - Detailed Plan

## Summary
Two main formatting issues identified in sleap-roots API documentation:
1. **Duplicate headings** - Manual markdown headings (e.g., `#### get_primary_points`) conflict with mkdocstrings auto-generated headings
2. **Bullet lists not rendering** - "See Also" sections placed incorrectly relative to mkdocstrings blocks

## Affected Files

### Priority 1: Core Classes (MOST COMPLEX)
- **docs/api/core/series.md** (~650 lines)
  - Issue 1: Manual `#### get_primary_points` before `:::` blocks
  - Issue 2: "See Also" bullet lists appear directly after `:::` block content
  - Affected functions: `load`, `get_primary_points`, `get_lateral_points`, `get_crown_points`, `get_metadata`, `filter_plants`, `plot`
  
- **docs/api/core/pipelines.md** (~612 lines)
  - Issue 1: Manual headings like `### DicotPipeline` before `:::` blocks
  - Issue 2: "See Also" sections with broken bullet rendering
  - Affected pipelines: `DicotPipeline`, `MultipleDicotPipeline`, `YoungerMonocotPipeline`, `OlderMonocotPipeline`, `PrimaryRootPipeline`, `MultiplePrimaryRootPipeline`, `LateralRootPipeline`

### Priority 2: Trait Modules (SIMPLER)
- **docs/api/traits/bases.md**
  - No manual `####` headings (correct!)
  - Bullet lists appear AFTER `:::` blocks (correct positioning!)
  - No fixes needed
  
- **docs/api/traits/lengths.md**
  - No manual `####` headings (correct!)
  - Bullet lists appear AFTER `:::` blocks (correct positioning!)
  - Likely no fixes needed (but check all trait modules)

- Other trait modules: `angles.md`, `convhull.md`, `ellipse.md`, `networklength.md`, `points.md`, `scanline.md`, `tips.md`
  - Likely follow correct pattern but need verification

## Pattern Analysis

### INCORRECT Pattern (in series.md & pipelines.md):
```markdown
#### get_primary_points

::: sleap_roots.Series.get_primary_points
    options:
      show_source: true
      heading_level: 4

**Example**:
```python
...
```

**See Also**:
- [link1](#anchor1)
- [link2](#anchor2)

---
```

**Problems:**
1. Manual `####` heading duplicates the auto-generated heading from `heading_level: 4`
2. "See Also" bullet list appears right after the `:::` block closes
3. Result: mkdocstrings renders the heading, then the content/example, but the bullet list rendering breaks

### CORRECT Pattern (in traits modules):
```markdown
::: sleap_roots.bases.get_bases
    options:
      show_source: true

**Example**:
```python
...
```

**See Also**:
- [link1](#anchor1)
- [link2](#anchor2)

---
```

**Why it works:**
1. No manual heading (mkdocstrings handles it with default heading_level)
2. "See Also" section is clearly OUTSIDE/AFTER the `:::` block
3. Proper spacing between block and next section

## Fixing Strategy

### Step 1: Fix series.md
1. Remove all manual `#### function_name` headings (8 instances)
2. Verify mkdocstrings `heading_level: 4` will handle the heading
3. Move "See Also" sections to after any example content
4. Add blank line before bullet lists to ensure proper rendering
5. Test and commit

### Step 2: Fix pipelines.md
1. Remove manual `### PipelineName` headings (7 instances)
2. Keep `heading_level: 3` in mkdocstrings options
3. Move "See Also" sections appropriately
4. Add blank lines before bullet lists
5. Test and commit

### Step 3: Verify trait modules
1. Quick check of all trait modules to confirm correct pattern
2. No fixes expected, but document findings
3. List all verified modules

### Step 4: Test build
1. Run mkdocs build locally
2. Check rendered output for:
   - Single (non-duplicate) headings for each function
   - Properly formatted bullet lists in "See Also" sections
   - No formatting artifacts

## Detailed Changes Required

### series.md - 8 headings to remove:
- Line 57: `#### load`
- Line 129: `#### get_primary_points`
- Line 165: `#### get_lateral_points`
- Line 218: `#### get_crown_points`
- (Continue for remaining functions with manual headings)

### pipelines.md - 7 headings to remove:
- Manual section headings for each pipeline class that conflicts with mkdocstrings

## Rollback Plan
If issues arise:
1. Git history preserves original formatting
2. Easy to revert specific files with git checkout
3. Can also restore manual headings if mkdocstrings auto-generation fails

## Success Criteria
- [x] No duplicate headings in rendered docs
- [x] "See Also" bullet lists render properly
- [x] No formatting artifacts
- [x] mkdocs build completes without warnings
- [x] Visual inspection confirms correct layout
