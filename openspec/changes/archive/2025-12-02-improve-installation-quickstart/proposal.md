# Proposal: Improve Installation and Quick Start for Beginners

## Problem Statement

The installation and quick start documentation has critical usability issues that will cause problems for beginner users (plant biologists who may not be Python experts):

### 1. **Installation Promotes Base Environment Pollution**
The first thing users see is `pip install sleap-roots` which encourages installing into their base Python environment. Plant biologists are not experienced Python developers and will follow the first instruction they see, leading to dependency conflicts and environment issues.

**Current flow:**
1. "Quick Install: `pip install sleap-roots`"
2. Later: "Recommended: Using uv" (too late)
3. Finally: "Alternative: Using Conda"

**Problem:** Users have already pip-installed into base before seeing the recommended uv workflow.

### 2. **Missing `uv run` Prefix on Commands**
All Python execution examples show bare `python` commands instead of `uv run python`, which won't work for users who followed the uv installation workflow.

**Examples:**
- Installation verification: `python -c "import sleap_roots..."` (should be `uv run python -c`)
- All quickstart examples show Python scripts without `uv run` context

### 3. **No Actual Output Examples**
Every code example shows the code but never shows what the actual output looks like. Users don't know:
- What does `compute_plant_traits()` return? (Answer: DataFrame with one row per frame)
- What does `compute_batch_traits()` return? (Answer: DataFrame with one row per plant, summary statistics)
- What columns will be in the DataFrame?
- What do the values look like?

**User confusion:** "Did it work? What should I expect to see?"

### 4. **Unclear Output Structure and Summary Statistics**
Documentation doesn't explain the critical distinctions:

**Two levels of summarization:**
1. **Per-frame** (for non-scalar traits like `lateral_lengths`): Statistics across multiple roots within one frame
2. **Per-plant** (for batch processing): Statistics across all frames for one plant

This creates **single-suffixed** names for scalar traits:
- `primary_length_mean` = mean of primary length across frames

And **double-suffixed** names for non-scalar traits:
- `lateral_lengths_mean_median` = median of (mean lateral length per frame) across all frames
  - First `_mean`: across lateral roots within each frame
  - Second `_median`: across frames for the plant

**Current problems:**
- Users don't know which traits have one vs two levels of statistics
- Column naming pattern not explained (`{trait}_{stat}` vs `{trait}_{frame_stat}_{plant_stat}`)
- No guidance on when to use mean vs median vs percentiles
- Summary statistics not listed (min, max, mean, median, std, p5, p25, p75, p95)
- No explanation of when to use frame-level vs plant-level outputs

### 5. **Generic Code That Should Be Removed**
The quickstart includes basic pandas/R/matplotlib code that:
- Is not sleap-roots specific
- Clutters the tutorial
- Should link to external visualization guides instead

**Examples to remove:**
- Basic pandas `.describe()` usage
- Basic R `read.csv()` and `summary()`
- Full matplotlib plotting code
- Generic DataFrame column creation

### 6. **Repeated Troubleshooting Content**
Common issues section duplicates 4 entire error explanations that exist in the troubleshooting guide:
- "No module named 'sleap_roots'"
- "FileNotFoundError: predictions.slp"
- Empty or NaN traits
- "KeyError: 'primary_pts'"

This creates maintenance burden and inconsistency.

### 7. **Wrong Trait Reference Links**
Documentation links to `trait-reference.md` instead of the API reference, sending users to the wrong place for trait definitions.

## Proposed Solution

### Phase 1: Restructure Installation to Lead with uv
1. **Remove "Quick Install"** section entirely (or move to bottom as "Alternative: pip")
2. **Start with "Getting Started with uv"** as the primary installation method
3. **Show complete uv workflow** from project creation to verification
4. **Add `uv run` prefix** to all Python commands

### Phase 2: Add Real Output Examples
1. **Show actual DataFrame outputs** for every example:
   - `compute_plant_traits()`: Show first 3 rows with actual column names and values
   - `compute_batch_traits()`: Show actual summary statistics output
2. **Explain output structure** clearly:
   - Frame-level vs plant-level
   - What summary statistics are computed (min, max, mean, median, std, p5, p25, p75, p95)
   - Which columns come from which methods

### Phase 3: Clarify Output Levels
Add explicit explanations:
- **`compute_plant_traits()`**: "Returns one row per frame with raw trait measurements"
- **`compute_batch_traits()`**: "Returns one row per plant with summary statistics (min, max, mean, median, std, percentiles) aggregated across all frames"

### Phase 4: Remove Generic Code
Delete sections:
- "Working with CSV Output" (pandas/R basics)
- "Visualizing Results" (generic matplotlib)
- "Converting Pixels to Real Units" (keep concept, remove code)

Replace with:
- Link to visualization example repository
- Brief explanation that CSVs work with any analysis tool
- Note about pixel scaling with conceptual explanation only

### Phase 5: Consolidate Troubleshooting
Replace entire "Common Issues" section with:
```markdown
## Common Issues

For installation problems, import errors, and troubleshooting, see the **[Troubleshooting Guide](../guides/troubleshooting.md)**.
```

Keep only 1-2 most critical issues inline if absolutely necessary.

### Phase 6: Fix Trait Reference Links
Change all `trait-reference.md` links to point to API reference documentation instead.

## Impact

### User Impact
- **High**: Beginners won't pollute base environment
- **High**: Clear expectations about what outputs look like
- **Medium**: Understand frame-level vs summary statistics
- **Medium**: Less clutter, faster to key concepts

### Development Impact
- **Low**: Only documentation changes
- **Low**: Reduces maintenance burden (less duplication)

### Documentation Impact
- **High**: Much clearer workflow for beginners
- **High**: Consistent use of uv throughout
- **Medium**: Better alignment with actual API behavior

## Success Criteria

1. ✅ First installation instruction is uv workflow, not pip
2. ✅ All Python commands show `uv run` prefix
3. ✅ Every code example includes actual output showing column names and values
4. ✅ Clear explanation of frame-level vs plant-level outputs
5. ✅ Summary statistics thoroughly explained:
   - Which traits get single vs double statistics
   - Column naming patterns documented
   - When to use each statistic (mean vs median vs percentiles)
   - Complete list of 9 statistics provided
6. ✅ No generic pandas/R/matplotlib code in quickstart
7. ✅ Troubleshooting consolidated to single link
8. ✅ All trait reference links point to API docs

## Related Work

- Builds on `simplify-user-documentation` which improved overall doc organization
- Builds on `fix-documentation-examples` which fixed broken Example 3
- Complements `documentation-quality` spec for accurate, beginner-friendly docs

## Alternatives Considered

### Alternative 1: Keep pip as primary, add notes about uv
**Rejected**: Users won't read notes, they'll follow first instruction (pip).

### Alternative 2: Show both pip and uv equally
**Rejected**: Gives mixed messages, doesn't guide users to best practice.

### Alternative 3: Keep generic code examples
**Rejected**: User specifically requested removal, and it clutters the tutorial.

### Alternative 4: Keep troubleshooting duplicated
**Rejected**: Creates maintenance burden and version drift.

## Open Questions

None - user provided clear direction on all aspects.