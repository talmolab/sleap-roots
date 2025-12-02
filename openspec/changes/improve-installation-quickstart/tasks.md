# Tasks: Improve Installation and Quick Start for Beginners

## Phase 1: Restructure installation.md to Lead with uv

- [ ] Remove "Quick Install" section (lines 5-13) or move to bottom as "Alternative: pip"
- [ ] Rename "Recommended: Using uv" to "Getting Started" and move to top
- [ ] Expand uv section to show complete beginner workflow:
  - Installing uv
  - Creating new project with `uv init`
  - Adding sleap-roots with `uv add`
  - Running scripts with `uv run`
- [ ] Update verification commands to use `uv run`:
  - Line 59: `python -c "..."` → `uv run python -c "..."`
  - Lines 45-54: Add `uv run` prefix to Python import example
- [ ] Move conda section further down (it's fine where it is as "Alternative")
- [ ] Add actual output examples to verification section

**Validation**: Installation guides users to uv first, all commands use `uv run` ✅

---

## Phase 2: Add Real Output Examples to quickstart.md

### Example 1: Single Plant Analysis (lines 29-58)

- [ ] After line 52 (`print(traits_df.head())`), add actual output:
  ```python
  # Output (first 3 rows):
  #   plant_name  frame_idx  primary_length  lateral_count  primary_angle  ...
  # 0    919QDUH          0          523.45             12          87.32  ...
  # 1    919QDUH          1          541.23             13          86.91  ...
  # 2    919QDUH          2          558.91             14          85.77  ...
  ```
- [ ] Add explanation: "Returns one row per frame with raw trait measurements"

### Example 2: Batch Processing (lines 60-86)

- [ ] After line 79 (`print(batch_df[['plant', 'frame_idx', ...]])`), add actual output:
  ```python
  # Output (plant-level summaries):
  #   plant_name  primary_length_mean  primary_length_std  lateral_count_median  ...
  # 0   plant_001               534.89               12.45                  13.0  ...
  # 1   plant_002               612.34               18.92                  15.0  ...
  ```
- [ ] Add explanation: "Returns one row per plant with summary statistics (min, max, mean, median, std, p5, p25, p75, p95) aggregated across all frames"
- [ ] Fix line 78: Should show summary stats columns, not frame_idx (batch doesn't have frame_idx)

### Example 3: Individual Functions (lines 88-115)

- [ ] After lines 113-114 (print statements), add actual output:
  ```python
  # Output:
  # Primary root length: 971.05 pixels
  # Primary root angle: 50.13 degrees
  ```

**Validation**: All examples show actual outputs with real column names and values ✅

---

## Phase 3: Clarify Output Structure

- [ ] Add section "Understanding Output Levels" after "Basic Workflow" (before Example 1):
  ```markdown
  ## Understanding Output Levels

  sleap-roots computes traits at two levels:

  - **Frame-level traits** (`compute_plant_traits()`): One row per frame with raw measurements for that frame
  - **Plant-level summaries** (`compute_batch_traits()`): One row per plant with summary statistics across all frames

  Summary statistics include: min, max, mean, median, std, p5, p25, p75, p95
  ```
- [ ] Update Example 1 description (line 31) to mention "frame-level traits"
- [ ] Update Example 2 description (line 62) to mention "plant-level summaries"

**Validation**: Clear explanation of what each method returns ✅

---

## Phase 4: Remove Generic Code Sections

### Remove "Working with CSV Output" (lines 157-175)

- [ ] Delete entire section including:
  - Python (pandas) subsection
  - R subsection
  - Excel subsection
- [ ] Replace with brief note:
  ```markdown
  ## Working with Results

  The CSV files are standard data frames compatible with any analysis tool (pandas, R, Excel, etc.).
  For visualization examples, see the [sleap-roots-vis repository](https://github.com/talmolab/sleap-roots-vis).
  ```

### Remove "Visualizing Results" (lines 177-196)

- [ ] Delete entire matplotlib visualization section
- [ ] Covered by link to visualization repository above

### Simplify "Converting Pixels to Real Units" (lines 198-216)

- [ ] Remove code example (lines 202-209)
- [ ] Keep conceptual explanation only:
  ```markdown
  ## Converting Pixels to Real Units

  SLEAP predictions are in pixels. To convert to real-world units:

  1. Include a ruler or reference object in your images
  2. Measure the known distance in pixels
  3. Calculate your scale factor (pixels per cm/mm)
  4. Divide all length measurements by the scale factor

  Angle measurements are already in degrees and don't need conversion.
  ```

**Validation**: No generic pandas/R/matplotlib code, links to external resources ✅

---

## Phase 5: Consolidate Troubleshooting

- [ ] Replace entire "Common Issues" section (lines 217-248) with:
  ```markdown
  ## Troubleshooting

  For help with common issues including:
  - Import errors and installation problems
  - File not found errors
  - Empty or NaN trait values
  - Pipeline selection and data loading

  See the **[Troubleshooting Guide](../guides/troubleshooting.md)**.
  ```
- [ ] Verify troubleshooting guide has all 4 removed issues covered
- [ ] Optionally keep 1 most critical issue inline if user requests

**Validation**: Troubleshooting consolidated, no duplication ✅

---

## Phase 6: Fix Trait Reference Links

- [ ] Line 155: Change `../guides/trait-reference.md` → `../reference/` (API docs)
- [ ] Line 253: Change `../guides/trait-reference.md` → `../reference/` (API docs)
- [ ] Verify API reference has trait documentation
- [ ] Update any other trait-reference.md links found in these files

**Validation**: All trait documentation links point to API reference ✅

---

## Phase 7: Add uv run Prefix Throughout

### quickstart.md Examples

- [ ] Add note at top of quickstart after "Prerequisites":
  ```markdown
  !!! tip "Running Examples"
      If you installed with uv (recommended), prefix all python commands with `uv run`:
      ```bash
      uv run python my_script.py
      ```
  ```
- [ ] Update Example 1 comment (line 33-34) to mention `uv run python` option
- [ ] Update Example 2 comment to mention `uv run python` option
- [ ] Update Example 3 comment to mention `uv run python` option

**Note:** Don't modify the actual example code (keep it clean), but add context that uv users should prefix with `uv run`

**Validation**: Clear guidance on how uv users run examples ✅

---

## Phase 8: Update Common Traits Section

- [ ] Line 147-154: Make it explicit these are column names:
  ```markdown
  ### Common Trait Columns

  Here are some commonly used trait columns in the output DataFrame:

  - **`primary_length`** – Length of primary root in pixels
  - **`lateral_length_total`** – Sum of all lateral root lengths
  - **`lateral_length_avg`** – Average lateral root length
  - **`lateral_count`** – Number of lateral roots detected
  - **`primary_tip_count`** – Number of primary root tips
  - **`crown_count`** – Number of crown roots (monocots)
  - **`primary_angle`** – Primary root angle from vertical (degrees)
  - **`lateral_angle_avg`** – Average lateral root angle
  - **`crown_angle_avg`** – Average crown root angle
  - **`convex_hull_area`** – Area of convex hull around roots (pixels²)
  - **`network_distribution_ratio`** – Distribution of root network

  For batch processing, each trait has summary statistics appended:
  `{trait_name}_min`, `{trait_name}_max`, `{trait_name}_mean`, `{trait_name}_median`,
  `{trait_name}_std`, `{trait_name}_p5`, `{trait_name}_p25`, `{trait_name}_p75`, `{trait_name}_p95`
  ```

**Validation**: Clear that these are exact column names, explains summary stat naming ✅

---

## Phase 9: Build and Validate

- [ ] Build documentation locally:
  ```bash
  uv run mkdocs build
  ```
- [ ] Check for warnings:
  ```bash
  uv run mkdocs build 2>&1 | grep -E "WARNING|ERROR"
  ```
- [ ] Verify all cross-references work
- [ ] Test that visualization repo link works
- [ ] Confirm API reference has trait docs

**Validation**: Clean build, all links work ✅

---

## Phase 10: Final Review

- [ ] Verify installation starts with uv workflow
- [ ] Verify all python commands show uv run context
- [ ] Verify all examples have actual outputs
- [ ] Verify frame-level vs plant-level distinction is clear
- [ ] Verify no generic code remains
- [ ] Verify troubleshooting is consolidated
- [ ] Verify trait links point to API reference
- [ ] Confirm all success criteria from proposal are met

**Validation**: Documentation is beginner-friendly and accurate ✅

---

## Dependencies

- Phase 1 (installation restructure) can be done independently
- Phase 2-3 (outputs) should be done together
- Phase 4-6 (cleanup) can be done in parallel
- Phase 7 (uv run) depends on Phase 1
- Phase 8 (common traits) can be done independently
- Phase 9 depends on all content phases
- Phase 10 is final review

## Notes

- Focus on beginner users (plant biologists, not Python experts)
- Be explicit about outputs - show what users will actually see
- Remove clutter - link to external resources for generic code
- Consistent uv workflow throughout
- Clear about data structure (frame vs plant level)