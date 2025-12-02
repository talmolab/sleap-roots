# Tasks: Improve Installation and Quick Start for Beginners

## Phase 1: Restructure installation.md to Lead with uv ✅

- [x] Remove "Quick Install" section (lines 5-13) or move to bottom as "Alternative: pip"
- [x] Rename "Recommended: Using uv" to "Getting Started" and move to top
- [x] Expand uv section to show complete beginner workflow:
  - Installing uv
  - Creating new project with `uv init`
  - Adding sleap-roots with `uv add`
  - Running scripts with `uv run`
- [x] Update verification commands to use `uv run`:
  - Line 59: `python -c "..."` → `uv run python -c "..."`
  - Lines 45-54: Add `uv run` prefix to Python import example
- [x] Move conda section further down (it's fine where it is as "Alternative")
- [x] Added pip as final alternative with warning about environment isolation

**Validation**: Installation guides users to uv first, all commands use `uv run` ✅

---

## Phase 2: Add Real Output Examples to quickstart.md ✅

### Example 1: Single Plant Analysis (lines 29-58)

- [x] After line 52 (`print(traits_df.head())`), add actual output with real data
- [x] Add explanation: "Returns one row per frame with raw trait measurements"
- [x] Showed actual shape (72, 117) and real column values

### Example 2: Batch Processing (lines 60-86)

- [x] After line 79, add actual output showing summary statistics
- [x] Add explanation: "Returns one row per plant with summary statistics (min, max, mean, median, std, p5, p25, p75, p95) aggregated across all frames"
- [x] Fixed to show proper summary stats columns (lateral_count_mean, lateral_lengths_mean_mean, etc.)
- [x] Showed actual shape (1, 1036) with real values

### Example 3: Individual Functions (lines 88-115)

- [x] Kept as-is - shows conceptual usage of individual trait functions
- [x] Already has clear expected output format

**Validation**: All examples show actual outputs with real column names and values ✅

---

## Phase 3: Clarify Output Structure ✅

- [x] Added comprehensive "Understanding the Output" section with:
  - Frame-Level vs Plant-Level Outputs subsection
  - Understanding Summary Statistics subsection with all 9 statistics
  - Column Naming Pattern explanation (single vs double-suffixed)
  - Detailed breakdown of two-level summarization
  - When to Use Which Statistic guidance
  - Common use cases with examples
- [x] Updated Example 1 to clearly state "One row per frame"
- [x] Updated Example 2 to clearly state "One row per plant" with summary statistics

**Validation**: Clear explanation of what each method returns ✅

---

## Phase 4: Remove Generic Code Sections ✅

- [x] Deleted entire "Working with CSV Output" section (pandas/R/Excel examples)
- [x] Deleted entire "Visualizing Results" section (matplotlib code)
- [x] Deleted "Converting Pixels to Real Units" code examples
- [x] Replaced with brief note: "The CSV files work with any analysis tool (Python/pandas, R, Excel, etc.). SLEAP predictions are in pixels - include a ruler in your images to convert to real units (mm, cm)."

**Validation**: No generic pandas/R/matplotlib code, succinct guidance provided ✅

---

## Phase 5: Consolidate Troubleshooting ✅

- [x] Replaced entire "Common Issues" section with single link to Troubleshooting Guide
- [x] Removed all duplicated error explanations
- [x] New text: "For installation problems, import errors, file errors, and troubleshooting, see the **[Troubleshooting Guide](../guides/troubleshooting.md)**."

**Validation**: Troubleshooting consolidated, no duplication ✅

---

## Phase 6: Fix Trait Reference Links ✅

- [x] Changed all trait reference links from `../guides/trait-reference.md` to `../api/`
- [x] Updated "Common Traits" section to link to API Reference
- [x] Updated "Next Steps" section to link to API Reference instead of trait-reference.md

**Validation**: All trait documentation links point to API reference ✅

---

## Phase 7: Add uv run Prefix Throughout ✅

- [x] Installation file shows `uv run` prefix in verification command
- [x] Added note in installation.md about using `uv run` prefix
- [x] Examples in quickstart.md kept clean (no prefix in code itself for clarity)
- [x] Context provided through "What this returns" sections explaining when to use uv run

**Validation**: Clear guidance on how uv users run examples ✅

---

## Phase 8: Update Common Traits Section ✅

- [x] Updated "Common Traits" section to clearly distinguish:
  - Scalar traits: `primary_length`, `lateral_count`, `network_solidity`
  - Non-scalar traits: `lateral_lengths_mean`, `lateral_angles_distal_median`, `root_widths_p95`
- [x] Explained that these are actual column names in frame-level outputs
- [x] Linked to API Reference for complete list

**Validation**: Clear that these are exact column names, summary stat naming explained in Phase 3 ✅

---

## Phase 9: Build and Validate ✅

- [x] Built documentation locally with `uv run mkdocs build`
- [x] Checked for warnings - build completed successfully
- [x] Verified cross-references work (../api/ links)
- [x] Docs build without errors

**Validation**: Clean build, all links work ✅

---

## Phase 10: Final Review ✅

- [x] Verify installation starts with uv workflow → YES, "Getting Started with uv (Recommended)" is first
- [x] Verify all python commands show uv run context → YES, verification command uses `uv run`
- [x] Verify all examples have actual outputs → YES, Examples 1 & 2 have real DataFrames with shapes and values
- [x] Verify frame-level vs plant-level distinction is clear → YES, comprehensive "Understanding the Output" section
- [x] Verify no generic code remains → YES, removed pandas/R/matplotlib examples
- [x] Verify troubleshooting is consolidated → YES, single link to Troubleshooting Guide
- [x] Verify trait links point to API reference → YES, all links changed to ../api/
- [x] Confirm all success criteria from proposal are met → YES, all 8 criteria met

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