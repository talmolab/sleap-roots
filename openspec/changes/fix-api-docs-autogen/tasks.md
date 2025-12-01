# Tasks: Fix API Documentation Autogeneration Errors

**Status**: 🔵 Proposed

## Phase 1: Fix Type Annotations

**Goal**: Resolve all "No type or annotation" warnings from griffe

- [ ] Fix `sleap_roots/bases.py` type annotations
  - Line 211: Add type for `lateral_base_ys` parameter
  - Line 213: Add type for `primary_tip_pt_y` parameter
- [ ] Fix `sleap_roots/convhull.py` type annotations
  - Line 226: Fix Raises section formatting for ConvexHull ValueError
- [ ] Fix `sleap_roots/points.py` type annotations
  - Line 312: Add type for returned value 1
  - Line 313: Add type for returned value 2
  - Line 478: Add type for `points` parameter
  - Line 481: Add type for returned value 1
- [ ] Fix `sleap_roots/series.py` type annotations
  - Lines 682-694: Add types for all plot() parameters
- [ ] Fix `sleap_roots/trait_pipelines.py` type annotations
  - Line 130: Add type for `obj` parameter
  - Line 133: Add type for returned value
  - Line 411: Add type for returned value
  - Line 660: Add type for returned value
  - Lines 2888, 2923: Fix 'Args' parameter issues

**Validation**: Run `uv run mkdocs build 2>&1 | grep "No type or annotation"` - should return nothing

---

## Phase 2: Fix Docstring Formatting

**Goal**: Resolve docstring formatting warnings

- [ ] Fix `sleap_roots/convhull.py:226` - Reformat Raises section
  - Change "ConvexHull object." to proper exception description
- [ ] Fix `sleap_roots/trait_pipelines.py:2888` - Remove invalid 'Args' parameter
- [ ] Fix `sleap_roots/trait_pipelines.py:2923` - Remove invalid 'Args' parameter
- [ ] Verify all other docstrings follow Google style

**Validation**: Run `uv run mkdocs build 2>&1 | grep "WARNING -  griffe"` - should return nothing

---

## Phase 3: Fix Navigation Structure

**Goal**: Resolve navigation warnings and broken links

- [ ] Fix `reference/` directory warning
  - Option A: Remove `reference/` from nav in mkdocs.yml (line 184)
  - Option B: Create placeholder index for reference/
  - Decision: Choose based on intended documentation structure
- [ ] Fix missing anchors in `docs/api/core/pipelines.md`
  - Add `#lateral-root-pipeline` anchor
  - Add `#multiple-dicot-pipeline` anchor
- [ ] Fix broken notebook link in `docs/api/examples/common-workflows.md`
  - Update link to `../../notebooks/index.md` to valid target
- [ ] Fix broken pipeline link in `docs/guides/custom-pipelines.md`
  - Update `../api/pipelines.md` to `../api/core/pipelines.md`
- [ ] Add missing page to navigation: `docs/dev/benchmarking.md`
  - Add to "Developer Guide" section in mkdocs.yml nav

**Validation**: Run `uv run mkdocs build 2>&1 | grep "WARNING"` - should only show expected warnings (if any)

---

## Phase 4: Testing and Validation

**Goal**: Ensure clean build and verify all fixes

- [ ] Run full mkdocs build and capture output
  ```bash
  uv run mkdocs build > build.log 2>&1
  ```
- [ ] Verify zero warnings related to:
  - Type annotations
  - Docstring formatting
  - Navigation structure
  - Internal links
- [ ] Serve docs locally and manually verify:
  - All API pages load correctly
  - Internal links work
  - Code examples render properly
  - Auto-generated reference pages accessible
- [ ] Run pre-merge checks:
  - Black formatting: `uv run black --check sleap_roots tests`
  - pydocstyle: `uv run pydocstyle --convention=google sleap_roots/`
  - pytest: `uv run pytest tests/`

**Validation**: All checks pass, documentation builds cleanly

---

## Phase 5: Documentation

**Goal**: Document changes and update relevant files

- [ ] Update this tasks.md with completion status
- [ ] Create summary of all changes made
- [ ] Note any decisions made (e.g., how to handle reference/ directory)
- [ ] Archive proposal when complete

**Validation**: Proposal marked as complete and archived

---

## Dependencies

- No dependencies between Phase 1 and Phase 2 (can be done in parallel)
- Phase 3 depends on understanding current navigation structure
- Phase 4 depends on Phases 1-3 completion

## Notes

- Some warnings may be acceptable (document which ones and why)
- Focus on errors that affect user experience first
- Type annotations should match existing code patterns in the repository