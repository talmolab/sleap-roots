# Tasks: Simplify User Documentation

## Phase 1: Audit and Count

- [x] Count actual user-facing traits across all pipelines
  - Run script or check trait_pipelines.py for TraitDef counts
  - Determine which are user-facing vs. intermediate
  - Document final count for reference: **174 TraitDef objects, using 100+ as user-facing claim**
- [x] Identify all conda references across documentation
  - Search all .md files in docs/
  - Document locations for systematic removal
- [x] Find all repeated error messages
  - List all files with common error documentation
  - Identify which should be canonical source

**Validation**: Complete inventory of all changes needed ✅

---

## Phase 2: Update Home Page Performance Section

- [x] Remove link to dev/benchmarking.md from home page
- [x] Create user-friendly performance summary on home page
  - Show 0.1-0.5s per plant timing
  - Add context: platform specs (GitHub Actions: 2 cores, 7GB RAM)
  - Add sample count context (how many samples benchmarked)
  - Simple explanation: "Fast enough for X plants per hour"
- [x] Add subtle link: "Developer Guide → Benchmarking" for infrastructure details
- [x] Update trait count on home page (2 locations)

**Validation**: Home page is user-focused, no dev infrastructure links ✅

---

## Phase 3: Update Trait Counts Throughout

- [x] Update docs/index.md trait references
  - Line 11: "Extract 50+ morphological traits" → 100+
  - Line 48: "50+ morphological traits" → 100+
- [x] Update docs/tutorials/dicot-pipeline.md
  - Line 9: "Compute 50+ morphological traits" → 100+
  - Line 100: "See the Trait Reference for all 50+ traits" → 100+
- [x] Update docs/getting-started/quickstart.md
  - Line 57: "40+ morphological traits" → 100+
- [x] Verify accuracy with actual pipeline code

**Validation**: All trait count claims match reality ✅

---

## Phase 4: De-emphasize Conda

### Installation Guide

- [x] Remove "Recommended: Conda Environment" section header
- [x] Rewrite as "Alternative: Using Conda" at bottom
- [x] Keep instructions minimal and consolidated
- [x] Remove conda from troubleshooting sections in installation.md

### Home Page

- [x] Change "Quick pip install or conda setup" → "Quick pip install or uv setup"

### Quick Start

- [x] Remove "Activate your conda environment:" section (lines 218-220)
- [x] Assume users followed recommended uv/pip approach

### Troubleshooting Guide

- [x] Remove conda-specific environment creation (lines 50-51)
- [x] Provide universal solutions only

### Verify No Conda in User Guides

- [x] Check home page: no conda mentions ✓
- [x] Check quick start: no conda mentions ✓
- [x] Check troubleshooting: no conda mentions ✓
- [x] Installation guide: conda mentioned once only, as alternative ✓

**Validation**: Conda appears exactly once in installation guide, nowhere in user flow ✅

---

## Phase 5: Consolidate Error Documentation

- [x] Make troubleshooting.md canonical source for common errors
- [x] Update installation.md: Replace error solution with link
  - Line 89: "ModuleNotFoundError" → link to troubleshooting
- [x] Update quickstart.md: Replace error solution with link
  - Line 216: Common Issues → link to troubleshooting
- [x] Ensure troubleshooting.md has comprehensive solutions
  - Import errors
  - Path issues
  - Version conflicts

**Validation**: No duplicated error solutions, single source of truth ✅

---

## Phase 6: Clarify Python Version Support

- [x] Update Python version table in installation.md
  - Ubuntu: "3.7-3.11 Fully supported" → "3.11 CI tested"
  - Add row: "3.7-3.12 Should work (not CI tested)"
- [x] Add explanatory note about CI testing
  ```markdown
  !!! info "Python Version Support"
      We officially test and support Python 3.11 on all platforms through CI.
      Other versions (3.7-3.12) should work based on dependencies, but are not continuously tested.
  ```
- [x] Consider adding note about why 3.11 is recommended

**Validation**: Python version claims match CI reality ✅

---

## Phase 7: Build and Validate

- [x] Build documentation locally
  ```bash
  uv run mkdocs build
  ```
- [x] Check for warnings
  ```bash
  uv run mkdocs build 2>&1 | grep -E "WARNING|ERROR"
  ```
- [x] Verify all cross-references work
- [x] Check that performance section is clear and user-friendly
- [x] Confirm no broken links

**Validation**: Clean build, all improvements implemented ✅

---

## Phase 8: Final Review

- [x] Review home page for user-friendliness
- [x] Verify installation guide flow (pip → uv → conda alternative)
- [x] Check troubleshooting guide is comprehensive
- [x] Ensure consistent trait counts everywhere (100+)
- [x] Confirm no developer-heavy content on user pages

**Validation**: Documentation is simple, accurate, and user-focused ✅

---

## Dependencies

- Phase 2-4 can be done in parallel
- Phase 5 depends on Phase 4 (error consolidation after conda removal)
- Phase 6 independent
- Phase 7 depends on all content phases (2-6)
- Phase 8 is final review

## Notes

- Prioritize simplicity - remove more than we add
- Keep developer content in Developer Guide only
- Single source of truth for everything
- Be accurate with claims (trait counts, Python versions)