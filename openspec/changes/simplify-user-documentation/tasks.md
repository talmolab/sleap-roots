# Tasks: Simplify User Documentation

## Phase 1: Audit and Count

- [ ] Count actual user-facing traits across all pipelines
  - Run script or check trait_pipelines.py for TraitDef counts
  - Determine which are user-facing vs. intermediate
  - Document final count for reference
- [ ] Identify all conda references across documentation
  - Search all .md files in docs/
  - Document locations for systematic removal
- [ ] Find all repeated error messages
  - List all files with common error documentation
  - Identify which should be canonical source

**Validation**: Complete inventory of all changes needed

---

## Phase 2: Update Home Page Performance Section

- [ ] Remove link to dev/benchmarking.md from home page
- [ ] Create user-friendly performance summary on home page
  - Show 0.1-0.5s per plant timing
  - Add context: platform specs (GitHub Actions: 2 cores, 7GB RAM)
  - Add sample count context (how many samples benchmarked)
  - Simple explanation: "Fast enough for X plants per hour"
- [ ] Add subtle link: "Developer Guide → Benchmarking" for infrastructure details
- [ ] Update trait count on home page (2 locations)

**Validation**: Home page is user-focused, no dev infrastructure links

---

## Phase 3: Update Trait Counts Throughout

- [ ] Update docs/index.md trait references
  - Line 11: "Extract 50+ morphological traits"
  - Line 48: "50+ morphological traits"
- [ ] Update docs/tutorials/dicot-pipeline.md
  - Line 9: "Compute 50+ morphological traits"
  - Line 100: "See the Trait Reference for all 50+ traits"
- [ ] Update docs/getting-started/quickstart.md
  - Line 57: "40+ morphological traits"
- [ ] Verify accuracy with actual pipeline code

**Validation**: All trait count claims match reality

---

## Phase 4: De-emphasize Conda

### Installation Guide

- [ ] Remove "Recommended: Conda Environment" section header
- [ ] Rewrite as "Alternative: Using Conda" at bottom
- [ ] Keep instructions minimal and consolidated
- [ ] Remove conda from troubleshooting sections in installation.md

### Home Page

- [ ] Change "Quick pip install or conda setup" → "Quick pip install or uv setup"

### Quick Start

- [ ] Remove "Activate your conda environment:" section (lines 218-220)
- [ ] Assume users followed recommended uv/pip approach

### Troubleshooting Guide

- [ ] Remove conda-specific environment creation (lines 50-51)
- [ ] Provide universal solutions only

### Verify No Conda in User Guides

- [ ] Check home page: no conda mentions ✓
- [ ] Check quick start: no conda mentions ✓
- [ ] Check troubleshooting: no conda mentions ✓
- [ ] Installation guide: conda mentioned once only, as alternative ✓

**Validation**: Conda appears exactly once in installation guide, nowhere in user flow

---

## Phase 5: Consolidate Error Documentation

- [ ] Make troubleshooting.md canonical source for common errors
- [ ] Update installation.md: Replace error solution with link
  - Line 89: "ModuleNotFoundError" → link to troubleshooting
- [ ] Update quickstart.md: Replace error solution with link
  - Line 216: Common Issues → link to troubleshooting
- [ ] Ensure troubleshooting.md has comprehensive solutions
  - Import errors
  - Path issues
  - Version conflicts

**Validation**: No duplicated error solutions, single source of truth

---

## Phase 6: Clarify Python Version Support

- [ ] Update Python version table in installation.md
  - Ubuntu: "3.7-3.11 Fully supported" → "3.11 CI tested"
  - Add row: "3.7-3.12 Should work (not CI tested)"
- [ ] Add explanatory note about CI testing
  ```markdown
  !!! info "Python Version Support"
      We officially test and support Python 3.11 on all platforms through CI.
      Other versions (3.7-3.12) should work based on dependencies, but are not continuously tested.
  ```
- [ ] Consider adding note about why 3.11 is recommended

**Validation**: Python version claims match CI reality

---

## Phase 7: Build and Validate

- [ ] Build documentation locally
  ```bash
  uv run mkdocs build
  ```
- [ ] Check for warnings
  ```bash
  uv run mkdocs build 2>&1 | grep -E "WARNING|ERROR"
  ```
- [ ] Verify all cross-references work
- [ ] Check that performance section is clear and user-friendly
- [ ] Confirm no broken links

**Validation**: Clean build, all improvements implemented

---

## Phase 8: Final Review

- [ ] Review home page for user-friendliness
- [ ] Verify installation guide flow (pip → uv → conda alternative)
- [ ] Check troubleshooting guide is comprehensive
- [ ] Ensure consistent trait counts everywhere
- [ ] Confirm no developer-heavy content on user pages

**Validation**: Documentation is simple, accurate, and user-focused

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