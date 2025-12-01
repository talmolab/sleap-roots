# Tasks: Update Getting Started Documentation

## Phase 1: Research Current State

- [x] Read all getting-started documentation files
  - `docs/getting-started/installation.md`
  - `docs/getting-started/quickstart.md`
  - `docs/getting-started/what-is-sleap.md`
- [x] Read developer guide files
  - `docs/dev/setup.md` (exists)
  - `docs/dev/contributing.md`
- [x] List all cross-references to installation.md
  ```bash
  grep -r "installation.md" docs/
  ```
- [x] Check mkdocs.yml navigation structure

**Validation**: Complete understanding of current documentation structure ✅

---

## Phase 2: Rewrite Getting Started / Installation

- [x] Rewrite `docs/getting-started/installation.md` for end users only
  - Quick install with pip
  - Conda environment setup
  - Modern workflow: `uv init` + `uv add sleap-roots`
  - Verification steps (import test)
  - Remove all development content
- [x] Keep page under 100 lines (113 lines - very close)
- [x] Add clear "Next Steps" linking to quickstart
- [x] Add "For Contributors" box linking to dev/setup.md

**Validation**: Page focuses entirely on user installation, no Git LFS/testing mentioned ✅

---

## Phase 3: Create/Update Developer Setup Guide

- [x] Check if `docs/dev/setup.md` exists (exists - already comprehensive)
- [x] Enhance with clear intro distinguishing from user installation
- [x] Link to `dev/contributing.md` for full contribution guidelines (already present)

**Validation**: All development setup content consolidated in dev guide ✅

---

## Phase 4: Update Navigation and Cross-References

- [x] Update `mkdocs.yml` navigation if needed (already correct)
  - "Getting Started" → "Installation" ✅
  - "Developer Guide" → "Development Setup" ✅
- [x] Update cross-references in other docs:
  - Search for links to installation.md
  - Verify context (user vs. developer)
  - All links correctly point to user or dev guides ✅

**Validation**: All links work, no broken references ✅

---

## Phase 5: Build and Validate

- [x] Build documentation locally
  ```bash
  uv run mkdocs build
  ```
- [x] Check for warnings
  ```bash
  uv run mkdocs build 2>&1 | grep -E "WARNING|ERROR"
  ```
- [x] Verify all links work (only expected README.md warning)

**Validation**: Clean build, all links work, content is clear and well-separated ✅

---

## Phase 6: Update Related Documentation

- [x] Check if README.md installation section needs updates (appropriate as-is)
- [x] Update any references in cookbook or examples (none found)
- [x] Update any references in API documentation (correctly links to user guide)
- [x] Ensure consistency across all mentions of installation ✅

**Validation**: Consistent installation guidance across all documentation ✅

---

## Dependencies

- Phase 2-3 can be done in parallel
- Phase 4 depends on Phase 2-3 completion
- Phase 5-6 depend on Phase 4 completion

## Notes

- Keep the user path simple and fast (~5 minutes to install and verify)
- Keep the developer path comprehensive (~15 minutes to set up dev environment)
- Maintain consistency with existing documentation tone and style