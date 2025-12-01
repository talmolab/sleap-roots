# Tasks: Update Getting Started Documentation

## Phase 1: Research Current State

- [ ] Read all getting-started documentation files
  - `docs/getting-started/installation.md`
  - `docs/getting-started/quickstart.md`
  - `docs/getting-started/what-is-sleap.md`
- [ ] Read developer guide files
  - `docs/dev/setup.md` (if exists)
  - `docs/dev/contributing.md`
- [ ] List all cross-references to installation.md
  ```bash
  grep -r "installation.md" docs/
  ```
- [ ] Check mkdocs.yml navigation structure

**Validation**: Complete understanding of current documentation structure

---

## Phase 2: Rewrite Getting Started / Installation

- [ ] Rewrite `docs/getting-started/installation.md` for end users only
  - Quick install with pip
  - Conda environment setup
  - Modern workflow: `uv init` + `uv add sleap-roots`
  - Verification steps (import test)
  - Remove all development content
- [ ] Keep page under 100 lines
- [ ] Add clear "Next Steps" linking to quickstart
- [ ] Add "For Contributors" box linking to dev/setup.md

**Validation**: Page focuses entirely on user installation, no Git LFS/testing mentioned

---

## Phase 3: Create/Update Developer Setup Guide

- [ ] Check if `docs/dev/setup.md` exists
- [ ] If exists: Enhance with moved content from installation.md
- [ ] If not: Create comprehensive `docs/dev/setup.md` with:
  - Prerequisites (Git, Git LFS, Python)
  - Clone repository and Git LFS setup
  - Using uv for development (`uv sync`)
  - Alternative conda workflow (`environment.yml`)
  - Running tests (`pytest`)
  - Code formatting (`black`, `pydocstyle`)
  - Pre-commit hooks
  - Contributing workflow overview
- [ ] Link to `dev/contributing.md` for full contribution guidelines

**Validation**: All development setup content consolidated in dev guide

---

## Phase 4: Update Navigation and Cross-References

- [ ] Update `mkdocs.yml` navigation if needed
  - Ensure "Getting Started" → "Installation" is clear
  - Ensure "Developer Guide" → "Development Setup" exists
- [ ] Update cross-references in other docs:
  - Search for links to installation.md
  - Verify context (user vs. developer)
  - Update to dev/setup.md where appropriate
- [ ] Update any "See installation" references in:
  - README.md (if it exists there)
  - Contributing guide
  - Other getting-started pages

**Validation**: All links work, no broken references

---

## Phase 5: Build and Validate

- [ ] Build documentation locally
  ```bash
  uv run mkdocs build
  ```
- [ ] Check for warnings
  ```bash
  uv run mkdocs build 2>&1 | grep -E "WARNING|ERROR"
  ```
- [ ] Serve documentation and manually review
  ```bash
  uv run mkdocs serve
  ```
- [ ] Walk through both paths:
  - User path: Home → Getting Started → Installation → Quickstart
  - Developer path: Developer Guide → Development Setup
- [ ] Verify all links work

**Validation**: Clean build, all links work, content is clear and well-separated

---

## Phase 6: Update Related Documentation

- [ ] Check if README.md installation section needs updates
- [ ] Update any references in cookbook or examples
- [ ] Update any references in API documentation
- [ ] Ensure consistency across all mentions of installation

**Validation**: Consistent installation guidance across all documentation

---

## Dependencies

- Phase 2-3 can be done in parallel
- Phase 4 depends on Phase 2-3 completion
- Phase 5-6 depend on Phase 4 completion

## Notes

- Keep the user path simple and fast (~5 minutes to install and verify)
- Keep the developer path comprehensive (~15 minutes to set up dev environment)
- Maintain consistency with existing documentation tone and style