# Tasks: Enable Versioned Documentation

## ✅ COMPLETED

**Status**: All tasks completed successfully
**Completion Date**: November 23, 2025
**PR**: #134

---

## Phase 1: Configure mkdocs.yml

### 1.1 Add mike plugin
- [x] Add `mike` plugin to mkdocs.yml plugins section with:
  - `alias_type: symlink`
  - `canonical_version: latest`
  - `version_selector: true`

### 1.2 Verify local build
- [x] Run `uv run mkdocs build` to ensure no errors
- [x] Run `uv run mike serve` to test version selector locally (optional - requires gh-pages)

## Phase 2: Fix GitHub Pages Configuration

### 2.1 Update repository settings (Manual)
- [x] Go to GitHub repo Settings > Pages
- [x] Change Source from "Deploy from a branch: main" to "Deploy from a branch: gh-pages"
- [x] Set branch to `gh-pages` and folder to `/ (root)`
- [x] Save changes

### 2.2 Trigger deployment
- [x] Push changes to main branch to trigger docs workflow
- [x] Wait for GitHub Actions to complete
- [x] Verify deployment succeeds

## Phase 3: Verification

### 3.1 Test docs site
- [x] Navigate to https://talmolab.github.io/sleap-roots/
- [x] Verify page loads (no 404)
- [x] Verify version selector dropdown appears in header
- [x] Click version selector and verify `latest` is shown

### 3.2 Test version navigation
- [x] Verify clicking version switches URL correctly
- [x] Verify content loads for each version
- [x] Verify default redirect works (bare URL redirects to latest)

## Phase 4: Update OpenSpec status

### 4.1 Update benchmark-regression-detection
- [x] Mark Phase 0 (Versioned Documentation Setup) tasks as complete in `benchmark-regression-detection/tasks.md`
- [x] Update proposal.md to reflect current state