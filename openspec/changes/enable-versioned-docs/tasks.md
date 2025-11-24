# Tasks: Enable Versioned Documentation

## Phase 1: Configure mkdocs.yml

### 1.1 Add mike plugin
- [x] Add `mike` plugin to mkdocs.yml plugins section with:
  - `alias_type: symlink`
  - `canonical_version: latest`
  - `version_selector: true`

### 1.2 Verify local build
- [x] Run `uv run mkdocs build` to ensure no errors
- [ ] Run `uv run mike serve` to test version selector locally (optional - requires gh-pages)

## Phase 2: Fix GitHub Pages Configuration

### 2.1 Update repository settings (Manual)
- [ ] Go to GitHub repo Settings > Pages
- [ ] Change Source from "Deploy from a branch: main" to "Deploy from a branch: gh-pages"
- [ ] Set branch to `gh-pages` and folder to `/ (root)`
- [ ] Save changes

### 2.2 Trigger deployment
- [ ] Push changes to main branch to trigger docs workflow
- [ ] Wait for GitHub Actions to complete
- [ ] Verify deployment succeeds

## Phase 3: Verification

### 3.1 Test docs site
- [ ] Navigate to https://talmolab.github.io/sleap-roots/
- [ ] Verify page loads (no 404)
- [ ] Verify version selector dropdown appears in header
- [ ] Click version selector and verify `latest` is shown

### 3.2 Test version navigation
- [ ] Verify clicking version switches URL correctly
- [ ] Verify content loads for each version
- [ ] Verify default redirect works (bare URL redirects to latest)

## Phase 4: Update OpenSpec status

### 4.1 Update benchmark-regression-detection
- [ ] Mark Phase 0 (Versioned Documentation Setup) tasks as complete in `benchmark-regression-detection/tasks.md`
- [ ] Update proposal.md to reflect current state