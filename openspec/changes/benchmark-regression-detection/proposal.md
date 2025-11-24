# Proposal: Benchmark Regression Detection and Historical Tracking

## Why

Currently, benchmarks only run on `main` branch pushes and store results as 30-day artifacts. This approach has critical limitations:

1. **No regression prevention**: Performance regressions can be merged into `main` before being detected
2. **No historical tracking**: Results are lost after 30 days, making it impossible to track performance trends over time
3. **No PR feedback**: Contributors don't know if their changes impact performance until after merge
4. **No accountability**: Without visible performance data on the docs site, users can't verify published claims

The published paper (Berrigan et al., 2024) reports specific performance metrics (~0.1-0.5s per plant), but these aren't systematically validated against each change. We need to catch performance regressions during PR review, not after merge.

## What Changes

This proposal adds three complementary capabilities:

### 0. Prerequisites: Versioned Documentation (Partial - Complete Setup)

**Current state:** `mike` is installed but not fully configured:
- ✅ `mike` in dev dependencies (pyproject.toml lines 55, 86)
- ✅ `extra.version.provider: mike` in mkdocs.yml (line 192)
- ❌ `mike` plugin not enabled in plugins list
- ❌ No GitHub Actions workflow using `mike deploy`
- ❌ No version selector configured

**Required changes** (following [lablink](https://github.com/talmolab/lablink/blob/main/mkdocs.yml) pattern):
1. Add `mike` plugin to mkdocs.yml plugins section:
   ```yaml
   - mike:
       alias_type: symlink
       canonical_version: latest
       version_selector: true
   ```
2. Create/update `.github/workflows/docs.yml` to use `mike deploy` instead of `mkdocs gh-deploy`
3. Configure version aliases (`latest`, `stable`, `dev`) for deployments
4. Test version selector appears in docs UI

**Rationale:** Benchmark historical data must be tied to specific versions:
- Each release should show benchmarks from that version
- Users can compare performance across versions (e.g., v0.1.0 vs v0.2.0)
- Benchmark trends make sense only within version context
- Links from benchmark history to docs must resolve to correct version

**Blockers:** Historical tracking (capability #2) depends on this being implemented first. PR regression detection (capability #1) can proceed independently.

### 1. PR-based Regression Detection
- Run benchmarks on all PRs targeting `main`
- Compare PR benchmark results against latest `main` baseline
- Fail CI if performance regresses beyond configurable threshold (15% slower)
- Post benchmark comparison summary as PR comment via `/review-pr` command
- Store both baseline and PR results as artifacts for investigation

### 2. Historical Performance Tracking

**Depends on:** Versioned docs infrastructure (#0)

- Publish benchmark results to versioned GitHub Pages documentation site
- Create `/benchmarks/` page showing performance trends over time (per version)
- Auto-generate charts comparing current vs. historical performance within version
- Store benchmark history in Git (committed to repo after main branch builds)
- Make performance data transparent and searchable
- Version-aware benchmark pages showing trends for each release

### 3. Enhanced Review Integration
- Extend `.claude/commands/review-pr.md` slash command to:
  - Fetch benchmark artifacts from PR checks
  - Parse and format benchmark comparison data
  - Include performance summary in review comments
  - Flag significant regressions for reviewer attention

## Impact

**Affected specs:**
- `testing` (modified) - Add regression detection requirements
- `ci-cd` (new capability) - PR benchmark workflow and artifact handling
- `documentation` (modified) - Add benchmark results page to docs site, versioned docs infrastructure

**Affected code:**
- `pyproject.toml` - Add `mike` to dev dependencies for versioned docs
- `.github/workflows/docs.yml` - Update to use `mike deploy` for versioned publishing
- `.github/workflows/ci.yml` - Add PR benchmark job with baseline comparison
- `.github/workflows/publish-benchmarks.yml` - New workflow to publish results to GitHub Pages
- `tests/benchmarks/conftest.py` - Add baseline loading and comparison utilities
- `.claude/commands/review-pr.md` - Add benchmark summary to review workflow
- `docs/benchmarks/index.md` - New page for historical benchmark results (per version)
- `mkdocs.yml` - Add benchmarks section to navigation, configure version selector
- `scripts/generate-benchmark-history.py` - Script to generate performance charts (version-aware)

**Breaking changes:** None. This only adds new CI jobs, docs pages, and enhances the review command.

**Performance threshold:** 15% regression tolerance (configurable via CI env var)

**Storage strategy:**
- Benchmark JSON stored in `.benchmarks/` directory at repo root (gitignored locally, committed in CI)
- GitHub Pages deploys from `gh-pages` branch using `mike`
- Version-specific benchmark histories (e.g., `v0.1.0/benchmarks/`, `latest/benchmarks/`)
- 90-day history retained per version (older results archived)

**Implementation sequence:**
1. **Phase 0**: Set up versioned docs with `mike` (prerequisite)
2. **Phase 1-2**: Implement baseline infrastructure and regression detection (works without versioning)
3. **Phase 3**: Integrate with review workflow (works without versioning)
4. **Phase 4**: Implement historical tracking (requires versioned docs from Phase 0)