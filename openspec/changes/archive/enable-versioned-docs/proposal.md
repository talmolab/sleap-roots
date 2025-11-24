# Proposal: Enable Versioned Documentation with Mike

## Why

The versioned documentation infrastructure is partially configured but not working:

**Current state:**
- `mike` is already in dev dependencies (pyproject.toml)
- `extra.version.provider: mike` is set in mkdocs.yml (line 192)
- GitHub Actions workflow (`.github/workflows/docs.yml`) already uses `mike deploy`
- gh-pages branch exists with mike-deployed content
- `mike list` shows `latest [dev]` version exists

**The problem:** GitHub Pages is configured to deploy from `main` branch instead of `gh-pages`:
```json
"source": {"branch": "main", "path": "/"}
```

This means the mike-deployed versions on gh-pages are not being served, resulting in 404 errors.

**Additional issue:** The `mike` plugin is not in mkdocs.yml plugins list, which means:
- No version selector dropdown appears in the docs UI
- Users can't switch between versions

## What Changes

### 1. Add mike plugin to mkdocs.yml

Add the mike plugin to the plugins section to enable the version selector:
```yaml
plugins:
  - mike:
      alias_type: symlink
      canonical_version: latest
      version_selector: true
```

### 2. Fix GitHub Pages source branch

Change GitHub Pages to deploy from `gh-pages` branch instead of `main`. This is a manual configuration change in the repository settings (Settings > Pages > Source).

### 3. Verify version selector works

After deployment, verify:
- Version selector dropdown appears in docs UI
- Clicking versions navigates correctly
- Default version (`latest`) loads properly

## Impact

**Affected code:**
- `mkdocs.yml` - Add mike plugin configuration

**Manual steps required:**
- Update GitHub Pages source branch setting (cannot be done via code)

**Breaking changes:** None. This enables new functionality.

**Dependencies:** None - mike is already installed.

**Prerequisite for:**
- `benchmark-regression-detection` Phase 4 (Historical tracking with versioned benchmark pages)