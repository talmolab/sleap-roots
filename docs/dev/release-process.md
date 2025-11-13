# Release Process

This guide documents the process for releasing new versions of sleap-roots.

## Overview

sleap-roots follows semantic versioning (SemVer) and uses automated workflows for building, testing, and publishing releases.

## Version Numbering

### Semantic Versioning

Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples:
- `0.1.0` → `0.1.1`: Bug fix
- `0.1.1` → `0.2.0`: New pipeline added
- `0.2.0` → `1.0.0`: Breaking API change

### Pre-release Versions

- `0.1.0rc1`: Release candidate 1
- `0.1.0a1`: Alpha version
- `0.1.0b1`: Beta version

## Release Checklist

### Pre-Release (1-2 weeks before)

- [ ] Review open issues and PRs
- [ ] Merge ready features for the release
- [ ] Update dependencies if needed
- [ ] Run full test suite locally
- [ ] Check CI passes on main branch
- [ ] Update CHANGELOG.md with new features, fixes, and breaking changes
- [ ] Review and update documentation

### Release Day

- [ ] Create release branch
- [ ] Update version number
- [ ] Update CHANGELOG.md
- [ ] Create release commit
- [ ] Push and create PR
- [ ] Review and merge release PR
- [ ] Create Git tag
- [ ] Create GitHub release
- [ ] Verify PyPI publication
- [ ] Deploy documentation
- [ ] Announce release

### Post-Release

- [ ] Monitor for issues
- [ ] Address critical bugs with patch release if needed
- [ ] Plan next release features

## Step-by-Step Release Process

### 1. Prepare Release Branch

```bash
# Ensure main is up to date
git checkout main
git pull origin main

# Create release branch
git checkout -b release/v0.2.0

# Or use GitHub CLI
gh issue create --title "Release v0.2.0" --body "Release tracking issue"
```

### 2. Update Version Number

Update version in `sleap_roots/__init__.py`:

```python
# sleap_roots/__init__.py

__version__ = "0.2.0"
```

Verify version is correct:

```bash
python -c "import sleap_roots; print(sleap_roots.__version__)"
```

### 3. Update CHANGELOG

Edit `CHANGELOG.md` following [Keep a Changelog](https://keepachangelog.com/) format:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-01-15

### Added
- New `MultipleDicotPipeline` for multi-plant analysis
- Root sinuosity trait computation
- Batch processing utilities
- Comprehensive tutorials for all pipelines

### Changed
- Improved performance of length calculations (20% faster)
- Updated trait reference documentation
- Migrated to uv for package management

### Fixed
- Fixed bug in lateral root angle calculation (#123)
- Corrected convex hull computation for edge cases (#145)

### Deprecated
- `old_function` will be removed in v0.3.0

## [0.1.4] - 2024-01-01

### Fixed
- Fixed critical bug in primary root detection

...
```

### 4. Run Pre-Release Tests

```bash
# Run full test suite
pytest tests/ -v

# Check code style
black --check sleap_roots tests
pydocstyle sleap_roots/

# Build documentation locally
mkdocs build

# Test package build
python -m build
```

### 5. Commit and Push Release Branch

```bash
# Commit version bump and changelog
git add sleap_roots/__init__.py CHANGELOG.md
git commit -m "Bump version to v0.2.0"

# Push release branch
git push origin release/v0.2.0
```

### 6. Create Release PR

Create PR from `release/v0.2.0` to `main`:

```bash
# Using GitHub CLI
gh pr create \
  --title "Release v0.2.0" \
  --body "$(cat <<EOF
## Release v0.2.0

### Summary
This release includes...

### Changes
- New MultipleDicotPipeline
- Performance improvements
- Bug fixes

### Checklist
- [x] Version bumped
- [x] CHANGELOG updated
- [x] Tests passing
- [x] Documentation updated

Closes #XX
EOF
)"
```

**PR Description should include**:
- Summary of major changes
- Link to CHANGELOG section
- Breaking changes (if any)
- Migration guide (for breaking changes)
- Checklist of completed tasks

### 7. Review and Merge

**Review checklist**:
- [ ] Version number is correct
- [ ] CHANGELOG is complete and accurate
- [ ] All tests pass in CI
- [ ] Documentation builds successfully
- [ ] No unintended changes

**Merge the PR** after approval.

### 8. Create Git Tag

```bash
# After merging release PR, checkout main
git checkout main
git pull origin main

# Create annotated tag
git tag -a v0.2.0 -m "Release version 0.2.0

Major changes:
- Added MultipleDicotPipeline
- Improved performance
- Bug fixes

See CHANGELOG.md for details."

# Push tag to trigger release workflow
git push origin v0.2.0
```

### 9. Create GitHub Release

GitHub Actions will automatically create a draft release. Edit and publish it:

1. Go to [Releases](https://github.com/talmolab/sleap-roots/releases)
2. Find draft release for `v0.2.0`
3. Edit release notes:

```markdown
## sleap-roots v0.2.0

### Highlights

🌱 **New Pipeline**: `MultipleDicotPipeline` for analyzing multiple plants simultaneously

⚡ **Performance**: 20% faster length calculations

🐛 **Bug Fixes**: Corrected lateral root angle computation

### What's Changed

#### Added
- MultipleDicotPipeline for multi-plant setups (#150)
- Root sinuosity trait (#155)
- Comprehensive pipeline tutorials (#160)

#### Improved
- Vectorized length computation (#148)
- Enhanced trait documentation (#152)

#### Fixed
- Lateral root emergence angle calculation (#123)
- Convex hull edge case handling (#145)

### Installation

```bash
pip install sleap-roots==0.2.0
```

### Documentation

Full documentation: https://talmolab.github.io/sleap-roots/

### Contributors

Thanks to @contributor1, @contributor2 for their contributions!

**Full Changelog**: https://github.com/talmolab/sleap-roots/compare/v0.1.4...v0.2.0
```

4. Check **Set as latest release**
5. Click **Publish release**

### 10. Verify PyPI Publication

GitHub Actions automatically publishes to PyPI when a release is created.

**Verify publication**:

```bash
# Wait a few minutes, then check PyPI
pip install sleap-roots==0.2.0

# Verify installation
python -c "import sleap_roots; print(sleap_roots.__version__)"
# Should print: 0.2.0
```

**If publication fails**:
1. Check [GitHub Actions](https://github.com/talmolab/sleap-roots/actions)
2. Review error logs
3. Fix issues and manually publish if needed:

```bash
# Build package
python -m build

# Upload to PyPI (requires credentials)
python -m twine upload dist/sleap_roots-0.2.0*
```

### 11. Deploy Documentation

Documentation is automatically deployed by GitHub Actions.

**Verify deployment**:

1. Visit https://talmolab.github.io/sleap-roots/
2. Check version selector shows `v0.2.0`
3. Verify new features are documented

**Manual deployment** (if automated deployment fails):

```bash
# Install mike for version management
pip install mike

# Deploy documentation for this version
mike deploy --push --update-aliases 0.2.0 latest

# Set default version
mike set-default --push latest
```

### 12. Announce Release

**GitHub Discussions**:
```markdown
Title: sleap-roots v0.2.0 Released!

We're excited to announce sleap-roots v0.2.0!

## Highlights
- 🌱 New MultipleDicotPipeline for multi-plant analysis
- ⚡ 20% performance improvement
- 🐛 Critical bug fixes

## Installation
```bash
pip install sleap-roots==0.2.0
```

## Documentation
https://talmolab.github.io/sleap-roots/

## Changelog
https://github.com/talmolab/sleap-roots/releases/tag/v0.2.0

Please report any issues or feedback!
```

**SLEAP Slack** (if applicable):
- Post announcement in #plant-phenotyping channel
- Highlight key new features
- Link to release notes

**Twitter/X** (if applicable):
```
🌱 sleap-roots v0.2.0 is out!

✨ New: MultipleDicotPipeline
⚡ Faster trait computation
🐛 Bug fixes

📦 pip install sleap-roots==0.2.0
📖 https://talmolab.github.io/sleap-roots/

#plantphenotyping #computervision #sleap
```

## Hotfix Releases

For critical bugs that need immediate patching:

### 1. Create Hotfix Branch

```bash
# Branch from latest release tag
git checkout -b hotfix/v0.2.1 v0.2.0
```

### 2. Fix the Bug

```bash
# Make minimal changes to fix the bug
git add fixed_file.py
git commit -m "Fix critical bug in lateral root detection"
```

### 3. Update Version and Changelog

```python
# sleap_roots/__init__.py
__version__ = "0.2.1"
```

```markdown
# CHANGELOG.md

## [0.2.1] - 2024-01-20

### Fixed
- Critical bug in lateral root detection causing crashes (#170)
```

### 4. Release Hotfix

```bash
# Commit changes
git add sleap_roots/__init__.py CHANGELOG.md
git commit -m "Bump version to v0.2.1 (hotfix)"

# Merge to main
git checkout main
git merge --no-ff hotfix/v0.2.1

# Tag and push
git tag -a v0.2.1 -m "Hotfix release v0.2.1"
git push origin main
git push origin v0.2.1
```

## Automation

### GitHub Actions Workflows

#### `.github/workflows/release.yml`

```yaml
name: Release

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install build twine
      - name: Build package
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: twine upload dist/*
```

#### `.github/workflows/docs.yml`

```yaml
name: Deploy Docs

on:
  push:
    branches: [main]
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Install dependencies
        run: |
          pip install mkdocs-material mike
      - name: Deploy docs
        run: |
          mike deploy --push --update-aliases ${{ github.ref_name }} latest
```

## Troubleshooting

### PyPI Upload Fails

**Issue**: `403 Forbidden` error

**Solution**: Verify PyPI API token in repository secrets

1. Go to PyPI → Account settings → API tokens
2. Create token with scope for sleap-roots
3. Add token to GitHub secrets as `PYPI_API_TOKEN`

### Version Conflict

**Issue**: Version already exists on PyPI

**Solution**: Increment patch version

```bash
# Instead of 0.2.0, use 0.2.1
# Update version and re-release
```

### Documentation Doesn't Deploy

**Issue**: Mike deployment fails

**Solution**: Check GitHub Pages settings

1. Repository → Settings → Pages
2. Source: Deploy from a branch
3. Branch: `gh-pages` / `root`

## Release Schedule

Suggested release cadence:

- **Major releases** (X.0.0): Yearly or when breaking changes accumulate
- **Minor releases** (0.X.0): Every 1-2 months with new features
- **Patch releases** (0.0.X): As needed for bug fixes

**Example timeline**:
- Jan: v0.2.0 (new pipeline)
- Feb: v0.2.1 (hotfix)
- Mar: v0.3.0 (new traits)
- May: v0.4.0 (performance improvements)
- Dec: v1.0.0 (stable API)

## Next Steps

- Review [Contributing Guide](contributing.md) for development workflow
- See [Testing Guide](testing.md) for test requirements
- Check [Development Setup](setup.md) for environment configuration
- Read [Architecture](architecture.md) for understanding codebase structure