---
description: Guide through the complete release process for the sleap-roots package
---

# Release Process for sleap-roots

Comprehensive workflow for releasing a new version of the `sleap-roots` package to PyPI.

> **Planned migration (#248):** the target release flow is `uv build` + `uv publish` with
> PyPI **trusted publishing** (OIDC, no token). Until #248 lands, CI publishes via
> `twine` + `PYPI_TOKEN` and the version is bumped by editing `sleap_roots/__init__.py`.
> This command documents the **current** flow; update it alongside #248.

## Purpose

This command guides you through the complete release process, ensuring:

1. All pre-release checks pass (tests, coverage, linting, CI)
2. Version is bumped correctly following semantic versioning
3. Changes are documented in the changelog and committed properly
4. A GitHub release is created with appropriate notes
5. The package is published to PyPI automatically by `.github/workflows/build.yml`
6. The release is verified

## Tooling

This project uses **uv** as its package manager and **setuptools** as its build backend:

- **`uv run <tool>`** — run dev-group tools (pytest, black, pydocstyle)
- **`uv build`** — build wheel + sdist (invokes the setuptools backend)
- **`uvx twine`** — one-off metadata check / upload (no install)
- Version is **setuptools-dynamic**: `version = {attr = "sleap_roots.__version__"}` in
  `pyproject.toml`, sourced from `sleap_roots/__init__.py`.

## Prerequisites

- You are on `main` with the latest changes; working tree clean
- All PRs intended for this release are merged; CI green on `main`
- You have maintainer permissions and `gh` is authenticated
- `uv` is installed

## Usage

```bash
/prepare-release            # interactive
/prepare-release patch      # bug fixes      (0.1.0 -> 0.1.1)
/prepare-release minor      # new features   (0.1.0 -> 0.2.0)
/prepare-release major      # breaking       (0.1.0 -> 1.0.0)
```

**Arguments:** `$ARGUMENTS`

## Release Workflow

### Step 1: Pre-Release Validation

```bash
git branch --show-current        # should be 'main'
git status                        # clean
git pull origin main
gh run list --branch main --limit 5
```

Run the CI gate locally (mirrors `.github/workflows/ci.yml`):

```bash
uv sync --frozen
uv run black --check sleap_roots tests
uv run pydocstyle --convention=google sleap_roots/
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/
```

Build and validate the package:

```bash
uv build
uvx twine check dist/*
```

**Stop if any check fails.** Fix issues before proceeding.

### Step 2: Determine Version Number

Follow semantic versioning (https://semver.org), `MAJOR.MINOR.PATCH`:

- **PATCH** (0.1.0 → 0.1.1): bug fixes, doc updates
- **MINOR** (0.1.0 → 0.2.0): new pipelines/traits, backward-compatible
- **MAJOR** (0.1.0 → 1.0.0): breaking changes (e.g. changed trait calculations or CSV output)

> **Published-results caution:** changes to trait calculations affect reproducibility of
> published papers. Treat any algorithm change as at least MINOR and document it thoroughly
> with validation data.

Review changes since the last release:

```bash
LAST_TAG=$(git tag -l | sort -V | tail -1)
echo "Last release: $LAST_TAG"
git log "$LAST_TAG"..HEAD --oneline --no-merges
```

### Step 3: Metadata Completeness Check

Read `pyproject.toml` and verify:

- `[project]`: name, description, readme, authors, `requires-python`
- `license` matches the LICENSE file (**BSD-3-Clause**)
- `classifiers` include the supported Python versions
- `[project.urls]`: Homepage, Repository
- `[project.scripts]`: `sleap-roots = "sleap_roots.cli:main"`
- `[build-system]`: setuptools

Fix any missing metadata.

### Step 4: Documentation Audit

1. **Changelog** (`docs/changelog.md`): has content in `[Unreleased]`; no duplicate headers;
   no placeholder dates. (Use `/update-changelog` to maintain it.)
2. **README.md**: install instructions and any version badge are correct.
3. **Version source**: `sleap_roots/__init__.py` `__version__` is the single source of truth.

### Step 5: Update the Changelog

Move `[Unreleased]` content into a new dated version section in `docs/changelog.md` and
update the comparison links at the bottom. See `/update-changelog`.

### Step 6: Bump the Version

Edit `sleap_roots/__init__.py`:

```python
__version__ = "X.Y.Z"
```

(No separate `pyproject.toml` edit needed — it reads the attr dynamically.)

### Step 7: Build and Test Release Artifacts

```bash
rm -rf dist/ build/
uv build
ls -lh dist/
# sleap_roots-X.Y.Z-py3-none-any.whl  and  sleap_roots-X.Y.Z.tar.gz

# Smoke-test the wheel in an isolated env
uv run --isolated --with dist/*.whl python -c "import sleap_roots as sr; print('OK', sr.__version__)"
```

### Step 8: Commit the Version Bump + Changelog

```bash
git add sleap_roots/__init__.py docs/changelog.md
git commit -m "chore: bump version to vX.Y.Z

- Update __version__ in sleap_roots/__init__.py
- Update docs/changelog.md with release notes"
git push origin main
```

Open and merge a PR if `main` is protected (run `/pre-merge-check` first).

### Step 9: Create the GitHub Release

**Guardrails before releasing:**

1. Confirm `docs/changelog.md` has the new `## [X.Y.Z]` section (today's date) with real content
   and an emptied `[Unreleased]` section.
2. Confirm `sleap_roots.__version__` matches the tag you are about to create.

```bash
git checkout main && git pull origin main

gh release create vX.Y.Z \
  --title "sleap-roots vX.Y.Z" \
  --notes "$(cat <<'NOTES'
## Installation

```bash
pip install sleap-roots==X.Y.Z
# or
uv add sleap-roots==X.Y.Z
```

## What's Changed

<paste the docs/changelog.md section for this version, excluding the header line>

**Full Changelog**: https://github.com/talmolab/sleap-roots/commits/vX.Y.Z

🤖 Generated with [Claude Code](https://claude.com/claude-code)
NOTES
)"
```

Publishing `vX.Y.Z` triggers `.github/workflows/build.yml`, which:

1. Builds wheel + sdist with `uv build`
2. Runs `uvx twine check` and wheel/sdist install smoke tests
3. Publishes to PyPI via `uvx twine upload` using `PYPI_TOKEN`
   *(→ migrating to `uv publish` + trusted publishing in #248)*

### Step 10: Verify the Release

```bash
gh run watch                                  # watch the Build workflow
curl -s https://pypi.org/pypi/sleap-roots/json \
  | python -c "import sys,json; print(sorted(json.load(sys.stdin)['releases'])[-5:])"
uvx --from "sleap-roots==X.Y.Z" python -c "import sleap_roots as sr; print(sr.__version__)"
```

## Rollback

### Before PyPI upload

```bash
gh release delete vX.Y.Z --yes
git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z
git revert <version-bump-commit> && git push origin main
```

### After PyPI upload

You cannot delete a PyPI release, only **yank** it via the PyPI web UI
(https://pypi.org/manage/project/sleap-roots/releases/), or ship a patch release.

## Integration with Other Commands

- `/run-ci-locally` - run the exact CI checks locally
- `/lint` - quick formatting + docstring check
- `/coverage` - detailed coverage analysis
- `/update-changelog` - maintain `docs/changelog.md`
- `/pre-merge-check` - comprehensive pre-merge validation
- `/cleanup-merged` - clean up after the release PR merges
