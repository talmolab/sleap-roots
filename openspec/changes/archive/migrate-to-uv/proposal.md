# Proposal: Migrate to uv for Package and Dependency Management

## Overview

Migrate sleap-roots from conda/pip-based dependency management to [uv](https://docs.astral.sh/uv/), a modern Python package and project manager. This aligns with best practices demonstrated in the Ariadne project and provides significant performance and reproducibility benefits.

## Problem Statement

### Current Issues

1. **Slow dependency resolution**: conda environment creation takes 5-15 minutes
2. **Mixed tooling**: Using both conda (environment.yml) and pip (pyproject.toml) creates confusion
3. **No lockfile**: Builds aren't fully reproducible across different machines/times
4. **CI complexity**: CI workflows use different installation methods (conda, pip)
5. **Developer friction**: New contributors must install conda/miniconda
6. **Inconsistent environments**: Different resolution across conda vs pip can cause issues

### Current Stack

- **Environment management**: conda (environment.yml)
- **Package metadata**: pyproject.toml with optional-dependencies
- **CI**: Mixed conda and pip installations
- **Build**: setuptools via pip
- **No lockfile**: Dependencies can drift over time

## Proposed Solution

Migrate to uv as the single tool for:

- Python version management (`uv python install`)
- Dependency resolution and installation (`uv sync`)
- Virtual environment management (`uv venv`)
- Package building (`uv build`)
- Script running (`uv run`)
- Lockfile-based reproducibility (`uv.lock`)

### Architecture

```
pyproject.toml
├── [project.dependencies]           # Runtime dependencies (numpy, pandas, etc.)
├── [dependency-groups]               # Dev dependencies (uv-specific, for contributors/CI)
│   └── dev = [pytest, black, ...]
└── [project.optional-dependencies]   # PyPI extras (for end users)
    └── dev = [pytest, black, ...]    # Mirror of dependency-groups for pip compatibility
```

**Dual dependency specification** (following Ariadne pattern):
- `[dependency-groups]` for uv users (contributors, CI)
- `[project.optional-dependencies]` for pip users (end users, backwards compatibility)

## Benefits

### 1. Performance

- **10-100x faster** than conda for dependency resolution
- Environment setup: ~15 minutes → ~30 seconds
- CI runs: Faster with uv caching

### 2. Reproducibility

- **Lockfile (`uv.lock`)**: Exact versions pinned for all dependencies
- Cross-platform lockfile: Same versions on Ubuntu, macOS, Windows
- CI verification: `uv sync --frozen` ensures lockfile isn't stale

### 3. Developer Experience

- **Single tool**: No need to install conda/miniconda
- **Simpler commands**: `uv sync` instead of `conda env create -f environment.yml && pip install -e .[dev]`
- **Faster iteration**: Quick dependency updates with `uv add`
- **Better error messages**: Clear conflict resolution

### 4. CI Improvements

- **Consistent across workflows**: Same uv commands for lint, test, docs, build
- **Caching**: GitHub Actions cache with `astral-sh/setup-uv@v5`
- **Matrix testing**: Easy to test multiple Python versions

### 5. Modern Python Standards

- **PEP 735**: Native support for `[dependency-groups]`
- **Standards-based**: Works with any PEP 517 build backend
- **Future-proof**: Active development, fast adoption

## Migration Strategy

### Phase 1: Add uv Support (Parallel Installation)

Keep conda working while adding uv:

1. Add `[dependency-groups]` to pyproject.toml
2. Mirror dev dependencies in `[project.optional-dependencies]` for backwards compat
3. Generate `uv.lock` with `uv lock`
4. Update README with uv installation instructions (alongside conda)
5. Test locally with uv

### Phase 2: Migrate CI to uv

Update GitHub Actions workflows:

1. CI workflow (`.github/workflows/ci.yml`)
2. Docs workflow (`.github/workflows/docs.yml`)
3. Build workflow (`.github/workflows/build.yml`)

Keep conda instructions in README for users who prefer it.

### Phase 3: Deprecate conda (Optional)

After 1-2 releases:

1. Move conda instructions to "Legacy Installation" section
2. Recommend uv as primary method
3. Eventually remove `environment.yml`

## Technical Details

### pyproject.toml Changes

```toml
# Current: project.optional-dependencies only
[project.optional-dependencies]
dev = [
    "pytest",
    "pytest-cov",
    "black",
    "pydocstyle",
    "toml",
    "twine",
    "build",
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.20",
    # ... more
]

# Proposed: Add dependency-groups (uv-native)
[dependency-groups]
dev = [
    "pytest",
    "pytest-cov",
    "black",
    "pydocstyle",
    "toml",
    "twine",
    "build",
    "mkdocs>=1.5",
    "mkdocs-material>=9.0",
    "mkdocstrings[python]>=0.20",
    # ... more
]

# Keep optional-dependencies for pip users
[project.optional-dependencies]
dev = [
    # ... same list as dependency-groups
]
```

### CI Workflow Example (from Ariadne)

```yaml
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-22.04, windows-2022, macos-14]
        python: ["3.7", "3.8", "3.9", "3.10", "3.11"]
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true
          github-token: ${{ secrets.GITHUB_TOKEN }}
          cache-dependency-glob: "**/uv.lock"

      - name: Install Python ${{ matrix.python }}
        run: uv python install ${{ matrix.python }}

      - name: Sync deps
        run: uv sync --frozen

      - name: Run tests
        run: uv run pytest tests/
```

### Developer Workflow

**Current (conda):**
```bash
conda env create -f environment.yml
conda activate sleap-roots
# Wait 10-15 minutes...
```

**Proposed (uv):**
```bash
uv sync
# Done in ~30 seconds!

# Run commands
uv run pytest tests/
uv run black sleap_roots tests
uv run mkdocs serve

# Add dependency
uv add numpy@latest
uv lock  # Update lockfile
```

## Compatibility

### Backwards Compatibility

- **End users**: Can still use `pip install sleap-roots`
- **Optional extras**: `pip install sleap-roots[dev]` still works
- **PyPI publishing**: No changes to package distribution
- **conda users**: Can continue using conda if preferred (Phase 1-2)

### Python Version Support

- Current: Python 3.7+
- uv supports: Python 3.7+ (same range)
- Can test multiple versions easily in CI matrix

## Risks and Mitigations

### Risk 1: Learning Curve for Contributors

**Mitigation:**
- Comprehensive README updates with uv examples
- Update CLAUDE.md and AGENTS.md with uv context
- Side-by-side conda/uv instructions during transition
- Video tutorial or documentation

### Risk 2: uv is Relatively New

**Mitigation:**
- uv is backed by Astral (also makes ruff, used by >100k projects)
- Rapid adoption in Python ecosystem
- Fallback: Keep pip/conda instructions available
- Test thoroughly before full migration

### Risk 3: Lockfile Merge Conflicts

**Mitigation:**
- Git attributes for uv.lock (similar to package-lock.json)
- Documentation on resolving conflicts (`uv lock`)
- Most conflicts auto-resolve with re-lock

### Risk 4: Platform-Specific Dependencies

**Mitigation:**
- uv.lock is cross-platform by default
- Test on Ubuntu, macOS, Windows in CI (already doing this)
- Platform markers in pyproject.toml if needed

## Alternatives Considered

### 1. Keep conda

**Pros:**
- No migration needed
- Familiar to scientific Python users
- Handles non-Python dependencies (though we don't have any)

**Cons:**
- Slow (10-15 minute environment creation)
- No lockfile (mamba has experimental support)
- Mixed conda/pip is error-prone
- CI complexity

### 2. Poetry

**Pros:**
- Mature, widely used
- Lockfile support

**Cons:**
- Slower than uv (uses same resolver as pip)
- More opinionated (custom pyproject.toml sections)
- Doesn't manage Python versions
- Larger install size

### 3. PDM

**Pros:**
- PEP 582 support
- Lockfile support

**Cons:**
- Smaller community than Poetry
- Slower than uv
- Less GitHub Actions integration

### 4. pip-tools

**Pros:**
- Minimal, pip-based
- `requirements.txt` lockfiles

**Cons:**
- Separate tools for compilation vs installation
- No Python version management
- Manual workflow (compile → sync)

**Why uv wins:**
- Fastest performance (written in Rust)
- All-in-one tool (Python + deps + build + run)
- Modern standards (PEP 735 dependency-groups)
- Excellent GitHub Actions support
- Active development and rapid iteration

## Success Criteria

### Phase 1: uv Support Added

- [ ] `uv sync` installs all dependencies correctly
- [ ] `uv run pytest tests/` passes all tests
- [ ] `uv run mkdocs build` builds documentation
- [ ] `uv.lock` committed and CI validates it
- [ ] README has uv installation instructions

### Phase 2: CI Migrated

- [ ] All CI workflows use uv
- [ ] CI runs as fast or faster than before
- [ ] Cross-platform tests pass (Ubuntu, macOS, Windows)
- [ ] Python version matrix works (3.7-3.11)

### Phase 3: Adoption

- [ ] Contributors successfully use uv for development
- [ ] No regression in development experience
- [ ] Faster CI runs (measured improvement)
- [ ] Updated documentation and guides

## Timeline

### Week 1: Phase 1 - Add uv Support

- Day 1-2: Add `[dependency-groups]` to pyproject.toml
- Day 2-3: Generate and test `uv.lock`
- Day 3-4: Update README and documentation
- Day 5: Test locally on all platforms

### Week 2: Phase 2 - Migrate CI

- Day 1-2: Update ci.yml workflow
- Day 2-3: Update docs.yml workflow
- Day 3-4: Update build.yml workflow
- Day 5: Test and verify all workflows

### Week 3: Phase 3 - Documentation and Rollout

- Day 1-2: Update developer guides
- Day 2-3: Update Claude commands (`/validate-env`, etc.)
- Day 4-5: Announce to team, gather feedback

### Week 4+: Phase 4 - Optional Deprecation

- Monitor adoption
- Decide on timeline for conda deprecation
- Update guides accordingly

**Total Estimated Time:** 15-20 hours over 3-4 weeks

## Impact on OpenSpec Workflow

uv integrates well with OpenSpec methodology:

- Faster iteration on proposals (quick environment setup)
- Reproducible testing (lockfile ensures consistency)
- Easier for AI agents to work with (single tool, clear commands)
- Better CI integration (slash commands can use `uv run`)

## References

- [uv Documentation](https://docs.astral.sh/uv/)
- [PEP 735: Dependency Groups](https://peps.python.org/pep-0735/)
- [Ariadne Migration Example](https://github.com/Salk-Harnessing-Plants-Initiative/Ariadne)
- [setup-uv GitHub Action](https://github.com/astral-sh/setup-uv)

## Decision

**Recommendation:** Proceed with migration to uv.

The benefits (10-100x faster, reproducible builds, better DX) far outweigh the risks (learning curve, new tool). The Ariadne project has successfully demonstrated this pattern, and uv's rapid adoption in the ecosystem suggests it's the future of Python package management.

**Next Steps:**
1. Review and approve this proposal
2. Create feature branch `infra/migrate-to-uv`
3. Implement Phase 1 (parallel installation)
4. Test thoroughly across platforms
5. Submit PR for review