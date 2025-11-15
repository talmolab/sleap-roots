# Design: Migrate to uv

## Architecture Overview

This design document details the technical implementation of migrating sleap-roots from conda/pip to uv for dependency management.

## System Architecture

### Current State

```
┌─────────────────────────────────────────┐
│          Developer Machine              │
├─────────────────────────────────────────┤
│  conda (environment management)         │
│    ↓                                    │
│  environment.yml → conda env            │
│    ↓                                    │
│  pip install -e .[dev]                  │
│                                         │
│  No lockfile = non-reproducible builds  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           GitHub Actions CI             │
├─────────────────────────────────────────┤
│  actions/setup-python                   │
│    ↓                                    │
│  pip install -e .[dev]                  │
│    ↓                                    │
│  Run tests, linting, docs build         │
│                                         │
│  Different resolution than local        │
└─────────────────────────────────────────┘
```

### Proposed State

```
┌─────────────────────────────────────────┐
│          Developer Machine              │
├─────────────────────────────────────────┤
│  uv (single tool for everything)        │
│    ↓                                    │
│  pyproject.toml + uv.lock               │
│    ↓                                    │
│  uv sync --frozen                       │
│    ↓                                    │
│  Exact versions from lockfile           │
│                                         │
│  ✅ Reproducible builds                 │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│           GitHub Actions CI             │
├─────────────────────────────────────────┤
│  astral-sh/setup-uv@v5                  │
│    ↓                                    │
│  uv python install ${{ matrix.python }} │
│    ↓                                    │
│  uv sync --frozen                       │
│    ↓                                    │
│  uv run pytest / black / mkdocs         │
│                                         │
│  ✅ Same versions as local              │
└─────────────────────────────────────────┘
```

## Component Design

### 1. pyproject.toml Structure

Following Ariadne's dual specification pattern:

```toml
[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "sleap-roots"
# ... metadata ...
dependencies = [
    "numpy",
    "h5py",
    "attrs",
    "pandas",
    "matplotlib",
    "seaborn",
    "sleap-io>=0.0.11",
    "scikit-image",
    "shapely"
]

# ---- uv dev group for contributors/CI ----
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
    "mkdocs-gen-files",
    "mkdocs-literate-nav",
    "mkdocs-section-index",
    "mike"
]

# ---- PyPI extras for end users who opt in ----
[project.optional-dependencies]
dev = [
    # Mirror of dependency-groups for pip compatibility
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
    "mkdocs-gen-files",
    "mkdocs-literate-nav",
    "mkdocs-section-index",
    "mike"
]
```

**Rationale:**
- `[dependency-groups]`: PEP 735 standard, uv-native, fast resolution
- `[project.optional-dependencies]`: Backwards compatible with pip
- Duplicated intentionally to support both ecosystems

### 2. Lockfile Management

**File**: `uv.lock`

**Generation**: `uv lock`

**Properties**:
- Cross-platform (single lockfile for Linux, macOS, Windows)
- Includes transitive dependencies with exact versions
- Includes hashes for integrity verification
- TOML format (human-readable)

**Update Strategy**:
```bash
# Add new dependency
uv add numpy@latest

# Update specific dependency
uv lock --upgrade-package numpy

# Update all dependencies
uv lock --upgrade

# Sync to lockfile
uv sync --frozen  # Fail if lockfile is out of date
```

### 3. CI Workflow Design

#### Matrix Strategy

Test across Python versions and platforms:

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [ubuntu-22.04, windows-2022, macos-14]
    python: ["3.7", "3.8", "3.9", "3.10", "3.11"]
    exclude:
      # Optional: exclude specific combinations if needed
      - os: macos-14  # ARM Mac
        python: "3.7"  # Not available on ARM
```

#### Workflow Steps

```yaml
- name: Checkout
  uses: actions/checkout@v4

- name: Set up uv
  uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    github-token: ${{ secrets.GITHUB_TOKEN }}
    cache-dependency-glob: "**/uv.lock"

- name: Install Python ${{ matrix.python }}
  run: uv python install ${{ matrix.python }}

- name: Sync deps (runtime + dev group)
  run: uv sync --frozen

- name: Verify lockfile unchanged
  run: |
    git diff --exit-code uv.lock || {
      echo "ERROR: uv.lock was modified. Run 'uv lock' locally and commit."
      exit 1
    }

- name: Environment info
  run: |
    uv --version
    uv run python -c "import sys; print(sys.version)"
    uv tree

- name: Run tests
  run: uv run pytest -q --cov=sleap_roots --cov-report=xml
```

#### Caching Strategy

GitHub Actions cache key based on:
- OS
- Python version
- uv.lock hash (via `cache-dependency-glob`)

**Cache hit**: Skip dependency installation (~95% faster)
**Cache miss**: Install deps (~30 seconds)

### 4. Developer Workflow

#### Initial Setup

```bash
# Clone repo
git clone https://github.com/talmolab/sleap-roots.git
cd sleap-roots

# Install uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install dependencies
uv sync

# Run tests
uv run pytest tests/

# Run formatters
uv run black sleap_roots tests

# Build docs
uv run mkdocs serve
```

**Time comparison**:
- conda: 10-15 minutes
- uv: 30-60 seconds (first time), 5-10 seconds (cached)

#### Daily Development

```bash
# Activate is NOT needed with uv!
# Just prefix commands with 'uv run'

# Run tests
uv run pytest tests/test_lengths.py

# Run all tests with coverage
uv run pytest --cov

# Format code
uv run black sleap_roots

# Lint
uv run pydocstyle sleap_roots/

# Build docs
uv run mkdocs serve

# Run scripts
uv run python scripts/analyze_data.py
```

#### Dependency Management

```bash
# Add new dependency
uv add scikit-learn

# Add dev dependency
uv add --dev ipython

# Remove dependency
uv remove scikit-learn

# Update all dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package numpy

# Sync after pulling changes
uv sync
```

### 5. Migration Path

#### Parallel Installation Phase

During migration, both conda and uv work:

```
Repository Root
├── environment.yml        # ← Keep for backwards compat
├── pyproject.toml         # ← Add [dependency-groups]
├── uv.lock               # ← New lockfile
├── README.md             # ← Add both installation methods
└── .github/workflows/
    ├── ci.yml            # ← Migrated to uv
    ├── docs.yml          # ← Migrated to uv
    └── build.yml         # ← Migrated to uv
```

**README Installation Section**:
```markdown
## Installation

### Recommended: uv (fast, reproducible)

# ... uv instructions ...

### Alternative: conda (legacy)

# ... conda instructions ...
```

#### Full Migration Phase

After 1-2 releases, optionally remove conda:

```
Repository Root
├── pyproject.toml         # ← Only dependency source
├── uv.lock               # ← Lockfile
└── README.md             # ← uv only
```

### 6. Command Mapping

| Current (conda/pip) | Proposed (uv) | Notes |
|---------------------|---------------|-------|
| `conda env create -f environment.yml` | `uv sync` | 95% faster |
| `conda activate sleap-roots` | *(not needed)* | Use `uv run` instead |
| `pip install -e .[dev]` | `uv sync` | Includes dev group |
| `pytest tests/` | `uv run pytest tests/` | Explicit runner |
| `black sleap_roots` | `uv run black sleap_roots` | Explicit runner |
| `mkdocs serve` | `uv run mkdocs serve` | Explicit runner |
| `pip install numpy` | `uv add numpy` | Updates lock |
| `pip freeze` | `uv tree` | Better visualization |

### 7. Error Handling

#### Lockfile Out of Sync

**Error**:
```
error: The lockfile is out of sync with pyproject.toml
```

**Fix**:
```bash
uv lock
git add uv.lock
git commit -m "chore: update lockfile"
```

#### Platform-Specific Dependencies

If a dependency is platform-specific:

```toml
[project.dependencies]
pywin32 = { version = ">=306", markers = "sys_platform == 'win32'" }
```

uv.lock will include it only for Windows.

#### Python Version Incompatibility

If testing Python 3.7 on Apple Silicon (not supported):

```yaml
matrix:
  exclude:
    - os: macos-14  # ARM Mac
      python: "3.7"
```

### 8. Performance Optimization

#### uv Caching

uv maintains a global cache in `~/.cache/uv/`:

- Downloaded wheels
- Built source distributions
- Extracted packages

**Benefits**:
- Shared across projects
- Faster subsequent installs
- Survives `uv sync --reinstall`

#### GitHub Actions Caching

```yaml
- uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    cache-dependency-glob: "**/uv.lock"
```

**Cache key**: `os-uv-<hash of uv.lock>`

**Typical CI times**:
- Cold cache: 60-90 seconds
- Warm cache: 10-20 seconds

### 9. Testing Strategy

#### Unit Tests

No changes needed - tests work identically:

```bash
# Before
pytest tests/

# After
uv run pytest tests/
```

#### Integration Tests

Test uv-specific functionality:

```bash
# Test lockfile is up to date
uv lock --locked  # Fail if out of sync

# Test sync works
rm -rf .venv
uv sync --frozen

# Test build works
uv build

# Test scripts work
uv run python -c "import sleap_roots; print(sleap_roots.__version__)"
```

#### CI Testing Matrix

Full matrix coverage:

```yaml
matrix:
  os: [ubuntu-22.04, windows-2022, macos-14]
  python: ["3.7", "3.8", "3.9", "3.10", "3.11"]
```

**Total combinations**: 15 (3 OS × 5 Python versions)

### 10. Documentation Updates

#### Files to Update

1. **README.md**
   - Add uv installation instructions
   - Add uv workflow examples
   - Performance comparison

2. **docs/getting-started/installation.md**
   - Detailed uv setup
   - Platform-specific instructions
   - Troubleshooting

3. **docs/dev/setup.md**
   - Developer workflow with uv
   - Dependency management
   - Common commands

4. **docs/dev/contributing.md**
   - Pull request workflow with uv
   - Lockfile commit guidelines

5. **Claude commands** (`.claude/commands/*.md`)
   - Update all to use `uv run`

6. **CLAUDE.md / openspec/project.md**
   - Update tech stack
   - Update dependency management section

### 11. Rollback Strategy

If migration fails, rollback is simple:

```bash
# Remove uv changes
git revert <commit-hash>

# Remove lockfile
rm uv.lock

# Revert CI workflows
git checkout main -- .github/workflows/

# Revert pyproject.toml
git checkout main -- pyproject.toml

# Back to conda
conda env create -f environment.yml
```

**Backwards compatibility preserved**:
- `pip install sleap-roots` still works
- `pip install sleap-roots[dev]` still works
- PyPI package unchanged

## Security Considerations

### Dependency Hashing

uv.lock includes SHA256 hashes for all packages:

```toml
[[distribution]]
name = "numpy"
version = "1.24.3"
source = { registry = "https://pypi.org/simple" }
wheels = [
    { url = "https://...", hash = "sha256:..." },
]
```

Verifies integrity on install.

### Supply Chain Security

uv verifies:
- Package hashes match lockfile
- No tampering during download
- Reproducible builds

## Performance Benchmarks

### Expected Performance (Based on Ariadne Migration)

| Operation | conda | uv | Speedup |
|-----------|-------|-----|---------|
| Initial install (cold) | 10-15 min | 30-60 sec | **10-30x** |
| Install (warm cache) | 5-10 min | 5-10 sec | **30-60x** |
| CI run (cold) | 3-5 min | 60-90 sec | **2-3x** |
| CI run (cached) | 2-3 min | 10-20 sec | **6-9x** |
| Add dependency | 2-5 min | 5-10 sec | **12-30x** |

### Disk Space

- uv cache: ~500 MB (shared across all projects)
- conda env: ~2 GB (per environment)

### Network Usage

- First install: Similar to conda (downloads wheels)
- Subsequent: Minimal (uses cache)

## Alternatives Considered

See proposal.md section "Alternatives Considered" for detailed comparison of Poetry, PDM, pip-tools, and keeping conda.

**Decision**: uv wins on performance, modern standards, and developer experience.

## Open Questions

1. **Should we keep environment.yml indefinitely?**
   - Recommendation: Keep for 2 releases, then deprecate

2. **Should we require uv for contributions?**
   - Recommendation: Yes, but pip still works for end users

3. **What about non-Python dependencies?**
   - Not applicable - sleap-roots has no C dependencies beyond wheels

4. **How to handle lockfile merge conflicts?**
   - Document in CONTRIBUTING.md: re-run `uv lock` and test

## Success Criteria

✅ Migration is successful if:

1. All CI jobs pass with uv
2. CI is ≥50% faster
3. Fresh `git clone → uv sync → uv run pytest` works in <2 minutes
4. No regression in test coverage
5. No regression in developer experience
6. Documentation is clear and comprehensive
7. Lockfile prevents dependency drift (verified in CI)

## Conclusion

The migration to uv provides significant benefits:

- **10-100x faster** dependency resolution
- **Reproducible builds** via lockfile
- **Simpler workflows** (single tool)
- **Better CI** (faster, cached, consistent)
- **Future-proof** (modern standards)

With careful phased rollout and backwards compatibility, migration risk is minimal.