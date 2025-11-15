# Tasks: Migrate to uv

## Overview

Migrate sleap-roots from conda/pip to uv for faster, more reproducible dependency management.

## Task List

### Phase 1: Add uv Support (Parallel Installation)

#### 1.1 Update pyproject.toml with dependency-groups
**Priority**: High
**Estimated Time**: 1 hour

- [ ] Add `[dependency-groups]` section to pyproject.toml
- [ ] Copy all dev dependencies from `[project.optional-dependencies.dev]`
- [ ] Keep `[project.optional-dependencies.dev]` for backwards compatibility
- [ ] Ensure both lists are identical
- [ ] Add comment explaining dual specification pattern

**Deliverable**: pyproject.toml with both dependency-groups and optional-dependencies

**Files Modified**:
- `pyproject.toml`

---

#### 1.2 Generate and Commit uv.lock
**Priority**: High
**Estimated Time**: 1 hour

- [ ] Install uv locally: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [ ] Generate lockfile: `uv lock`
- [ ] Review uv.lock for correctness
- [ ] Test installation: `uv sync`
- [ ] Verify all dependencies installed correctly
- [ ] Add uv.lock to git (do not add to .gitignore)
- [ ] Commit uv.lock

**Deliverable**: Committed uv.lock with all dependencies pinned

**Files Created**:
- `uv.lock`

---

#### 1.3 Test uv Locally
**Priority**: High
**Estimated Time**: 2 hours

- [ ] Test on Ubuntu (if available)
- [ ] Test on macOS
- [ ] Test on Windows (if available)
- [ ] Verify `uv sync` installs correctly
- [ ] Verify `uv run pytest tests/` passes
- [ ] Verify `uv run black --check sleap_roots tests` works
- [ ] Verify `uv run mkdocs build` works
- [ ] Compare performance vs conda (timing)

**Deliverable**: Confirmed working uv setup on all platforms

---

#### 1.4 Update README with uv Instructions
**Priority**: High
**Estimated Time**: 1.5 hours

- [ ] Add "Installation with uv (Recommended)" section
- [ ] Add uv installation instructions for macOS, Linux, Windows
- [ ] Add `uv sync` example
- [ ] Add common uv commands (run, add, remove)
- [ ] Keep conda installation section as alternative
- [ ] Add performance comparison note (30 seconds vs 15 minutes)
- [ ] Update "Development" section with uv workflow

**Deliverable**: README with comprehensive uv documentation

**Files Modified**:
- `README.md`

---

#### 1.5 Update Documentation
**Priority**: Medium
**Estimated Time**: 1 hour

- [ ] Update `docs/getting-started/installation.md` with uv instructions
- [ ] Update `docs/dev/setup.md` with uv workflow
- [ ] Add uv to recommended tools list
- [ ] Update troubleshooting section

**Deliverable**: Documentation updated with uv instructions

**Files Modified**:
- `docs/getting-started/installation.md`
- `docs/dev/setup.md`

---

### Phase 2: Migrate CI to uv

#### 2.1 Update CI Workflow (ci.yml)
**Priority**: High
**Estimated Time**: 2 hours

- [ ] Replace conda setup with `astral-sh/setup-uv@v5`
- [ ] Add Python version matrix (3.7, 3.8, 3.9, 3.10, 3.11)
- [ ] Use `uv python install ${{ matrix.python }}`
- [ ] Replace pip install with `uv sync --frozen`
- [ ] Add lockfile verification step
- [ ] Update all command invocations to use `uv run`
- [ ] Add environment info step (`uv --version`, `uv tree`)
- [ ] Enable uv caching with `cache-dependency-glob: "**/uv.lock"`
- [ ] Test workflow on feature branch

**Deliverable**: Working ci.yml using uv

**Files Modified**:
- `.github/workflows/ci.yml`

**Example Changes**:
```yaml
# Before
- uses: actions/setup-python@v5
  with:
    python-version: "3.11"
- run: pip install -e .[dev]

# After
- uses: astral-sh/setup-uv@v5
  with:
    enable-cache: true
    cache-dependency-glob: "**/uv.lock"
- run: uv python install 3.11
- run: uv sync --frozen
```

---

#### 2.2 Update Documentation Workflow (docs.yml)
**Priority**: High
**Estimated Time**: 1.5 hours

- [ ] Replace pip setup with uv setup
- [ ] Use `uv sync --frozen`
- [ ] Update mkdocs command to `uv run mkdocs build`
- [ ] Update mike deployment commands to use `uv run`
- [ ] Enable uv caching
- [ ] Test on feature branch

**Deliverable**: Working docs.yml using uv

**Files Modified**:
- `.github/workflows/docs.yml`

---

#### 2.3 Update Build Workflow (build.yml)
**Priority**: High
**Estimated Time**: 1 hour

- [ ] Replace pip setup with uv setup
- [ ] Use `uv build` instead of `python -m build`
- [ ] Test wheel installation with uv
- [ ] Test sdist installation with uv
- [ ] Use `uvx twine` for package checking
- [ ] Enable uv caching

**Deliverable**: Working build.yml using uv

**Files Modified**:
- `.github/workflows/build.yml`

---

#### 2.4 Add Package Verification Job (Following Ariadne Pattern)
**Priority**: Medium
**Estimated Time**: 2 hours

- [ ] Add new CI job: `package`
- [ ] Build sdist and wheel with `uv build`
- [ ] Verify metadata with `uvx twine check dist/*`
- [ ] Test wheel installation in fresh environment
- [ ] Test sdist with dev extras installation
- [ ] Verify imports work correctly

**Deliverable**: Comprehensive package testing in CI

**Files Modified**:
- `.github/workflows/ci.yml` or new `.github/workflows/package.yml`

---

### Phase 3: Update Developer Tools and Documentation

#### 3.1 Update Claude Commands
**Priority**: Medium
**Estimated Time**: 2 hours

- [ ] Update `/validate-env` to check for uv installation
- [ ] Update `/validate-env` to verify uv.lock is present
- [ ] Update `/run-ci-locally` to use uv commands
- [ ] Update `/lint` to use `uv run black` and `uv run pydocstyle`
- [ ] Update `/test` to use `uv run pytest`
- [ ] Update `/coverage` to use `uv run pytest --cov`
- [ ] Update `/fix-formatting` to use `uv run black`
- [ ] Add uv installation check to validation

**Deliverable**: All Claude commands use uv

**Files Modified**:
- `.claude/commands/validate-env.md`
- `.claude/commands/run-ci-locally.md`
- `.claude/commands/lint.md`
- `.claude/commands/test.md`
- `.claude/commands/coverage.md`
- `.claude/commands/fix-formatting.md`

---

#### 3.2 Update CLAUDE.md and AGENTS.md
**Priority**: Low
**Estimated Time**: 30 minutes

- [ ] Add uv to tech stack section
- [ ] Update dependency management description
- [ ] Update development workflow examples
- [ ] Add uv performance notes

**Deliverable**: Updated project context files

**Files Modified**:
- `CLAUDE.md`
- `openspec/project.md`

---

#### 3.3 Update Jupyter Notebooks (if applicable)
**Priority**: Low
**Estimated Time**: 1 hour

- [ ] Review notebooks/ for setup instructions
- [ ] Update any conda references to uv
- [ ] Test notebooks work with uv-installed packages

**Deliverable**: Notebooks work with uv environment

**Files Modified**:
- `notebooks/*.ipynb` (as needed)

---

### Phase 4: Testing and Validation

#### 4.1 Cross-Platform Testing
**Priority**: High
**Estimated Time**: 2 hours

- [ ] Test full workflow on Ubuntu 22.04
- [ ] Test full workflow on macOS (Intel and/or ARM)
- [ ] Test full workflow on Windows
- [ ] Verify all Python versions work (3.7-3.11)
- [ ] Document any platform-specific issues

**Deliverable**: Confirmed working on all platforms

---

#### 4.2 Performance Benchmarking
**Priority**: Low
**Estimated Time**: 1 hour

- [ ] Measure conda environment creation time
- [ ] Measure uv sync time (cold cache)
- [ ] Measure uv sync time (warm cache)
- [ ] Measure CI runtime before/after migration
- [ ] Document performance improvements

**Deliverable**: Performance comparison data

---

#### 4.3 Fresh Install Testing
**Priority**: High
**Estimated Time**: 1 hour

- [ ] Clone repo on fresh machine
- [ ] Follow new README instructions with uv
- [ ] Verify all tests pass
- [ ] Verify docs build
- [ ] Document any issues

**Deliverable**: Confirmed working for new users

---

### Phase 5: Optional Conda Deprecation

#### 5.1 Deprecate environment.yml (Optional - Future)
**Priority**: Low
**Estimated Time**: 30 minutes

- [ ] Move conda instructions to "Legacy Installation"
- [ ] Add deprecation notice
- [ ] Set timeline for removal (e.g., 2 releases)
- [ ] Update README

**Deliverable**: Conda marked as legacy

**Files Modified**:
- `README.md`

---

#### 5.2 Remove environment.yml (Optional - Future)
**Priority**: Low
**Estimated Time**: 15 minutes

- [ ] Remove environment.yml
- [ ] Remove conda references from documentation
- [ ] Update all guides to use uv exclusively

**Deliverable**: Conda fully removed

**Files Removed**:
- `environment.yml`

---

## Summary

### Total Estimated Time: 18-22 hours

**Phase Breakdown**:
- Phase 1 (Add uv support): 6.5 hours
- Phase 2 (Migrate CI): 6.5 hours
- Phase 3 (Update tools/docs): 3.5 hours
- Phase 4 (Testing): 4 hours
- Phase 5 (Optional deprecation): 0.75 hours

### Dependencies

- Phase 2 depends on Phase 1 completion
- Phase 3 can run parallel to Phase 2
- Phase 4 requires Phase 1-3 completion
- Phase 5 is optional and can happen later

### Critical Path

1. Update pyproject.toml (1.1)
2. Generate uv.lock (1.2)
3. Test locally (1.3)
4. Update CI workflows (2.1, 2.2, 2.3)
5. Validate everything works (4.1, 4.3)

## Success Metrics

- [ ] CI runs complete successfully with uv
- [ ] CI runs are faster (>50% improvement)
- [ ] All tests pass on all platforms
- [ ] Documentation builds successfully
- [ ] Fresh clone → uv sync → tests pass in <2 minutes
- [ ] No regression in developer experience
- [ ] Lockfile prevents dependency drift

## Rollback Plan

If issues arise:

1. **Phase 1**: Keep using conda, remove uv.lock
2. **Phase 2**: Revert CI workflows to conda/pip
3. **Phase 3**: Revert Claude commands
4. **Phase 4**: Document issues for future attempt

All changes are reversible as we maintain backwards compatibility.