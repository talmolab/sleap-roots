# Simplify User Documentation

## Why

The user-facing documentation currently contains several issues that make it confusing and inconsistent:

### 1. Developer Content on User-Facing Pages

**Problem**: The home page links to developer-focused benchmarking infrastructure (`dev/benchmarking.md` - 399 lines of CI/testing details) instead of simple performance visualizations.

**Impact**: Users don't care about pytest-benchmark internals or regression detection workflows. They just want to know "is it fast?" with simple examples.

### 2. Inaccurate Trait Count Claims

**Problem**: Documentation claims "50+ traits" but the actual count is **174 TraitDef objects** (likely 100+ user-facing traits).

**Locations**:
- Home page mentions "50+" twice
- Tutorials mention "50+"
- Quick start mentions "40+"

**Impact**: Undersells the package's actual capabilities and creates inconsistency.

### 3. Conda Over-Recommendation

**Problem**: Despite recommending `uv init` + `uv add` as best practice, conda is mentioned extensively throughout documentation (6 files).

**Current state**:
- Installation guide: "Recommended: Conda Environment" section
- Home page: "Quick pip install or conda setup"
- Quick start: conda activation instructions
- Troubleshooting: conda environment creation
- Multiple places with conda-specific solutions

**Impact**: Contradicts the modern workflow recommendation and clutters documentation with outdated approach.

### 4. Repeated Error Messages

**Problem**: "ModuleNotFoundError: No module named 'sleap_roots'" is documented in 3 separate places with varying solutions.

**Locations**:
- Installation guide (conda solution)
- Troubleshooting guide (pip solution)
- Quick start (conda solution)

**Impact**: Confusing for users - which solution is correct? Creates maintenance burden.

### 5. Python Version Confusion

**Problem**: Documentation claims broad Python version support that isn't tested in CI.

**Claims**:
- "Ubuntu 22.04: 3.7-3.11 ✅ Fully supported"
- "macOS/Windows: 3.11 ✅ Fully supported"

**Reality**:
- CI only tests Python 3.11 on all platforms
- pyproject.toml includes 3.12 in classifiers but documentation doesn't mention it
- "Fully supported" is misleading when only 3.11 is CI-tested

**Impact**: Sets incorrect expectations about Python version support.

## What Changes

### 1. Replace Developer Benchmarking Link with User-Friendly Performance Section

**On home page**:
- Remove link to `dev/benchmarking.md`
- Add simple performance visualization or summary table
- Show actual benchmark results (0.1-0.5s per plant) with context:
  - Number of samples benchmarked
  - Platform (GitHub Actions: 2 cores, 7GB RAM)
  - Simple "what this means" explanation
- Add link to "Developer Guide → Benchmarking" for those interested in infrastructure

**In developer guide**:
- Keep `dev/benchmarking.md` as-is for contributors
- Ensure it's easy to find from Developer Guide section

### 2. Correct Trait Count Throughout

**Action**: Count actual user-facing traits and update all references.

**Update locations**:
- Home page: "50+" → accurate count (e.g., "100+")
- Tutorials: Update trait count claims
- Quick start: Consistent trait count

**Verify**: Check with actual pipeline trait definitions to ensure accuracy.

### 3. De-emphasize Conda, Promote uv Best Practices

**Installation guide** (`docs/getting-started/installation.md`):
- Remove "Recommended: Conda Environment" section
- Keep single "Alternative: Using Conda" section at bottom (collapsed or minimal)
- Lead with pip and uv workflows only

**Home page**:
- "Quick pip install or conda setup" → "Quick pip install or uv setup"

**Quick start**:
- Remove conda activation instructions from main flow
- Assume users followed the recommended uv/pip approach

**Troubleshooting**:
- Remove conda-specific solutions
- Provide universal solutions (pip install, check PATH, etc.)

**Developer setup**:
- Keep conda as alternative for developers (it's fine for contributors)

**Principle**: Mention conda exactly once as an alternative, with full instructions in that one place only.

### 4. Consolidate Error Troubleshooting

**Action**: Create single source of truth for common errors.

**Implementation**:
- Keep detailed solutions in `docs/guides/troubleshooting.md` only
- Installation guide: Link to troubleshooting for import errors (don't repeat solution)
- Quick start: Link to troubleshooting (don't repeat solution)
- Use cross-references instead of duplication

**Pattern**:
```markdown
!!! warning "Import Errors?"
    If you encounter `ModuleNotFoundError`, see the [Troubleshooting Guide](../guides/troubleshooting.md#import-errors).
```

### 5. Clarify Python Version Support

**Action**: Align documentation with CI reality.

**Update table**:
```markdown
| Platform | Python Versions | Status |
|----------|----------------|--------|
| **Ubuntu 22.04** | 3.11 | ✅ CI tested |
| **macOS** | 3.11 | ✅ CI tested |
| **Windows** | 3.11 | ✅ CI tested |
| **All platforms** | 3.7-3.12 | ⚠️ Should work (not CI tested) |
```

**Add note**: "We officially test and support Python 3.11. Other versions (3.7-3.12) should work based on dependencies, but are not continuously tested."

## Impact

**Users Affected**:
- ✅ New users get clearer, simpler path with modern tools
- ✅ Advanced users can still find conda/alternative approaches
- ✅ Accurate feature claims (trait counts)
- ✅ Honest Python version support expectations

**Breaking Changes**: None (documentation only)

## Success Criteria

- [ ] Home page has user-friendly performance section (no dev/benchmarking.md link)
- [ ] Accurate trait count throughout (based on actual pipeline counts)
- [ ] Conda mentioned exactly once in installation guide as "Alternative"
- [ ] No conda references in home page, quick start, or troubleshooting
- [ ] Single source of truth for common errors (troubleshooting guide)
- [ ] Python version table matches CI reality
- [ ] Documentation builds cleanly

## Scope

**In Scope**:
- Update home page performance section
- Count and update trait references
- Remove/consolidate conda references
- Consolidate error troubleshooting
- Update Python version table

**Out of Scope**:
- Changing actual CI test matrix (separate decision)
- Adding new benchmark visualizations (can be follow-up)
- Removing conda from developer setup (appropriate for contributors)

## Dependencies

None. This is a documentation-only change.

## References

- Current home page: `docs/index.md`
- Installation guide: `docs/getting-started/installation.md`
- Troubleshooting: `docs/guides/troubleshooting.md`
- Quick start: `docs/getting-started/quickstart.md`
- CI configuration: `.github/workflows/ci.yml`
- Package metadata: `pyproject.toml`