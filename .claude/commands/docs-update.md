# Update Documentation

Systematic workflow for reviewing and updating project documentation to ensure accuracy and completeness.

## Quick Start

```bash
# Review and update documentation
```

This command helps you:
1. Identify outdated documentation
2. Find missing documentation
3. Update docs to match current code
4. Verify examples still work
5. Check for broken links

## What Gets Checked

### Core Documentation Files

- **README.md** - Project overview, installation, usage examples
- **CHANGELOG.md** - Release history and version notes
- **CLAUDE.md** - AI assistant instructions and context
- **AGENTS.md** - OpenSpec agent instructions
- **openspec/project.md** - Project context and conventions

### Code Documentation

- **Docstrings** - All public functions and classes
- **Module docstrings** - Top-level module descriptions
- **Type hints** - Function signatures

### External Documentation

- **HackMD Trait Docs** - https://hackmd.io/DMiXO2kXQhKH8AIIcIy--g
- **API documentation** - https://roots.sleap.ai (MkDocs with mkdocstrings)
- **GitHub Wiki** (if exists)

## Commands for Finding Docs

```bash
# Find all markdown files
find . -name "*.md" -not -path "*/node_modules/*" -not -path "*/.git/*" | sort

# Search for TODO/FIXME in docs
grep -r "TODO\|FIXME\|TBD" --include="*.md" .

# Find recently modified docs
find . -name "*.md" -mtime -7 -exec ls -lh {} \;

# Check for broken internal links
grep -r "\[.*\](.*\.md)" --include="*.md" .
```

## Documentation Review Checklist

### 1. README.md

- [ ] Installation instructions are current
- [ ] All pipeline types are listed
- [ ] Usage examples work with current API
- [ ] Dependencies are up to date
- [ ] Links to external resources work
- [ ] Badge status is current (CI, coverage, PyPI)

**Check:**
```bash
# Verify installation commands work
conda create -n test-install python=3.11
conda activate test-install
# Copy-paste install commands from README
pip install sleap-roots

# Verify usage examples
python -c "$(grep -A 10 'import sleap_roots' README.md | head -15)"
```

### 2. Pipeline Documentation

- [ ] All pipeline classes documented in README
- [ ] Each pipeline has usage example
- [ ] Trait lists are accurate

**Check:**
```python
# List all pipeline classes
from sleap_roots import trait_pipelines
import inspect

pipelines = [
    name for name, obj in inspect.getmembers(trait_pipelines)
    if inspect.isclass(obj) and name.endswith('Pipeline')
]

print("Pipelines in code:", pipelines)
# Compare with README list
```

### 3. Trait Documentation

- [ ] All traits are documented
- [ ] Trait descriptions are accurate
- [ ] Units are specified
- [ ] Formulas/methods are explained

**Check:**
```python
# List all traits from a pipeline
from sleap_roots import DicotPipeline

pipeline = DicotPipeline()
traits = pipeline.get_trait_definitions()

print(f"DicotPipeline has {len(traits)} traits")
for trait in traits[:5]:
    print(f"  - {trait.name}: {trait.description}")

# Compare with HackMD trait docs
```

### 4. CHANGELOG.md

- [ ] Latest version documented
- [ ] Unreleased section exists
- [ ] Version links work
- [ ] Follows Keep a Changelog format

**Check:**
```bash
# View CHANGELOG sections
grep "^## " CHANGELOG.md

# Check version matches code
python -c "import sleap_roots; print(f'Code version: {sleap_roots.__version__}')"
grep "^\[" CHANGELOG.md | head -3
```

### 5. Docstrings

- [ ] All public functions have docstrings
- [ ] Google-style format used
- [ ] Args, Returns, Raises documented
- [ ] Examples included where helpful

**Check:**
```bash
# Find functions missing docstrings
pydocstyle --convention=google sleap_roots/

# Count documented vs undocumented
python -c "
import sleap_roots.lengths as m
import inspect

funcs = [f for f in dir(m) if not f.startswith('_')]
with_docs = sum(1 for f in funcs if getattr(m, f).__doc__)
print(f'{with_docs}/{len(funcs)} functions documented')
"
```

## Update Workflow

### Step 1: Identify What Changed

```bash
# Check recent code changes
git log --oneline -10

# Check files modified since last release
git diff v0.1.4..HEAD --stat

# Find changed modules
git diff v0.1.4..HEAD --name-only | grep "sleap_roots/"
```

### Step 2: Update Affected Documentation

**If pipeline classes changed:**
- Update README.md pipeline list
- Update usage examples
- Update trait counts

**If trait computation changed:**
- Update HackMD trait documentation
- Update docstrings
- Update CHANGELOG.md

**If API changed:**
- Update README.md examples
- Update docstrings
- Mark breaking changes in CHANGELOG.md

**If dependencies changed:**
- Update README.md installation
- Update environment.yml notes
- Update CHANGELOG.md

### Step 3: Verify Examples Work

```bash
# Extract code examples from README
grep -A 20 "^```python" README.md > /tmp/examples.py

# Test they execute
python /tmp/examples.py
```

Or manually test each example:

```python
# Example from README
import sleap_roots as sr

series = sr.Series.load(
    series_name="919QDUH",
    h5_path="tests/data/canola_7do/919QDUH.h5",
    primary_path="tests/data/canola_7do/919QDUH.primary.slp",
    lateral_path="tests/data/canola_7do/919QDUH.lateral.slp"
)

pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series, write_csv=True)

# Verify this works!
assert traits is not None
```

### Step 4: Update Version References

```bash
# Find all version references
grep -r "0\.1\.4" . --include="*.md"

# Update to current version if needed
# (or leave as examples of old versions)
```

### Step 5: Check External Links

```bash
# List all external links
grep -oh "https://[^)]*" README.md

# Check each manually or with curl
curl -I https://hackmd.io/DMiXO2kXQhKH8AIIcIy--g
```

### Step 6: Build and Check MkDocs Site

```bash
# Build MkDocs documentation
uv run mkdocs build

# Check for warnings (should be zero except README.md exclusion)
uv run mkdocs build 2>&1 | grep -E "WARNING|ERROR"

# Serve locally to preview
uv run mkdocs serve
# Visit http://127.0.0.1:8000

# Check for broken internal links automatically
uv run mkdocs build --strict  # Fails on warnings

# Manual link verification for external links
grep -roh "https://[^)]*" docs/ | sort -u | while read url; do
  echo "Checking: $url"
  curl -s -o /dev/null -w "%{http_code}" "$url" || echo "FAILED"
done
```

**MkDocs Documentation Structure:**
- `mkdocs.yml` - Site configuration and navigation
- `docs/` - Markdown documentation files
- `docs/gen_ref_pages.py` - Auto-generates API reference from docstrings
- Site URL: https://roots.sleap.ai

## Common Documentation Tasks

### Task 1: New Pipeline Added

Update these files:
```markdown
# README.md
## Usage

Trait pipelines supported:
- `DicotPipeline` – Primary + lateral roots
- `YoungerMonocotPipeline` – Primary + crown roots
- `OlderMonocotPipeline` – Crown roots only
- `PrimaryRootPipeline` – Primary root only
- `LateralRootPipeline` – Lateral roots only
- `MultipleDicotPipeline` – Multi-plant dicot setup
- `NewPipeline` – Description  # ADD THIS
```

Add usage example:
```markdown
### Example: New Pipeline

```python
import sleap_roots as sr

pipeline = sr.NewPipeline()
# ... example code
```
\`\`\`
```

### Task 2: Trait Computation Changed

Update CHANGELOG.md:
```markdown
## [Unreleased]

### Changed

- Improved angle calculation accuracy by using dot product clipping
- **BREAKING**: Angle units changed from radians to degrees
  - Migration: Use `np.radians(angle)` to convert back
```

Update trait docs (HackMD):
- Document new calculation method
- Note any changes to expected values
- Add validation data if available

### Task 3: Dependency Updated

Update README.md:
```markdown
## Installation

Requires:
- Python 3.7+
- sleap-io >= 0.0.11  # Updated version
```

Update environment.yml if needed.

### Task 4: Breaking Change

Document in CHANGELOG.md:
```markdown
## [2.0.0] - 2025-XX-XX

### Changed

- **BREAKING**: `compute_traits()` now requires `frame_index` parameter
  - Migration: Add `frame_index=0` to maintain previous behavior
  - Reason: Support multi-frame analysis
```

Update README.md examples to use new API.

## Documentation Style Guide

### Markdown Formatting

- Use `#` for main heading (one per file)
- Use `##` for sections
- Use `###` for subsections
- Use code blocks with language: \`\`\`python
- Use inline code for: `variable_names`, `function_names`, `'strings'`

### Code Examples

- Always test code examples before committing
- Use real data from `tests/data/` when possible
- Include expected output in comments
- Keep examples concise but complete

```python
# Good example - complete and tested
import sleap_roots as sr

series = sr.Series.load(
    series_name="919QDUH",
    primary_path="tests/data/canola_7do/919QDUH.primary.slp"
)
pipeline = sr.DicotPipeline()
traits = pipeline.compute_plant_traits(series)
# Output: DataFrame with 52+ trait columns
```

### Docstring Style

Use Google-style docstrings:

```python
def get_root_lengths(points: np.ndarray) -> float:
    """Calculate total Euclidean length of root from points.

    Computes the sum of distances between consecutive points along
    the root skeleton.

    Args:
        points: Array of shape (n, 2) containing x,y coordinates.

    Returns:
        Total length as a float (in pixels or mm if calibrated).

    Raises:
        ValueError: If points array is empty or malformed.

    Examples:
        >>> pts = np.array([[0, 0], [3, 4], [6, 8]])
        >>> get_root_lengths(pts)
        10.0
    """
    pass
```

## Tools for Documentation

### Generate Trait List

```python
# List all traits from all pipelines
from sleap_roots import trait_pipelines
import inspect

for name, cls in inspect.getmembers(trait_pipelines):
    if inspect.isclass(cls) and name.endswith('Pipeline'):
        try:
            pipeline = cls()
            traits = pipeline.get_trait_definitions()
            print(f"\n{name}: {len(traits)} traits")
            for t in traits:
                print(f"  - {t.name}")
        except:
            pass
```

### Check Documentation Coverage

```bash
# Check module docstrings
python -c "
import sleap_roots
import pkgutil

for importer, modname, ispkg in pkgutil.iter_modules(sleap_roots.__path__):
    module = __import__(f'sleap_roots.{modname}')
    mod = getattr(module, modname)
    if mod.__doc__:
        print(f'✓ {modname}')
    else:
        print(f'✗ {modname} - missing docstring')
"
```

### Find Outdated Examples

```bash
# Find API patterns that might be outdated
grep -n "Series.load" README.md
grep -n "compute_traits" README.md

# Compare with current API
python -c "
from sleap_roots import Series
import inspect
sig = inspect.signature(Series.load)
print('Current signature:', sig)
"
```

## Documentation Checklist for PRs

Before creating a PR with code changes:

- [ ] README.md updated if API changed
- [ ] Docstrings added/updated for new/changed functions
- [ ] CHANGELOG.md updated with changes
- [ ] Examples tested and working
- [ ] Breaking changes clearly documented
- [ ] Type hints added to new functions

## External Documentation

### HackMD Trait Docs

Update at: https://hackmd.io/DMiXO2kXQhKH8AIIcIy--g

When to update:
- New traits added
- Trait calculations changed
- New pipeline added

What to include:
- Trait name and description
- Computation method/formula
- Units (pixels, mm, degrees, etc.)
- Biological meaning
- Example values

### GitHub Releases

When creating a release:
- Copy CHANGELOG.md section to release notes
- Highlight major features
- Note breaking changes prominently
- Link to full CHANGELOG.md

## Automation Ideas

### Pre-commit Hook for Docs

```bash
# Check if code changes require doc updates
if git diff --cached | grep -q "def "; then
    echo "⚠️  Code changes detected - update documentation!"
    echo "   - Update docstrings"
    echo "   - Update CHANGELOG.md"
    echo "   - Update README.md if API changed"
fi
```

### CI Check for Example Code

```yaml
# .github/workflows/docs.yml
name: Docs
on: [pull_request]
jobs:
  check-examples:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Extract and test README examples
        run: |
          # Extract code blocks
          # Execute them
          # Fail if they error
```

## Related Commands

- `/changelog` - Update CHANGELOG.md specifically
- `/doc-traits` - Auto-generate trait documentation (future)
- `/pr-description` - Document changes in PR
- `/release` - Update docs for release

## Tips

1. **Update continuously**: Don't batch documentation updates
2. **Test examples**: Always verify code examples work
3. **Link to code**: Reference specific modules/functions
4. **Be specific**: "Fixed angle calculation" not "Fixed bug"
5. **Think of users**: What would help someone understand this?
6. **Keep consistent**: Follow existing patterns
7. **Link external docs**: Don't duplicate, link to HackMD/wiki

## Common Issues

### "README example doesn't work"

**Cause:** API changed but docs not updated

**Fix:**
```bash
# Test all examples
python -c "$(grep -A 10 'import sleap_roots' README.md)"
# Update examples to match current API
```

### "Docstring format is wrong"

**Cause:** Not following Google-style

**Fix:**
```bash
# Check with pydocstyle
pydocstyle --convention=google sleap_roots/module.py

# Fix to match Google-style template (see above)
```

### "CHANGELOG is empty for recent changes"

**Cause:** Forgot to update CHANGELOG during development

**Fix:**
```bash
# Review commits since last release
git log v0.1.4..HEAD --oneline

# Add to CHANGELOG.md [Unreleased] section
# Use /changelog command for help
```

### "Trait count doesn't match docs"

**Cause:** New traits added but docs not updated

**Fix:**
```python
# Count current traits
pipeline = DicotPipeline()
print(f"DicotPipeline has {len(pipeline.get_trait_definitions())} traits")

# Update README.md and HackMD docs
```