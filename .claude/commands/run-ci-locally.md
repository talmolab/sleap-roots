# Run CI Checks Locally

Run the exact same CI checks locally that run on GitHub Actions before pushing your code.

## Quick Start

```bash
# Run all CI checks (linting + tests)
# This mirrors what runs in .github/workflows/ci.yml
```

When you run this command, it will execute:

1. **Black formatting check** - Ensures code follows PEP 8 via Black
2. **Pydocstyle check** - Verifies Google-style docstrings
3. **Pytest** - Runs full test suite
4. **Coverage** - Generates coverage report (if on Ubuntu/macOS)

## What This Command Does

This command runs the exact CI workflow from `.github/workflows/ci.yml`:

```bash
# Step 1: Lint checks
echo "Running Black formatting check..."
uv run black --check sleap_roots tests

echo "Running pydocstyle docstring check..."
uv run pydocstyle --convention=google sleap_roots/

# Step 2: Run tests
echo "Running pytest..."
uv run pytest tests/

# Step 3: Coverage (Ubuntu/macOS only)
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Running pytest with coverage..."
    uv run pytest --cov=sleap_roots --cov-report=xml --cov-report=term tests/
fi
```

## Why Use This?

**Benefits:**
- ✅ Catch CI failures before pushing
- ✅ Faster feedback loop (run locally in ~30s vs waiting for CI)
- ✅ Exactly matches CI environment
- ✅ Prevents "oops, forgot to run Black" commits
- ✅ Reduces CI build queue time

**When to use:**
- Before every `git push`
- Before creating a PR
- After making significant changes
- When you want confidence your PR will pass CI

## Expected Output

### ✅ Success (All Checks Pass)

```
================================
Running CI Checks Locally
================================

[1/4] Black formatting check...
All done! ✨ 🍰 ✨
23 files would be left unchanged.
✅ Black check passed

[2/4] Pydocstyle docstring check...
✅ Pydocstyle check passed

[3/4] Running tests...
================================ test session starts =================================
collected 87 items

tests/test_lengths.py ........                                              [  9%]
tests/test_angles.py .......                                                [ 17%]
...
================================ 87 passed in 12.34s =================================
✅ Tests passed

[4/4] Running coverage...
-------------------------------- coverage: platform darwin, python 3.11 --------------
Name                          Stmts   Miss  Cover
-------------------------------------------------
sleap_roots/lengths.py           45      0   100%
sleap_roots/angles.py            38      0   100%
...
-------------------------------------------------
TOTAL                           892      5    99%
✅ Coverage check passed

================================
✅ ALL CI CHECKS PASSED!
================================

Your code is ready to push! 🚀
```

### ❌ Failure (Checks Failed)

```
================================
Running CI Checks Locally
================================

[1/4] Black formatting check...
would reformat sleap_roots/lengths.py
Oh no! 💥 💔 💥
1 file would be reformatted, 22 files would be left unchanged.
❌ Black check FAILED

[2/4] Pydocstyle docstring check...
sleap_roots/angles.py:42 in public function `get_angle`:
        D400: First line should end with a period
❌ Pydocstyle check FAILED

Stopping checks (2 failures detected)

================================
❌ CI CHECKS FAILED
================================

Fix the issues above before pushing.

Quick fixes:
- Run 'black sleap_roots tests' to auto-fix formatting
- Fix docstring issues manually or use /fix-formatting
```

## Integration with Git Workflow

### Before Pushing

```bash
# 1. Make your changes
git add sleap_roots/lengths.py

# 2. Run CI checks locally
/run-ci-locally

# 3. If checks pass, commit and push
git commit -m "feat: improve length calculation"
git push
```

### Pre-Commit Hook (Optional)

You can set up a git pre-commit hook to run this automatically:

```bash
# Create .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "Running CI checks before commit..."
black --check sleap_roots tests || exit 1
pydocstyle --convention=google sleap_roots/ || exit 1
echo "✅ Pre-commit checks passed"
EOF

chmod +x .git/hooks/pre-commit
```

## Comparison with Individual Commands

| Command | What it does | When to use |
|---------|-------------|-------------|
| `/lint` | Just Black + pydocstyle | Quick formatting check |
| `/test` | Just pytest | Testing specific changes |
| `/coverage` | Pytest + coverage report | Checking test coverage |
| **`/run-ci-locally`** | **All of the above** | **Before pushing/PR** |

## Platform Notes

### Ubuntu/macOS
All checks run, including coverage report uploaded to Codecov format (XML).

### Windows
All checks run except coverage XML report (coverage HTML still available).

## CI Configuration Reference

This command mirrors `.github/workflows/ci.yml`:

```yaml
# Lint job
- name: Run Black
  run: uv run black --check sleap_roots tests

- name: Run pydocstyle
  run: uv run pydocstyle --convention=google sleap_roots/

# Test job
- name: Test with pytest
  run: uv run pytest tests/

# Coverage (Ubuntu only in CI)
- name: Test with pytest (with coverage)
  run: uv run pytest --cov=sleap_roots --cov-report=xml tests/
```

## Troubleshooting

### "Black not found"
```bash
# Install dev dependencies
uv sync
```

### "Tests fail locally but pass in CI"
- Check Git LFS data is pulled: `git lfs pull`
- Verify Python version: `uv run python --version` (should match .python-version)
- Check dependencies are synced: `uv sync`

### "Command takes too long"
- Skip coverage: Just run `/lint` and `/test` separately
- Use pytest filtering: `pytest tests/test_lengths.py` for quick checks
- Coverage adds ~20% overhead, mostly on large test suites

## Tips

1. **Run frequently**: Don't wait until you're done - run after each logical change
2. **Fix formatting first**: Black failures are fastest to fix (`uv run black sleap_roots tests`)
3. **Use `/fix-formatting`**: Auto-fixes most issues instead of checking
4. **Parallel development**: Run this while working on next task (takes ~30s)
5. **CI queue optimization**: Running locally reduces wasted CI cycles

## Related Commands

- `/lint` - Just formatting and docstring checks
- `/test` - Just run tests without coverage
- `/coverage` - Full coverage analysis with HTML report
- `/fix-formatting` - Auto-fix formatting instead of checking