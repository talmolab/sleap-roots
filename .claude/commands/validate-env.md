# Validate Development Environment

Check that your development environment is correctly set up and ready for sleap-roots development.

## Quick Start

```bash
# Run full environment validation
```

This checks:
1. Python version
2. Conda/mamba environment
3. Package installation
4. Required dependencies
5. Git LFS configuration
6. Test data availability
7. Import smoke test

## What Gets Checked

### 1. Python Version
- ✅ Python version matches .python-version file
- ❌ Python version mismatch

### 2. uv Installation
- ✅ uv installed and accessible
- ✅ uv version compatible

### 3. Package Installation
- ✅ Dependencies synced via uv.lock
- ✅ All core dependencies present
- ✅ Dev dependencies present (pytest, black, etc.)

### 4. Git LFS
- ✅ Git LFS installed
- ✅ LFS filters configured
- ✅ Test data files downloaded (not pointers)

### 5. Test Data
- ✅ Test data files exist
- ✅ Files are actual data (not LFS pointers)
- ✅ Files can be loaded

### 6. Smoke Test
- ✅ Package can be imported
- ✅ Core modules load without errors
- ✅ Basic pipeline can be instantiated

## Expected Output

### ✅ Fully Configured Environment

```
================================
Environment Validation
================================

[1/7] Python Version
✅ Python 3.11 (matches .python-version)

[2/7] uv Installation
✅ uv 0.5.0 installed

[3/7] Package Installation
✅ Dependencies synced from uv.lock
   Location: /Users/you/repos/sleap-roots/.venv
✅ All core dependencies installed:
   numpy 1.24.3
   pandas 2.0.1
   h5py 3.8.0
   sleap-io 0.0.11
   ... (10 more)
✅ All dev dependencies installed:
   pytest 7.4.0
   black 23.3.0
   pydocstyle 6.3.0

[4/7] Git LFS
✅ Git LFS 3.3.0 installed
✅ LFS filters configured
✅ LFS tracking: *.h5, *.slp

[5/7] Test Data
✅ Test data directory exists: tests/data/
✅ Checking test files (23 files)...
   ✅ tests/data/canola_7do/919QDUH.h5 (45.2 MB)
   ✅ tests/data/canola_7do/919QDUH.primary.slp (1.2 MB)
   ... (21 more)
✅ All test data files downloaded (not LFS pointers)

[6/7] Smoke Test
✅ Package imports successfully
✅ Core modules loaded: lengths, angles, tips, bases, trait_pipelines
✅ DicotPipeline instantiated successfully

================================
✅ ENVIRONMENT VALID
================================

Your environment is ready for development! 🚀

Next steps:
  - Run tests: uv run pytest tests/
  - Check formatting: uv run black --check sleap_roots tests
  - Start developing!
```

### ❌ Issues Found

```
================================
Environment Validation
================================

[1/7] Python Version
✅ Python 3.11.0

[2/7] uv Installation
❌ uv not found

FIX: Install uv with:
     curl -LsSf https://astral.sh/uv/install.sh | sh

[3/7] Package Installation
⏭  Skipped (environment not active)

[4/7] Git LFS
✅ Git LFS 3.3.0 installed
❌ LFS filters not configured

FIX: Run: git lfs install

[5/7] Test Data
⚠️  Test data directory exists but some files are LFS pointers:
    ❌ tests/data/rice_3do/YR39SJX.h5 (130 bytes - LFS pointer!)

FIX: Pull LFS files with:
     git lfs pull

[6/7] Smoke Test
⏭  Skipped (package not installed)

================================
❌ ENVIRONMENT HAS ISSUES
================================

Found 3 issues. Fix them using the commands above.
```

## Common Issues & Fixes

### Issue: "uv not found"

**Cause:** uv not installed

**Fix:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Issue: "Dependencies not synced"

**Cause:** Virtual environment not created or dependencies not installed

**Fix:**
```bash
uv sync
```

### Issue: "Git LFS not configured"

**Cause:** Git LFS not installed or initialized

**Fix:**
```bash
# macOS
brew install git-lfs

# Ubuntu
sudo apt-get install git-lfs

# Then initialize
git lfs install
```

### Issue: "Test data files are LFS pointers"

**Cause:** LFS files not downloaded (shows as small ~130 byte files)

**Fix:**
```bash
git lfs pull
```

### Issue: "Import errors during smoke test"

**Cause:** Missing dependencies or broken installation

**Fix:**
```bash
# Recreate virtual environment
rm -rf .venv
uv sync
```

## When to Run This

### Initial Setup
Run after cloning the repository for the first time.

### After Environment Changes
- After updating `pyproject.toml` dependencies
- After `uv lock`
- After installing new dependencies

### Troubleshooting
- When tests fail unexpectedly
- When imports don't work
- When getting "module not found" errors
- After switching machines

### Onboarding
- Help new contributors verify setup
- Include in onboarding documentation

## Detailed Checks Explained

### Python Version Check
```bash
uv run python --version
# Should match the version in .python-version file
```

The Python version is automatically managed by uv based on the .python-version file.

### Virtual Environment Check
```bash
ls -la .venv
# Should show the .venv directory if it exists
```

### Dependency Sync Check
```bash
uv tree
# Should show all dependencies installed from uv.lock
```

Dependencies are managed by uv and installed in the .venv directory.

### Git LFS Check
```bash
git lfs install
git lfs ls-files
# Should show: *.h5 and *.slp files tracked
```

Test data (898 MB) is stored in Git LFS to keep the repo size manageable.

### LFS Pointer Detection
```bash
# LFS pointer files are ~130 bytes and start with:
# "version https://git-lfs.github.com/spec/v1"

# Actual files are larger (MB range)
```

### Smoke Test
```python
import sleap_roots
from sleap_roots import DicotPipeline

pipeline = DicotPipeline()
print("✅ Basic functionality works")
```

## Platform-Specific Notes

### macOS
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Git LFS: `brew install git-lfs`
- Everything else should work

### Ubuntu
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Git LFS: `sudo apt-get install git-lfs`
- May need build tools: `sudo apt-get install build-essential`

### Windows
- Install uv: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Git LFS: Download from https://git-lfs.github.com/
- Use Git Bash or PowerShell for git commands

## Integration with Other Commands

```bash
# 1. First time setup
git clone https://github.com/talmolab/sleap-roots.git
cd sleap-roots

# 2. Install dependencies
uv sync

# 3. Validate environment
/validate-env
# Fix any issues it identifies

# 4. Run tests to verify
/test

# 5. Start development!
```

## Automated Fix (Advanced)

For automated setup, you can chain commands:

```bash
# Auto-fix common issues
uv sync
git lfs install
git lfs pull

# Then validate
/validate-env
```

## Troubleshooting Guide

### "uv: command not found"

Install uv:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### "Dependency resolution is slow"

uv is already fast, but you can try:
```bash
# Clear cache and re-sync
uv cache clean
uv sync
```

### "Test data files are empty"

Git LFS issue - files are pointers instead of actual data:
```bash
git lfs install
git lfs pull --include="tests/data/**"
```

### "Import errors for sleap_io"

Dependency version mismatch:
```bash
uv lock --upgrade-package sleap-io
uv sync
```

### "Environment exists but validation fails"

Environment may be corrupted - recreate it:
```bash
rm -rf .venv
uv sync
```

## Output Format

The command outputs:
- ✅ Green checkmark: Validation passed
- ❌ Red X: Validation failed (with fix instructions)
- ⚠️  Yellow warning: Non-critical issue
- ⏭  Skipped: Check skipped due to previous failure

## Related Commands

- `/setup-env` - Interactive environment creation (future)
- `/test` - Run tests (requires valid environment)
- `/run-ci-locally` - Run all CI checks (requires valid environment)

## Tips

1. **Run after long breaks**: Environment can drift over time
2. **Before filing bug reports**: Attach validation output to bugs
3. **Team onboarding**: Send validation output to verify new contributors
4. **CI debugging**: Compare local validation with CI environment
5. **After system updates**: OS/Python updates can break environment