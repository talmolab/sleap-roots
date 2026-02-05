# Fix Viewer Code Quality

## Why

PR #142 received Copilot code review feedback identifying unused imports, silent exception handling, case-sensitive file extension matching, and incomplete test coverage. These issues affect correctness on case-sensitive filesystems and reduce code maintainability.

## What Changes

- Remove unused imports (`numpy`, `seaborn`, `Optional`) and unused variable (`ax`) in renderer.py
- Remove unused imports (`io`, `ScanInfo`) in test_viewer.py
- Add warning message to bare `except: pass` clause in `_find_and_remap_video()`
- Add case-insensitive image extension matching for cross-platform compatibility
- Improve `sort_key` function to handle non-numeric filenames with stable ordering
- Implement placeholder test for `FrameLimitExceededError`

## Impact

- Affected specs: `html-prediction-viewer` (MODIFIED requirement for video source support)
- Affected code: `sleap_roots/viewer/renderer.py`, `sleap_roots/viewer/generator.py`, `tests/test_viewer.py`