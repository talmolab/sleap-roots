## 1. Tests (TDD)

- [x] 1.1 Write test for case-insensitive image extension discovery
- [x] 1.2 Write test for non-numeric filename sorting
- [x] 1.3 Write test for warning on video existence check failure
- [x] 1.4 Implement placeholder test for FrameLimitExceededError with mock

## 2. Implementation

- [x] 2.1 Remove unused imports in renderer.py (numpy, seaborn, Optional already done)
- [x] 2.2 Remove unused variable `ax` in renderer.py
- [x] 2.3 Remove unused import `io` in test_viewer.py (ScanInfo kept - used in new mock test)
- [x] 2.4 Add case-insensitive extension matching in generator.py
- [x] 2.5 Improve sort_key for non-numeric filenames in generator.py
- [x] 2.6 Add warning to bare except clause in generator.py

## 3. Verification

- [x] 3.1 Run all tests and verify they pass (64 passed)
- [x] 3.2 Run black formatting check (passed)
- [x] 3.3 Run pydocstyle check (passed)