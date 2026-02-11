# Tasks: Normalize Confidence Badge Display

## 1. Tests (TDD)

- [x] 1.1 Add test that `confidence_to_hex` returns valid hex
- [x] 1.2 Add test that viridis produces distinct colors for low/high
- [x] 1.3 Add test that scan-level confidence is normalized to 0-1
- [x] 1.4 Add test that HTML badge shows "Score:" label
- [x] 1.5 Add test that HTML includes tooltip text
- [x] 1.6 Add test that badge uses viridis hex color

## 2. Implementation

- [x] 2.1 Add `confidence_to_hex()` utility in generator.py
- [x] 2.2 Collect global min/max confidence during generation
- [x] 2.3 Normalize scan-level and frame-level confidences after rendering
- [x] 2.4 Update template badge to use viridis color, "Score:" label, and tooltip
- [x] 2.5 Update template frame stats to show "Score:" and tooltip
- [x] 2.6 Remove green/yellow/red CSS classes (replaced by inline viridis)

## 3. Validation

- [x] 3.1 Run all viewer tests (58 passed)

## Files Changed

- `sleap_roots/viewer/generator.py` - Added `confidence_to_hex()`, normalization pass
- `sleap_roots/viewer/templates/viewer.html` - Viridis badge, "Score:" label, tooltip
- `tests/test_viewer.py` - Added `TestNormalizedConfidenceBadge` class with 6 tests