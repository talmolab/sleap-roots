## 1. Video Path Remapping Fix (Critical Bug)

- [x] 1.1 Write test for multi-timepoint video remapping
- [x] 1.2 Fix `_find_and_remap_video()` to use multiple path components
- [x] 1.3 Verify backwards compatibility with existing single-directory tests

## 2. Plant Name Display

- [x] 2.1 Extract plant name (QR code) from matched image directory path
- [x] 2.2 Add plant_name field to scan data structure
- [x] 2.3 Update gallery cards to display plant name prominently
- [x] 2.4 Display scan ID as secondary info (tooltip/small text)
- [x] 2.5 Test plant name extraction with various directory structures

## 3. Gallery Timepoint Grouping

- [x] 3.1 Add group extraction logic to generator (parse parent directory)
- [x] 3.2 Include group metadata in viewer data structure
- [x] 3.3 Update JavaScript to render grouped sections
- [x] 3.4 Add CSS for group headers and collapsible sections
- [x] 3.5 Test gallery grouping with multi-timepoint data

## 4. Timepoint Filtering Fix

- [x] 4.1 Add `--timepoint` CLI argument
- [x] 4.2 Write failing tests for flat prediction directory filtering
  - [x] Test: Basic group field filtering (`test_filter_scans_by_group_basic`)
  - [x] Test: Multiple `--timepoint` flags use OR logic (`test_filter_scans_by_group_multiple_patterns_or_logic`)
  - [x] Test: Pattern matches nothing → warning + empty viewer (`test_filter_scans_warns_when_no_matches`)
  - [x] Test: Case insensitivity (`day0` matches `Day0`) (`test_filter_scans_case_insensitive`)
  - [x] Test: Scan with failed video remap (group=None) excluded (`test_filter_scans_excludes_none_group`)
  - [x] Test: scans_data and scans_template stay in sync after filter (`test_filter_scans_keeps_data_and_template_in_sync`)
- [x] 4.3 Move filter to post-processing (use discovered `group` field)
  - Created `_filter_scans_by_timepoint()` helper function
  - Filter runs AFTER scan processing when `group` is available
- [x] 4.4 Add warning when no scans match pattern
- [x] 4.5 Document `--timepoint` option in docstrings and help

## 5. Fix Stale _index After Filtering (Critical Bug)

**Problem**: After timepoint filtering, scan cards have stale `_index` values that don't match
the filtered `scansData` array positions. Clicking cards calls `openScan(16)` but `scansData[16]`
doesn't exist.

- [x] 5.1 Write TDD test: `test_filter_scans_reassigns_index_values`
- [x] 5.2 Write TDD test: `test_filter_scans_indices_match_data_positions`
- [x] 5.3 Fix: Reassign `_index` values in `_filter_scans_by_timepoint()` after filtering
- [x] 5.4 Run tests and verify fix (142 tests pass)
- [x] 5.5 Regenerate viewers and verify clicking works

## 6. Documentation & Cleanup

- [x] 6.1 Update prediction-viewer.md with new features
- [x] 6.2 Update spec with new requirements (documented in design.md)
- [x] 6.3 Run full test suite (142 tests pass)
- [x] 6.4 Verify fix with Alfalfa GWAS dataset