# Proposal: Fix Stale _index After Timepoint Filtering

## Problem Statement

After applying `--timepoint` filtering to the HTML viewer, clicking on scan cards fails to open the scan detail view. The click handler calls `openScan(index)` but the index values are stale.

### Root Cause

When filtering scans by timepoint pattern:

1. Scans are processed and assigned `_index` values based on their position in the original array
2. Filtering removes non-matching scans from both `scans_data` and `scans_template`
3. **The `_index` values are NOT updated** to reflect new positions in filtered arrays
4. JavaScript `openScan(16)` tries to access `scansData[16]` which doesn't exist

### Evidence

From `viewer_Day9.html`:
```html
<div class="scan-card" data-scan-index="16" onclick="openScan(16)">
<div class="scan-card" data-scan-index="22" onclick="openScan(22)">
<div class="scan-card" data-scan-index="31" onclick="openScan(31)">
```

But `scansData` only has 42 entries (indices 0-41), not the original 292.

## Proposed Solution

Update `_filter_scans_by_timepoint()` to reassign `_index` values after filtering:

```python
def _filter_scans_by_timepoint(scans_data, scans_template, patterns):
    # ... existing filtering logic ...

    # Filter both lists
    filtered_data = [s for s in scans_data if s["name"] in matching_names]
    filtered_template = [s for s in scans_template if s["name"] in matching_names]

    # FIX: Reassign _index values to match new array positions
    for i, scan in enumerate(filtered_template):
        scan["_index"] = i

    return filtered_data, filtered_template
```

## Impact

- **Scope**: `sleap_roots/viewer/generator.py` - `_filter_scans_by_timepoint()` function
- **Risk**: Low - isolated change to filtering function
- **Backwards Compatibility**: No impact on existing behavior (filtering is opt-in via `--timepoint`)

## Test Plan (TDD)

### Test 1: Verify indices are reassigned after filtering
```python
def test_filter_scans_reassigns_index_values():
    """Test that _index values are updated to match filtered array positions."""
    scans_data = [
        {"name": "scan1", "group": "Day0"},
        {"name": "scan2", "group": "Day3"},  # Will be filtered out
        {"name": "scan3", "group": "Day0"},
        {"name": "scan4", "group": "Day5"},  # Will be filtered out
    ]
    scans_template = [
        {"name": "scan1", "_index": 0},
        {"name": "scan2", "_index": 1},
        {"name": "scan3", "_index": 2},
        {"name": "scan4", "_index": 3},
    ]

    filtered_data, filtered_template = _filter_scans_by_timepoint(
        scans_data, scans_template, ["Day0*"]
    )

    # After filtering, indices should be 0, 1 (not 0, 2)
    assert filtered_template[0]["_index"] == 0
    assert filtered_template[1]["_index"] == 1
    assert filtered_template[0]["name"] == "scan1"
    assert filtered_template[1]["name"] == "scan3"
```

### Test 2: Verify indices match scansData array positions
```python
def test_filter_scans_indices_match_data_positions():
    """Test that _index in template matches position in data array."""
    scans_data = [{"name": f"scan{i}", "group": f"Day{i%3}"} for i in range(10)]
    scans_template = [{"name": f"scan{i}", "_index": i} for i in range(10)]

    filtered_data, filtered_template = _filter_scans_by_timepoint(
        scans_data, scans_template, ["Day0*"]
    )

    # Each template's _index should be valid index into filtered_data
    for i, scan_t in enumerate(filtered_template):
        assert scan_t["_index"] == i
        assert filtered_data[i]["name"] == scan_t["name"]
```

## Verification

After fix:
1. Regenerate `viewer_Day9.html` with `--timepoint "Day9*"`
2. Open in browser
3. Click on any scan card (e.g., Fado_3)
4. Verify scan detail view opens correctly

## Timeline

- Implementation: ~15 minutes
- Testing: ~10 minutes
- Regenerate viewers: ~30 minutes (background)