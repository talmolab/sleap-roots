## Context

The HTML viewer is used to review SLEAP predictions for plant root phenotyping experiments. Multi-timepoint experiments track the same plants across multiple days to observe root growth over time. Each timepoint is a separate scan with its own `.slp` predictions file.

**Current failure mode**:
```
Embedded path: /workspace/images/Wave1/Day0_2025-11-27/Fado_1/1.jpg
Code extracts: Fado_1
Glob search:   **/Fado_1
Match found:   ./images/Wave1/Day7_2025-12-04/Fado_1  (WRONG DAY!)
```

**Stakeholders**: Researchers reviewing predictions before running trait extraction

## Goals / Non-Goals

### Goals
- Fix video path remapping to correctly match image directories in multi-timepoint datasets
- Make gallery browsing intuitive for multi-timepoint experiments
- Support filtering to specific timepoints when reviewing

### Non-Goals
- Automatic timepoint detection/grouping by date parsing (keep it simple)
- Complex hierarchical navigation (multiple levels of grouping)
- Comparison views between timepoints

## Decisions

### Decision 1: Path Matching Strategy

**Decision**: Use progressive path component matching, trying more specific matches first.

**Approach**:
1. Extract up to 4 path components from embedded path (e.g., `Wave1/Day0/Fado_1`)
2. Try matching with all 4 components first (most specific)
3. Fall back to fewer components if not found
4. Preserve current single-component fallback for backwards compatibility

**Rationale**:
- Preserves unique path context without requiring exact path knowledge
- Works with varying directory structures
- Backwards compatible with simpler datasets

**Alternatives considered**:
- Full path matching: Too brittle, paths often differ between pipeline and local
- Parent directory name only: Would miss cases like `images/Fado_1` vs `images_backup/Fado_1`

### Decision 2: Display Plant Name Instead of Scan ID

**Decision**: Use plant name (QR code) from image directory path instead of scan ID in gallery.

**Approach**:
- Extract plant name from the image directory path (leaf directory, e.g., `Fado_1`)
- Display plant name prominently on scan cards
- Keep scan ID as secondary identifier (tooltip or small text)
- This is derived during video remapping when we find the local image directory

**Rationale**:
- Plant names (QR codes) are meaningful to researchers
- Scan IDs are internal pipeline identifiers, not useful for browsing
- Same plant across timepoints becomes visually identifiable

**Implementation**:
- Add `plant_name` field to scan data structure
- Extract from matched local image directory path
- Display in gallery cards and frame view header

### Decision 3: Gallery Organization

**Decision**: Add optional grouping by parent directory (timepoint folder).

**Approach**:
- Parse the parent directory name (e.g., `Day0_2025-11-27`) from scan paths
- Group scan cards by this parent directory in the gallery
- Display group headers (collapsible sections)
- Default to grouped view when multiple unique parent directories exist

**Implementation**:
- Add `group_by_parent` parameter to generator (default: `True`)
- JavaScript in viewer handles collapsible group rendering
- Each group shows count of scans

### Decision 4: Timepoint Filtering

**Decision**: Add `--timepoint` argument to filter scans by timepoint pattern.

**Problem discovered**: Initial implementation filtered scan paths before processing, but for
flat prediction directories (e.g., `predictions/scan_123.slp`), the timepoint info is only
in the embedded video paths discovered during video remapping.

**Approach (revised)**:
- Accept glob-style pattern: `--timepoint "Day0*"` or `--timepoint "Day3*"`
- Filter AFTER processing, using the discovered `group` field from video remapping
- Multiple patterns supported via multiple `--timepoint` flags (OR logic)
- Case-insensitive matching for user-friendliness

**Implementation**:
1. Remove early scan path filtering (doesn't work for flat dirs)
2. Process all scans (extracting group during video remapping)
3. Filter `scans_data` and `scans_template` by matching `group` against patterns
4. Warn user if no scans match the pattern

**Edge cases**:
- `group=None` (video remap failed): Excluded from filter matches
- Empty pattern list: No filtering, include all scans
- All scans filtered out: Empty viewer with warning
- Case sensitivity: Use `fnmatch` with `.lower()` for case-insensitive

**Rationale**:
- Works with both nested and flat prediction directory structures
- Uses already-computed group info (no duplicate .slp loading)
- Simple implementation with clear data flow

## Risks / Trade-offs

### Risk: Gallery grouping makes viewer complex
**Mitigation**: Keep grouping simple (single level), use progressive enhancement (works without JS grouping)

### Risk: Path matching may still fail for unusual structures
**Mitigation**: Fall back gracefully, log warnings about failed remaps, document expected path structures

### Risk: Large multi-timepoint experiments generate huge viewers
**Mitigation**: Already have `--max-frames` and `--no-limit`; timepoint filtering reduces scope

### BUG DISCOVERED: Stale _index after filtering (Critical)

**Problem**: After timepoint filtering, clicking on scan cards fails because `_index` values
are stale. The scan cards have their original indices (e.g., `onclick="openScan(16)"`),
but `scansData` has been filtered to a shorter array where index 16 doesn't exist.

**Root cause**: `_filter_scans_by_timepoint()` filters the arrays but doesn't reassign
`_index` values to match the new array positions.

**Fix**: After filtering, reassign `_index` values:
```python
for i, scan in enumerate(filtered_template):
    scan["_index"] = i
```

## Implementation Notes

### Video Remapping Fix (`generator.py:_find_and_remap_video`)

Current:
```python
img_dir_name = first_path.parent.name  # Gets "Fado_1" only
for subdir in search_dir.glob(f"**/{img_dir_name}"):
```

Fixed:
```python
path_parts = first_path.parent.parts
for num_parts in range(min(4, len(path_parts)), 0, -1):
    suffix = "/".join(path_parts[-num_parts:])
    for subdir in search_dir.glob(f"**/{suffix}"):
```

### Gallery Grouping (JavaScript)

Add grouping data to scan objects:
```javascript
{
  id: "scan_12305180",
  group: "Day0_2025-11-27",  // Parent directory name
  ...
}
```

Render grouped sections:
```html
<div class="timepoint-group">
  <h3 class="group-header">Day0_2025-11-27 (15 scans)</h3>
  <div class="scan-cards">...</div>
</div>
```

## Open Questions

1. **Sort order within groups**: Should scans within a timepoint group be sorted alphabetically or by discovery order?
   - Proposal: Alphabetically by plant name (genotype/plant ID)

2. **Group order**: Should timepoint groups be sorted chronologically or alphabetically?
   - Proposal: Alphabetically (date format like `Day0_2025-11-27` will sort correctly)