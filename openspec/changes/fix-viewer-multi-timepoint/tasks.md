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

## 4. Timepoint Filtering

- [x] 4.1 Add `--timepoint` CLI argument
- [x] 4.2 Implement pattern matching filter in scan discovery
- [x] 4.3 Write tests for timepoint filtering
- [x] 4.4 Document `--timepoint` option in docstrings and help

## 5. Documentation & Cleanup

- [ ] 5.1 Update prediction-viewer.md with new features
- [ ] 5.2 Update spec with new requirements
- [ ] 5.3 Run full test suite
- [ ] 5.4 Verify fix with Alfalfa GWAS dataset