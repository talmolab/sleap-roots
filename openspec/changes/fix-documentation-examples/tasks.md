# Tasks: Fix Documentation Examples

## Phase 1: Verify Current State

- [ ] Read actual API signatures from source code
  - `sleap_roots/angle.py` - verify available functions
  - `sleap_roots/lengths.py` - verify available functions
  - `sleap_roots/series.py` - verify `Series.get_primary_points()` signature
- [ ] Document actual function signatures for reference
- [ ] Identify all broken imports and function calls
- [ ] Test each example against actual API

**Validation**: Complete inventory of all corrections needed ✅

---

## Phase 2: Fix Critical Bugs

### Fix quickstart.md Example 3

- [ ] Remove broken import: `from sleap_roots.angles import get_primary_angle`
- [ ] Add correct imports:
  ```python
  from sleap_roots.angle import get_node_ind, get_root_angle
  from sleap_roots.lengths import get_root_lengths
  ```
- [ ] Replace example code with working implementation:
  - Use `get_node_ind(pts, proximal=True)` to find node index
  - Use `get_root_angle(pts, node_ind, proximal=True, base_ind=0)` for angle
  - Show all arguments explicitly
- [ ] Add explanatory comments about what each function does
- [ ] Update output examples to match actual return values

**Validation**: Example 3 uses only functions that exist in the API ✅

---

## Phase 3: Improve Argument Clarity

### Update All Examples to Use Explicit Arguments

- [ ] Review index.md Quick Example
  - Verify `Series.load()` shows all used arguments explicitly
  - Add comment about DicotPipeline defaults if using `DicotPipeline()`
  - Ensure `write_csv=True` is explicit (already is)
- [ ] Review quickstart.md Example 1
  - Verify all `Series.load()` arguments explicit (already good)
  - Verify `compute_plant_traits()` arguments explicit (already good)
- [ ] Review quickstart.md Example 2
  - Make `h5s=True` explicit in `load_series_from_slps()` (already is)
  - Verify `compute_batch_traits()` arguments explicit (already good)
- [ ] Review quickstart.md Example 3 (after fixes)
  - Ensure `frame_idx=0` is explicit in `get_primary_points()`
  - Ensure `proximal=True` is explicit in `get_node_ind()`
  - Ensure `base_ind=0` is explicit in `get_root_angle()`

**Validation**: All function calls use named arguments for clarity ✅

---

## Phase 4: Add Explanatory Comments

- [ ] Add comment before `DicotPipeline()` explaining default parameters
- [ ] Add comment in Example 3 explaining the angle computation workflow
- [ ] Add comment about return types where not obvious
- [ ] Ensure comments are concise and helpful, not verbose

**Validation**: Examples are self-documenting with helpful comments ✅

---

## Phase 5: Test and Validate

- [ ] Create test script with all examples from index.md
- [ ] Create test script with all examples from quickstart.md
- [ ] Run both test scripts to verify they execute without errors
- [ ] Verify imports resolve correctly
- [ ] Verify function calls match actual signatures
- [ ] Build documentation and check for new warnings:
  ```bash
  uv run mkdocs build 2>&1 | grep -E "WARNING|ERROR"
  ```

**Validation**: All examples execute successfully, clean doc build ✅

---

## Phase 6: Verify Against Real Data

- [ ] Run Example 1 with actual test data (canola_7do/919QDUH)
- [ ] Run Example 2 with actual test data (soy_6do)
- [ ] Run Example 3 with actual series to verify output
- [ ] Confirm all examples produce expected results

**Validation**: Examples work with real data, not just syntactically correct ✅

---

## Phase 7: Final Review

- [ ] Review all changes for clarity and correctness
- [ ] Ensure consistent style across all examples
- [ ] Verify no redundancy between index.md and quickstart.md beyond what's intentional
- [ ] Check that examples progress from simple to complex appropriately
- [ ] Confirm all success criteria from proposal are met

**Validation**: Documentation examples are correct, clear, and user-friendly ✅

---

## Dependencies

- Phase 1 must complete before Phase 2 (need to know actual API)
- Phase 2 must complete before Phase 3 (fix bugs before improving clarity)
- Phases 3-4 can be done in parallel
- Phase 5 depends on Phases 2-4 (test after changes)
- Phase 6 depends on Phase 5 (verify functionality)
- Phase 7 is final review

## Notes

- Focus on correctness first, clarity second
- All changes are documentation-only, no code modifications
- Examples should reflect actual test usage patterns from codebase
- Explicit arguments help new users understand what's customizable