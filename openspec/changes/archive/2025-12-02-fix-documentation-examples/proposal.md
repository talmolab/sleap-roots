# Proposal: Fix Documentation Examples

## Problem Statement

The documentation contains code examples that have critical issues preventing users from successfully following the tutorials:

### 1. **Broken Import in quickstart.md**
Example 3 (Individual Trait Functions) imports a non-existent function:
```python
from sleap_roots.angles import get_primary_angle  # ❌ DOES NOT EXIST
```

**Issues:**
- Module is named `angle` (singular), not `angles` (plural)
- Function `get_primary_angle()` does not exist in the codebase
- The actual API provides `get_node_ind()` and `get_root_angle()` which require different usage patterns

This will cause immediate ImportError for users trying to follow the tutorial.

### 2. **Inconsistent Argument Patterns**
While most examples use explicit `arg=value` format, there are opportunities to improve clarity:
- Pipeline initialization uses defaults implicitly: `sr.DicotPipeline()`
- Some functions use positional arguments where named would be clearer
- New users may not understand what defaults are being used

### 3. **Missing Function Parameters**
Examples don't always show all available parameters, which can confuse users about:
- What options are available
- What the defaults actually are
- How to customize behavior

### 4. **Lack of API Verification**
Examples appear to have been written without verifying against the actual codebase, leading to:
- Non-existent functions being referenced
- Incorrect module names
- Return value assumptions that may not match reality

## Proposed Solution

### Phase 1: Fix Critical Bugs
1. **Replace broken Example 3** in quickstart.md with working code using actual API:
   - Use `get_node_ind()` to find proximal/distal node indices
   - Use `get_root_angle()` to calculate angles
   - Show proper usage pattern with all required arguments explicit

2. **Verify all imports** are correct:
   - `sleap_roots.angle` (singular) not `angles`
   - `sleap_roots.lengths` (plural) is correct

### Phase 2: Improve Argument Clarity
1. **Make all critical arguments explicit** in examples:
   - Show `proximal=True` explicitly in `get_node_ind()` calls
   - Show `base_ind=0` explicitly in `get_root_angle()` calls
   - Use named arguments instead of positional where it improves clarity

2. **Add comments explaining defaults** where pipeline/class initialization uses them:
   ```python
   # DicotPipeline uses default node names: primary_name="base", lateral_name="lateral"
   pipeline = sr.DicotPipeline()
   ```

### Phase 3: Consolidate and Verify
1. **Test all examples** against actual codebase:
   - Verify imports work
   - Verify function signatures match
   - Verify return types match usage

2. **Reduce redundancy** between index.md and quickstart.md:
   - Keep index.md as minimal teaser (current approach is good)
   - Keep quickstart.md as comprehensive tutorial
   - Ensure they complement rather than duplicate

## Impact

### User Impact
- **High**: Users can successfully follow tutorials without hitting ImportErrors
- **High**: Examples demonstrate actual working API usage
- **Medium**: Clearer argument patterns help users understand customization options

### Development Impact
- **Low**: Only documentation changes, no code modifications
- **Low**: Examples already exist, just need corrections

### Documentation Impact
- **High**: Critical fix for broken tutorials
- **Medium**: Improved clarity and usability

## Success Criteria

1. ✅ All code examples in docs/index.md can be copied and executed without errors
2. ✅ All code examples in docs/getting-started/quickstart.md can be executed without errors
3. ✅ All imports reference actual modules and functions that exist
4. ✅ Function calls use correct signatures with explicit critical arguments
5. ✅ Documentation builds without new warnings
6. ✅ Examples demonstrate realistic usage patterns from actual test code

## Related Work

- Builds on `simplify-user-documentation` change which improved overall doc organization
- Complements `documentation-quality` spec for zero-warning builds
- Supports user-facing documentation standards from `documentation-organization` spec

## Alternatives Considered

### Alternative 1: Remove Example 3 Entirely
**Rejected**: Users need to understand how to use individual trait functions for custom analyses.

### Alternative 2: Keep Generic Examples Without Specifics
**Rejected**: User specifically requested explicit `arg=value` format to avoid confusion about defaults.

### Alternative 3: Create New Wrapper Function `get_primary_angle()`
**Rejected**: Would add API complexity just to support a documentation example. Better to teach actual API.

## Open Questions

None - the actual API is clear and well-documented, we just need examples to match it.