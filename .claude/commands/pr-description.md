# PR Description Template

Use this template when creating pull requests to ensure comprehensive documentation.

## Quick Commands

```bash
# View current PR
gh pr view

# View PR diff
gh pr diff

# List changed files
gh pr diff --name-only

# View specific file changes
gh pr diff <file-path>

# Create PR with description
gh pr create --title "feat: add new pipeline" --body "$(cat description.md)"
```

## PR Description Template

```markdown
## Summary

[Brief 1-2 sentence description of what this PR does]

## Changes

- [Bullet point list of specific changes]
- [Group related changes together]
- [Use present tense: "Add X", "Fix Y", "Update Z"]

## Testing

- [ ] All existing tests pass (`pytest tests/`)
- [ ] Added new tests for new functionality
- [ ] Test coverage maintained or improved (`pytest --cov=sleap_roots tests/`)
- [ ] Tests pass on multiple platforms (Ubuntu, macOS, Windows)
- [ ] Manually tested with real data (if applicable)

## Linting & Formatting

- [ ] Black formatting applied (`black sleap_roots tests`)
- [ ] Pydocstyle checks pass (`pydocstyle --convention=google sleap_roots/`)
- [ ] Google-style docstrings for all new functions/classes

## Coverage

- [ ] Coverage meets thresholds (target: full coverage on new code)
- [ ] Coverage report reviewed (`pytest --cov=sleap_roots --cov-report=html tests/`)
- [ ] No decrease in overall coverage

## Breaking Changes

- [ ] No breaking changes
- [ ] Breaking changes documented below

[If breaking changes, describe migration path and update CHANGELOG.md]

## Related Issues

Closes #[issue number]
Related to #[issue number]

## Trait Computation Changes (if applicable)

- [ ] New traits are biologically meaningful
- [ ] Trait calculations validated against known data
- [ ] Changes maintain reproducibility of published results
- [ ] Updated trait documentation (HackMD) if needed

## Examples/Screenshots

[If UI/visualization changes, include screenshots]
[If new pipeline/traits, include example output]

## Reviewer Notes

[Specific areas you want reviewers to focus on]
[Any concerns or questions you have]
```

## Feature PR Example

```markdown
## Summary

Add lateral root only pipeline for analyzing plants with only lateral root predictions.

## Changes

- **sleap_roots/trait_pipelines.py**: Add `LateralRootPipeline` class
- **tests/test_trait_pipelines.py**: Add tests for lateral root pipeline
- **sleap_roots/__init__.py**: Export `LateralRootPipeline`
- **README.md**: Document lateral root pipeline usage

## Testing

- [x] All existing tests pass
- [x] Added 8 new test cases for `LateralRootPipeline`
- [x] Test coverage: 100% on new code
- [x] Manually tested with soy lateral root data

## Linting & Formatting

- [x] Black formatting applied
- [x] Pydocstyle checks pass
- [x] Added Google-style docstrings for new class and methods

## Coverage

- [x] Coverage: 98.5% overall (up from 98.2%)
- [x] `LateralRootPipeline`: 100% coverage

## Breaking Changes

- [ ] No breaking changes

## Related Issues

Closes #121

## Trait Computation Changes

- [x] Traits computed: lateral root lengths, tip counts, base counts
- [x] Validated against manual measurements (error < 1%)
- [x] No changes to existing trait computations

## Examples

Example output for soy lateral roots:
```csv
frame,lateral_length_total,lateral_count,lateral_tips
0,245.6,12,12
1,267.3,14,14
2,289.1,15,15
```

## Reviewer Notes

Please review the trait definitions in `LateralRootPipeline.get_trait_definitions()` to ensure they match expected lateral root phenotyping needs.
```

## Bug Fix PR Example

```markdown
## Summary

Fix angle calculation returning NaN for collinear points.

## Changes

- **sleap_roots/angle.py**: Add epsilon tolerance for dot product clipping
- **tests/test_angle.py**: Add regression test for collinear points

## Testing

- [x] All tests pass
- [x] Added regression test that reproduces the bug
- [x] Verified fix with real data where bug occurred

## Linting & Formatting

- [x] Black formatting applied
- [x] Pydocstyle checks pass

## Coverage

- [x] Coverage maintained at 98.5%
- [x] New test covers the previously untested code path

## Breaking Changes

- [ ] No breaking changes

## Related Issues

Fixes #142

## Trait Computation Changes

- [x] Angle calculations now handle edge case correctly
- [x] No changes to angle values for non-collinear points
- [x] Validated that fix doesn't affect existing results
```

## GitHub CLI Tips

```bash
# Create PR with template
gh pr create --title "feat: add lateral root pipeline" --body-file pr-description.md

# Create PR interactively
gh pr create

# Edit PR description
gh pr edit --body-file updated-description.md

# Add labels
gh pr edit --add-label "type:feature"
gh pr edit --add-label "area:pipelines"

# Request review
gh pr edit --add-reviewer @username

# Check CI status
gh pr checks

# View PR in browser
gh pr view --web
```

## Labels

Common labels for sleap-roots PRs:

**Type**:
- `type:feature` - New feature
- `type:bugfix` - Bug fix
- `type:enhancement` - Improvement to existing feature
- `type:docs` - Documentation changes
- `type:test` - Test-only changes
- `type:refactor` - Code refactoring

**Area**:
- `area:pipelines` - Changes to pipeline classes
- `area:traits` - New traits or trait modifications
- `area:core` - Core computation modules (lengths, angles, etc.)
- `area:io` - Data loading/saving
- `area:ci` - CI/CD changes

**Priority**:
- `priority:high` - Critical fixes
- `priority:medium` - Important enhancements
- `priority:low` - Nice to have

## Writing Good PR Descriptions

### Good Examples

**Clear Summary**:
```markdown
## Summary
Add support for analyzing crown roots in older monocot plants without primary roots.
```

**Specific Changes**:
```markdown
## Changes
- Add `OlderMonocotPipeline` class with crown-only trait computation
- Implement crown root length, tip count, and network length calculations
- Add 15 test cases using rice 10do test data
- Update README with OlderMonocotPipeline usage example
```

**Thorough Testing Notes**:
```markdown
## Testing
- [x] All 87 existing tests pass
- [x] Added 15 new tests for OlderMonocotPipeline
- [x] Manually validated against 5 rice samples (10 days old)
- [x] Compared trait outputs to manual measurements (< 2% error)
- [x] Tested on Ubuntu, macOS, and Windows
```

### Less Helpful Examples

❌ **Vague Summary**: "Update pipeline"
✅ **Better**: "Add OlderMonocotPipeline for crown-only root analysis"

❌ **Unclear Changes**: "Fix stuff and add things"
✅ **Better**: "Fix NaN in angle calculation for collinear points"

❌ **No Testing Details**: "Tested it"
✅ **Better**: "Added 8 test cases covering edge cases (empty arrays, single points, collinear points)"

## Cross-Platform Considerations

sleap-roots must work on Ubuntu, macOS, and Windows. Note any platform-specific concerns:

```markdown
## Platform Notes

- Uses `pathlib.Path` for cross-platform file paths
- No platform-specific dependencies
- Tests pass on all three platforms in CI
```

## Scientific Accuracy

For trait computation changes, emphasize validation:

```markdown
## Validation

- Compared outputs to manual measurements on 10 samples
- Error < 1% for length measurements
- Error < 0.5° for angle measurements
- Matches published results from Berrigan et al. 2024
```

## Tips

1. **Write PR description as you code**: Don't wait until the end
2. **Link to issues**: Use `Closes #123` to auto-close issues on merge
3. **Be specific**: "Fix angle calculation" not "Fix bug"
4. **Include examples**: Show expected output for new features
5. **Explain why**: Not just what changed, but why it was needed
6. **Note trade-offs**: If you made design decisions, explain them
7. **Request specific feedback**: Tell reviewers what to focus on