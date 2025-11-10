# Review Pull Request

Systematic workflow for reviewing PRs and responding to comments.

## Quick Review Commands

```bash
# List open PRs
gh pr list

# View specific PR
gh pr view <number>

# View PR diff
gh pr diff <number>

# Checkout PR locally
gh pr checkout <number>

# View PR checks
gh pr checks <number>

# Add review comment
gh pr review <number> --comment --body "Great work!"

# Approve PR
gh pr review <number> --approve --body "LGTM! Excellent test coverage."

# Request changes
gh pr review <number> --request-changes --body "Please address the comments on trait calculations."
```

## Review Checklist

### 1. Code Quality

- [ ] Code follows PEP 8 conventions (enforced by Black)
- [ ] Functions have clear, single responsibilities
- [ ] Variable/function names are descriptive
- [ ] No commented-out code or debug print statements
- [ ] Error handling is appropriate
- [ ] No hardcoded values that should be configurable

### 2. Type Safety & Documentation

- [ ] Type hints used for function arguments and returns
- [ ] Google-style docstrings for all public functions/classes
- [ ] Docstrings include Args, Returns, Raises sections
- [ ] Examples provided in docstrings where helpful
- [ ] Module-level docstrings present

### 3. Testing

- [ ] New features have test coverage
- [ ] Tests are clear and descriptive
- [ ] Edge cases are tested (empty arrays, single points, invalid inputs)
- [ ] Coverage maintained or improved
- [ ] Tests pass on all platforms (Ubuntu, Windows, macOS)

### 4. Scientific Accuracy

- [ ] Trait computations are biologically meaningful
- [ ] Algorithm references credible sources (papers, textbooks)
- [ ] Calculations validated against known data or manual measurements
- [ ] Changes maintain reproducibility of published results
- [ ] Units and coordinate systems are documented

### 5. Cross-Platform Compatibility

- [ ] Uses `pathlib.Path` for file paths (not string concatenation)
- [ ] No platform-specific dependencies
- [ ] Tests pass in CI on all platforms
- [ ] File handling respects OS differences

### 6. Performance

- [ ] No unnecessary loops or redundant calculations
- [ ] Large arrays handled efficiently (vectorized NumPy operations)
- [ ] Batch processing doesn't load all data into memory at once
- [ ] No blocking operations that could hang

### 7. Breaking Changes

- [ ] Breaking changes are clearly documented
- [ ] Migration path provided for users
- [ ] CHANGELOG.md updated with breaking changes
- [ ] Version number follows SemVer (major bump for breaking changes)

## Review Response Workflow

### As a Reviewer

1. **Read the PR description** - Understand the purpose and scope
2. **Check CI status** - Don't review if CI is failing
3. **Review diff file by file** - Start with test files to understand intent
4. **Test locally** - Checkout the branch and run tests
5. **Leave specific comments** - Reference line numbers and suggest alternatives
6. **Approve or request changes** - Be clear and constructive

### As a PR Author

1. **Address all comments** - Don't ignore any feedback
2. **Respond to each comment** - Explain your reasoning or agree to change
3. **Push fixes** - Make requested changes in new commits
4. **Mark resolved** - Resolve conversations after addressing them
5. **Request re-review** - Notify reviewers when ready

## Example Review Comments

### Good Comments

```markdown
**Line 42 in sleap_roots/lengths.py**: Consider using `np.linalg.norm(axis=1)`
instead of manual calculation for better performance and readability.

**Line 87 in sleap_roots/angles.py**: This calculation assumes points are in
image coordinates (y-down). Should we add a docstring note about coordinate system?

**test_trait_pipelines.py**: Excellent test coverage! Could you add a test case
for plants with missing lateral roots (primary only)?

**General**: Great work on the lateral root pipeline! The code is clean and
well-tested. Just a few minor suggestions above.
```

### Less Helpful Comments

```markdown
This doesn't look right. ❌
Why did you do it this way? ❌
Use a different method. ❌
```

### Constructive Feedback

```markdown
✅ "This approach works, but using `np.diff()` might be clearer. What do you think?"
✅ "Great start! For edge case handling, consider what happens when points is empty."
✅ "The logic here is correct, but the function name could be more descriptive. Maybe `calculate_lateral_root_angles`?"
```

## GitHub CLI Review Examples

```bash
# Start a review
gh pr review 42 --comment --body "Starting review..."

# Approve with message
gh pr review 42 --approve --body "LGTM! Excellent test coverage on the new pipeline."

# Request changes
gh pr review 42 --request-changes --body "Please address the comments about trait validation in test_trait_pipelines.py"

# View review comments
gh pr view 42 --comments
```

## Responding to Review Comments

```bash
# View PR with comments
gh pr view 42 --comments

# Checkout PR to make fixes
gh pr checkout 42

# Make changes, commit, push
git add sleap_roots/lengths.py
git commit -m "fix: address review comments on length calculation"
git push

# Notify reviewer
gh pr comment 42 --body "✅ Addressed all review comments. Ready for re-review!"
```

## Common Review Patterns for sleap-roots

### Pattern 1: Trait Computation Changes

When reviewing changes to trait calculations:

1. **Check validation** - Are calculations validated against known data?
2. **Verify units** - Are units documented (pixels, mm, degrees, etc.)?
3. **Review edge cases** - What happens with empty arrays, single points?
4. **Check reproducibility** - Will this change published results?

Example:
```markdown
**sleap_roots/angles.py**: Great addition! A few questions:

1. Is the angle in degrees or radians? Please add to docstring.
2. What happens when vectors are collinear? I see the clipping, but a test case would be good.
3. Can you add a reference to the algorithm used (e.g., "dot product method as per...")?
```

### Pattern 2: New Pipeline Classes

When reviewing new pipeline classes:

1. **Check trait definitions** - Are all traits biologically meaningful?
2. **Verify test data** - Does test data match the pipeline's plant type?
3. **Review output format** - Is CSV output consistent with other pipelines?
4. **Check documentation** - Is README updated with usage example?

Example:
```markdown
**sleap_roots/trait_pipelines.py**: Nice work on `LateralRootPipeline`!

Suggestions:
1. Line 234: Consider adding a `min_length` parameter to filter short false positives
2. Test coverage is excellent (100%)!
3. Could you add a usage example to README.md similar to `DicotPipeline`?
4. Trait names match existing conventions ✓
```

### Pattern 3: Bug Fixes

When reviewing bug fixes:

1. **Verify regression test** - Does the test reproduce the original bug?
2. **Check for side effects** - Could the fix break other functionality?
3. **Review test coverage** - Does the fix increase coverage?
4. **Validate fix scope** - Is the fix minimal and focused?

Example:
```markdown
**sleap_roots/angles.py**: Good catch on the NaN issue!

Review notes:
1. The epsilon tolerance fix looks correct ✓
2. Regression test clearly demonstrates the bug ✓
3. Consider testing with very small angles (< 0.1°) as well
4. No impact on other angle calculations verified ✓
```

## Domain-Specific Review Criteria

### Plant Root Phenotyping Context

When reviewing code, consider:

1. **Biological validity**: Are trait measurements meaningful for plant biologists?
2. **Developmental stages**: Does the code handle different growth stages appropriately?
3. **Root types**: Are primary, lateral, and crown roots handled correctly?
4. **Image coordinate systems**: Are y-down image coordinates handled consistently?
5. **SLEAP integration**: Are `.slp` files loaded correctly with `sleap-io`?

### Research Reproducibility

Be especially careful with changes that could affect:

1. **Published results**: Changes to existing trait calculations
2. **Data formats**: Changes to CSV output structure
3. **Coordinate systems**: Changes to how points are interpreted
4. **Default parameters**: Changes to thresholds, filters, etc.

If in doubt, request validation against known good data.

## Tips for Effective Reviews

1. **Be timely** - Review within 24-48 hours if possible
2. **Be specific** - Reference line numbers and suggest concrete alternatives
3. **Be kind** - Assume positive intent, use constructive language
4. **Test locally** - Don't just read code, run it
5. **Focus on substance** - Don't nitpick style (Black handles that)
6. **Explain why** - Help the author learn, don't just point out issues
7. **Approve quickly** - If it's good, say so and approve

## When to Request Changes vs Comment

- **Request Changes**: Test failures, incorrect algorithms, missing validation, scientific inaccuracy
- **Comment**: Style suggestions, performance optimizations, nice-to-haves, questions

## Escalation

If a PR discussion is getting stuck:

1. Jump on a call or video chat to discuss
2. Create a GitHub Discussion for architectural questions
3. Update `openspec/project.md` or `CLAUDE.md` with decision for future reference
4. Consult with domain experts (plant biologists) for trait validation

## Review Approval Criteria

Only approve if:

- [ ] All CI checks pass (tests, linting, coverage)
- [ ] Code quality meets standards
- [ ] Tests adequately cover new code
- [ ] Documentation is sufficient
- [ ] No unresolved questions or concerns
- [ ] Scientific accuracy validated (if applicable)

## Post-Review

After approval:

1. **Merge the PR** - Use squash or merge based on project preference
2. **Delete the branch** - Clean up merged branches
3. **Archive OpenSpec** - If applicable, use `/cleanup-merged`
4. **Update CHANGELOG** - Use `/changelog` if releasing soon