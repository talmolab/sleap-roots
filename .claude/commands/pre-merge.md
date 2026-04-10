# Pre-Merge Checks

**Comprehensive pre-merge verification workflow**

Run all quality checks, create PR, review feedback, and update changelog before merging.

## Your Task

Perform a complete pre-merge check following this workflow:

### Phase 1: Code Quality

1. **Formatting & Style**
   - Run `uv run black --check sleap_roots tests`
   - Run `uv run pydocstyle --convention=google sleap_roots/`
   - If failures: fix with `uv run black sleap_roots tests`, re-run

### Phase 2: Tests & Coverage

2. **Unit Tests**
   - Run `uv run pytest tests/`
   - Run `uv run pytest --cov=sleap_roots --cov-report=term-missing tests/`
   - If an OpenSpec change exists: `openspec validate --strict`
   - If benchmark-relevant: `uv run pytest tests/benchmarks/ --benchmark-only` (note #143 may block baseline -- use `gh run download` for manual comparison)

### Phase 3: Documentation

3. **Documentation Review**
   - Verify docstrings are current for all changed code
   - Check OpenSpec tasks completed: `openspec list`
   - README up-to-date if public API changed

### Phase 4: PR Creation

4. **Create or Update PR**
   - Run `gh pr create --title "<title>" --body "<body>"`
   - Include in description: summary, test results, breaking changes, OpenSpec proposal link (if applicable)

### Phase 5: CI Monitoring

5. **Monitor GitHub Actions**
   - Run `gh pr checks <PR_NUMBER>`
   - Watch for cross-platform failures (Ubuntu/Windows/macOS)
   - If any fail: investigate, use `/debug-test`

### Phase 6: Review Feedback

6. **Review PR Comments**
   - Run `gh pr view <PR_NUMBER> --json comments --jq '.comments[] | "\(.author.login): \(.body)"'`
   - Run `gh pr view <PR_NUMBER> --json reviews --jq '.reviews[] | "\(.author.login) (\(.state)): \(.body)"'`
   - Check: Copilot, Codecov, reviewer feedback

### Phase 7: Changelog

7. **Update Changelog**
   - Run `/changelog` command

### Phase 8: Final Verification

8. **Final Check**
   - Re-run local CI: `uv run black --check sleap_roots tests`, `uv run pydocstyle --convention=google sleap_roots/`, `uv run pytest tests/`
   - Push changes: `git push`
   - Verify CI: `gh pr checks <PR_NUMBER>`
   - Confirm up-to-date: `git fetch origin main && git merge-base --is-ancestor origin/main HEAD`

## Output Format

Provide results in this format:

```markdown
# Pre-Merge Check Results

## Code Quality
- [x] Black formatting: PASS
- [x] Pydocstyle: PASS

## Testing
- [x] Unit Tests: X passed, Y skipped
- [x] Coverage: X% (maintained/improved)
- [x] Benchmarks: No regressions (or N/A -- #143)

## Documentation
- [x] Docstrings current
- [x] OpenSpec completed (or N/A)
- [x] OpenSpec validated (or N/A)

## Pull Request
- [x] PR created: #X
- [x] All checks passing

## Changelog
- [x] Entry added (or N/A)

## Status: READY TO MERGE
```

If any checks fail, provide:
- Clear explanation of the failure
- Proposed fix
- Steps to implement
- Re-run instructions

## Planning Mode Template

When using planning mode to address issues, use this template:

```markdown
# Pre-Merge Action Plan

## Current Status
- Branch: [branch-name]
- PR: #[number]
- Target: [main/develop]

## Issues Found

### Critical (Must Fix)
1. [ ] [Issue description]
   - Impact: [description]
   - Fix: [approach]

### Important (Should Fix)
1. [ ] [Issue description]
   - Impact: [description]
   - Fix: [approach]

### Nice-to-Have (Optional)
1. [ ] [Issue description]
   - Impact: [description]
   - Fix: [approach]

## Implementation Plan

### Step 1: [Category]
- Action: [what to do]
- Commands: [commands to run]
- Verification: [how to verify]

### Step 2: [Category]
- Action: [what to do]
- Commands: [commands to run]
- Verification: [how to verify]

## Verification Checklist
- [ ] Local CI passes
- [ ] GitHub CI passes
- [ ] All comments addressed
- [ ] Coverage maintained
- [ ] Documentation updated (if needed)
- [ ] Changelog updated (if needed)

## Ready to Merge
- [ ] All critical issues fixed
- [ ] All important issues addressed or deferred
- [ ] All checks green
- [ ] Approved by reviewer(s)
```

## Troubleshooting

### Issue: "Checks keep failing"
- Review the specific failing check
- Use `/debug-test` for test failures
- Check logs in GitHub Actions

### Issue: "Copilot comment unclear"
- Ask reviewer for clarification
- Check Copilot documentation
- Make best judgment and document decision

### Issue: "Coverage decreased"
- Use `/coverage` to find untested code
- Write tests for new functionality
- Explain if coverage drop is acceptable

### Issue: "Merge conflicts"
- Rebase on target branch: `git rebase main`
- Resolve conflicts
- Re-run all checks

## Integration with Other Commands

This command orchestrates these other commands:

- `/run-ci-locally` - Run all CI checks locally
- `/test` - Run test suite
- `/coverage` - Analyze test coverage
- `/lint` - Check code style
- `/fix-formatting` - Auto-fix style issues
- `/debug-test` - Debug failing tests
- `/review-pr` - Comprehensive PR review
- `/changelog` - Update changelog