# Pre-Merge Checks

Comprehensive pre-merge workflow that runs all necessary checks, reviews PR comments (including GitHub Copilot), and ensures the code is ready for merging.

## What This Command Does

This command orchestrates a complete pre-merge workflow:

1. **Run Local CI Checks** - Verify all tests and linting pass
2. **Check PR Status** - Review GitHub CI status and check for failures
3. **Review PR Comments** - Analyze all comments including GitHub Copilot suggestions
4. **Address Concerns** - Use planning mode to fix any issues found
5. **Verify Ready to Merge** - Confirm all checks pass and concerns are addressed

## Workflow Steps

### Phase 1: Local CI Checks ✓

Run the full CI suite locally to catch issues before they hit GitHub:

```bash
# Formatting check
uv run black --check sleap_roots tests

# Docstring check
uv run pydocstyle --convention=google sleap_roots/

# Run all tests
uv run pytest tests/

# Optional: Run with coverage
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/
```

**Expected outcome:** All checks pass locally

### Phase 2: PR Status Check ✓

Check the PR's CI status and review any failures:

```bash
# View PR checks
gh pr checks <PR_NUMBER>

# View detailed PR status
gh pr view <PR_NUMBER>

# Check for failing workflows
gh pr checks <PR_NUMBER> --watch
```

**Expected outcome:** All GitHub Actions checks pass (green checkmarks)

### Phase 3: Review PR Comments ✓

Analyze all PR comments, including bot feedback:

```bash
# Get all PR comments
gh pr view <PR_NUMBER> --json comments --jq '.comments[] | "\(.author.login): \(.body)"'

# Get review comments on specific lines
gh pr view <PR_NUMBER> --json reviews --jq '.reviews[] | "\(.author.login) (\(.state)): \(.body)"'

# Check for Copilot comments
gh pr view <PR_NUMBER> --json comments --jq '.comments[] | select(.author.login | contains("copilot")) | .body'
```

**What to look for:**
- GitHub Copilot suggestions for improvements
- Codecov coverage reports
- Security vulnerability warnings
- Reviewer feedback and requested changes
- Bot comments (dependabot, etc.)

### Phase 4: Address Concerns (Planning Mode) ✓

Use planning mode to systematically address all identified issues:

```markdown
## Issues to Address:

### From Copilot Comments:
- [ ] Issue 1: [Description]
- [ ] Issue 2: [Description]

### From CI Failures:
- [ ] Failing test: [Test name]
- [ ] Linting error: [Error description]

### From Code Review:
- [ ] Reviewer comment 1
- [ ] Reviewer comment 2

### From Coverage Report:
- [ ] Low coverage areas
- [ ] Untested edge cases
```

**Planning mode approach:**
1. Categorize all issues by type (critical, important, nice-to-have)
2. Create a plan to address each issue
3. Implement fixes systematically
4. Re-run checks after each fix
5. Verify all issues resolved

### Phase 5: Final Verification ✓

Before declaring ready to merge, verify everything:

```bash
# Re-run local CI
uv run black --check sleap_roots tests
uv run pydocstyle --convention=google sleap_roots/
uv run pytest tests/

# Push any fixes
git push

# Wait for CI to complete
gh pr checks <PR_NUMBER> --watch

# Get final PR status
gh pr view <PR_NUMBER>
```

**Merge criteria:**
- ✅ All local tests pass
- ✅ All GitHub CI checks pass (lint, test, build, coverage)
- ✅ No unresolved review comments
- ✅ No Copilot warnings unaddressed
- ✅ Coverage maintained or improved
- ✅ All conversations resolved

## Common Issue Categories

### 1. GitHub Copilot Comments

Copilot typically flags:
- **Code quality issues**: Complex functions, potential bugs
- **Security concerns**: Unsafe patterns, credential exposure
- **Performance issues**: Inefficient algorithms, unnecessary loops
- **Best practices**: Missing error handling, unclear variable names

**Action:** Review each suggestion and either:
- Implement the suggested improvement
- Add a comment explaining why the suggestion doesn't apply
- Request clarification if the suggestion is unclear

### 2. CI Failures

Common CI failures:
- **Formatting**: Black or pydocstyle errors
- **Tests**: Failing test cases
- **Coverage**: Coverage drop below threshold
- **Build**: Package build failures

**Action:** Use `/debug-test` for failing tests, `/fix-formatting` for style issues

### 3. Code Review Comments

Reviewer feedback may include:
- Architecture suggestions
- Logic errors or edge cases
- Documentation improvements
- Test coverage requests

**Action:** Address each comment and respond with your approach

### 4. Coverage Issues

Codecov may report:
- Decreased overall coverage
- New code not covered by tests
- Missing edge case tests

**Action:** Use `/coverage` to identify gaps and write tests

## Example Workflow

### Scenario: PR Ready for Final Review

```bash
# 1. Check PR number
gh pr status
# Shows: #132 (current branch)

# 2. Run local checks
uv run black --check sleap_roots tests
uv run pydocstyle --convention=google sleap_roots/
uv run pytest tests/
# ✅ All pass

# 3. Check CI status
gh pr checks 132
# ✅ All checks passing

# 4. Review comments
gh pr view 132 --json comments --jq '.comments[] | "\(.author.login): \(.body)"'
# Shows:
# - codecov: Coverage maintained at 83.98%
# - No Copilot warnings
# - No reviewer comments

# 5. Verify conversations resolved
gh pr view 132 --json reviews
# All conversations marked as resolved

# ✅ Ready to merge!
```

### Scenario: Issues Found

```bash
# 1. Run checks - finds issues
uv run black --check sleap_roots tests
# ❌ FAIL: 2 files would be reformatted

# 2. Check comments
gh pr view 132 --json comments
# Copilot: "Function complexity high in foo.py line 42"
# Reviewer: "Please add tests for edge case X"

# 3. Create action plan (Planning Mode)
## Plan:
# 1. Fix formatting issues
# 2. Refactor complex function per Copilot
# 3. Add tests for edge case X
# 4. Re-run all checks

# 4. Execute plan
uv run black sleap_roots tests  # Fix formatting
# [Edit foo.py to reduce complexity]
# [Write new test for edge case X]

# 5. Verify fixes
uv run pytest tests/
git add -u
git commit -m "fix: address pre-merge feedback"
git push

# 6. Wait for CI
gh pr checks 132 --watch

# 7. Confirm ready
gh pr view 132
# ✅ All checks pass, all comments addressed
```

## Integration with Other Commands

This command orchestrates these other commands:

- `/run-ci-locally` - Run all CI checks locally
- `/test` - Run test suite
- `/coverage` - Analyze test coverage
- `/lint` - Check code style
- `/fix-formatting` - Auto-fix style issues
- `/debug-test` - Debug failing tests
- `/review-pr` - Comprehensive PR review

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

## Advanced Checks

### Security Scan

```bash
# Check for security vulnerabilities
pip-audit  # or similar security scanner

# Check for secrets
git secrets --scan
```

### Dependency Check

```bash
# Verify lockfile is up to date
uv lock --locked

# Check for dependency issues
uv tree
```

### Documentation Check

```bash
# Build docs locally (if applicable)
mkdocs build --strict

# Check for broken links
# [Link checker command]
```

## Automation Opportunities

Consider adding these checks to your workflow:

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
uv run black sleap_roots tests
uv run pytest tests/ -x
```

### GitHub Actions Check

Add a workflow that comments on PRs with pre-merge checklist.

### Branch Protection

Require these before merge:
- All CI checks pass
- At least one approval
- Up-to-date with base branch
- Conversations resolved

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

## Success Criteria

PR is ready to merge when:

✅ **All automated checks pass**
- Lint (Black + pydocstyle)
- Tests (all platforms)
- Build (package builds successfully)
- Coverage (maintained or improved)

✅ **All comments addressed**
- Reviewer feedback implemented or discussed
- Copilot suggestions reviewed and addressed
- Bot comments (codecov, etc.) acknowledged

✅ **Code quality verified**
- No obvious bugs or issues
- Edge cases tested
- Documentation updated
- Follows project conventions

✅ **Approvals obtained**
- Required reviewers approved
- No outstanding requested changes
- Conversations resolved

## Related Commands

- `/run-ci-locally` - Local CI checks
- `/review-pr` - PR review workflow
- `/test` - Run tests
- `/coverage` - Coverage analysis
- `/lint` - Linting checks
- `/fix-formatting` - Auto-fix formatting
- `/debug-test` - Debug test failures

## Tips

1. **Start early**: Run pre-merge checks before requesting review
2. **Be systematic**: Address issues in order of priority
3. **Communicate**: Respond to comments promptly
4. **Test thoroughly**: Don't skip edge cases
5. **Document decisions**: Explain why you made certain choices
6. **Keep it clean**: Rebase and clean up commit history if needed
7. **Be responsive**: Address feedback quickly to keep momentum

## Next Steps After Merge

1. **Delete branch**: `gh pr close <PR_NUMBER> --delete-branch`
2. **Update local**: `git checkout main && git pull`
3. **Verify deployment**: Check that changes deploy correctly (if applicable)
4. **Monitor**: Watch for any issues in production
5. **Celebrate**: 🎉 Your code is merged!

---

**Remember**: The goal is not just to pass checks, but to ensure the code is high quality, maintainable, and ready for production use.