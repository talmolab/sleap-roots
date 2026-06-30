# Clean Up Merged Branch

Clean up after a PR merge by deleting the branch and archiving OpenSpec changes.

## Workflow Steps

### 1. Verify Merge Status

First, confirm the PR has been merged:

```bash
# View recent merged PRs
gh pr list --state merged --limit 10

# View specific PR status
gh pr view <number>
```

Ask the user for the branch name if needed (e.g., `feat/add-lateral-pipeline`).

### 2. Switch to Main and Pull

```bash
git checkout main
git pull
```

**CRITICAL**: You must be on the `main` branch (after pulling the merged PR) before archiving. Archiving on a feature branch will not update the base specs on main.

### 3. Delete Feature Branch

Delete both local and remote tracking references:

```bash
# Delete local branch (safe delete — requires branch to be merged)
git branch -d <branch-name>

# Clean up remote tracking references
git remote prune origin
```

**Important**: The `-d` flag (not `-D`) ensures the branch has been merged. If this fails, see Troubleshooting below.

### 4. Archive OpenSpec Change (if applicable)

If this was an OpenSpec-tracked change, **delegate to `/openspec:archive`**.

The `/openspec:archive` skill handles:
- Determining and validating the change ID
- Running `openspec archive <id> --yes` to move the change and apply spec updates
- Reviewing output to confirm specs were updated and the change landed in `changes/archive/`
- Validating with `openspec validate --strict`

#### Archive Best Practices

Per `openspec/AGENTS.md` (the source of truth for archive conventions):

- **Default is full spec application**: Run `openspec archive <change-id> --yes` which moves the change directory and applies spec deltas.
- **`--skip-specs` only with explicit justification**: Use `openspec archive <change-id> --skip-specs --yes` only for tooling-only changes with zero spec deltas. Always document why specs were skipped.
- **Dependency order for multiple archives**: When archiving multiple changes that modify the same capability specs, archive in dependency order — parent/base changes first, then changes that build on them. For example, if `add-pipeline` introduces a requirement and `fix-pipeline` modifies it, archive `add-pipeline` first so the base spec exists.
- **Validate after all archives**: Run `openspec validate --strict` after all archives are complete to confirm everything passes.

### 5. Commit and Push

```bash
# Stage archive changes
git add openspec/

# Commit with descriptive message
git commit -m "openspec: Archive <change-name> change

Archived completed OpenSpec change after PR #<number> merge.
Applied spec updates to openspec/specs/.

Related: PR #<pr-number>"

# Push to main
git push
```

### 6. Verify Cleanup

Confirm cleanup is complete:

```bash
# Branch should not appear
git branch -a | grep <branch-name> || echo "Branch deleted"

# OpenSpec should be in archive (if applicable)
ls openspec/changes/archive/

# Validate specs
openspec validate --strict
```

## Summary Checklist

Provide a summary when done:

- Switched to main and pulled latest
- Branch deleted (local + remote tracking pruned)
- OpenSpec change archived via `/openspec:archive` (if applicable)
- Spec updates applied (if applicable)
- Changes committed and pushed (if applicable)
- Main branch clean and up-to-date

## Common Scenarios

### Scenario 1: Simple bug fix (no OpenSpec)

1. Verify merge status
2. Switch to main, pull
3. Delete branch
4. Done

```bash
gh pr view <number>
git checkout main
git pull
git branch -d fix/angle-nan-bug
git remote prune origin
```

### Scenario 2: Feature with OpenSpec change

1. Verify merge status
2. Switch to main, pull
3. Delete branch
4. Delegate to `/openspec:archive` for archiving
5. Commit and push archive changes
6. Verify cleanup

```bash
gh pr view <number>
git checkout main
git pull
git branch -d feat/add-lateral-pipeline
git remote prune origin
# Then invoke /openspec:archive with the change ID
# After archive completes:
git add openspec/
git commit -m "openspec: Archive add-lateral-pipeline change

Archived completed OpenSpec change after PR #<number> merge.
Applied spec updates to openspec/specs/.

Related: PR #<number>"
git push
```

## Troubleshooting

### "Branch not fully merged"

**Error**: `error: The branch '<branch>' is not fully merged.`

**Cause**: Git does not recognize the branch as merged because the commit SHAs differ. This commonly happens with **squash merges** on GitHub — the squashed commit on main has a different SHA than the branch commits, so git thinks the branch was never merged even though the PR shows as merged.

**Solution**: First verify the PR is actually merged on GitHub:

```bash
gh pr view <number>
```

If the PR is confirmed merged, you can safely force delete:

```bash
git branch -D <branch-name>
```

Only use `-D` after confirming the PR was merged. Do NOT use `-D` if the PR is still open.

### "Remote ref does not exist"

**Error**: `error: unable to delete '<branch>': remote ref does not exist`

**Cause**: The remote branch was already deleted (GitHub auto-deletes branches on merge when configured).

**Solution**: This is expected. Skip remote deletion and just clean up the stale tracking reference:

```bash
git remote prune origin
```

### OpenSpec archive fails

If the `/openspec:archive` skill reports an error, check:

```bash
# List exact change IDs
openspec list

# Verify the change exists and is not already archived
openspec show <change-id>
```

## Related Commands

- `/openspec:archive` - Archive skill (delegated to from step 4)
- `/pr-description` - Template used before merge
- `/review-pr` - Checklist used during review
- `/update-changelog` - Update changelog after merge
- `openspec/AGENTS.md` - Source of truth for OpenSpec conventions