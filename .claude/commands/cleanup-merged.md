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

Confirm you're on the latest main branch.

### 3. Delete Feature Branch

Delete both local and remote tracking references:

```bash
# Delete local branch
git branch -d <branch-name>

# Clean up remote tracking references
git remote prune origin
```

**Important**: The `-d` flag (not `-D`) ensures the branch has been merged. If this fails, the branch hasn't been fully merged yet.

### 4. Archive OpenSpec Change (if applicable)

If this was an OpenSpec-tracked change, archive it:

#### Check for OpenSpec directory

```bash
ls openspec/changes/
```

#### Identify the change

Look for the change directory (e.g., `add-claude-commands`, `fix-angle-calculation`, etc.).

#### Move to archive

```bash
git mv openspec/changes/<change-name> openspec/changes/archive/<change-name>
```

#### Update Archive README

Edit `openspec/changes/archive/README.md` to add the archived change:

```markdown
### <change-name> (Month Year)

**Status**: ✅ Completed - Merged in PR #<number>

Brief description of what was implemented.

- **Proposal**: [proposal.md](<change-name>/proposal.md)
- **Design**: [design.md](<change-name>/design.md) (if exists)
- **Tasks**: [tasks.md](<change-name>/tasks.md)
- **Related Issue**: #<issue-number> (if applicable)

**Key Deliverables**:

- Bullet point summary
- Of main deliverables
- And outcomes

**Timeline**: <actual-time> (vs. <estimated-time> estimate)
```

### 5. Commit and Push

```bash
git add -A
git commit -m "chore: archive <change-name> OpenSpec change

Moved completed OpenSpec change to archive after PR #<number> merge.
Updated archive README with summary of deliverables.

Related: #<issue>, PR #<pr-number>"

git push
```

### 6. Verify Cleanup

Confirm cleanup is complete:

```bash
# Branch should not appear
git branch -a | grep <branch-name> || echo "✅ Branch deleted"

# OpenSpec should be in archive (if applicable)
ls openspec/changes/archive/<change-name>
```

## Summary Checklist

Provide a summary when done:

- ✅ Switched to main and pulled latest
- ✅ Branch deleted (local + remote tracking)
- ✅ OpenSpec change archived (if applicable)
- ✅ Archive README updated (if applicable)
- ✅ Changes committed and pushed (if applicable)
- ✅ Main branch clean and up-to-date

## Common Scenarios

### Scenario 1: Simple bug fix (no OpenSpec)

1. Switch to main, pull
2. Delete branch
3. Done ✅

```bash
git checkout main
git pull
git branch -d fix/angle-nan-bug
git remote prune origin
```

### Scenario 2: Feature with OpenSpec documentation

1. Switch to main, pull
2. Delete branch
3. Archive OpenSpec change
4. Update archive README
5. Commit and push

```bash
git checkout main
git pull
git branch -d feat/add-lateral-pipeline
git remote prune origin
git mv openspec/changes/add-lateral-pipeline openspec/changes/archive/add-lateral-pipeline
# Edit openspec/changes/archive/README.md
git add -A
git commit -m "chore: archive add-lateral-pipeline OpenSpec change"
git push
```

### Scenario 3: Branch not yet merged

If `git branch -d` fails with "not fully merged" error:

```bash
# Check merge status
gh pr view <number>

# If PR is actually merged on GitHub but git doesn't know:
git fetch origin
git checkout main
git pull

# Try delete again
git branch -d <branch-name>

# If still failing, PR wasn't actually merged
# Ask user to merge PR first
# Do NOT use -D (force delete)
```

## Notes

- **Only archive OpenSpec changes** if they exist in `openspec/changes/`
- **Not all PRs have OpenSpec documentation** - that's okay
- **Use `git mv`** instead of shell `mv` to preserve git history
- **If branch can't be deleted with `-d`**, it means it hasn't been merged
- **Don't force delete** with `-D` unless explicitly instructed

## Example: Full Workflow

```bash
# 1. Verify merge
gh pr list --state merged --limit 5
# Output shows: #130 docs/openspec-project-context merged

# 2. Switch to main
git checkout main
git pull

# 3. Delete branch
git branch -d docs/openspec-project-context
git remote prune origin

# 4. Check for OpenSpec
ls openspec/changes/
# Output shows: add-claude-commands/

# 5. Archive (if this was an OpenSpec change)
git mv openspec/changes/add-claude-commands openspec/changes/archive/add-claude-commands

# 6. Update archive README
# (Edit openspec/changes/archive/README.md)

# 7. Commit
git add -A
git commit -m "chore: archive add-claude-commands OpenSpec change

Moved completed OpenSpec change to archive after PR #135 merge.
Updated archive README with summary of deliverables.

Related: PR #135"

# 8. Push
git push

# 9. Verify
git branch -a | grep "openspec-project-context" || echo "✅ Branch deleted"
ls openspec/changes/archive/add-claude-commands && echo "✅ OpenSpec archived"
```

## Archive README Template

When updating `openspec/changes/archive/README.md`:

```markdown
### add-claude-commands (November 2024)

**Status**: ✅ Completed - Merged in PR #135

Added Claude Code slash commands for streamlined developer workflows including
linting, testing, coverage, PR management, and changelog maintenance.

- **Proposal**: [proposal.md](add-claude-commands/proposal.md)
- **Design**: [design.md](add-claude-commands/design.md)
- **Tasks**: [tasks.md](add-claude-commands/tasks.md)

**Key Deliverables**:

- 7 slash commands: /lint, /coverage, /test, /pr-description, /review-pr, /cleanup-merged, /changelog
- Commands adapted from cosmos-azul for Python/pytest context
- Comprehensive templates and checklists for PR workflow
- Scientific accuracy validation guidelines for trait computations

**Timeline**: 5 hours (vs. 4-5 hour estimate)
```

## Tips

1. **Always verify PR is merged** before deleting branches
2. **Use `-d` not `-D`** for safety (prevents deleting unmerged work)
3. **Archive OpenSpec systematically** to maintain project history
4. **Update archive README** with meaningful summary, not just copy-paste
5. **Preserve git history** with `git mv` instead of `mv`
6. **Keep main clean** by regularly pruning merged branches