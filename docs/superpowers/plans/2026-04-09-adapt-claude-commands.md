# Adapt Claude Commands from bloom-desktop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade 3 existing Claude commands (review-pr, cleanup-merged, pre-merge) and create 2 new files (new-feature command, openspec-review skill) adapted from bloom-desktop for the sleap-roots Python scientific library context.

**Architecture:** Each deliverable is an independent markdown file. No code dependencies between them. The files are Claude Code command/skill definitions — markdown with embedded prompt text, bash commands, and workflow instructions.

**Tech Stack:** Markdown, GitHub CLI (`gh`), OpenSpec CLI (`openspec`), Python tooling (`uv run pytest`, `uv run black`, `uv run pydocstyle`)

**Spec:** `docs/superpowers/specs/2026-04-09-adapt-claude-commands-design.md`

**Source material (bloom-desktop originals to adapt from):**
- `c:\repos\bloom-desktop\.claude\commands\review-pr.md`
- `c:\repos\bloom-desktop\.claude\commands\cleanup-branch.md`
- `c:\repos\bloom-desktop\.claude\commands\pre-merge.md`
- `c:\repos\bloom-desktop\.claude\commands\new-feature.md`
- `c:\repos\bloom-desktop\.claude\skills\openspec-review\SKILL.md`

**Current sleap-roots files to preserve content from:**
- `c:\repos\sleap-roots\.claude\commands\review-pr.md`
- `c:\repos\sleap-roots\.claude\commands\cleanup-merged.md`
- `c:\repos\sleap-roots\.claude\commands\pre-merge.md`

**Cross-cutting rules (apply to ALL tasks):**
- All `gh` commands: prefix with `unset GITHUB_TOKEN &&`
- GitHub org/repo: `talmolab/sleap-roots`
- All Python commands: prefix with `uv run`
- OpenSpec: reference `openspec/AGENTS.md` as source of truth
- Default archive: `openspec archive <change-id> --yes` (full spec application). `--skip-specs` only with explicit justification.
- No TypeScript/Electron/React/build/packaging references
- No UX references — use "interpretability and performance" instead
- Scientific values: reproducibility, interpretability, performance, memory usage, metadata preservation

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.claude/commands/review-pr.md` | Rewrite | 5-subagent parallel PR review with full prompt templates |
| `.claude/commands/cleanup-merged.md` | Rewrite | Branch cleanup + delegate to `/openspec:archive` |
| `.claude/commands/pre-merge.md` | Rewrite | Phased pre-merge checks with output template |
| `.claude/commands/new-feature.md` | Create | End-to-end feature workflow (branch, OpenSpec, TDD) |
| `.claude/skills/openspec-review/SKILL.md` | Create | 5-subagent parallel OpenSpec proposal review |

---

### Task 1: Rewrite review-pr.md — 5-Subagent PR Review

**Files:**
- Rewrite: `.claude/commands/review-pr.md`
- Reference (bloom-desktop): `c:\repos\bloom-desktop\.claude\commands\review-pr.md` (for subagent structure and synthesis/posting pattern)
- Reference (current sleap-roots): `c:\repos\sleap-roots\.claude\commands\review-pr.md` (for domain-specific content to preserve)

- [ ] **Step 1: Read both source files**

Read the bloom-desktop `review-pr.md` for the subagent structure (Step 1 gather context, Step 2 launch subagents, Step 3 synthesize and post). Read the current sleap-roots `review-pr.md` for domain-specific content to preserve (review patterns for trait computation, pipeline classes, bug fixes, performance; benchmark artifact download; review response workflow; domain-specific criteria).

- [ ] **Step 2: Write the new review-pr.md**

Rewrite `.claude/commands/review-pr.md` with this structure:

**Header:** Title and description matching bloom-desktop's format — "PR Code Review — Subagent Team" with sleap-roots context line:

```markdown
# PR Code Review — Subagent Team

You are a senior scientific programmer reviewing a pull request for sleap-roots
(Python library for plant root phenotyping using SLEAP pose estimation). You value
testing, code quality, reproducibility, metadata preservation, traceability,
interpretability, and performance above all else.

## How This Skill Works

This skill launches **5 specialized subagents in parallel** to critically review the PR.
Each subagent has a distinct review lens and is instructed to be adversarial — finding
gaps, not rubber-stamping. After all subagents return, synthesize findings into a unified
review and post it to GitHub.
```

**Step 1: Gather PR Context** — Adapt bloom-desktop's Step 1. All `gh` commands get `unset GITHUB_TOKEN &&` prefix. GraphQL query uses `talmolab/sleap-roots`:

```bash
# Get PR metadata
unset GITHUB_TOKEN && gh pr view $PR_NUMBER --json title,body,baseRefName,headRefName,author,labels,files

# Get the full diff
unset GITHUB_TOKEN && gh pr diff $PR_NUMBER

# Get CI status
unset GITHUB_TOKEN && gh pr checks $PR_NUMBER

# Get any existing Copilot review comments
unset GITHUB_TOKEN && gh api graphql -f query='
query {
  repository(owner: "talmolab", name: "sleap-roots") {
    pullRequest(number: '$PR_NUMBER') {
      reviews(first: 10) {
        nodes {
          author { login }
          comments(first: 50) {
            nodes { path line body }
          }
        }
      }
    }
  }
}
' --jq '.data.repository.pullRequest.reviews.nodes[] | select(.author.login | contains("opilot")) | .comments.nodes[] | "File: \(.path):\(.line)\n\(.body)"'
```

Also read any OpenSpec proposal linked in the PR body (look for `openspec/changes/` paths).

**Step 2: Launch 5 subagents** — Write FULL prompt templates for each (not just bullet points). Each subagent prompt must include:
- The sleap-roots project description (Python library, plant root phenotyping, SLEAP pose estimation)
- Architecture context: pure Python library with attrs dataclasses, numpy trait computations, networkx trait dependency graph in pipelines, sleap-io for .slp file loading
- The full PR diff, description, CI status, and Copilot comments (as template variables `{PR_DIFF}`, `{PR_BODY}`, `{CI_STATUS}`, `{COPILOT_COMMENTS}`)
- Specific check items from the spec (see spec Deliverable 1, items 1-5)
- Return format: BLOCKING/IMPORTANT/SUGGESTIONS + score

The 5 subagent prompts:

**Subagent 1: Code Quality & Architecture** — Adapt bloom-desktop's subagent 1. Replace Electron/React/IPC/preload context with:
- Architecture: Python library with attrs dataclasses, numpy vectorized trait computations, networkx DAG for pipeline trait dependencies, sleap-io for .slp/.h5 file loading
- Naming: snake_case Python, kebab-case filenames
- Check: PEP 8/Black, Google docstrings, attrs patterns, type hints, numpy idioms, `# type: ignore`/`# noqa`/`np.errstate`/`warnings.filterwarnings` justification, error handling, ripple effects, dead code, sleap-io >= 0.0.11 compatibility
- Return: BLOCKING/IMPORTANT/SUGGESTIONS + code quality score 1-10

**Subagent 2: Testing Strategy** — Adapt bloom-desktop's subagent 2. Replace Vitest/Playwright/IPC coverage with:
- Testing infrastructure: pytest (`tests/test_*.py` for unit, `tests/test_trait_pipelines.py` for integration-level pipeline tests, `tests/benchmarks/` for performance), pytest-cov (~84% tracked via Codecov), pytest-benchmark (15% regression threshold, note #143), test data in `tests/data/` (Git LFS), CI on Ubuntu/Windows/macOS
- Check: TDD evidence, right test level (unit vs integration-level pipeline), metadata preservation tests, data flow correctness tests (intermediate results passed correctly through pipeline DAG), edge cases (empty arrays, NaN, single points), cross-platform CI, benchmark regression (fallback: `unset GITHUB_TOKEN && gh run download` if no baseline), existing test breakage
- Return: BLOCKING/IMPORTANT/SUGGESTIONS + TDD verdict

**Subagent 3: Scientific Rigor & Reproducibility** — Adapt bloom-desktop's subagent 3. Replace session/idle timer/scan context with:
- Scientific values: trait computation accuracy, algorithm references (papers/textbooks), units (pixels/mm/degrees/radians), coordinate systems (y-down image coordinates), impact on published results, metadata preservation (all pipeline parameters in output), numerical stability (NaN propagation, floating point precision, `warnings.filterwarnings` suppression justification), sleap-io >= 0.0.11, data format stability (CSV column names/order changes are breaking for downstream research scripts)
- Return: BLOCKING/IMPORTANT/SUGGESTIONS

**Subagent 4: Performance, Memory & Cross-Platform** — Adapt bloom-desktop's subagent 4. Replace Electron/IPC/timer context with:
- Check: numpy vectorization vs Python loops, benchmark regressions (>15%), memory usage (large datasets, pipeline OOM prevention), batch processing (don't load all data into memory), pathlib.Path (not string concatenation), platform differences, blocking operations, thread safety (global `warnings.filterwarnings` is process-global side effect)
- Return: BLOCKING/IMPORTANT/SUGGESTIONS

**Subagent 5: Behavioral Correctness & Edge Cases** — Adapt bloom-desktop's subagent 5. Replace Electron component lifecycle/IPC context with:
- Check: spec-implementation match, empty/NaN/single-point inputs, SLEAP file loading edge cases, data integrity under partial failure (NaN/empty propagation returning scientifically defensible results, not silently masking errors), pipeline error propagation between stages, memory behavior with large Series (stream vs batch, 10k+ frames), idempotency and statelessness of trait functions
- Return: BLOCKING/IMPORTANT/SUGGESTIONS

**Step 3: Synthesize and Post Review** — Copy bloom-desktop's Step 3 structure verbatim, changing only:
- Repo description line: `*Review by Claude Code subagent team (Code Quality · Testing · Scientific Rigor · Performance/Memory · Behavioural Correctness)*`
- Own-PR fallback pattern: keep exactly as bloom-desktop (attempt `--request-changes` or `--approve`, fall back to `--comment` with verdict note)
- All `gh` commands get `unset GITHUB_TOKEN &&` prefix

**After Step 3:** Append preserved sleap-roots domain content:

```markdown
## Domain-Specific Review Patterns

### Pattern 1: Trait Computation Changes
[Preserve lines 199-216 from current sleap-roots review-pr.md]

### Pattern 2: New Pipeline Classes
[Preserve lines 218-236 from current sleap-roots review-pr.md]

### Pattern 3: Bug Fixes
[Preserve lines 238-257 from current sleap-roots review-pr.md]

### Pattern 4: Performance Changes
[Preserve lines 259-295 from current sleap-roots review-pr.md — includes benchmark artifact download, when to accept/reject regressions]
```

- [ ] **Step 3: Verify the file**

Read the written file back. Check:
- No Electron/TypeScript/React/IPC references remain
- All `gh` commands have `unset GITHUB_TOKEN &&` prefix
- GraphQL query uses `talmolab/sleap-roots`
- All 5 subagent prompts are complete (not just bullet points)
- Domain-specific patterns from current sleap-roots version are preserved
- Own-PR review fallback pattern is included

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/review-pr.md
git commit -m "feat: upgrade review-pr to 5-subagent parallel review team

Adapted from bloom-desktop's subagent PR review pattern for
sleap-roots Python scientific library context. Includes lenses
for code quality, testing, scientific rigor, performance/memory,
and behavioral correctness.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Rewrite cleanup-merged.md — OpenSpec Archive Best Practices

**Files:**
- Rewrite: `.claude/commands/cleanup-merged.md`
- Reference (bloom-desktop): `c:\repos\bloom-desktop\.claude\commands\cleanup-branch.md` (for openspec archive CLI pattern)
- Reference (current sleap-roots): `c:\repos\sleap-roots\.claude\commands\cleanup-merged.md` (for verification checklist, troubleshooting)
- Reference: `c:\repos\sleap-roots\.claude\commands\openspec\archive.md` (existing archive skill — cleanup-merged delegates to this)

- [ ] **Step 1: Read source files**

Read bloom-desktop's `cleanup-branch.md`, current sleap-roots `cleanup-merged.md`, and the existing `openspec/archive.md` skill.

- [ ] **Step 2: Write the new cleanup-merged.md**

Rewrite `.claude/commands/cleanup-merged.md` with this structure:

```markdown
# Clean Up Merged Branch

Clean up after a PR merge by deleting the branch and archiving OpenSpec changes using OpenSpec best practices.

## Workflow Steps

### 1. Verify Merge Status

First, confirm the PR has been merged:

` ` `bash
# View recent merged PRs
unset GITHUB_TOKEN && gh pr list --state merged --limit 10

# View specific PR status
unset GITHUB_TOKEN && gh pr view <number>
` ` `

Ask the user for the branch name if needed.

### 2. Switch to Main and Pull

` ` `bash
git checkout main
git pull
` ` `

**CRITICAL**: You must be on `main` with the merged PR pulled before archiving. Never archive on feature branches.

### 3. Delete Feature Branch

` ` `bash
# Delete local branch (safe — fails if not merged)
git branch -d <branch-name>

# Clean up remote tracking references
git remote prune origin
` ` `

**Important**: Use `-d` (not `-D`) to ensure the branch has been merged. If this fails, the branch hasn't been fully merged yet — see Troubleshooting.

### 4. Archive OpenSpec Change (if applicable)

If this was an OpenSpec-tracked change, delegate to the archive skill:

**Run `/openspec:archive`** — this skill handles:
- Identifying the change ID
- Running `openspec archive <change-id> --yes` (full spec application by default)
- Validating with `openspec validate --strict`
- Reviewing output to confirm specs updated

**Archive best practices** (per `openspec/AGENTS.md`):
- **Default**: `openspec archive <change-id> --yes` — always applies spec deltas
- **Exception**: `--skip-specs` only when explicitly justified for tooling-only changes with zero spec deltas
- **Dependency order**: When archiving multiple changes that modify the same capability specs, archive parent changes first, then children
- **Validation**: `openspec validate --strict` after all archives complete

### 5. Commit and Push

` ` `bash
git add openspec/
git commit -m "openspec: Archive <change-name> change

Archived completed OpenSpec change after PR #<number> merge.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"

git push origin main
` ` `

### 6. Verify Cleanup

` ` `bash
# Branch should not appear
git branch -a | grep <branch-name> || echo "Branch deleted"

# OpenSpec should be in archive (if applicable)
ls openspec/changes/archive/

# Validation passes
openspec validate --strict
` ` `

## Summary Checklist

- Switched to main and pulled latest
- Branch deleted (local + remote tracking pruned)
- OpenSpec change archived via `/openspec:archive` (if applicable)
- Archives committed and pushed (if applicable)
- `openspec validate --strict` passes
- Main branch clean and up-to-date

## Common Scenarios

### Scenario 1: Simple bug fix (no OpenSpec)

1. Switch to main, pull
2. Delete branch
3. Done

### Scenario 2: Feature with OpenSpec documentation

1. Switch to main, pull
2. Delete branch
3. Run `/openspec:archive` to archive the change
4. Commit and push archive changes

## Troubleshooting

### "Branch not fully merged"

**Error**: `error: The branch '<branch>' is not fully merged.`

**Cause**: Git doesn't recognize the branch as merged (different commit SHAs due to squash merge on GitHub)

**Solution**:
` ` `bash
# Verify PR is actually merged on GitHub
unset GITHUB_TOKEN && gh pr view <number>

# If PR is merged but git doesn't know, fetch and try again
git fetch origin
git checkout main
git pull
git branch -d <branch-name>

# If still failing, the PR wasn't actually merged
# Ask user to merge PR first — do NOT use -D (force delete)
` ` `

### "Remote ref does not exist"

**Cause**: Branch already deleted on remote (GitHub auto-deletes on merge)

**Solution**: Skip remote deletion, just prune tracking refs:
` ` `bash
git remote prune origin
` ` `

## Related Commands

- `/review-pr` — PR review before merge
- `/pre-merge` — Pre-merge checks
- `/changelog` — Update changelog after merge
```

Note: Replace the triple-backtick placeholders above with real triple backticks (shown with spaces to avoid markdown escaping in this plan).

- [ ] **Step 3: Verify the file**

Read the written file back. Check:
- No `git mv` references remain
- No archive README template remains
- `/openspec:archive` delegation is clear
- `--skip-specs` mentioned only as exception with justification requirement
- All `gh` commands have `unset GITHUB_TOKEN &&` prefix
- `git branch -d` (not `-D`) is the default
- `openspec/AGENTS.md` referenced as source of truth

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/cleanup-merged.md
git commit -m "feat: upgrade cleanup-merged to use openspec archive best practices

Replaced git mv with openspec archive CLI delegation.
Default is full spec application. References openspec/AGENTS.md
as source of truth for archive conventions.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Rewrite pre-merge.md — Clean Phased Structure

**Files:**
- Rewrite: `.claude/commands/pre-merge.md`
- Reference (bloom-desktop): `c:\repos\bloom-desktop\.claude\commands\pre-merge.md` (for phased structure and output template)
- Reference (current sleap-roots): `c:\repos\sleap-roots\.claude\commands\pre-merge.md` (for planning mode template, troubleshooting, command references)

- [ ] **Step 1: Read source files**

Read bloom-desktop's `pre-merge.md` and current sleap-roots `pre-merge.md`.

- [ ] **Step 2: Write the new pre-merge.md**

Rewrite `.claude/commands/pre-merge.md` with 8 phases from the spec. Key structure:

**Header:**
```markdown
# Pre-Merge Checks

**Comprehensive pre-merge verification workflow**

Run all quality checks, create PR, review feedback, and update changelog before merging.

## Your Task

Perform a complete pre-merge check following this workflow:
```

**Phase 1: Code Quality**
```bash
uv run black --check sleap_roots tests
uv run pydocstyle --convention=google sleap_roots/
```
If failures: fix with `uv run black sleap_roots tests`, re-run.

**Phase 2: Tests & Coverage**
```bash
uv run pytest tests/
uv run pytest --cov=sleap_roots --cov-report=term-missing tests/
```
If OpenSpec change exists: `openspec validate --strict`
If benchmark-relevant changes: `uv run pytest tests/benchmarks/ --benchmark-only` (note: #143 may block baseline comparison — use `unset GITHUB_TOKEN && gh run download` for manual comparison)

**Phase 3: Documentation**
- Verify docstrings are current for changed code
- Check OpenSpec tasks completed (if applicable): `openspec list`
- README up-to-date if public API changed

**Phase 4: PR Creation**
```bash
unset GITHUB_TOKEN && gh pr create --title "<title>" --body "<body>"
```
Include: summary of changes, test results, breaking changes, OpenSpec proposal link (if applicable).

**Phase 5: CI Monitoring**
```bash
unset GITHUB_TOKEN && gh pr checks <PR_NUMBER>
```
Watch for cross-platform failures (Ubuntu, Windows, macOS). If any fail: investigate logs, use `/debug-test` for test failures.

**Phase 6: Review Feedback**
```bash
unset GITHUB_TOKEN && gh pr view <PR_NUMBER> --json comments --jq '.comments[] | "\(.author.login): \(.body)"'
unset GITHUB_TOKEN && gh pr view <PR_NUMBER> --json reviews --jq '.reviews[] | "\(.author.login) (\(.state)): \(.body)"'
```
Check: Copilot suggestions, Codecov coverage, reviewer feedback. Address all concerns.

**Phase 7: Changelog** — Run `/changelog` command.

**Phase 8: Final Verification**
```bash
# Re-run local CI
uv run black --check sleap_roots tests
uv run pydocstyle --convention=google sleap_roots/
uv run pytest tests/

# Push fixes
git push

# Wait for CI
unset GITHUB_TOKEN && gh pr checks <PR_NUMBER>

# Check branch is up-to-date with main
git fetch origin main && git merge-base --is-ancestor origin/main HEAD
```

**Output Format** — the markdown checklist template from the spec:
```markdown
# Pre-Merge Check Results

## Code Quality
- [x] Black formatting: PASS
- [x] Pydocstyle: PASS

## Testing
- [x] Unit Tests: X passed, Y skipped
- [x] Coverage: X% (maintained/improved)
- [x] Benchmarks: No regressions (or N/A — #143)

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

**Failure output** — if any checks fail, provide: clear explanation, proposed fix, steps to implement, re-run instructions.

**Preserve from current sleap-roots version:**
- Planning mode template (the markdown template for categorizing issues as Critical/Important/Nice-to-have with fix approaches — lines 262-309 from current file)
- Troubleshooting section (lines 371-397: checks keep failing, Copilot unclear, coverage decreased, merge conflicts)
- Integration with other commands: `/run-ci-locally`, `/test`, `/coverage`, `/lint`, `/fix-formatting`, `/debug-test`, `/review-pr`, `/changelog`

- [ ] **Step 3: Verify the file**

Read the written file back. Check:
- 8 phases present (Code Quality, Tests, Docs, PR, CI, Review, Changelog, Final)
- All commands use `uv run` prefix
- All `gh` commands have `unset GITHUB_TOKEN &&` prefix
- Output template included
- `openspec validate --strict` in Phase 2
- No Electron/TypeScript/build/packaging references
- Planning mode template preserved
- Troubleshooting preserved

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/pre-merge.md
git commit -m "feat: upgrade pre-merge to clean phased structure with output template

Streamlined from verbose format to 8 clear phases with
markdown checklist output. Added openspec validation and
benchmark awareness.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Create new-feature.md — Feature Workflow Command

**Files:**
- Create: `.claude/commands/new-feature.md`
- Reference (bloom-desktop): `c:\repos\bloom-desktop\.claude\commands\new-feature.md`

- [ ] **Step 1: Read bloom-desktop source**

Read bloom-desktop's `new-feature.md` for structure.

- [ ] **Step 2: Write new-feature.md**

Create `.claude/commands/new-feature.md`:

```markdown
---
name: New Feature
description: End-to-end workflow for scoping, proposing, reviewing, and implementing a new feature using OpenSpec and TDD.
category: Development
tags: [feature, openspec, tdd, workflow]
---

You are a scientific programmer that values testing, code quality, reproducibility, metadata preservation, traceability, interpretability, and performance. You are starting a new feature workflow. The user's feature request is: $ARGUMENTS

**Guardrails**

- Do NOT write any implementation code until the proposal is approved.
- Follow OpenSpec conventions strictly (see `openspec/AGENTS.md`).
- Use TDD when implementing (tests before implementation code).
- Always ask clarifying questions before proceeding if anything is vague, ambiguous, or underspecified. Do not assume.

**Steps**

1. **Ensure feature branch**: Check if you are on a feature branch (not `main`). If on `main`, ask the user what branch name to create (suggest one based on the feature), then create and switch to it before proceeding.

2. **Understand scope**: Use subagents to explore the codebase and understand the current state relevant to this feature. Investigate:
   - Existing trait modules in `sleap_roots/` (lengths, angles, tips, bases, etc.)
   - Pipeline classes and the networkx trait dependency graph in `sleap_roots/trait_pipelines.py`
   - Test fixtures in `tests/fixtures/` and test data in `tests/data/`
   - Related specs in `openspec/specs/`
   - Active changes in `openspec/changes/`

3. **Ask clarifying questions**: Based on what you learned from the codebase exploration, ask the user any clarifying questions about:
   - Requirements and expected behavior
   - Edge cases (empty arrays, NaN, single points)
   - Biological validity and scientific accuracy
   - Data handling and coordinate systems
   - Impact on published results or reproducibility
   - Memory considerations for large datasets
   - Which plant types are affected (dicot, monocot, etc.)
   - Which test data in `tests/data/` is relevant
   Do not proceed until you have clear answers.

4. **Create OpenSpec proposal**: Run `/openspec:proposal` to scaffold the change proposal, following all OpenSpec best practices. Ground the proposal in what you learned from steps 2-3. The proposal's `tasks.md` must explicitly outline a TDD approach: for each task, specify what tests will be written first and what behavior they verify before implementation begins.

5. **Review the proposal**: Run the openspec-review skill to have the proposal critically reviewed by 5 specialized subagents. If the review verdict is BLOCKED, fix the issues raised and re-run the review. Repeat until the verdict is APPROVED or NEEDS REVISION.

6. **Get user approval**: Present the reviewed proposal to the user and wait for explicit approval before proceeding to implementation.

7. **Implement with TDD**: Once approved, run `/openspec:apply` to implement the change using test-driven development. Write tests before implementation code.
```

- [ ] **Step 3: Verify the file**

Read the written file back. Check:
- YAML frontmatter present (name, description, category, tags)
- `$ARGUMENTS` placeholder for user input
- Scientific values line: "reproducibility, metadata preservation, traceability, interpretability, and performance" (not UX)
- All 7 steps present
- Step 2 mentions networkx trait dependency graph
- Step 3 mentions plant types, test data, memory
- Step 5 has iteration loop (re-run if BLOCKED)
- `/openspec:proposal`, `/openspec:apply` referenced as skills
- No Electron/TypeScript references

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/new-feature.md
git commit -m "feat: add new-feature command for end-to-end feature workflow

Adapted from bloom-desktop for sleap-roots scientific context.
Workflow: branch -> explore -> clarify -> OpenSpec proposal ->
review -> approve -> TDD implementation.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Create openspec-review Skill — 5-Subagent Proposal Review

**Files:**
- Create: `.claude/skills/openspec-review/SKILL.md`
- Reference (bloom-desktop): `c:\repos\bloom-desktop\.claude\skills\openspec-review\SKILL.md`

- [ ] **Step 1: Read bloom-desktop source**

Read bloom-desktop's `SKILL.md` for structure and subagent prompt templates.

- [ ] **Step 2: Create directory**

```bash
mkdir -p .claude/skills/openspec-review
```

- [ ] **Step 3: Write SKILL.md**

Create `.claude/skills/openspec-review/SKILL.md` with this structure. Adapt bloom-desktop's skill, replacing all Electron/React/TypeScript/IPC context with sleap-roots Python scientific library context.

**Frontmatter:**
```yaml
---
name: openspec-review
description: |
  Critically review an OpenSpec proposal using a team of specialized subagents.
  Use when: reviewing proposals before approval, validating spec quality, checking TDD plans,
  ensuring scientific rigor (metadata, reproducibility, traceability), and verifying GitHub issue alignment.
  Launches 5 parallel subagents for deep, adversarial review.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write, Agent, TodoWrite
---
```

**Body structure:**

```markdown
# OpenSpec Proposal Review — Subagent Team

You are a senior scientific programmer reviewing an OpenSpec proposal for sleap-roots,
a Python library for plant root phenotyping using SLEAP pose estimation. You value testing,
code quality, reproducibility, metadata preservation, traceability, interpretability, and
performance above all else.

## How This Skill Works
[Same as bloom-desktop — 5 parallel subagents, adversarial, synthesize]

## Step 1: Identify the Proposal
[Same as bloom-desktop — user specifies or ls openspec/changes/]

## Step 2: Gather Context
[Same as bloom-desktop — read proposal.md, tasks.md, design.md, delta specs, current specs]

## Step 3: Launch Subagent Review Team
```

**5 subagent prompts** — each must be a FULL prompt template (not bullets). Follow bloom-desktop's pattern of embedding proposal text as template variables (`{PROPOSAL_MD}`, `{TASKS_MD}`, `{DELTA_SPECS}`, `{CURRENT_SPECS}`, `{AFFECTED_FILES_LIST}`, `{ISSUE_NUMBERS}`, `{SEARCH_KEYWORDS}`).

**Subagent 1: Spec Quality & OpenSpec Best Practices** — Adapt bloom-desktop's subagent 1. Replace Electron architecture with:
- Architecture: Python library with attrs dataclasses, numpy trait computations, networkx trait dependency graph, sleap-io for file loading
- Keep all OpenSpec format rules verbatim from bloom-desktop (they're the same — delta headers, requirement/scenario format, MODIFIED full text)
- Add: flag `--skip-specs` usage as requiring explicit justification
- Add: reference `openspec/AGENTS.md` for authoritative rules
- Return: PASS/FAIL per check, line-level issues, concrete rewrites, quality score 1-10

**Subagent 2: Code & Architecture Feasibility** — Adapt bloom-desktop's subagent 2. Replace Electron/IPC/Prisma/preload context with:
- Architecture: Python library — `sleap_roots/` modules for trait computations, `sleap_roots/trait_pipelines.py` for pipeline orchestration with networkx DAG, `sleap_roots/series.py` for data loading, `tests/` for pytest, attrs dataclasses, numpy arrays
- Check: read affected files, verify code claims, numpy/attrs patterns, networkx trait graph placement, breaking changes, ripple effects, cross-platform (pathlib), memory feasibility (loading all frames into single array?)
- Return: incorrect claims, missing files, architecture violations, compatibility risks, concrete code snippets

**Subagent 3: GitHub Issues & Requirements Alignment** — Adapt bloom-desktop's subagent 3 directly. Change only:
- `gh` commands get `unset GITHUB_TOKEN &&` prefix
- Repo context: "sleap-roots, a Python library for plant root phenotyping"
- Return format: same as bloom-desktop

**Subagent 4: TDD & Testing Strategy** — Adapt bloom-desktop's subagent 4. Replace Vitest/Playwright/IPC/E2E context with:
- Testing infrastructure: pytest (`tests/test_*.py` for unit, `tests/test_trait_pipelines.py` for integration-level pipeline tests, `tests/benchmarks/` for performance), pytest-cov (~84% via Codecov), pytest-benchmark (15% threshold), test data in `tests/data/` (Git LFS), CI on Ubuntu/Windows/macOS
- Add: test level distinction (unit vs integration-level), metadata preservation tests, data flow correctness tests (intermediate results through pipeline DAG), verification check gates between sections
- Return: missing tests, wrong framework, CI incompatible, scenario gaps, TDD violations, commit safety issues, existing test breakage

**Subagent 5: Scientific Rigor & Data Integrity** — Adapt bloom-desktop's subagent 5. Replace Basler camera/DAQ/scan/session context with:
- Scientific values: trait computation accuracy, reproducibility (units, defaults), published results impact, coordinate systems, algorithm references, numerical stability (NaN propagation, floating point, `warnings.filterwarnings` justification), memory for large datasets (OOM), data format stability (CSV column names/order), migration paths, sleap-io >= 0.0.11 compatibility
- Return: metadata gaps, missing migrations, traceability gaps, scientific concerns, hardware claims needing verification

**Step 4: Synthesize Review** — Same structure as bloom-desktop: deduplicate, prioritize (BLOCKING/IMPORTANT/SUGGESTION), unified review with sections per subagent, plus a "Commit Safety & CI Health" section from subagent 4's commit discipline findings.

**Step 5: Offer to Fix** — Same as bloom-desktop: ask user if they want automatic fixes, revised proposal, or GitHub issues.

- [ ] **Step 4: Verify the file**

Read the written file back. Check:
- YAML frontmatter correct (name, description, allowed-tools)
- All 5 subagent prompts are FULL templates (not bullet points)
- No Electron/TypeScript/React/IPC/Prisma references
- Template variables used (`{PROPOSAL_MD}`, etc.)
- `unset GITHUB_TOKEN &&` on all `gh` commands
- `openspec/AGENTS.md` referenced
- `--skip-specs` flagging included in subagent 1
- Memory feasibility in subagent 2
- Metadata preservation and data flow in subagent 4
- Numerical stability in subagent 5
- Synthesis structure includes all sections

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/openspec-review/SKILL.md
git commit -m "feat: add openspec-review skill with 5-subagent parallel review

Adapted from bloom-desktop for sleap-roots scientific context.
Subagents review: spec quality, code feasibility, issue alignment,
TDD strategy, and scientific rigor/data integrity.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final Verification

**Files:** All 5 files from Tasks 1-5.

- [ ] **Step 1: Verify all files exist**

```bash
ls -la .claude/commands/review-pr.md .claude/commands/cleanup-merged.md .claude/commands/pre-merge.md .claude/commands/new-feature.md .claude/skills/openspec-review/SKILL.md
```

- [ ] **Step 2: Grep for stale references**

Search all 5 files for terms that should NOT appear:

```bash
grep -rni "electron\|typescript\|react\|vitest\|playwright\|prisma\|preload\|renderer\|IPC handler\|npx\|npm run\|Basler\|DAQ\|scanner\|camera\|session reset\|idle timer" .claude/commands/review-pr.md .claude/commands/cleanup-merged.md .claude/commands/pre-merge.md .claude/commands/new-feature.md .claude/skills/openspec-review/SKILL.md
```

Expected: zero matches. If any found, fix them.

- [ ] **Step 3: Grep for required patterns**

Verify key patterns are present:

```bash
# All gh commands have unset GITHUB_TOKEN
grep -c "unset GITHUB_TOKEN" .claude/commands/review-pr.md .claude/commands/cleanup-merged.md .claude/commands/pre-merge.md .claude/skills/openspec-review/SKILL.md

# uv run prefix used
grep -c "uv run" .claude/commands/pre-merge.md

# openspec/AGENTS.md referenced
grep -c "AGENTS.md" .claude/commands/cleanup-merged.md .claude/commands/new-feature.md .claude/skills/openspec-review/SKILL.md

# talmolab/sleap-roots
grep -c "talmolab" .claude/commands/review-pr.md .claude/skills/openspec-review/SKILL.md
```

- [ ] **Step 4: Verify git status is clean**

```bash
git status
git log --oneline -5
```

Expected: 5 commits from Tasks 1-5, clean working tree.
