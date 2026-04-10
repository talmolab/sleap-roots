# Adapt Claude Commands from bloom-desktop

**Date:** 2026-04-09
**Approach:** Selective merge (Approach B) — keep existing sleap-roots context, surgically upgrade with bloom-desktop patterns.

## Context

sleap-roots already has `.claude/commands/` with project-specific content (benchmark sections, test data locations, pydocstyle guidance). bloom-desktop has more advanced patterns — particularly the subagent team reviews and OpenSpec CLI integration. This design merges the best of both.

## Cross-Cutting Decisions

- All `gh` commands prefixed with `unset GITHUB_TOKEN &&` (workaround for talmolab org token lifetime issue). Demonstrated in command templates where `gh` is used.
- GitHub org/repo: `talmolab/sleap-roots`
- **OpenSpec best practices:** Follow `openspec/AGENTS.md` as source of truth. Default archive command is `openspec archive <change-id> --yes` (full spec application). `--skip-specs` is allowed only when explicitly justified for tooling-only changes with zero spec deltas — never the default.
- Benchmark regression threshold: 15% (note: #143 blocks baseline auto-commit due to branch protection). When no baseline exists, manually download and compare artifacts via `gh run download`.
- Scientific context: reproducibility, interpretability, performance, memory usage (not UX — this is a library)
- Python-only: no TypeScript/Electron/build/packaging phases from bloom-desktop
- **Test level distinction:** `test_trait_pipelines.py` = integration-level (full pipeline data flow, metadata output, trait correctness). `test_*.py` (individual modules) = unit-level (single trait computations). Tests must verify metadata preservation and correct data passing between pipeline stages.
- Each subagent receives the full PR diff independently — do not optimize by giving subagents partial diffs.
- Commands that reference `/openspec:proposal`, `/openspec:apply`, `/openspec:archive` are invoking skills (via the Skill tool), not CLI commands.

## Deliverable 1: review-pr.md — Subagent Team Review

**Action:** Replace manual checklist with 5-subagent parallel review system.

### Structure

**Step 1: Gather PR context** — Run in parallel:
- `gh pr view $PR_NUMBER --json title,body,baseRefName,headRefName,author,labels,files`
- `gh pr diff $PR_NUMBER`
- `gh pr checks $PR_NUMBER`
- Copilot comments via GraphQL (org: `talmolab`, repo: `sleap-roots`)
- Read any OpenSpec proposal linked in PR body

**Step 2: Launch 5 subagents in parallel** — Each gets full diff, PR description, CI status.

1. **Code Quality & Architecture**
   - PEP 8/Black compliance, Google-style docstrings (pydocstyle)
   - attrs patterns, type hints, numpy idioms
   - No `any` types, no dead code, single responsibility
   - Check for `# type: ignore`, `# noqa`, `np.errstate`, and `warnings.filterwarnings` — all require justification
   - Error handling — errors surfaced or silently swallowed?
   - Ripple effects in files NOT changed by the PR
   - `sleap-io` API compatibility — changes to .slp loading must remain compatible with sleap-io >= 0.0.11

2. **Testing Strategy**
   - pytest coverage (tracked via Codecov, ~84% overall)
   - Right test level: unit tests (`test_*.py`) for individual trait modules, integration-level tests (`test_trait_pipelines.py`) for full pipeline data flow
   - **Metadata preservation:** Do tests verify metadata is correctly written and preserved through pipeline stages?
   - **Data flow correctness:** Do tests verify intermediate results are passed correctly between trait computations in the pipeline DAG?
   - Edge cases: empty arrays, NaN, single points, extreme values
   - Cross-platform CI: Ubuntu, Windows, macOS
   - Test data fixtures in `tests/data/` (Git LFS)
   - TDD evidence: test files in earlier commits?
   - Benchmark regression check (15% threshold). If #143 is unresolved and no baseline exists, manually compare via `gh run download`.

3. **Scientific Rigor & Reproducibility**
   - Trait computation accuracy — biologically meaningful?
   - Algorithm references (papers, textbooks)
   - Units documented (pixels, mm, degrees, radians)
   - Coordinate systems (y-down image coordinates)
   - Impact on published results / reproducibility
   - Metadata preservation — all pipeline parameters captured in output
   - **Numerical stability** — NaN propagation correctness, floating point precision in geometric computations, justification for any `warnings.filterwarnings` suppressions
   - SLEAP integration — `.slp` files loaded correctly with `sleap-io` >= 0.0.11?
   - **Data format stability** — CSV column names, column order, schema changes are breaking for downstream research scripts

4. **Performance, Memory & Cross-Platform**
   - numpy vectorization vs Python loops
   - Benchmark regressions (>15% threshold)
   - Memory usage — large datasets, pipeline OOM prevention
   - Batch processing — doesn't load all data into memory at once
   - `pathlib.Path` usage (not string concatenation)
   - Platform-specific behavior differences
   - No blocking operations that could hang
   - Thread safety for batch processing contexts — global `warnings.filterwarnings` calls are process-global side effects

5. **Behavioral Correctness & Edge Cases**
   - Spec-implementation match (does code do what PR description claims?)
   - Empty array, NaN, single-point inputs
   - SLEAP file loading edge cases
   - **Data integrity under partial failure** — are NaN/empty propagation paths returning scientifically defensible results rather than silently masking errors?
   - Pipeline error propagation — do errors surface or get swallowed between stages?
   - **Memory behavior with large Series** — does the pipeline stream or batch frames? What happens with 10,000+ frames?
   - Idempotency and statelessness of trait functions

**Step 3: Synthesize** — Deduplicate, prioritize (BLOCKING/IMPORTANT/SUGGESTION), determine verdict (APPROVE/COMMENT/REQUEST_CHANGES), post to GitHub with own-PR fallback.

**Kept from current sleap-roots version:**
- Domain-specific review patterns (trait computation, pipeline classes, bug fixes)
- Benchmark artifact download instructions
- Review response workflow guidance
- When to accept regressions vs request optimization

## Deliverable 2: cleanup-merged.md — OpenSpec Archive CLI

**Action:** Replace `git mv` with `openspec archive` CLI following OpenSpec best practices.

### Changes

- **Delegate archive to `/openspec:archive`** — cleanup-merged calls the existing openspec archive skill rather than reimplementing archive logic
- **Default archive:** `openspec archive <change-id> --yes` (full spec application). `--skip-specs` only when explicitly justified for tooling-only changes with zero spec deltas.
- **Dependency order:** When archiving multiple changes that modify the same capability specs, archive parent changes first, then children
- **Validation:** `openspec validate --strict` after all archives complete
- **Must be on main:** Switch to main and pull before archiving (never archive on feature branches)
- **Commit archives:** Stage and commit archive changes, push to main
- **Reference `openspec/AGENTS.md`** as the authoritative source for archive best practices
- Remove all references to `git mv` and manual directory moves
- Drop the archive README template — `openspec archive` handles spec updates and archive metadata lives in the archive directory itself

### Kept from current version

- Verification checklist
- Common scenarios (simple bug fix vs OpenSpec feature)
- Troubleshooting for "branch not fully merged"
- `git branch -d` (not `-D`) safety guidance

## Deliverable 3: pre-merge.md — Clean Phased Structure

**Action:** Tighten verbose version with bloom-desktop's phased approach and output template.

### Phases

1. **Code Quality** — `uv run black --check sleap_roots tests`, `uv run pydocstyle --convention=google sleap_roots/`
2. **Tests & Coverage** — `uv run pytest tests/`, coverage check, `openspec validate --strict` (if OpenSpec change exists), benchmark run if applicable (note #143)
3. **Documentation** — README, docstrings, OpenSpec task completion check
4. **PR Creation** — `gh pr create` with comprehensive description
5. **CI Monitoring** — `gh pr checks`, watch for cross-platform failures (Ubuntu/Windows/macOS)
6. **Review Feedback** — Copilot comments, reviewer feedback, codecov reports
7. **Changelog** — Run `/changelog`
8. **Final Verification** — All green, branch up-to-date, no conflicts

### Output format

Markdown summary template with checkmarks (from bloom-desktop):

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

## Pull Request
- [x] PR created: #X
- [x] All checks passing

## Changelog
- [x] Entry added (or N/A)

## Status: READY TO MERGE
```

### Removed

- TypeScript type checking, build & package verification, E2E tests (bloom-desktop specific)

### Kept from current version

- Planning mode template for addressing issues
- Troubleshooting section
- Integration with other `/commands`

## Deliverable 4: new-feature.md — New Command

**Action:** Create new command adapted from bloom-desktop.

### Content

**Intro:** "You are a scientific programmer that values testing, code quality, reproducibility, metadata preservation, traceability, interpretability, and performance."

**Guardrails:**
- Do NOT write implementation code until proposal is approved
- Follow OpenSpec conventions strictly (see `openspec/AGENTS.md`)
- Use TDD when implementing (tests before implementation code)
- Always ask clarifying questions — do not assume

**Steps:**
1. **Ensure feature branch** — Check not on `main`, suggest branch name based on feature, create if needed
2. **Understand scope** — Use subagents to explore codebase: existing trait modules, pipeline classes (including the networkx trait dependency graph in `trait_pipelines.py`), test fixtures, related specs
3. **Ask clarifying questions** — Requirements, edge cases, biological validity, data handling, coordinate systems, impact on published results, memory considerations for large datasets, which plant types affected (dicot/monocot), relevant test data in `tests/data/`
4. **Create OpenSpec proposal** — Run `/openspec:proposal`, ground in codebase exploration, TDD approach in `tasks.md`
5. **Review proposal** — Run openspec-review skill for 5-subagent adversarial review. If BLOCKED, fix issues and re-run review until APPROVED or NEEDS REVISION.
6. **Get user approval** — Present reviewed proposal, wait for explicit approval
7. **Implement with TDD** — Run `/openspec:apply`, tests before implementation code

## Deliverable 5: openspec-review Skill — 5-Subagent Proposal Review

**Action:** Create new skill at `.claude/skills/openspec-review/SKILL.md`.

### Structure

**Step 1: Identify proposal** — User specifies change ID, or `ls openspec/changes/` and ask.

**Step 2: Gather context** — Read proposal.md, tasks.md, design.md (if exists), delta specs, current specs being modified.

**Step 3: Launch 5 subagents in parallel:**

1. **Spec Quality & OpenSpec Best Practices**
   - Delta sections use correct headers (`## ADDED|MODIFIED|REMOVED Requirements`)
   - Requirements use `### Requirement: Name`, scenarios use `#### Scenario: Name`
   - Every requirement has at least one scenario
   - GIVEN/WHEN/THEN format with bold markers
   - MODIFIED requirements include FULL existing text
   - Scenarios specific enough to write a test from
   - Change ID is verb-led kebab-case
   - Missing edge case scenarios
   - **Flag any `--skip-specs` usage in proposals or tasks as requiring explicit justification**
   - Reference `openspec/AGENTS.md` for authoritative format rules

2. **Code & Architecture Feasibility**
   - Read every file listed in Impact section
   - Verify proposal's claims about current code state
   - Check numpy/attrs/pipeline patterns respected
   - Understand the networkx trait dependency graph in `trait_pipelines.py` — where do new traits fit?
   - Identify breaking changes proposal might have missed
   - Ripple effects in unlisted files
   - Cross-platform consistency (pathlib, no hardcoded paths)
   - **Memory feasibility** — will the design load all frames/data into a single array? Does it stream or batch?

3. **GitHub Issues & Requirements Alignment**
   - `unset GITHUB_TOKEN && gh issue view` for each linked issue
   - Search for related issues
   - Check proposal fully addresses issue requirements
   - Check for scope gaps and scope creep
   - Verify no contradictions with issue discussions

4. **TDD & Testing Strategy**
   - Tests written before implementation (TDD order)
   - Right framework for each test (pytest unit, pytest-benchmark)
   - Right test level: unit tests (`test_*.py`) for individual trait modules, integration-level tests (`test_trait_pipelines.py`) for full pipeline data flow
   - **Metadata preservation:** Do proposed tests verify metadata is correctly written and preserved through pipeline stages?
   - **Data flow correctness:** Do proposed tests verify intermediate results are passed correctly between trait computations in the pipeline DAG?
   - Missing tests: error paths, boundary values, empty arrays, NaN
   - Scenario-to-test 1:1 mapping
   - Commit safety: subsections committable independently
   - Existing test breakage check
   - Verification check gates between sections
   - CI platform differences (Ubuntu/Windows/macOS)

5. **Scientific Rigor & Data Integrity**
   - Trait computation accuracy
   - Reproducibility — units explicit, defaults documented
   - Impact on published results
   - Coordinate system consistency
   - Algorithm references
   - **Numerical stability** — NaN propagation, floating point precision, `warnings.filterwarnings` suppression justification
   - Memory usage for large datasets (OOM prevention)
   - Data format stability — CSV column names/order changes are breaking
   - Migration path for breaking changes
   - `sleap-io` version compatibility (>= 0.0.11)

**Step 4: Synthesize** — Deduplicate, BLOCKING/IMPORTANT/SUGGESTION, verdict (APPROVED/NEEDS REVISION/BLOCKED).

**Step 5: Offer to fix** — Ask user if they want automatic fixes, revised proposal, or GitHub issues for discussion items.
