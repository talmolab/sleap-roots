---
description: Critically review an OpenSpec proposal using a team of specialized subagents before approval
---

# OpenSpec Proposal Review — Subagent Team

You are a senior scientific programmer reviewing an OpenSpec proposal for `sleap-roots`,
a pure Python library for plant root phenotyping that extracts root traits from SLEAP pose
estimation predictions. You value testing, code quality, reproducibility, biological and
scientific validity, metadata preservation, traceability, correctness, and documentation
that is clear, succinct, and DRY.

This skill launches **5 specialized subagents in parallel** to critically review an OpenSpec proposal.
Each subagent has a distinct review lens and is instructed to be **adversarial** — finding gaps, not rubber-stamping.
After all subagents return, you synthesize their findings into a unified review verdict.

**Arguments:** `$ARGUMENTS` (the change-id to review)

## Step 1: Identify the Proposal

Determine which proposal to review:

- If the user specifies a change ID via `$ARGUMENTS`, use it directly
- Otherwise, run `openspec list` to find active proposals and ask the user which one to review
- Read the proposal's `proposal.md`, `tasks.md`, `design.md` (if exists), and all delta spec files under `specs/`

## Step 2: Gather Context

Before launching subagents, collect essential context that each agent will need:

1. Read the full proposal files (proposal.md, tasks.md, design.md, delta specs)
2. Read the CURRENT specs being modified (from `openspec/specs/`)
3. Read `openspec/AGENTS.md` for OpenSpec conventions
4. Read `openspec/project.md` for project conventions (tech stack, architecture, testing, constraints)
5. Note the affected code files listed in the Impact section
6. Note any related GitHub issues mentioned

Embed the full proposal text, current spec text, and file lists into each subagent prompt.

**Do not trust any counts, version numbers, or infrastructure details from memory or from this
document.** Every subagent is instructed to READ the actual repo files (`.github/workflows/`,
`pyproject.toml`, `docs/`, `tests/`, `openspec/`) and report what is actually there.

## Step 3: Launch Subagent Review Team

Launch ALL 5 subagents **in a single message** (parallel execution). Each subagent gets the full proposal
text embedded in its prompt. Each agent MUST read the actual files it needs — do not rely on summaries.

---

### Subagent 1: Spec Quality & OpenSpec Best Practices

```
subagent_type: "general-purpose"
description: "Review OpenSpec format quality"
```

**Prompt template:**

> You are reviewing an OpenSpec proposal for `sleap-roots`, a pure Python library for plant root
> phenotyping built on SLEAP pose estimation.
> Your role: **Spec Quality & OpenSpec Best Practices Reviewer**.
>
> IMPORTANT: Be critical. Find problems. Do NOT rubber-stamp.
>
> First, read `openspec/AGENTS.md` to understand the full OpenSpec format rules.
> Then read the proposal files and current specs being modified.
>
> **Format rules to check:**
>
> - Delta sections MUST use: `## ADDED Requirements`, `## MODIFIED Requirements`, `## REMOVED Requirements`
> - Requirements use `### Requirement: Name` (3 hashtags)
> - Scenarios use `#### Scenario: Name` (4 hashtags)
> - Every requirement MUST have at least one scenario
> - Scenarios MUST use **WHEN**/**THEN** format with bold markers
> - MODIFIED requirements MUST include the FULL existing text (partial deltas lose detail at archive)
> - Requirements use SHALL/MUST for normative language
>
> **Proposal rules:**
>
> - `proposal.md` must have: ## Why, ## What Changes, ## Impact
> - ## Why should be 1-2 sentences explaining the problem/opportunity
> - ## Impact must list: affected specs AND affected code files
> - BREAKING changes must be marked with **BREAKING**
> - Change ID must be verb-led kebab-case
>
> **Tasks rules:**
>
> - Must follow TDD order: tests FIRST, then implementation, then verification
> - Tasks must be small, verifiable work items (suitable for atomic commits)
> - Each task must have a checkbox `- [ ]`
> - Task groups should map to logical commit boundaries
>
> **Check for:**
>
> 1. Are any scenarios vague or untestable? (e.g., "should work correctly")
> 2. Are WHEN/THEN conditions specific enough to write a test from?
> 3. Do MODIFIED requirements include the FULL original text or just fragments?
> 4. Are there requirements without scenarios?
> 5. Are there missing edge case scenarios? (error paths, empty/NaN inputs, single-point geometry, boundary values)
> 6. Does the Impact section list ALL affected specs and code files? (e.g. the relevant trait module
>    in `sleap_roots/`, `sleap_roots/trait_pipelines.py` if the trait DAG changes, fixtures/test data)
> 7. Could any requirements be split into smaller, more focused requirements?
> 8. Is the change ID appropriate (verb-led, descriptive)?
> 9. Run `openspec validate {CHANGE_ID} --strict` and report the result
>
> **Proposal to review:**
> {PROPOSAL_MD}
>
> **Tasks:**
> {TASKS_MD}
>
> **Delta specs:**
> {DELTA_SPECS}
>
> **Current specs being modified:**
> {CURRENT_SPECS}
>
> Return a structured review with:
> - PASS/FAIL verdict for each check
> - Specific issues found with suggested rewrites
> - Overall quality score (1-10) with justification

---

### Subagent 2: TDD & Testing Strategy

```
subagent_type: "general-purpose"
description: "Review TDD and testing plan"
```

**Prompt template:**

> You are reviewing an OpenSpec proposal's testing strategy for `sleap-roots`.
> Your role: **TDD & Testing Strategy Reviewer**.
>
> IMPORTANT: Be critical. The test plan must be concrete, complete, and CI-feasible.
> Do NOT trust any test counts or fixture counts you may have seen elsewhere — READ the repo
> and report what is actually there.
>
> **Discover the project testing infrastructure by reading the actual files:**
>
> - `.github/workflows/ci.yml` — the CI jobs, the OS/Python matrix, lint vs test vs benchmark jobs,
>   how coverage is collected, and how benchmarks are run/compared
> - `docs/dev/testing.md` — the testing guide (framework, conventions, how to run)
> - `pyproject.toml` — the `[dependency-groups] dev` / `[project.optional-dependencies] dev` test deps
>   (pytest, pytest-cov, pytest-benchmark, etc.) and `[tool.black]` / `[pydocstyle]` config
> - `tests/` — the actual test layout: one `test_<module>.py` per `sleap_roots/<module>.py`,
>   `tests/test_trait_pipelines.py` for pipeline-level tests, `tests/benchmarks/` for performance,
>   `tests/conftest.py` and `tests/fixtures/` for fixtures, `tests/data/` for real SLEAP `.slp`/`.h5`
>   test data (stored via Git LFS)
>
> Report the ACTUAL framework, markers, matrix, fixture locations, and test data conventions you find —
> do not assume.
>
> **Review the tasks.md for:**
>
> 1. **TDD ordering**: Are tests written BEFORE implementation? The tasks.md should have:
>    - Write failing test → Implement feature → Verify test passes
>    - NOT: Implement feature → Write tests after
> 2. **Test specificity**: Is each test specific enough to implement? Not vague like "verify it works".
>    Good: "returns NaN for an all-NaN input array", "returns empty array for zero roots".
> 3. **Correct test level**: Are the right tools used?
>    - Pure trait function logic (numpy math, geometry) → unit test in `tests/test_<module>.py`
>    - Full pipeline integration (trait DAG, CSV output) → `tests/test_trait_pipelines.py`
>    - Performance-sensitive paths → benchmark test in `tests/benchmarks/`
>    - CLI behaviour (`sleap-roots` entry point) → subprocess / CLI invocation tests
> 4. **Missing tests**:
>    - Empty arrays (zero landmarks, zero frames, zero roots)
>    - NaN inputs (missing keypoints from SLEAP) and mixed valid/NaN inputs
>    - Single-point / degenerate geometry (can't compute angles or lengths)
>    - Metadata preservation through pipeline stages (trace a CSV row back to its source)
>    - Backward compatibility (existing pipelines, existing CSV column names/order)
>    - Regression tests for any bug being fixed
> 5. **CI feasibility**: Will these tests run in CI?
>    - Do any tests require network access, real PyPI, or external services?
>    - Are tests cross-platform safe across the matrix in `ci.yml` (path separators via `pathlib`,
>      line endings, no hardcoded `/`)?
>    - Do tests rely on Git-LFS test data that is actually present in `tests/data/`?
> 6. **Scenario-to-test mapping**: Do delta spec scenarios map 1:1 to tests in tasks.md?
>    - Every scenario SHOULD have a corresponding test. Flag any scenario without a test.
> 7. **Verification section completeness**: Does the tasks.md verification section run the same
>    checks CI runs (read `ci.yml` to confirm the exact commands)? At minimum expect:
>    - `uv run pytest tests/`
>    - `uv run black --check sleap_roots tests`
>    - `uv run pydocstyle --convention=google sleap_roots/`
>    - coverage and (if performance-relevant) `uv run pytest tests/benchmarks/ --benchmark-only`
>
> **Tasks to review:**
> {TASKS_MD}
>
> **Delta specs (scenarios to match against tests):**
> {DELTA_SPECS}
>
> **Proposal summary:**
> {PROPOSAL_MD}
>
> Report:
> - Missing tests (with concrete descriptions of what to add)
> - TDD ordering violations (where implementation comes before tests)
> - Scenarios without corresponding tests (gap analysis)
> - Verification checklist gaps
> - Suggested additional test tasks with exact wording

---

### Subagent 3: CI/CD & Build Infrastructure

```
subagent_type: "general-purpose"
description: "Review CI/CD and build changes"
```

**Prompt template:**

> You are reviewing an OpenSpec proposal for `sleap-roots`.
> Your role: **CI/CD & Build Infrastructure Reviewer**.
>
> IMPORTANT: Be critical. Read the ACTUAL workflow and packaging files. Find real problems.
> Do NOT assume any particular publishing flow, action versions, or build backend — report what
> the files actually contain.
>
> **Read these files and describe the ACTUAL setup before reviewing:**
>
> - `.github/workflows/ci.yml` — lint, test (OS matrix), and benchmark/benchmark-comparison jobs
> - `.github/workflows/build.yml` — what triggers it (e.g. `release: published`), how it builds
>   (`uv build`), how it validates (twine check, wheel/sdist install tests), and HOW it publishes
>   to PyPI (read carefully: is it token-based via a secret, or OIDC trusted publishing? Report the
>   actual mechanism — do not assume.)
> - `.github/workflows/docs.yml` — docs build/deploy (mkdocs / mike), and what triggers it
> - `pyproject.toml` — `[build-system]` (build backend), `[project]` metadata, `dynamic = [...]`
>   (how is the version derived? e.g. `[tool.setuptools.dynamic] version = {attr = ...}`), the
>   `[project.scripts]` entry point, and the dev dependency group
>
> **Review the proposal for:**
>
> 1. **build.yml correctness**: If the proposal changes the build/publish flow:
>    - Will the build (`uv build`) and the validation steps (twine check, wheel install, sdist+dev
>      install) still pass?
>    - Is the publish mechanism handled correctly for the way THIS repo actually publishes (per the
>      file you read)? If it is token-based, is the secret referenced correctly and not exposed in
>      logs? If it is OIDC, are the permissions/environment scoped correctly?
>    - Are there race conditions or failure modes not addressed? Is `--skip-existing` needed on retry?
> 2. **Version derivation**: The version is dynamic (read pyproject.toml + `sleap_roots/__init__.py`
>    / `__version__` to confirm how). If the proposal touches versioning, does the build still resolve
>    a valid version, and do install-smoke-tests that print `sr.__version__` still work?
> 3. **GitHub Actions versions**: Are action versions pinned and internally consistent with what the
>    repo already uses (e.g. `astral-sh/setup-uv`, `actions/checkout`, `codecov/codecov-action`)?
>    Flag any version the proposal introduces that diverges from the rest of the workflows.
> 4. **CI trigger paths**: `ci.yml` only runs on changes to certain paths (read the `on:` block).
>    Will the proposal's changed files actually trigger CI? If it adds a new top-level dir that CI
>    should watch, is the trigger path updated?
> 5. **Cross-platform safety**: Do any workflow shell steps work across the OS matrix? Are bash-only
>    constructs (grep/sed/find) confined to ubuntu jobs?
> 6. **Coverage / benchmarks**: If the change affects coverage upload (Codecov) or the benchmark
>    baseline/comparison jobs, is that handled? (Note the benchmark jobs and the 15% regression
>    threshold defined in `ci.yml`.)
> 7. **Migration risk**: Will these changes break if a release/PR is triggered on the current
>    workflows before the new ones merge?
>
> Read these files:
> - `.github/workflows/ci.yml`
> - `.github/workflows/build.yml`
> - `.github/workflows/docs.yml`
> - `pyproject.toml`
> - `sleap_roots/__init__.py` (for `__version__`)
>
> **Proposal to review:**
> {PROPOSAL_MD}
>
> **Tasks:**
> {TASKS_MD}
>
> Report:
> - Incorrect assumptions about CI behavior (cite the actual file/line)
> - Missing failure handling
> - Security concerns (token exposure, permission scope)
> - Compatibility issues (action versions, trigger paths, build backend)
> - Suggested workflow improvements with concrete YAML

---

### Subagent 4: Documentation Quality (Clear, Succinct, DRY)

```
subagent_type: "general-purpose"
description: "Review documentation impact"
```

**Prompt template:**

> You are reviewing an OpenSpec proposal for `sleap-roots`.
> Your role: **Documentation Quality Reviewer** — you enforce clear, succinct, DRY documentation.
>
> IMPORTANT: Be critical. Read the ACTUAL documentation files. Find real inconsistencies.
> Do NOT trust any embedded line numbers, license claims, or version strings from this prompt —
> open the files and verify against what is actually written.
>
> **Discover and read the documentation that exists** (the docs live under `docs/`; list the
> directory first, then read the relevant files). Typical files to check:
>
> - `docs/changelog.md` — changelog (audit for duplicate headers, placeholder dates, stale or
>   misplaced entries, and whether the [Unreleased] section is organized)
> - `docs/dev/release-process.md` — release guide (compare against the ACTUAL `build.yml` / publish
>   mechanism)
> - `docs/dev/contributing.md` — contributor guide (Python version, setup, uv commands)
> - `docs/dev/testing.md` — testing guide (does it match the actual `ci.yml` and `tests/` layout?)
> - `docs/dev/adding-traits.md`, `docs/dev/creating-pipelines.md`, `docs/dev/architecture.md`,
>   `docs/dev/code-style.md` — read whichever are relevant to the trait/pipeline changes in the proposal
> - `README.md` — project readme (badges, install instructions, version/API references, usage examples)
> - `openspec/project.md` — project conventions (Python version, dependency info, architecture)
> - `mkdocs.yml` — nav structure (does any new doc page need to be added to the nav?)
> - `docs/api/` — generated API reference; new public functions should surface here
>
> **Review for:**
>
> 1. **Completeness**: Does the proposal identify ALL documentation that needs updating?
>    - Python version appears in multiple places (README badge, `docs/dev/contributing.md`,
>      `openspec/project.md`, `pyproject.toml` `requires-python` + classifiers) — find them all and
>      check they agree.
>    - License appears in `LICENSE`, `pyproject.toml`, and possibly README/changelog — confirm they agree.
>    - A new public trait function or pipeline needs: a docstring (Google style), an entry that
>      surfaces in the API docs, and often a README / cookbook / tutorial usage example.
> 2. **DRY violations**: Where is the same information stated in multiple places that could drift?
>    (version numbers, Python versions, dependency lists, trait lists). Prefer a single source of truth
>    plus cross-references.
> 3. **Accuracy after changes**: Will the proposed changes introduce NEW inconsistencies?
>    - If trait names, CSV columns, or pipeline behaviour change, do the docs/tutorials that show
>      example output still match?
>    - If the build/release flow changes, does `docs/dev/release-process.md` still describe reality?
> 4. **Succinctness**: Are any docs verbose, redundant, or describing features that don't exist?
> 5. **Changelog quality**: Is there a clear, dated entry plan for this change in `docs/changelog.md`,
>    following the existing Keep-a-Changelog format? Flag any concrete defects you actually find in the
>    file (duplicate sections, placeholder dates, wrong license text) with the real line numbers.
>
> **Proposal to review:**
> {PROPOSAL_MD}
>
> **Tasks:**
> {TASKS_MD}
>
> Report:
> - Documentation files the proposal MISSED (needs updating but not listed)
> - DRY violations that should be addressed
> - Inaccuracies that will be introduced by the proposed changes
> - Suggested fixes with concrete rewrites

---

### Subagent 5: Git Workflow & Commit Strategy

```
subagent_type: "general-purpose"
description: "Review git workflow plan"
```

**Prompt template:**

> You are reviewing an OpenSpec proposal for `sleap-roots`.
> Your role: **Git Workflow & Commit Strategy Reviewer**.
>
> IMPORTANT: Be critical. Commits should be small, focused, and CI-safe.
>
> **Discover the project's git conventions** by reading `git log --oneline -20` for the actual
> commit-message style, and `openspec/project.md` for the documented workflow. Report what you find
> (conventional prefixes like `feat:` / `fix:` / `docs:` / `chore:`, branch-off-main, squash vs merge,
> OpenSpec archive-after-merge) rather than assuming.
>
> **Review the tasks.md for commit strategy:**
>
> 1. **Atomic commits**: Can each task group be committed independently with CI staying green after each?
>    - Good: separate commits for the trait module change, the pipeline-DAG wiring, tests, and docs.
>    - Bad: one giant commit touching multiple `sleap_roots/` modules + `trait_pipelines.py` +
>      fixtures + tests + docs at once.
> 2. **Commit ordering**: Are there dependencies between tasks?
>    - Must a new fixture / test-data addition land before the tests that consume it? (yes)
>    - Must a trait function exist before `trait_pipelines.py` wires it into the DAG? (yes)
>    - Must tests be committed before/with the implementation to preserve TDD evidence?
> 3. **CI safety**: Will CI stay green between commits (lint + cross-platform pytest + coverage)?
>    - If a trait is wired into a pipeline before its function exists, does the pipeline test break?
>    - Does any intermediate commit drop coverage below the threshold or break pydocstyle/black?
> 4. **Suggested commit plan**: Propose a sequence of small commits with:
>    - Clear conventional commit messages
>    - Files affected per commit
>    - CI state after each commit (green/yellow/red)
>    - Dependencies noted
> 5. **PR strategy**:
>    - Single PR or multiple PRs? If single: is it reviewable (not too large)? If multiple: merge order?
> 6. **Risk mitigation**:
>    - What if a change to `trait_pipelines.py` or the trait DAG breaks an existing pipeline's output?
>    - Is there a rollback plan for each commit?
>
> **Tasks to review:**
> {TASKS_MD}
>
> **Proposal summary:**
> {PROPOSAL_MD}
>
> **Recent commit style** (run `git log --oneline -20`):
> Check the repo for actual commit message conventions.
>
> Report:
> - Tasks that are too large for a single commit
> - Ordering dependencies the proposal missed
> - CI breakage risks at each step
> - Concrete commit plan with messages and file lists
> - PR strategy recommendation

---

## Step 4: Synthesize Review

After ALL subagents return, synthesize their findings:

1. **Deduplicate**: Merge overlapping findings from multiple reviewers
2. **Prioritize**: Categorize issues as:
   - **BLOCKING** — Must fix before approval (spec errors, missing tests, scientific/data integrity risks, CI breakage)
   - **IMPORTANT** — Should fix before implementation (missing edge cases, unclear scenarios, doc gaps)
   - **SUGGESTION** — Nice to have (style improvements, additional context)
3. **Create a unified review** with this structure:

```markdown
# OpenSpec Review: {change-id}

## Verdict: APPROVED / NEEDS REVISION / BLOCKED

## Summary
[2-3 sentence overall assessment]

## Blocking Issues
[Issues that MUST be resolved before approval]

## Important Issues
[Issues that SHOULD be resolved before implementation]

## Suggestions
[Optional improvements]

## Proposed Commit Plan
1. `type: message` — [files affected, CI state after]
2. `type: message` — [files affected, CI state after]
...

## TDD Plan
For each testable change:
1. Test to write first → expected failure → implementation to pass it

## Risk Assessment
- CI breakage risk: LOW/MEDIUM/HIGH — [explanation]
- Regression risk (published results / CSV format): LOW/MEDIUM/HIGH — [explanation]
- Documentation drift risk: LOW/MEDIUM/HIGH — [explanation]

## Review Details by Agent
### 1. Spec Quality
### 2. TDD & Testing
### 3. CI/CD & Build
### 4. Documentation
### 5. Git Workflow
```

## Step 5: Present and Iterate

Present the synthesized review and ask:

1. Do you want to address blocking issues now (update proposal, tasks, and specs)?
2. Do you want to approve with important issues noted as additional tasks?
3. Do you want to revise the proposal first?

If revising, update `proposal.md`, `tasks.md`, and delta specs based on the agreed-upon changes.
Run `openspec validate {change-id} --strict` after any updates.
