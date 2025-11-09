# Tasks: Add Claude Commands

## Overview
Implement 7 Claude Code slash commands to streamline developer workflows in sleap-roots.

## Task List

### 1. Create `/lint` Command
**Priority**: High  
**Estimated Time**: 30 minutes

- [ ] Create `.claude/commands/lint.md`
- [ ] Add commands for Black formatting check
- [ ] Add commands for pydocstyle docstring checks
- [ ] Include guidance on fixing common issues
- [ ] Add "What to do after running" section
- [ ] Test command execution

**Deliverable**: Working `/lint` command that runs Black and pydocstyle checks

---

### 2. Create `/coverage` Command
**Priority**: High  
**Estimated Time**: 30 minutes

- [ ] Create `.claude/commands/coverage.md`
- [ ] Add pytest coverage commands
- [ ] Include coverage report interpretation guidance
- [ ] Document coverage goals (target full coverage, current tracked via Codecov)
- [ ] Add commands to view HTML coverage report
- [ ] Test command execution

**Deliverable**: Working `/coverage` command with coverage analysis guidance

---

### 3. Create `/test` Command
**Priority**: High  
**Estimated Time**: 20 minutes

- [ ] Create `.claude/commands/test.md`
- [ ] Add pytest commands with common flags
- [ ] Include filtering options (by module, test name)
- [ ] Add cross-platform notes (Windows, macOS, Ubuntu)
- [ ] Document test data location (tests/data/ with Git LFS)
- [ ] Test command execution

**Deliverable**: Working `/test` command for running pytest

---

### 4. Create `/pr-description` Command
**Priority**: Medium  
**Estimated Time**: 45 minutes

- [ ] Create `.claude/commands/pr-description.md`
- [ ] Adapt cosmos-azul template for single-package project (not monorepo)
- [ ] Include Python/pytest-specific checklist items
- [ ] Add sections for: Summary, Changes, Testing, Linting, Coverage, Breaking Changes
- [ ] Include examples for feature PRs and bug fix PRs
- [ ] Add GitHub CLI commands for PR creation
- [ ] Remove monorepo-specific sections

**Deliverable**: PR description template adapted for sleap-roots

---

### 5. Create `/review-pr` Command
**Priority**: Medium  
**Estimated Time**: 45 minutes

- [ ] Create `.claude/commands/review-pr.md`
- [ ] Add review checklist for code quality, testing, documentation
- [ ] Include Python-specific checks (type hints, docstrings, Google-style)
- [ ] Add scientific accuracy considerations (trait computations, reproducibility)
- [ ] Include cross-platform compatibility notes
- [ ] Add GitHub CLI review commands
- [ ] Document review response workflow
- [ ] Include examples of good vs less helpful review comments

**Deliverable**: Comprehensive PR review workflow command

---

### 6. Create `/cleanup-merged` Command
**Priority**: Medium  
**Estimated Time**: 30 minutes

- [ ] Create `.claude/commands/cleanup-merged.md`
- [ ] Add workflow for switching to main and pulling
- [ ] Include branch deletion commands (local + remote)
- [ ] Add OpenSpec change archiving steps
- [ ] Include verification commands
- [ ] Document common scenarios (with/without OpenSpec)
- [ ] Test workflow steps

**Deliverable**: Post-merge cleanup automation command

---

### 7. Create `/changelog` Command
**Priority**: Low  
**Estimated Time**: 45 minutes

- [ ] Create `.claude/commands/changelog.md`
- [ ] Document Keep a Changelog format
- [ ] Add git commands for viewing changes since last release
- [ ] Include change categorization guidance (Added, Changed, Fixed, etc.)
- [ ] Add SemVer quick reference
- [ ] Include release checklist
- [ ] Adapt examples for sleap-roots (remove monorepo-specific content)
- [ ] Note current version location (`sleap_roots/__version__`)

**Deliverable**: Changelog maintenance workflow command

---

### 8. Update Documentation
**Priority**: Low  
**Estimated Time**: 15 minutes

- [ ] Check if README.md should reference new commands
- [ ] Update CLAUDE.md if necessary
- [ ] Document command discovery (`ls .claude/commands/`)

**Deliverable**: Documentation references new slash commands if appropriate

---

### 9. Test All Commands
**Priority**: High  
**Estimated Time**: 30 minutes

- [ ] Test `/lint` executes correctly
- [ ] Test `/coverage` executes correctly
- [ ] Test `/test` executes correctly
- [ ] Verify `/pr-description` template is comprehensive
- [ ] Verify `/review-pr` checklist is complete
- [ ] Verify `/cleanup-merged` workflow is accurate
- [ ] Verify `/changelog` guidance is helpful
- [ ] Test on different platforms if applicable

**Deliverable**: All commands verified working

---

## Dependencies

- Task 8 (documentation) depends on tasks 1-7 being complete
- Task 9 (testing) should run continuously as commands are created

## Parallelization Opportunities

- Tasks 1-7 can be done in parallel (independent commands)
- Task 8 must wait for tasks 1-7
- Task 9 runs concurrently

## Validation

Each command should:
- Execute without errors
- Provide clear, actionable guidance
- Include relevant examples
- Be adapted for sleap-roots context (Python, pytest, single package)
- Follow the structure from cosmos-azul where applicable

## Timeline

- **Phase 1** (Core): Tasks 1-3 (lint, coverage, test) - ~1.5 hours
- **Phase 2** (PR Workflow): Tasks 4-6 (PR desc, review, cleanup) - ~2 hours
- **Phase 3** (Polish): Tasks 7-9 (changelog, docs, testing) - ~1.5 hours

**Total**: ~4-5 hours
