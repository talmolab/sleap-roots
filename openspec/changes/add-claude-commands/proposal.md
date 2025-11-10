# Proposal: Add Claude Commands for Developer Workflow

## Change ID
`add-claude-commands`

## Summary
Add Claude Code slash commands to streamline common developer workflows in sleap-roots, including linting, testing, coverage analysis, PR management, changelog updates, and branch cleanup.

## Problem Statement
Currently, developers and AI assistants working on sleap-roots must manually remember and execute common workflow commands. There are no standardized shortcuts for frequent tasks like:

- Running lint checks (Black, pydocstyle)
- Executing tests with coverage
- Creating well-formatted PR descriptions
- Reviewing and responding to PRs
- Updating changelogs following Keep a Changelog format
- Cleaning up merged branches and archiving OpenSpec changes

This increases cognitive load and can lead to inconsistent practices across the team.

## Proposed Solution
Create a set of Claude Code slash commands in `.claude/commands/` that provide:

1. **Workflow automation**: Pre-configured commands for common tasks
2. **Best practice guidance**: Templates and checklists embedded in commands
3. **Context-aware help**: Commands tailored to sleap-roots' Python/pytest/conda stack
4. **Consistency**: Standardized processes for all contributors

## Commands to Add

### Core Development Commands

1. **`/lint`** - Run Black formatting and pydocstyle checks
   - Execute `black --check sleap_roots tests`
   - Execute `pydocstyle --convention=google sleap_roots/`
   - Provide guidance on fixing issues

2. **`/coverage`** - Run tests with coverage analysis
   - Execute `pytest --cov=sleap_roots --cov-report=xml tests/`
   - Display coverage summary
   - Identify untested code areas
   - Check against coverage thresholds

3. **`/test`** - Run test suite with helpful output
   - Execute `pytest tests/` with appropriate flags
   - Support filtering by module or test name
   - Display failures clearly

### PR & Git Workflow Commands

4. **`/pr-description`** - Generate comprehensive PR descriptions
   - Template adapted for sleap-roots (not monorepo)
   - Include checklist for: tests, linting, coverage, breaking changes
   - Reference related issues
   - Python/pytest-specific sections

5. **`/review-pr`** - Systematic PR review workflow
   - Checklist for code quality, testing, documentation
   - Python-specific concerns (type hints, docstrings)
   - Scientific accuracy validation (trait computations)
   - Cross-platform compatibility checks

6. **`/cleanup-merged`** - Clean up after PR merge
   - Switch to main and pull latest
   - Delete merged branch (local + remote)
   - Archive OpenSpec changes if applicable
   - Update archive README

### Documentation Commands

7. **`/changelog`** - Maintain CHANGELOG.md following Keep a Changelog format
   - View recent changes since last tag
   - Template for categorizing changes (Added, Changed, Fixed, etc.)
   - Guidance on version numbering (SemVer)
   - Commands for release preparation

## Benefits

- **Reduced cognitive load**: Common tasks are one command away
- **Consistency**: All contributors use the same workflows
- **Onboarding**: New contributors learn best practices through command templates
- **AI efficiency**: AI assistants can quickly execute standard workflows
- **Quality**: Built-in checklists ensure nothing is missed

## Scope

**In Scope**:
- Commands for lint, test, coverage, PR workflow, changelog, cleanup
- Templates and checklists adapted for sleap-roots
- Python/pytest-specific guidance
- Integration with existing CI/CD (GitHub Actions)

**Out of Scope**:
- Modifying existing CI/CD configuration
- Creating new testing infrastructure
- Changing code style or conventions
- Monorepo-specific features (sleap-roots is a single package)

## Dependencies

- Existing: Black, pydocstyle, pytest, pytest-cov, gh CLI
- No new dependencies required

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Commands become outdated as project evolves | Version control tracks changes; regular reviews during OpenSpec updates |
| Commands don't match actual workflow | Based on proven cosmos-azul commands and sleap-roots CI config |
| Too opinionated for some users | Commands are helpers, not requirements; users can still use direct commands |

## Success Criteria

- [ ] All 7 commands implemented and documented
- [ ] Commands successfully execute their primary workflows
- [ ] Templates include relevant checklists and guidance
- [ ] Commands work on Ubuntu, macOS, Windows (where applicable)
- [ ] Documentation updated to reference new commands

## Alternatives Considered

1. **Just use direct commands**: Rejected - increases cognitive load and inconsistency
2. **Makefile/scripts**: Rejected - slash commands integrate better with Claude Code workflow
3. **Full documentation only**: Rejected - commands provide active assistance vs passive reference

## Related Issues

N/A (new capability)

## Estimated Effort

- **Design**: 1 hour (adapt cosmos-azul commands to sleap-roots)
- **Implementation**: 2-3 hours (create 7 command files)
- **Testing**: 1 hour (verify commands work correctly)
- **Documentation**: 30 minutes (update README if needed)

**Total**: ~4-5 hours
