# Design: Add Claude Commands for Developer Workflow

## Overview
This change adds 7 Claude Code slash commands to `.claude/commands/` that streamline common developer workflows in sleap-roots.

## Architecture

### Command Structure
Each command is a markdown file in `.claude/commands/` that follows this structure:

```markdown
# Command Title

Brief description of what the command does.

## Commands

```bash
# Actual shell commands to execute
command --flags
```

## What to do after running

Step-by-step guidance on interpreting results and next actions.

## Context-Specific Notes

Project-specific information (e.g., test data location, CI requirements).

## Examples

Concrete examples showing expected usage.
```

### Command Categories

#### 1. Quality Assurance Commands
- `/lint` - Code formatting and style checks
- `/coverage` - Test coverage analysis
- `/test` - Test execution

**Design Pattern**: Execute CI checks locally before pushing

#### 2. PR Workflow Commands
- `/pr-description` - Comprehensive PR templates
- `/review-pr` - Review checklists and workflow
- `/cleanup-merged` - Post-merge cleanup

**Design Pattern**: Standardize git/GitHub workflows

#### 3. Documentation Commands
- `/changelog` - Changelog maintenance following Keep a Changelog

**Design Pattern**: Maintain documentation as code evolves

## Adaptation from cosmos-azul

The commands are based on proven patterns from cosmos-azul but adapted for sleap-roots:

### Key Differences

| Aspect | cosmos-azul | sleap-roots |
|--------|-------------|-------------|
| **Package Type** | Monorepo (Turborepo) | Single package |
| **Language** | TypeScript | Python |
| **Test Framework** | Vitest | pytest |
| **Formatter** | Prettier | Black |
| **Linter** | ESLint | pydocstyle |
| **Package Manager** | pnpm | conda/pip |
| **Build System** | Turborepo | setuptools |

### Adaptation Strategy

1. **Remove monorepo context**: No package-specific sections, no workspace references
2. **Replace TypeScript tooling**: Prettier → Black, ESLint → pydocstyle, type-check → mypy (if used)
3. **Python-specific additions**: 
   - Google-style docstring requirements
   - Cross-platform conda/pip notes
   - Git LFS test data considerations
4. **Scientific context**: 
   - Trait computation accuracy validation
   - Reproducibility concerns for published results
   - Domain-specific review criteria (root phenotyping)

## Command Specifications

### `/lint`
**Purpose**: Run Black and pydocstyle checks to ensure code quality

**Commands**:
```bash
black --check sleap_roots tests
pydocstyle --convention=google sleap_roots/
```

**Key Features**:
- Explains Black formatting standard (88 char line length)
- Guides on fixing docstring issues
- Notes that CI runs these checks

### `/coverage`
**Purpose**: Analyze test coverage and identify untested code

**Commands**:
```bash
pytest --cov=sleap_roots --cov-report=xml tests/
pytest --cov=sleap_roots --cov-report=html tests/
```

**Key Features**:
- Displays coverage summary interpretation
- Notes Codecov integration
- Explains coverage goals (target full coverage)
- Shows how to open HTML report

### `/test`
**Purpose**: Run pytest test suite with helpful options

**Commands**:
```bash
pytest tests/
pytest tests/test_lengths.py
pytest -k "test_primary_root"
pytest -v tests/
```

**Key Features**:
- Shows filtering options (by file, by test name)
- Notes cross-platform testing (Ubuntu, Windows, macOS)
- References test data in `tests/data/` with Git LFS

### `/pr-description`
**Purpose**: Generate comprehensive PR descriptions

**Template Sections**:
- Summary (1-2 sentences)
- Changes (bulleted list)
- Testing checklist (pytest, coverage)
- Linting checklist (Black, pydocstyle)
- Coverage verification
- Breaking changes
- Related issues
- Reviewer notes

**Key Features**:
- Python/pytest-specific checklist items
- Scientific accuracy considerations
- Cross-platform testing notes
- Examples for feature and bug fix PRs

### `/review-pr`
**Purpose**: Systematic PR review workflow

**Review Checklist**:
- Code quality (PEP 8, type hints, docstrings)
- Testing (pytest coverage, edge cases)
- Documentation (README, docstrings, breaking changes)
- Scientific accuracy (trait computation validation)
- Cross-platform compatibility
- Security (no secrets, sensitive data handling)

**Key Features**:
- Python-specific checks (Google-style docstrings, type hints)
- Domain-specific validation (root phenotyping accuracy)
- GitHub CLI commands for review workflow
- Examples of constructive review comments

### `/cleanup-merged`
**Purpose**: Clean up after PR merge

**Workflow**:
1. Verify merge status with `gh pr list --state merged`
2. Switch to main and pull: `git checkout main && git pull`
3. Delete branch: `git branch -d <branch> && git remote prune origin`
4. Archive OpenSpec change (if applicable)
5. Update archive README
6. Commit and push

**Key Features**:
- Handles both simple PRs and OpenSpec-documented changes
- Uses `git mv` to preserve history
- Verification commands

### `/changelog`
**Purpose**: Maintain CHANGELOG.md following Keep a Changelog

**Key Features**:
- Keep a Changelog format explanation
- Git commands to view changes since last tag
- Change categorization (Added, Changed, Fixed, etc.)
- SemVer quick reference
- Release checklist
- Examples adapted for sleap-roots

## Implementation Details

### File Locations
```
.claude/
└── commands/
    ├── lint.md
    ├── coverage.md
    ├── test.md
    ├── pr-description.md
    ├── review-pr.md
    ├── cleanup-merged.md
    ├── changelog.md
    └── openspec/
        ├── proposal.md
        ├── apply.md
        └── archive.md
```

### Command Discovery
Commands are discovered automatically by Claude Code. Users can:
- Type `/` to see available commands
- Use tab completion
- List commands with `ls .claude/commands/`

### Integration with Existing Workflow

**CI/CD Integration**:
- Commands mirror CI checks (`.github/workflows/ci.yml`)
- Allows developers to catch issues before pushing
- No changes to CI configuration required

**Git Workflow Integration**:
- `/pr-description` integrates with GitHub CLI (`gh pr create`)
- `/review-pr` uses `gh pr review`
- `/cleanup-merged` uses git and gh commands

**OpenSpec Integration**:
- `/cleanup-merged` knows how to archive OpenSpec changes
- Commands reference `openspec/project.md` for context

## Testing Strategy

### Manual Testing
Each command should be tested by:
1. Executing the command in Claude Code
2. Verifying commands execute without errors
3. Checking output is helpful and actionable
4. Confirming examples work as documented

### Cross-Platform Considerations
- **Linux/macOS**: Standard bash commands work
- **Windows**: Commands should work in Git Bash or PowerShell where applicable
- Note platform-specific differences in command docs

## Maintenance

### Keeping Commands Updated
- Review commands when CI configuration changes
- Update when new tools are added (e.g., mypy for type checking)
- Revise as project conventions evolve

### Version Control
- Commands are tracked in git
- Changes follow standard PR workflow
- Document command changes in CHANGELOG.md

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Commands become outdated | Medium | Regular review, version control |
| Commands don't work cross-platform | Low | Test on multiple platforms, document limitations |
| Too prescriptive for some workflows | Low | Commands are helpers, not requirements |
| Duplication with existing docs | Low | Commands provide active workflows vs passive docs |

## Success Metrics

- Commands successfully execute their workflows
- Developers use commands regularly
- Fewer CI failures due to local pre-checking
- More consistent PR descriptions and reviews
- Faster onboarding for new contributors

## Future Enhancements

Potential future commands (not in scope for this change):
- `/release` - Automated release workflow
- `/benchmark` - Performance benchmarking for trait computation
- `/validate-traits` - Validate trait outputs against known good data
- `/docs` - Generate or update documentation
