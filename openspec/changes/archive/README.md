# Archived OpenSpec Changes

This directory contains completed OpenSpec changes that have been implemented and merged.

## Active Archives

### add-html-prediction-viewer (February 2026)

**Status**: ✅ Completed - Merged in PR #142

Added a self-contained HTML viewer for visual QC of SLEAP root predictions with multiple output modes, keyboard navigation, and confidence visualization.

- **Proposal**: [proposal.md](2026-02-11-add-html-prediction-viewer/proposal.md)
- **Design**: [design.md](2026-02-11-add-html-prediction-viewer/design.md)
- **Tasks**: [tasks.md](2026-02-11-add-html-prediction-viewer/tasks.md)

**Key Deliverables**:

- `sleap-roots viewer` CLI command with three output modes
- Client-render mode (default) for fast generation with Canvas overlays
- Pre-rendered mode (`--render`) for matplotlib-rendered shareable images
- Embedded mode (`--embed`) for single self-contained HTML files
- ZIP archive support (`--zip`) for easy sharing
- Keyboard navigation (arrow keys, Enter, Escape, C for mode toggle)
- Dual visualization: root-type coloring and confidence-based colormap
- Frame sampling with configurable limits
- Comprehensive user documentation

**Timeline**: ~5 days (iterative development across multiple sub-changes)

---

### add-external-image-mode (February 2026)

**Status**: ✅ Completed - Merged in PR #142

Implemented the three-mode output system (client-render, pre-rendered, embedded) and ZIP archive support for the HTML prediction viewer.

- **Proposal**: [proposal.md](2026-02-11-add-external-image-mode/proposal.md)
- **Tasks**: [tasks.md](2026-02-11-add-external-image-mode/tasks.md)

**Key Deliverables**:

- Client-render mode with JSON prediction data and Canvas rendering
- Pre-rendered mode saving matplotlib images to disk as JPEG/PNG
- Embedded mode maintaining backwards compatibility with base64
- ZIP archive creation for all three modes
- H5 video frame extraction for client-render mode
- Overlay toggle (show/hide predictions) in client-render mode
- CLI options: `--render`, `--embed`, `--format`, `--quality`, `--zip`

---

### add-viewer-documentation (February 2026)

**Status**: ✅ Completed - Merged in PR #142

Added user documentation for the HTML prediction viewer feature.

- **Proposal**: [proposal.md](2026-02-11-add-viewer-documentation/proposal.md)
- **Tasks**: [tasks.md](2026-02-11-add-viewer-documentation/tasks.md)

**Key Deliverables**:

- User guide at `docs/guides/prediction-viewer.md`
- CLI reference with all options documented
- Keyboard navigation reference
- Output mode explanations and use cases
- ZIP archive portability documentation
- README.md updated with viewer section
- docs/guides/index.md updated with link

---

### fix-viewer-code-quality (February 2026)

**Status**: ✅ Completed - Merged in PR #142

Addressed GitHub Copilot code review feedback for the HTML prediction viewer.

- **Proposal**: [proposal.md](2026-02-11-fix-viewer-code-quality/proposal.md)
- **Tasks**: [tasks.md](2026-02-11-fix-viewer-code-quality/tasks.md)

**Key Deliverables**:

- Removed unused imports from multiple files
- Added warnings for silent exception handlers
- Case-insensitive image extension matching
- Improved sort_key for non-numeric filenames
- Security fix: changed `|safe` to `|tojson` in template
- Added sampled_frame_count for accurate frame navigation
- Test coverage improvements (94%+ on viewer modules)

---

### migrate-to-uv (November 2025)

**Status**: ✅ Completed - Merged in PR #132

Migrated sleap-roots from conda/pip-based dependency management to uv, a modern Python package manager providing 10-100x faster dependency resolution and reproducible builds via lockfiles.

- **Proposal**: [proposal.md](migrate-to-uv/proposal.md)
- **Design**: [design.md](migrate-to-uv/design.md)
- **Tasks**: [tasks.md](migrate-to-uv/tasks.md)

**Key Deliverables**:

- Complete migration from conda to uv for dependency management
- PEP 735 dependency groups for dev dependencies
- Lockfile-based reproducibility with uv.lock
- Dual dependency specification (uv + pip compatibility)
- Updated CI workflows to use uv exclusively
- Comprehensive documentation updates
- Developer workflow improvements

**Technical Changes**:
- Removed environment.yml, migrated all deps to pyproject.toml
- Added [dependency-groups] for uv (dev, test, lint, docs)
- Maintained [project.optional-dependencies] for pip users
- Updated all GitHub Actions workflows to use uv
- Created uv.lock for reproducible environments
- Updated README, contributing docs, and installation guides
- Modified all Claude commands to use `uv run`

**Performance Improvements**:
- Dependency resolution: 5-15 minutes → 5-10 seconds (60-180x faster)
- CI installation time: ~3 minutes → ~20 seconds (9x faster)
- Full environment setup: ~10 minutes → ~30 seconds (20x faster)

**Timeline**: ~4 hours (vs. 4-5 hour estimate)
- OpenSpec planning and proposal: 0.5 hours
- Migration implementation: 2 hours
- CI/CD updates: 0.5 hours
- Documentation updates: 0.5 hours
- Testing and validation: 0.5 hours

**Impact**:
- Dramatically faster developer onboarding
- Reproducible builds across all platforms
- Simplified dependency management (single tool)
- Reduced CI costs with faster builds
- Better developer experience with instant feedback
- Foundation for modern Python development practices

---

### add-mkdocs-documentation (November 2025)

**Status**: ✅ Completed - Merged in PR #131

Implemented comprehensive MkDocs-based documentation infrastructure with Material theme, auto-generated API reference, user guides, developer documentation, and automated deployment to GitHub Pages.

- **Proposal**: [proposal.md](add-mkdocs-documentation/proposal.md)
- **Design**: [design.md](add-mkdocs-documentation/design.md)
- **Tasks**: [tasks.md](add-mkdocs-documentation/tasks.md)

**Key Deliverables**:

- **Phase 1 Complete**: Full infrastructure setup with automated deployment
- MkDocs Material theme with plant-themed green styling
- Auto-generated API reference using mkdocstrings
- Complete navigation structure with 7 pipeline tutorials
- GitHub Actions CI/CD workflow for automated deployment
- Comprehensive installation guide with uv best practices
- Developer setup guide (445 lines)
- Complete changelog from v0.0.1 through v0.1.4
- Cookbook recipes for common tasks
- Custom CSS and MathJax configuration

**Documentation Structure Created**:
- Getting Started (installation, quick start, SLEAP intro)
- Tutorials (7 pipeline types with examples)
- User Guide (pipelines, data formats, troubleshooting)
- Developer Guide (contributing, setup, architecture, testing)
- API Reference (auto-generated from docstrings)
- Cookbook (filtering, custom traits, batch optimization, exporting)
- Changelog (following Keep a Changelog format)

**Timeline**: ~9 hours (vs. 9 hour Phase 1 estimate)
- Infrastructure setup: 3 hours
- Documentation content: 4 hours
- CI/CD deployment: 2 hours
- Documentation improvements (uv, changelog, dev guide): 2 hours
- GitHub Copilot feedback resolution: 0.5 hours

**Deployment**:
- Live at: https://talmolab.github.io/sleap-roots/
- Automated deployment via GitHub Actions
- Multi-version support with mike
- Strict build mode to catch errors

**Impact**:
- Professional documentation site for users and developers
- Improved onboarding with comprehensive guides
- Auto-generated API reference from docstrings
- Reduced support burden with troubleshooting guides
- Foundation for future documentation phases (API details, more tutorials)

**Next Phases Planned**:
- Phase 2: Complete API documentation for all 13 modules
- Phase 3: Detailed user guides and workflow documentation
- Phase 4: Comprehensive trait reference
- Phase 5: Developer architecture documentation
- Phase 6: Cookbook expansion with Jupyter notebooks
- Phase 7: Polish, versioning, and cross-linking

---

### add-claude-commands (November 2025)

**Status**: ✅ Completed - Merged in PR #130

Added comprehensive Claude Code slash command suite for streamlined developer workflows, including linting, testing, coverage analysis, PR management, changelog maintenance, and documentation updates.

- **Proposal**: [proposal.md](add-claude-commands/proposal.md)
- **Design**: [design.md](add-claude-commands/design.md)
- **Tasks**: [tasks.md](add-claude-commands/tasks.md)

**Key Deliverables**:

- 13 slash commands total (7 workflow + 5 development + 1 documentation)
- Commands adapted from cosmos-azul for Python/pytest/single-package context
- Comprehensive templates and checklists for PR workflow, testing, and development
- Scientific accuracy validation guidelines for trait computations
- Cross-platform support documentation (Ubuntu, Windows, macOS)

**Commands Implemented**:
1. `/lint` - Black formatting and pydocstyle checks
2. `/coverage` - Pytest coverage analysis
3. `/test` - Run pytest test suite
4. `/pr-description` - Comprehensive PR templates
5. `/review-pr` - Systematic PR review workflow
6. `/cleanup-merged` - Post-merge branch cleanup and OpenSpec archiving
7. `/changelog` - Maintain CHANGELOG.md following Keep a Changelog format
8. `/run-ci-locally` - Run exact CI checks locally before pushing
9. `/fix-formatting` - Auto-fix Black formatting issues
10. `/validate-env` - Check development environment setup
11. `/new-pipeline` - Scaffold new pipeline classes with tests
12. `/debug-test` - Enhanced test debugging with pytest flags
13. `/docs-update` - Systematic documentation maintenance

**OpenSpec Documentation**:
- Created `openspec/project.md` with comprehensive project context
- Added `CLAUDE.md` and `AGENTS.md` for AI assistant instructions
- Documented tech stack, conventions, architecture patterns
- Added scientific context for plant root phenotyping domain

**Timeline**: ~6 hours total (vs. 4-5 hour initial estimate)
- OpenSpec planning and proposal: 1 hour
- Initial 7 commands implementation: 3 hours
- Development commands (5): 1.5 hours
- Documentation command: 0.5 hour

**Impact**:
- Reduced CI failures through local pre-checking
- Improved developer productivity with automation
- Standardized PR and review workflows
- Better onboarding experience for new contributors
- Comprehensive documentation for project context