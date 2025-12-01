# Fix API Documentation Autogeneration Errors

## Summary

Fix all warnings and errors in the MkDocs API documentation autogeneration process to ensure clean builds and properly formatted documentation.

## Problem

Running `uv run mkdocs build` currently produces multiple warnings:

1. **Type annotation warnings** (griffe): Missing type hints in function signatures
2. **Docstring formatting warnings** (griffe): Malformed docstring sections
3. **Navigation warnings**: Broken links to `reference/` directory
4. **Cross-reference warnings**: Missing anchor tags in internal links

These warnings indicate issues that could confuse users and affect documentation quality.

## Goals

1. Fix all type annotation warnings in source code
2. Fix all docstring formatting issues
3. Resolve navigation structure issues
4. Fix broken internal links
5. Achieve zero-warning MkDocs build

## Non-Goals

- Major refactoring of documentation structure (preserve current organization)
- Adding new API documentation (only fix existing)
- Changing mkdocstrings configuration (unless required for fixes)

## Scope

**In Scope:**
- Source code type annotations (sleap_roots/*.py)
- Docstring formatting in source files
- Navigation structure in mkdocs.yml
- Internal link references in docs/api/*.md
- Auto-generated reference pages setup

**Out of Scope:**
- Content improvements beyond fixing errors
- Tutorial or guide updates
- Adding missing documentation for undocumented features

## Impact

**Users Affected:**
- Documentation readers (better quality, no broken links)
- Contributors (clean builds, no confusing warnings)
- Maintainers (easier to spot real issues)

**Breaking Changes:**
- None (documentation fixes only)

## Success Criteria

- [ ] `uv run mkdocs build` completes with zero warnings
- [ ] All internal links resolve correctly
- [ ] All type annotations present where required
- [ ] All docstrings follow Google style correctly
- [ ] Auto-generated reference pages build successfully

## Dependencies

- Requires current mkdocstrings and griffe versions
- No external dependencies

## Timeline

- Phase 1: Fix type annotations (1-2 hours)
- Phase 2: Fix docstring formatting (1 hour)
- Phase 3: Fix navigation and links (30 min)
- Total: ~3 hours of focused work

## References

- MkDocs build output showing warnings
- griffe documentation: https://mkdocstrings.github.io/griffe/
- mkdocstrings documentation: https://mkdocstrings.github.io/
- Google docstring style: https://google.github.io/styleguide/pyguide.html