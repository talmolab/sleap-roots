# Update Getting Started Documentation

## Why

The current getting-started documentation (`docs/getting-started/installation.md`) is confusing for end users because it mixes user installation with development workflows. Key problems:

1. **Development-heavy focus**: 80% of the page covers development installation (uv, conda, Git LFS)
2. **Repo cloning confusion**: Users don't need to clone the repository to use sleap-roots
3. **Missing modern user workflow**: No guidance on using `uv init` + `uv add sleap-roots` for new projects
4. **Poor separation of concerns**: Developer setup should be in the Developer Guide, not Getting Started

This creates friction for researchers who just want to install and use the package.

## What Changes

### 1. Simplify User Installation (Getting Started)

Move the getting-started guide to focus entirely on **end-user installation and first use**:

- Quick install with pip
- Modern project setup with `uv init` + `uv add sleap-roots`
- Conda environment setup
- First-time usage verification
- Link to quickstart tutorial

**No development content** (no cloning, no Git LFS, no test running).

### 2. Create Comprehensive Developer Guide

Add a new "Development Setup" page under the Developer Guide section:

- Clone repository and Git LFS setup
- Using uv for development (`uv sync`)
- Alternative conda workflow
- Running tests and pre-commit checks
- Contributing workflow

This consolidates scattered development instructions into one place.

### 3. Update Navigation Structure

Ensure clear user vs. developer paths:

- Getting Started → For end users
- Developer Guide → Development Setup → For contributors

## Impact

**Users Affected**:
- ✅ End users get clearer, faster path to installation
- ✅ Contributors get comprehensive development guide in one place

**Breaking Changes**: None (documentation reorganization only)

## Success Criteria

- [ ] Getting Started page has <50 lines focused on user installation
- [ ] No mention of Git LFS or test running in Getting Started
- [ ] Developer Guide has complete development setup instructions
- [ ] Navigation clearly separates user vs. developer content
- [ ] All existing links updated to new structure

## Scope

**In Scope**:
- Rewrite `docs/getting-started/installation.md` for end users
- Create `docs/dev/setup.md` for developers (or enhance existing)
- Update navigation in `mkdocs.yml`
- Update cross-references in other docs

**Out of Scope**:
- Changing actual installation procedures
- Adding new development tooling
- Updating README.md (separate PR if needed)

## Dependencies

None. This is a documentation-only change.

## References

- Current installation docs: `docs/getting-started/installation.md`
- Existing developer guide: `docs/dev/`
- MkDocs navigation: `mkdocs.yml`