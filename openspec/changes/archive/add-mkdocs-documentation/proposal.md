# Proposal: Add MkDocs Documentation

## Change ID
`add-mkdocs-documentation`

## Summary
Add comprehensive MkDocs-based documentation for sleap-roots, providing both user-friendly guides and developer API reference, following the successful pattern established in sleap-io.

## Problem Statement

Currently, sleap-roots documentation is scattered and incomplete:

**Current State:**
- README.md has basic usage examples but limited depth
- Trait documentation lives in external HackMD (hard to maintain, not version-controlled)
- No structured API documentation (users must read source code)
- No guided tutorials for different use cases
- Developer documentation is minimal
- No searchable documentation site

**Pain Points:**
1. **Users struggle to get started** - README examples don't cover all pipelines or edge cases
2. **Trait meanings unclear** - External HackMD docs get out of sync with code
3. **No API reference** - Developers can't easily look up function signatures, parameters, return values
4. **Hard to discover features** - Users don't know about utility functions in individual modules
5. **Scientific reproducibility** - Trait computation methods not documented in detail
6. **Maintenance burden** - Manual HackMD updates are error-prone

## Proposed Solution

Create a comprehensive MkDocs documentation site (similar to sleap-io) with:

### For Users
- **Getting Started Guide** - Installation, quickstart, first analysis
- **Pipeline Tutorials** - Detailed guides for each pipeline type
- **Trait Reference** - Auto-generated from code, always up-to-date
- **Cookbook** - Common recipes and use cases
- **Troubleshooting** - FAQ and common issues

### For Developers
- **API Reference** - Auto-generated from docstrings using mkdocstrings
- **Contributing Guide** - How to add pipelines, traits, tests
- **Architecture** - Pipeline design patterns, trait computation flow
- **Testing Guide** - How to write tests, use fixtures

### Infrastructure
- **MkDocs + Material theme** - Modern, searchable, responsive
- **Auto-generated API docs** - mkdocstrings pulls from Google-style docstrings
- **Versioned docs** - mike for version management
- **CI/CD integration** - Auto-deploy on release
- **Jupyter notebook integration** - Include interactive examples

## Benefits

### For Users
- ✅ Comprehensive, searchable documentation
- ✅ Clear trait definitions (auto-synced with code)
- ✅ Step-by-step tutorials for each plant type
- ✅ Easy discovery of features
- ✅ Better scientific reproducibility (detailed trait computation docs)

### For Developers  
- ✅ Complete API reference
- ✅ Clear contribution guidelines
- ✅ Architecture documentation
- ✅ Reduced support burden (self-service docs)

### For Maintainers
- ✅ Single source of truth (code → docs)
- ✅ Version-controlled documentation
- ✅ Automated API docs (no manual updates)
- ✅ CI integration (docs tested on every PR)

## Scope

### In Scope

**Documentation Content:**
- User guide (installation, quickstart, tutorials, cookbook)
- Developer guide (contributing, architecture, testing)
- API reference (auto-generated from all modules)
- Trait reference (auto-generated from TraitDef objects)
- Migration of HackMD content to MkDocs

**Infrastructure:**
- MkDocs configuration (mkdocs.yml)
- Material theme setup
- mkdocstrings for API docs
- mike for versioning
- GitHub Actions workflow for deployment
- Custom CSS/assets for branding

**Organization:**
- `docs/` directory with markdown files
- `docs/guides/` - User and developer guides
- `docs/api/` - Auto-generated API reference
- `docs/traits/` - Trait reference
- `docs/assets/` - Images, diagrams, custom CSS

### Out of Scope

- Video tutorials (future enhancement)
- Interactive widget demos (future enhancement)
- Multi-language support (English only for now)
- Blog/news section (not needed yet)
- Community forum integration (use GitHub Discussions)

## Technical Approach

### MkDocs Configuration

Based on sleap-io's successful setup:

```yaml
site_name: sleap-roots
site_url: https://talmolab.github.io/sleap-roots
repo_url: https://github.com/talmolab/sleap-roots

theme:
  name: material
  palette:
    - scheme: default
      primary: green  # Plant-themed
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - scheme: slate
      primary: green
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  features:
    - navigation.tracking
    - navigation.sections
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.code.annotate

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          options:
            show_source: true
            docstring_style: google
  - mike  # Version management
  - gen-files  # Auto-generate API pages
  - literate-nav  # Navigation from SUMMARY.md

markdown_extensions:
  - pymdownx.highlight
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
  - pymdownx.arithmatex:
      generic: true
  - admonition
  - attr_list
  - md_in_html
```

### Documentation Structure

```
docs/
├── index.md                    # Landing page
├── installation.md             # Detailed installation guide
├── quickstart.md               # First analysis tutorial
├── guides/
│   ├── user/
│   │   ├── pipelines/
│   │   │   ├── dicot.md        # DicotPipeline guide
│   │   │   ├── monocot.md      # Monocot pipeline guides
│   │   │   ├── multiple.md     # Multi-plant pipelines
│   │   │   └── custom.md       # Creating custom pipelines
│   │   ├── data-preparation.md
│   │   ├── batch-processing.md
│   │   ├── visualization.md
│   │   └── troubleshooting.md
│   └── developer/
│       ├── contributing.md
│       ├── architecture.md
│       ├── adding-pipelines.md
│       ├── adding-traits.md
│       ├── testing.md
│       └── release-process.md
├── traits/
│   ├── index.md                # Trait reference overview
│   ├── lengths.md              # Length-based traits
│   ├── angles.md               # Angular traits
│   ├── topology.md             # Network/topology traits
│   └── automated.md            # Auto-generated full trait list
├── api/
│   ├── index.md                # API overview
│   ├── pipelines.md            # Pipeline classes
│   ├── series.md               # Series class
│   ├── traits/                 # Individual trait modules
│   │   ├── lengths.md
│   │   ├── angles.md
│   │   ├── tips.md
│   │   └── ...
│   └── utils.md                # Utility functions
├── cookbook/
│   ├── index.md
│   ├── filtering-data.md
│   ├── custom-traits.md
│   ├── batch-optimization.md
│   └── exporting-results.md
├── changelog.md                # Link to CHANGELOG.md
└── assets/
    ├── css/
    │   └── custom.css          # Custom styling
    └── images/
        ├── logo.png
        └── diagrams/
```

### Auto-Generated Content

**Trait Reference:**
Use gen-files plugin to generate trait documentation from code:

```python
# docs/gen_trait_docs.py
from sleap_roots import trait_pipelines
import inspect

for name, cls in inspect.getmembers(trait_pipelines):
    if name.endswith('Pipeline'):
        pipeline = cls()
        traits = pipeline.get_trait_definitions()
        # Generate markdown for each trait
```

**API Reference:**
Use mkdocstrings to auto-generate from docstrings:

```markdown
# docs/api/lengths.md

::: sleap_roots.lengths
    options:
      show_source: true
      members:
        - get_root_lengths
        - get_max_length_pts
```

### Integration with Existing Docs

**Migrate HackMD Content:**
- Copy trait descriptions from HackMD to markdown
- Enhance with auto-generated examples
- Add cross-references to API docs
- Keep HackMD link for historical reference

**Leverage README:**
- Move detailed content to docs
- Keep README concise with links to full docs
- Maintain quick-start section in README

## Dependencies

**New Python Dependencies (dev):**
- mkdocs (>= 1.5)
- mkdocs-material (>= 9.0)
- mkdocstrings[python] (>= 0.20)
- mkdocs-gen-files
- mkdocs-literate-nav
- mkdocs-section-index
- mike (for versioning)

**CI/CD:**
- GitHub Actions workflow for deployment
- Deploy to GitHub Pages on releases
- Version tags managed with mike

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Docs get out of sync with code | High | Auto-generate API docs and trait reference from code |
| Maintenance burden | Medium | Automate as much as possible, use CI to catch issues |
| Initial effort is large | Medium | Phased rollout: API docs first, then guides |
| External HackMD still used | Low | Clear migration plan, redirect from HackMD to MkDocs |
| Versioning complexity | Medium | Use mike for version management, follow sleap-io pattern |

## Success Criteria

- [ ] MkDocs site deployed to GitHub Pages
- [ ] All modules have API documentation
- [ ] All pipelines have tutorial guides
- [ ] Trait reference is auto-generated and comprehensive
- [ ] CI/CD workflow deploys docs on releases
- [ ] HackMD content migrated
- [ ] Users can find answers without asking maintainers
- [ ] New contributors can onboard using docs

## Alternatives Considered

### Alternative 1: Sphinx (Traditional Python Docs)
**Rejected** - MkDocs is simpler, Material theme is more modern, better for mixed user/dev docs

### Alternative 2: Keep HackMD
**Rejected** - Not version-controlled, hard to maintain, no API integration

### Alternative 3: Wiki Only
**Rejected** - Wikis don't integrate with code, no auto-generation, hard to version

### Alternative 4: Read the Docs (Sphinx-based)
**Rejected** - MkDocs + Material provides better UX and simpler setup

## Implementation Phases

### Phase 1: Infrastructure (Week 1)
- Set up MkDocs configuration
- Configure Material theme
- Set up CI/CD deployment
- Create basic navigation structure

### Phase 2: API Documentation (Week 2)
- Configure mkdocstrings
- Auto-generate API docs for all modules
- Enhance docstrings where needed
- Add code examples to docstrings

### Phase 3: User Guides (Week 3)
- Write pipeline tutorials
- Create quickstart guide
- Develop cookbook recipes
- Add troubleshooting section

### Phase 4: Trait Reference (Week 4)
- Auto-generate trait reference
- Migrate HackMD content
- Add detailed computation descriptions
- Include validation data and formulas

### Phase 5: Developer Docs (Week 5)
- Write contributing guide
- Document architecture
- Create testing guide
- Add release process docs

### Phase 6: Polish & Deploy (Week 6)
- Custom styling and branding
- Cross-linking and navigation
- Search optimization
- Version management setup
- Final deployment

## Estimated Effort

**Total Time:** 5-6 weeks

- **Infrastructure setup:** 1 week
- **API documentation:** 1 week  
- **User guides:** 1 week
- **Trait reference:** 1 week
- **Developer docs:** 1 week
- **Polish and deployment:** 0.5-1 week

**Note:** Can be done incrementally; API docs provide immediate value.

## Related Issues

N/A (new capability)

## References

- sleap-io MkDocs: https://github.com/talmolab/sleap-io/blob/main/mkdocs.yml
- Material for MkDocs: https://squidfunk.github.io/mkdocs-material/
- mkdocstrings: https://mkdocstrings.github.io/
- Current HackMD docs: https://hackmd.io/DMiXO2kXQhKH8AIIcIy--g
