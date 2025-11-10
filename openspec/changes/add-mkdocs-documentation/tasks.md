# Tasks: Add MkDocs Documentation

## Overview
Implement comprehensive MkDocs-based documentation for sleap-roots with user guides, developer docs, and auto-generated API reference.

## Task List

### Phase 1: Infrastructure Setup (Week 1)

#### 1.1 Install and Configure MkDocs
**Priority**: High  
**Estimated Time**: 3 hours

- [ ] Add MkDocs dependencies to `pyproject.toml` dev dependencies
- [ ] Create `mkdocs.yml` configuration file
- [ ] Configure Material theme with plant-themed colors (green primary)
- [ ] Set up dark/light mode toggle
- [ ] Configure search, navigation, and code highlighting features
- [ ] Test local build: `mkdocs serve`

**Deliverable**: Working local MkDocs site with Material theme

---

#### 1.2 Create Documentation Structure
**Priority**: High  
**Estimated Time**: 2 hours

- [ ] Create `docs/` directory structure
- [ ] Create subdirectories: `guides/`, `api/`, `traits/`, `cookbook/`, `assets/`
- [ ] Create placeholder `index.md` (landing page)
- [ ] Set up navigation structure in `mkdocs.yml`
- [ ] Add logo and branding assets

**Deliverable**: Complete directory structure with navigation

---

#### 1.3 Set Up CI/CD Deployment
**Priority**: High  
**Estimated Time**: 4 hours

- [ ] Create `.github/workflows/docs.yml` workflow
- [ ] Configure deployment to GitHub Pages
- [ ] Set up mike for version management
- [ ] Test deployment to staging branch
- [ ] Configure custom domain (if desired)

**Deliverable**: Automated docs deployment on push to main

---

### Phase 2: API Documentation (Week 2)

#### 2.1 Configure mkdocstrings
**Priority**: High  
**Estimated Time**: 2 hours

- [ ] Add mkdocstrings plugin to `mkdocs.yml`
- [ ] Configure Python handler with Google-style docstrings
- [ ] Set up auto-generation templates
- [ ] Test with one module (e.g., `lengths.py`)

**Deliverable**: Working API doc generation

---

#### 2.2 Generate API Documentation Pages
**Priority**: High  
**Estimated Time**: 6 hours

- [ ] Create `docs/api/index.md` (API overview)
- [ ] Create API page for each module:
  - [ ] `docs/api/pipelines.md` - Pipeline classes
  - [ ] `docs/api/series.md` - Series class
  - [ ] `docs/api/lengths.md` - Length calculations
  - [ ] `docs/api/angles.md` - Angle calculations
  - [ ] `docs/api/tips.md` - Tip detection
  - [ ] `docs/api/bases.md` - Base detection
  - [ ] `docs/api/convhull.md` - Convex hull
  - [ ] `docs/api/ellipse.md` - Ellipse fitting
  - [ ] `docs/api/networklength.md` - Network metrics
  - [ ] `docs/api/scanline.md` - Scanline analysis
  - [ ] `docs/api/points.md` - Point utilities
  - [ ] `docs/api/summary.md` - Summary functions
- [ ] Add navigation entries to `mkdocs.yml`

**Deliverable**: Complete API reference for all modules

---

#### 2.3 Enhance Docstrings Where Needed
**Priority**: Medium  
**Estimated Time**: 4 hours

- [ ] Review all public functions for docstring completeness
- [ ] Add missing docstrings
- [ ] Add code examples to key functions
- [ ] Ensure Args, Returns, Raises sections are complete
- [ ] Add cross-references between related functions

**Deliverable**: Comprehensive, well-documented API

---

### Phase 3: User Guides (Week 3)

#### 3.1 Create Installation and Quickstart
**Priority**: High  
**Estimated Time**: 3 hours

- [ ] Write `docs/installation.md` (detailed installation guide)
- [ ] Write `docs/quickstart.md` (first analysis tutorial)
- [ ] Include conda and pip installation methods
- [ ] Add troubleshooting for common install issues
- [ ] Include environment setup verification steps

**Deliverable**: Complete getting-started documentation

---

#### 3.2 Write Pipeline Tutorials
**Priority**: High  
**Estimated Time**: 8 hours

- [ ] Create `docs/guides/user/pipelines/` directory
- [ ] Write tutorial for each pipeline:
  - [ ] `dicot.md` - DicotPipeline (primary + lateral)
  - [ ] `younger-monocot.md` - YoungerMonocotPipeline
  - [ ] `older-monocot.md` - OlderMonocotPipeline
  - [ ] `primary-root.md` - PrimaryRootPipeline
  - [ ] `lateral-root.md` - LateralRootPipeline
  - [ ] `multiple-dicot.md` - MultipleDicotPipeline
  - [ ] `multiple-primary.md` - MultiplePrimaryRootPipeline
- [ ] Include full code examples for each
- [ ] Add expected output samples
- [ ] Include when to use each pipeline

**Deliverable**: Tutorial for every pipeline

---

#### 3.3 Create User Workflow Guides
**Priority**: Medium  
**Estimated Time**: 6 hours

- [ ] Write `docs/guides/user/data-preparation.md`
  - [ ] SLEAP prediction file requirements
  - [ ] H5 image series setup
  - [ ] File naming conventions
  - [ ] Data organization best practices
- [ ] Write `docs/guides/user/batch-processing.md`
  - [ ] Processing multiple plants/series
  - [ ] Parallel processing options
  - [ ] Memory management
  - [ ] Progress tracking
- [ ] Write `docs/guides/user/visualization.md`
  - [ ] Plotting predictions on images
  - [ ] Trait distribution plots
  - [ ] Time-series visualization
- [ ] Write `docs/guides/user/troubleshooting.md`
  - [ ] Common errors and solutions
  - [ ] Data quality issues
  - [ ] Performance problems
  - [ ] FAQ

**Deliverable**: Complete user workflow documentation

---

### Phase 4: Trait Reference (Week 4)

#### 4.1 Create Auto-Generation Script
**Priority**: High  
**Estimated Time**: 4 hours

- [ ] Create `docs/gen_trait_docs.py` script
- [ ] Extract traits from all Pipeline classes
- [ ] Generate markdown tables with:
  - [ ] Trait name
  - [ ] Description
  - [ ] Computation method
  - [ ] Units
  - [ ] Which pipelines include it
- [ ] Set up gen-files plugin to run script

**Deliverable**: Automated trait documentation generation

---

#### 4.2 Write Trait Category Guides
**Priority**: High  
**Estimated Time**: 5 hours

- [ ] Create `docs/traits/index.md` (trait overview)
- [ ] Write `docs/traits/lengths.md`
  - [ ] Explain length-based measurements
  - [ ] Include formulas (Euclidean distance)
  - [ ] Validation data and accuracy
- [ ] Write `docs/traits/angles.md`
  - [ ] Angular measurements and conventions
  - [ ] Coordinate system explanation
  - [ ] Example values and ranges
- [ ] Write `docs/traits/topology.md`
  - [ ] Network-based metrics
  - [ ] Convex hull analysis
  - [ ] Scanline methods
- [ ] Write `docs/traits/counts.md`
  - [ ] Tip and base counting
  - [ ] Detection methods

**Deliverable**: Comprehensive trait reference organized by category

---

#### 4.3 Migrate HackMD Content
**Priority**: Medium  
**Estimated Time**: 3 hours

- [ ] Review existing HackMD trait documentation
- [ ] Copy relevant content to MkDocs
- [ ] Update and enhance descriptions
- [ ] Add cross-references to API docs
- [ ] Add note in HackMD redirecting to MkDocs
- [ ] Link to original HackMD for historical reference

**Deliverable**: HackMD content integrated into MkDocs

---

### Phase 5: Developer Documentation (Week 5)

#### 5.1 Write Contributing Guide
**Priority**: High  
**Estimated Time**: 4 hours

- [ ] Create `docs/guides/developer/contributing.md`
- [ ] Environment setup instructions
- [ ] Code style and formatting (Black, pydocstyle)
- [ ] Git workflow and branching strategy
- [ ] PR process and review checklist
- [ ] Link to Claude commands for development

**Deliverable**: Complete contributing guide

---

#### 5.2 Document Architecture
**Priority**: High  
**Estimated Time**: 4 hours

- [ ] Create `docs/guides/developer/architecture.md`
- [ ] Pipeline design pattern explanation
- [ ] Series-centric data model
- [ ] Trait computation flow
- [ ] Dependency graphs (Mermaid diagrams)
- [ ] Module organization rationale

**Deliverable**: Architecture documentation with diagrams

---

#### 5.3 Create Development Guides
**Priority**: Medium  
**Estimated Time**: 6 hours

- [ ] Write `docs/guides/developer/adding-pipelines.md`
  - [ ] Step-by-step pipeline creation
  - [ ] Trait definition patterns
  - [ ] Testing requirements
  - [ ] Documentation requirements
- [ ] Write `docs/guides/developer/adding-traits.md`
  - [ ] Where to add trait computation code
  - [ ] Naming conventions
  - [ ] Validation requirements
  - [ ] Performance considerations
- [ ] Write `docs/guides/developer/testing.md`
  - [ ] Test organization
  - [ ] Fixture usage
  - [ ] Git LFS test data
  - [ ] Coverage requirements
- [ ] Write `docs/guides/developer/release-process.md`
  - [ ] Version bumping
  - [ ] Changelog updates
  - [ ] PyPI release steps
  - [ ] Documentation versioning

**Deliverable**: Complete developer guides

---

### Phase 6: Cookbook & Examples (Week 5-6)

#### 6.1 Create Cookbook Recipes
**Priority**: Medium  
**Estimated Time**: 5 hours

- [ ] Create `docs/cookbook/index.md`
- [ ] Write recipes:
  - [ ] `filtering-data.md` - Handling missing/bad data
  - [ ] `custom-traits.md` - Adding custom trait calculations
  - [ ] `batch-optimization.md` - Performance tuning
  - [ ] `exporting-results.md` - CSV, JSON, database export
  - [ ] `integrating-sleap.md` - Working with SLEAP workflow
  - [ ] `multi-timepoint.md` - Time-series analysis

**Deliverable**: Cookbook with practical recipes

---

#### 6.2 Add Jupyter Notebook Examples
**Priority**: Low  
**Estimated Time**: 4 hours

- [ ] Configure mkdocs-jupyter plugin
- [ ] Convert example notebooks to docs
- [ ] Add interactive examples
- [ ] Link to notebooks directory

**Deliverable**: Interactive notebook examples in docs

---

### Phase 7: Polish & Deployment (Week 6)

#### 7.1 Custom Styling and Branding
**Priority**: Medium  
**Estimated Time**: 3 hours

- [ ] Create `docs/assets/css/custom.css`
- [ ] Add Talmo Lab branding
- [ ] Custom color scheme (plant-themed greens)
- [ ] Logo and favicon
- [ ] Typography improvements

**Deliverable**: Polished, branded documentation

---

#### 7.2 Improve Navigation and Search
**Priority**: Medium  
**Estimated Time**: 3 hours

- [ ] Optimize navigation structure
- [ ] Add search keywords and tags
- [ ] Create section index pages
- [ ] Add "Edit on GitHub" links
- [ ] Set up breadcrumbs
- [ ] Add "last updated" timestamps

**Deliverable**: Easy-to-navigate documentation

---

#### 7.3 Cross-Linking and References
**Priority**: Medium  
**Estimated Time**: 3 hours

- [ ] Add cross-references between related docs
- [ ] Link API docs to user guides
- [ ] Link trait reference to pipeline tutorials
- [ ] Add "See Also" sections
- [ ] Create glossary of terms

**Deliverable**: Well-connected documentation

---

#### 7.4 Version Management Setup
**Priority**: High  
**Estimated Time**: 3 hours

- [ ] Configure mike for version management
- [ ] Set up version switcher in docs
- [ ] Document versioning strategy
- [ ] Create versions for existing releases
- [ ] Test version switching

**Deliverable**: Versioned documentation with switcher

---

#### 7.5 Final Review and Deployment
**Priority**: High  
**Estimated Time**: 4 hours

- [ ] Review all documentation for accuracy
- [ ] Test all code examples
- [ ] Check all links (internal and external)
- [ ] Verify mobile responsiveness
- [ ] Test search functionality
- [ ] Deploy to production (GitHub Pages)
- [ ] Announce documentation to users

**Deliverable**: Production-ready documentation site

---

### Phase 8: Maintenance Integration

#### 8.1 Update Development Workflows
**Priority**: High  
**Estimated Time**: 2 hours

- [ ] Update `/docs-update` command to include MkDocs
- [ ] Add docs checks to CI (build succeeds)
- [ ] Add docs preview to PR workflow
- [ ] Update contributing guide to mention docs

**Deliverable**: Docs integrated into development workflow

---

## Dependencies

- **Sequential dependencies**:
  - Phase 1 must complete before Phase 2
  - Phases 2-6 can be done in parallel after Phase 1
  - Phase 7 depends on Phases 2-6
  
- **External dependencies**:
  - Needs MkDocs and plugins installed
  - Requires GitHub Pages enabled
  - Needs approval for custom domain (if used)

## Parallelization Opportunities

- Phases 2-6 can be worked on simultaneously by different people
- Within each phase, individual task sections can be parallelized
- API docs (Phase 2) and User guides (Phase 3) are completely independent

## Validation Criteria

Each deliverable should be validated by:
- [ ] Local `mkdocs serve` builds without errors
- [ ] All links work (no 404s)
- [ ] Code examples execute successfully
- [ ] Docstrings render correctly in API docs
- [ ] Search finds expected content
- [ ] Mobile-responsive layout works

## Timeline

**Optimistic (full-time focus)**: 4-5 weeks  
**Realistic (part-time)**: 6-8 weeks  
**With team (2-3 people)**: 3-4 weeks

## Success Metrics

- [ ] MkDocs site live at https://talmolab.github.io/sleap-roots
- [ ] All 13 modules have API documentation
- [ ] All 7 pipelines have tutorials
- [ ] Trait reference auto-generated
- [ ] Developer guides complete
- [ ] HackMD content migrated
- [ ] CI/CD deploys docs automatically
- [ ] Version switching works
- [ ] Users report improved documentation in feedback
