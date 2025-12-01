# Spec: API Documentation Quality

## MODIFIED Requirements

### Requirement: Source code type annotations

All public functions and methods MUST have complete type annotations for mkdocstrings autogeneration.

#### Scenario: Function parameters have type annotations

- **Given** a public function in sleap_roots package
- **When** mkdocstrings generates API documentation
- **Then** all function parameters display type information
- **And** griffe does not emit "No type or annotation for parameter" warnings
- **And** generated docs show proper type hints in function signatures

#### Scenario: Return values have type annotations

- **Given** a public function returns one or more values
- **When** mkdocstrings generates API documentation
- **Then** all return values have type annotations
- **And** griffe does not emit "No type or annotation for returned value" warnings
- **And** tuple returns specify types for each position

#### Scenario: Plotting functions have complete annotations

- **Given** `Series.plot()` method with visualization parameters
- **When** mkdocstrings processes the method
- **Then** all parameters (instances, skeleton, cmap, color_by_track, tracks, **kwargs) have types
- **And** return value type is specified
- **And** no griffe warnings are emitted for this method

### Requirement: Docstring formatting compliance

All docstrings MUST follow Google style convention without formatting errors.

#### Scenario: Raises sections are properly formatted

- **Given** a function documents exceptions in Raises section
- **When** griffe parses the docstring
- **Then** each exception type has a proper description (not just "object.")
- **And** griffe successfully parses exception: description pairs
- **And** no "Failed to get 'exception: description' pair" warnings occur

#### Scenario: Docstring parameters match function signature

- **Given** a function has a docstring with Args section
- **When** griffe validates the docstring
- **Then** all documented parameters exist in function signature
- **And** no warnings about parameters not appearing in signature
- **And** all signature parameters are documented (or explicitly omitted for **kwargs)

#### Scenario: Special parameters are handled correctly

- **Given** pipeline functions with internal 'Args' documentation
- **When** griffe parses trait pipeline docstrings
- **Then** 'Args' is not treated as a parameter name
- **And** no "Parameter 'Args' does not appear in the function signature" warnings
- **And** trait pipeline documentation renders correctly

### Requirement: Documentation navigation structure

MkDocs navigation MUST be complete and free of broken references.

#### Scenario: All documentation pages are accessible

- **Given** documentation files exist in docs/ directory
- **When** mkdocs.yml navigation is evaluated
- **Then** all pages are either included in nav or intentionally excluded
- **And** important pages like benchmarking.md appear in navigation
- **And** no "not included in nav" warnings for user-facing documentation

#### Scenario: Navigation references resolve correctly

- **Given** mkdocs.yml navigation includes file references
- **When** MkDocs builds the site
- **Then** all nav references point to existing files or generated content
- **And** no "reference to X is not found" warnings
- **And** users can navigate to all referenced pages

#### Scenario: Auto-generated reference pages integrate properly

- **Given** gen-files plugin generates reference/ directory content
- **When** MkDocs builds documentation
- **Then** reference pages are accessible via navigation or search
- **And** no warnings about missing reference/ files
- **And** decision is documented: either include reference/ in nav with index or exclude from nav

### Requirement: Internal link integrity

All internal documentation links MUST resolve to valid targets.

#### Scenario: Cross-document links work

- **Given** documentation page links to another page
- **When** user clicks the link in rendered docs
- **Then** link navigates to correct target
- **And** no "link target not found" warnings during build
- **And** relative paths are correct (e.g., `../api/core/pipelines.md` not `../api/pipelines.md`)

#### Scenario: Anchor links resolve

- **Given** documentation links to specific section via anchor
- **When** MkDocs validates internal links
- **Then** all anchor references exist in target documents
- **And** anchors match heading IDs or explicitly defined anchors
- **And** no "does not contain anchor" warnings

#### Scenario: Missing anchors are added

- **Given** links reference `#lateral-root-pipeline` and `#multiple-dicot-pipeline`
- **When** target document is rendered
- **Then** anchors exist as heading IDs or explicit HTML anchors
- **And** links scroll to correct section when clicked
- **And** navigation is smooth and expected

## ADDED Requirements

### Requirement: Zero-warning documentation builds

MkDocs build process MUST complete without warnings that indicate errors.

#### Scenario: Clean build output

- **Given** developer runs `uv run mkdocs build`
- **When** build completes
- **Then** no griffe warnings about missing types
- **And** no griffe warnings about malformed docstrings
- **And** no MkDocs warnings about missing files
- **And** only informational messages remain (if any)

#### Scenario: Acceptable warnings are documented

- **Given** some warnings may be unavoidable or acceptable
- **When** build produces warnings
- **Then** each accepted warning has documented rationale
- **And** rationale is in README.md or proposal documentation
- **And** distinction is clear between errors and acceptable info messages

#### Scenario: CI validates documentation quality

- **Given** pull request modifies source code or documentation
- **When** CI runs documentation checks
- **Then** MkDocs build is tested for warnings
- **And** PR fails if new warnings are introduced
- **And** contributors are notified of documentation issues

### Requirement: Documentation maintenance guide

Contributors MUST have clear guidance on maintaining documentation quality.

#### Scenario: Type annotation requirements documented

- **Given** contributor adds new public function
- **When** contributor reads contributing guide
- **Then** guide explains type annotation requirements
- **And** guide shows examples of proper annotations
- **And** guide references Google style docstring requirements
- **And** guide explains how to test documentation locally

#### Scenario: Link checking workflow documented

- **Given** contributor updates documentation
- **When** contributor checks for broken links
- **Then** procedure uses `mkdocs build` to validate
- **And** procedure explains how to fix common link issues
- **And** procedure shows how to test locally with `mkdocs serve`