# beginner-friendly-docs Specification

## Purpose
User-facing documentation must guide beginners (plant biologists without extensive Python experience) to best practices and provide clear, concrete examples of what to expect.

## ADDED Requirements

### Requirement: Installation must guide users to best practices first

Documentation MUST present the recommended installation workflow prominently, before alternatives that may cause problems.

#### Scenario: uv workflow is primary installation method

- **Given** user visits installation page
- **When** user reads installation instructions
- **Then** uv workflow is presented first as recommended method
- **And** pip-only installation (base environment) comes later as alternative
- **And** conda installation is clearly marked as alternative
- **And** user is guided through complete uv workflow (init, add, run)

#### Scenario: Commands use recommended tooling

- **Given** documentation shows Python command execution
- **When** command is for script execution or REPL
- **Then** command shows `uv run python` prefix for uv users
- **And** context explains how to run for different installation methods
- **And** verification commands use recommended workflow

#### Scenario: Installation order prevents mistakes

- **Given** beginner user following documentation linearly
- **When** user completes first installation instruction
- **Then** user has NOT pip-installed into base environment
- **And** user has created isolated project environment
- **And** user understands how to run Python with their installation

### Requirement: Examples must show actual outputs

Code examples MUST include actual output showing real data, column names, and values so users know what to expect.

#### Scenario: Trait computation examples show DataFrame output

- **Given** example calls `compute_plant_traits()` or `compute_batch_traits()`
- **When** example includes print/display statement
- **Then** example shows actual DataFrame output with real column names
- **And** output shows realistic values (not placeholders)
- **And** output shows enough rows to understand structure (typically 2-3)

#### Scenario: Output format matches actual API behavior

- **Given** documentation shows example output
- **When** user runs same code with test data
- **Then** column names match exactly
- **And** value formats match (integers, floats, strings)
- **And** number of rows matches expected behavior

#### Scenario: Command outputs are demonstrated

- **Given** example runs shell command or Python one-liner
- **When** command produces terminal output
- **Then** example shows what output looks like
- **And** success indicators are clear (checkmarks, version numbers, etc.)

### Requirement: Output structure must be explained clearly

Documentation MUST distinguish between different levels of trait aggregation and explain what each method returns.

#### Scenario: Frame-level vs plant-level distinction is clear

- **Given** user reads about trait computation
- **When** documentation describes compute methods
- **Then** `compute_plant_traits()` clearly states "one row per frame"
- **And** `compute_batch_traits()` clearly states "one row per plant"
- **And** difference between raw measurements and summary statistics is explained

#### Scenario: Summary statistics are documented

- **Given** user reads about batch processing
- **When** `compute_batch_traits()` is described
- **Then** documentation lists summary statistics: min, max, mean, median, std, p5, p25, p75, p95
- **And** examples show actual summary statistic column names (e.g., `primary_length_mean`)
- **And** user understands summaries aggregate across frames

#### Scenario: Column naming conventions are clear

- **Given** documentation shows trait columns
- **When** examples list column names
- **Then** frame-level columns use base trait names (e.g., `primary_length`)
- **And** plant-level columns use suffixed names (e.g., `primary_length_mean`)
- **And** naming pattern is explicitly stated

### Requirement: Generic code must be removed

Beginner documentation MUST focus on sleap-roots-specific functionality and link to external resources for generic operations.

#### Scenario: No generic pandas usage in quickstart

- **Given** quickstart tutorial shows data manipulation
- **When** operation is generic pandas (not sleap-roots specific)
- **Then** code is not included in tutorial
- **And** link to pandas documentation or visualization guide is provided instead
- **And** users understand sleap-roots outputs are standard DataFrames

#### Scenario: No generic visualization code

- **Given** documentation discusses visualizing results
- **When** visualization requires matplotlib/seaborn/etc.
- **Then** full plotting code is not included
- **And** link to visualization example repository is provided
- **And** brief mention that CSVs work with standard tools

#### Scenario: Generic workflows are conceptual only

- **Given** documentation covers unit conversion or data export
- **When** operation is standard programming practice
- **Then** explanation is conceptual (steps, not code)
- **And** users are assumed to know or can look up generic operations
- **And** sleap-roots-specific aspects are highlighted

### Requirement: Documentation must avoid duplication

Common problems and reference material MUST have single source of truth with links from other locations.

#### Scenario: Troubleshooting is centralized

- **Given** user encounters common problem
- **When** documentation mentions the problem
- **Then** detailed solution is in troubleshooting guide only
- **And** other pages link to troubleshooting guide
- **And** at most 1-2 critical issues may have inline brief mention with link

#### Scenario: API documentation is authoritative

- **Given** user needs trait definitions or function signatures
- **When** documentation references traits or API
- **Then** links point to API reference documentation
- **And** trait reference guide (if exists) is supplementary, not primary
- **And** no duplicate trait lists in multiple places

#### Scenario: Maintenance updates propagate

- **Given** common issue solution is updated
- **When** solution is in troubleshooting guide
- **Then** all pages linking to guide get update automatically
- **And** no manual synchronization needed across pages
- **And** version drift is prevented

## Relationships

- **Extends** `documentation-organization` spec: User docs must be beginner-friendly
- **Extends** `documentation-accuracy` spec: Examples must show real behavior
- **Extends** `example-correctness` spec: Examples must work and show outputs
- **Complements** project conventions: uv is recommended workflow throughout