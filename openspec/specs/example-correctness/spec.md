# example-correctness Specification

## Purpose
TBD - created by archiving change fix-documentation-examples. Update Purpose after archive.
## Requirements
### Requirement: All code examples must be executable

Documentation examples MUST use only functions, modules, and classes that exist in the actual codebase.

#### Scenario: Imports resolve correctly

- **Given** user copies a code example from documentation
- **When** user runs the import statements
- **Then** all modules exist in `sleap_roots` package
- **And** all imported functions exist in those modules
- **And** module names match actual module files (e.g., `angle.py` not `angles.py`)

#### Scenario: Function calls match API signatures

- **Given** documentation example calls a function
- **When** user executes the example
- **Then** function exists with that exact name
- **And** arguments match the function's actual signature
- **And** required arguments are provided
- **And** optional arguments use valid values

#### Scenario: Examples produce expected output

- **Given** user runs a complete example from documentation
- **When** example executes with test data
- **Then** code runs without errors
- **And** output matches documented expectations
- **And** return types match usage in example

### Requirement: Critical arguments must be explicit

Examples MUST use named arguments (`arg=value`) instead of relying on positional arguments or implicit defaults, especially for parameters that users commonly customize.

#### Scenario: Optional parameters shown explicitly

- **Given** function has commonly customized optional parameters
- **When** example uses that function
- **Then** critical optional parameters are shown explicitly with `arg=value` syntax
- **And** example demonstrates realistic customization options
- **And** comments explain what the parameter controls if not obvious

#### Scenario: Boolean flags are explicit

- **Given** function accepts boolean flags
- **When** example calls that function
- **Then** boolean arguments use `flag=True` or `flag=False` explicitly
- **And** not implicit True/False positional arguments
- **And** users can clearly see what options are being set

#### Scenario: Default behavior is documented

- **Given** example uses default parameter values
- **When** initialization uses defaults (e.g., `DicotPipeline()`)
- **Then** comment explains what defaults are being used
- **And** users understand what the default behavior is
- **And** users know how to customize if needed

### Requirement: Examples progress from simple to complex

Documentation MUST structure examples to build user understanding progressively.

#### Scenario: Home page has minimal working example

- **Given** user visits home page (index.md)
- **When** user reads the "Quick Example" section
- **Then** example is under 15 lines of code
- **And** example shows basic workflow: load → pipeline → compute
- **And** example uses generic filenames, not specific test data paths
- **And** example is a teaser, not a complete tutorial

#### Scenario: Quick start has comprehensive examples

- **Given** user visits quickstart guide
- **When** user reads through examples
- **Then** examples progress from simple single-plant to complex batch processing
- **And** each example introduces new concepts
- **And** examples use real test data paths users can actually run
- **And** examples cover common use cases

#### Scenario: Examples demonstrate actual usage patterns

- **Given** example shows how to use a feature
- **When** pattern is demonstrated
- **Then** pattern matches usage in test code
- **And** pattern reflects real-world use cases
- **And** pattern uses the actual API, not simplified pseudocode

### Requirement: Module-level imports are accurate

Documentation examples MUST import from actual module structure as it exists in the package.

#### Scenario: Individual function imports work

- **Given** example imports individual functions
- **When** using `from sleap_roots.MODULE import FUNCTION` syntax
- **Then** MODULE matches actual module filename
- **And** FUNCTION exists in that module
- **And** function is publicly accessible (not private/internal)

#### Scenario: Package-level imports work

- **Given** example imports from package level
- **When** using `import sleap_roots as sr` syntax
- **Then** classes/functions accessed via `sr.ClassName` or `sr.function_name`
- **And** all accessed items are exposed in `__init__.py`
- **And** import pattern matches package structure

### Requirement: Examples must be tested before documentation release

New or modified examples MUST be verified to execute correctly before merging documentation changes.

#### Scenario: Example validation in PR process

- **Given** PR modifies code examples in documentation
- **When** reviewer evaluates the PR
- **Then** reviewer verifies imports resolve
- **And** reviewer verifies function signatures match
- **And** reviewer confirms example can execute with test data
- **And** reviewer checks for common pitfalls (wrong module names, non-existent functions)

#### Scenario: Automated example testing (future enhancement)

- **Given** CI pipeline runs for documentation changes
- **When** examples are modified
- **Then** automated script extracts code blocks
- **And** script attempts to execute examples
- **And** script reports failures for broken examples
- **And** PR cannot merge with failing examples

