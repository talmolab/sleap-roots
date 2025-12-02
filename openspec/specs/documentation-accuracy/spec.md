# documentation-accuracy Specification

## Purpose
TBD - created by archiving change simplify-user-documentation. Update Purpose after archive.
## Requirements
### Requirement: User-facing pages must not link to developer infrastructure

User-facing documentation pages (home, getting started, tutorials) MUST NOT link to developer infrastructure or internal tooling documentation.

#### Scenario: User reads home page performance section
```gherkin
Given a user is on the documentation home page
When they read the "High Performance" section
Then they should see simple performance metrics (e.g., "0.1-0.5s per plant")
And they should see context about benchmark conditions
And they should NOT see links to developer benchmarking infrastructure
And if they want infrastructure details, they can find them in Developer Guide
```

#### Scenario: User wants to understand benchmarking methodology
```gherkin
Given a user wants deep technical details on benchmarking
When they navigate to the Developer Guide
Then they should find a "Benchmarking" section
And it should contain all infrastructure and CI details
But this should NOT be linked from user-facing pages
```

### Requirement: Feature claims must match implementation

All quantitative claims about package capabilities (trait counts, performance metrics, platform support) MUST accurately reflect the actual implementation and CI testing.

#### Scenario: Documentation claims trait count
```gherkin
Given the documentation states a number of traits (e.g., "50+ traits")
When a user computes traits with any pipeline
Then the actual number of output traits should match or exceed the claim
And the claim should be accurate across all documentation pages
And different pages should use consistent numbers
```

#### Scenario: Documentation claims Python version support
```gherkin
Given the documentation claims platform and Python version support
When a user checks the CI configuration
Then the claimed "fully supported" versions should be CI-tested
And versions not CI-tested should be clearly marked as "should work" or "community supported"
And the documentation should not overstate support
```

### Requirement: Modern tooling should be primary recommendation

Documentation MUST recommend modern, fast tooling (uv, pip) as the primary approach, with legacy tools (conda) mentioned minimally as alternatives.

#### Scenario: User reads installation guide
```gherkin
Given a user reads the installation documentation
When they follow the recommended steps
Then they should see pip or uv as primary options
And conda should appear exactly once as "Alternative: Using Conda"
And conda should not be in the main installation flow
```

#### Scenario: User follows quick start tutorial
```gherkin
Given a user follows the quick start tutorial
When they need to set up their environment
Then they should see instructions for pip or uv
And they should NOT see conda activation steps in the main flow
And conda should only appear in installation guide as alternative
```

#### Scenario: User encounters import error
```gherkin
Given a user encounters an import error
When they check troubleshooting documentation
Then they should see universal solutions (pip install, check PATH)
And they should NOT see conda-specific solutions
And conda should not be recommended for error resolution
```

### Requirement: Error documentation must have single source of truth

Common errors and their solutions MUST be documented in exactly one canonical location, with other locations linking to that source.

#### Scenario: User encounters ModuleNotFoundError
```gherkin
Given a user encounters "ModuleNotFoundError: No module named 'sleap_roots'"
When they search for solutions in documentation
Then they should find ONE comprehensive solution in troubleshooting guide
And other pages (installation, quick start) should link to that solution
And the solution should be universal (not tool-specific)
```

#### Scenario: Documentation maintainer updates error solution
```gherkin
Given a documentation maintainer needs to update an error solution
When they edit the troubleshooting guide
Then the solution is updated in exactly one place
And all cross-references automatically point to the updated solution
And there are no duplicate solutions to maintain
```

### Requirement: Performance metrics must include context

Any performance claims (timing, throughput, benchmarks) MUST include sufficient context for users to understand applicability.

#### Scenario: User reads performance claims
```gherkin
Given a user reads "0.1-0.5s per plant" performance claim
When they look for context
Then they should see the platform specifications (cores, RAM)
And they should see the sample size (number of benchmarks)
And they should understand what this timing represents
And they should be able to estimate their own workload
```

#### Scenario: User compares with their own hardware
```gherkin
Given a user has different hardware than benchmark platform
When they read the performance documentation
Then they should see enough context to estimate their performance
And they should understand the benchmark conditions
And they should know if their results might differ
```

