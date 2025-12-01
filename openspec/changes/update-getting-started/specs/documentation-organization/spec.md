# Documentation Organization Spec

## ADDED Requirements

### Requirement: User documentation simplicity
User-facing installation documentation MUST focus exclusively on end-user workflows without mixing development setup instructions.

#### Scenario: User installs sleap-roots without cloning repository
```gherkin
Given a user wants to use sleap-roots for trait extraction
When they follow the Getting Started installation guide
Then they should see clear instructions for pip installation
And they should see instructions for uv-based installation
And they should NOT see Git LFS setup instructions
And they should NOT see development environment setup
And they should NOT be instructed to clone the repository
```

#### Scenario: User finds modern uv workflow
```gherkin
Given a user prefers modern Python tooling
When they read the installation documentation
Then they should find instructions for `uv init` workflow
And they should find instructions for `uv add sleap-roots`
And the workflow should be presented as a recommended option
```

#### Scenario: User verifies installation
```gherkin
Given a user has installed sleap-roots
When they follow the installation verification steps
Then they should be able to run a simple import test
And they should receive confirmation that installation succeeded
And the verification should take less than 30 seconds
```

### Requirement: Developer documentation completeness
Developer setup documentation MUST provide comprehensive instructions for contributing to the project.

#### Scenario: Developer clones repository for contribution
```gherkin
Given a developer wants to contribute to sleap-roots
When they follow the Developer Guide setup instructions
Then they should find instructions for Git and Git LFS setup
And they should find instructions for cloning the repository
And they should find instructions for installing development dependencies
And they should find instructions for running tests
And they should find links to contribution guidelines
```

#### Scenario: Developer uses uv for development
```gherkin
Given a developer has cloned the sleap-roots repository
When they follow the development setup guide
Then they should find instructions for `uv sync` to install dev dependencies
And they should find alternative conda workflow instructions
And both workflows should be clearly documented
```

#### Scenario: Developer sets up test environment
```gherkin
Given a developer needs to run tests
When they follow the developer setup guide
Then they should know how to install test dependencies
And they should know how to run pytest
And they should know how to check code formatting
And they should know how to set up pre-commit hooks
```

### Requirement: Clear navigation separation
Documentation navigation MUST clearly distinguish between user and developer paths.

#### Scenario: User navigates from home to installation
```gherkin
Given a user is on the documentation home page
When they click "Getting Started" → "Installation"
Then they should land on user-focused installation instructions
And the page should be under 100 lines
And the page should complete in under 5 minutes of reading
```

#### Scenario: Developer navigates to setup guide
```gherkin
Given a developer is on the documentation site
When they look for development setup instructions
Then they should find it under "Developer Guide" → "Development Setup"
And the guide should be comprehensive (>100 lines acceptable)
And it should link to contributing guidelines
```

#### Scenario: User is directed to developer resources when needed
```gherkin
Given a user is reading the Getting Started installation guide
When they want to contribute or use test data
Then they should see a clear callout box
And the box should link to "Developer Guide" → "Development Setup"
And the distinction should be obvious
```

### Requirement: Cross-reference consistency
All cross-references to installation documentation MUST point to the appropriate user or developer guide.

#### Scenario: README links to installation
```gherkin
Given a user reads the README.md file
When they click on installation instructions
Then they should be directed to the user-focused Getting Started guide
And NOT to developer setup instructions
```

#### Scenario: Contributing guide links to setup
```gherkin
Given a developer reads the contributing guide
When they need setup instructions
Then they should be directed to the Developer Guide setup page
And NOT to the user installation page
```

#### Scenario: API docs reference installation
```gherkin
Given a user is reading API documentation
When they encounter installation references
Then links should point to Getting Started (for users)
And developer-specific content should link to Developer Guide
```