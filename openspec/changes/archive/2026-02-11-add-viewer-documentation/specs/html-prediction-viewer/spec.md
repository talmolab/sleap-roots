## ADDED Requirements

### Requirement: User Documentation

The system SHALL provide user documentation for the HTML prediction viewer.

#### Scenario: User discovers viewer feature
- **WHEN** a user reads the README
- **THEN** they find a section describing the viewer command

#### Scenario: User learns viewer usage
- **WHEN** a user reads the prediction viewer guide
- **THEN** they find CLI examples, output mode descriptions, and keyboard navigation reference

#### Scenario: User finds viewer in guides index
- **WHEN** a user browses docs/guides/
- **THEN** they find a link to the prediction viewer guide

### Requirement: DRY Documentation

The system SHALL follow DRY principle for documentation.

#### Scenario: No duplicated CLI reference
- **WHEN** CLI options are documented
- **THEN** they appear in one canonical location (the guide)
- **AND** README links to the guide for details

#### Scenario: README provides overview only
- **WHEN** README describes the viewer
- **THEN** it provides a quick example and links to the full guide
- **AND** does not duplicate the full CLI reference