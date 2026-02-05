## ADDED Requirements

### Requirement: Frame Sampling for Performance

The viewer SHALL sample frames by default to ensure reasonable file sizes and browser performance.

#### Scenario: Default frame sampling
- **WHEN** user runs viewer without `--max-frames` option
- **THEN** 10 frames are sampled per scan (evenly distributed, including first and last)

#### Scenario: Custom frame sampling
- **WHEN** user specifies `--max-frames N`
- **THEN** N frames are sampled per scan
- **AND** if N is 0, all frames are included

#### Scenario: Warning for large output
- **WHEN** total frames across all scans exceeds 100
- **THEN** a warning message is displayed

#### Scenario: Hard limit protection
- **WHEN** total frames across all scans exceeds 1000
- **AND** `--no-limit` is not specified
- **THEN** generation fails with an error message
- **AND** user is instructed to use `--no-limit` or reduce `--max-frames`

#### Scenario: Override hard limit
- **WHEN** `--no-limit` is specified
- **THEN** no frame limit is enforced