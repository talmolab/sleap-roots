# multiple_arabidopsis_11do

This directory contains test fixtures for trait computation pipelines, specifically for the *MultipleDicotPipeline* and *MultiplePrimaryRootPipeline*. These fixtures include lateral and primary root predictions for four samples of 11-day-old Arabidopsis plants, a metadata CSV (with genotype and replicate information), and precomputed outputs used to validate pipeline behavior and ensure consistent results over time.

## Purpose

To improve maintainability and organization, a subdirectory is created for each trait computation pipeline that analyzes multiple plants per image. The outputs stored here serve as fixture data for pipeline testing.

## Notes

- All files in this directory are treated as static fixtures.
- If pipeline logic changes in a way that affects these outputs, both the tests and the corresponding fixture data should be reviewed and updated accordingly.