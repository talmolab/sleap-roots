# Tasks: Benchmark Regression Detection and Historical Tracking

## Phase 0: Versioned Documentation Setup (Prerequisite for Historical Tracking)

**✅ COMPLETED** - See archived `enable-versioned-docs` proposal (PR #134)

### Configure mike plugin
- [x] Add `mike` plugin to `mkdocs.yml` plugins section (following lablink pattern)
- [x] Configure `alias_type: symlink`, `canonical_version: latest`, `version_selector: true`
- [x] Test local build with `mike serve` to verify configuration
- [x] Check version selector appears in Material theme UI

### Create docs deployment workflow
- [x] Check if `.github/workflows/docs.yml` exists, create if needed
- [x] Configure workflow to use `mike deploy --push --update-aliases VERSION ALIAS`
- [x] Set up version aliases: `latest` for development, `stable` for releases
- [x] Add workflow trigger on pushes to main and tags matching `v*`
- [x] Test workflow by creating test tag

### Initial version deployment
- [x] Deploy current docs as `latest`: `mike deploy --push latest`
- [x] Verify `latest` version appears at https://talmolab.github.io/sleap-roots/latest/
- [x] Verify version selector dropdown works in docs UI
- [ ] Update README.md with link to versioned docs (optional)

### Documentation
- [ ] Document versioning strategy in `docs/dev/release-process.md` (optional)
- [ ] Add section explaining how versions map to releases (optional)
- [ ] Document how to deploy new versions manually if needed (optional)

## Phase 1: Baseline Infrastructure (Foundation)

### Setup baseline storage
17. Create `.benchmarks/baselines/` directory structure
18. Create `.benchmarks/history/` directory structure
19. Add `.benchmarks/` to `.gitignore` for local dev (not CI)
20. Document baseline JSON schema in README

### Update main branch workflow
21. Modify `.github/workflows/ci.yml` benchmark job to store baseline
22. Add step to copy results to `.benchmarks/baselines/main.json`
23. Add step to copy results to `.benchmarks/baselines/<commit-sha>.json`
24. Add step to copy results to `.benchmarks/history/<date>.json`
25. Add step to commit baseline files with "chore: update benchmark baselines [skip ci]"
26. Add step to cleanup baselines older than 90 days
27. Test baseline storage by pushing to main

## Phase 2: Regression Detection (Core)

### Add baseline comparison logic
28. Create `tests/benchmarks/conftest.py` with pytest hooks
29. Implement `pytest_benchmark_compare_machine_info` hook to load baseline
30. Implement regression detection logic comparing current to baseline
31. Add threshold check using `BENCHMARK_MAX_REGRESSION` env var
32. Write regression details to `benchmark-regressions.json` on failure
33. Add pytest marker for per-benchmark threshold overrides
34. Write unit tests for baseline comparison logic in `tests/benchmarks/test_baseline_comparison.py`
35. Write unit tests for regression detection in `tests/benchmarks/test_regression_detection.py`

### Create PR benchmark workflow
36. Add new `benchmark-pr` job to `.github/workflows/ci.yml`
37. Configure job to run on `pull_request` events
38. Add step to checkout PR branch
39. Add step to fetch baseline from main branch (`git checkout origin/main -- .benchmarks/baselines/main.json`)
40. Add step to run benchmarks with `--benchmark-compare` flag
41. Add step to check for regressions and fail job if detected
42. Add step to upload `benchmark-results.json` as artifact
43. Add step to upload `benchmark-regressions.json` as artifact (if exists)
44. Set artifact retention to 30 days

### Add PR commenting
45. Add GitHub Actions script step to post benchmark comparison as PR comment
46. Implement table formatting: Test | Main | PR | Change | Status
47. Add regression warning if any benchmarks exceed threshold
48. Add emoji indicators (✅ for OK, ⚠️ for regression)
49. Handle case where no baseline exists (new benchmarks)
50. Test PR commenting on dogfooding PR

## Phase 3: Review Integration (Workflow)

### Update review command
51. Update `.claude/commands/review-pr.md` with benchmark section
52. Add bash script to fetch benchmark artifact via `gh run download`
53. Add Python script to parse and format benchmark comparison
54. Add example output showing formatted comparison table
55. Document how to interpret benchmark results in review
56. Add checklist item for performance regressions
57. Test review workflow on PR with benchmarks

### Documentation updates
58. Update `docs/dev/benchmarking.md` with PR workflow section
59. Document regression threshold (15% default)
60. Explain baseline storage and management
61. Add troubleshooting section for false positives
62. Show example of PR comment with benchmark results
63. Document per-benchmark threshold override syntax
64. Link to OpenSpec proposal for design rationale

## Phase 4: Historical Tracking (Visibility) - Requires Phase 0

### Create benchmark history page
65. Create `docs/benchmarks/index.md` template
66. Add introduction explaining benchmark methodology
67. Add link to benchmarking guide
68. Add placeholder for performance charts

### Implement chart generation
69. Create `scripts/generate-benchmark-history.py` script
70. Implement data loading from `.benchmarks/history/*.json`
71. Generate line chart for each pipeline using matplotlib
72. Add ±1 stddev bands around mean
73. Annotate chart with version releases
74. Save charts as SVG in `docs/benchmarks/charts/`
75. Generate summary table: latest vs 7-day vs 30-day averages
76. Add trend indicators (📈 improving, 📊 stable, 📉 degrading)

### Integrate with docs site
77. Update `mkdocs.yml` to include benchmark history page in navigation
78. Add "Benchmarks" section under Developer Guide
79. Configure chart rendering in Material theme
80. Test local docs build with `uv run mkdocs serve`
81. Verify charts render correctly in browser

### Automate chart generation
82. Create `.github/workflows/publish-benchmarks.yml` workflow
83. Configure to run on main branch pushes (after benchmark job)
84. Add step to run `scripts/generate-benchmark-history.py`
85. Add step to commit generated charts to repo
86. Add step to trigger docs rebuild (if needed)
87. Set up GitHub Pages deployment (if not already configured)

### Data transparency
88. Add download link for raw JSON data on benchmark page
89. Create ZIP archive of `.benchmarks/history/` directory
90. Add README.txt explaining JSON schema
91. Link individual data points to GitHub commits
92. Link chart data points to CI runs

## Phase 5: Testing and Validation

### Create test PRs
93. Create PR with intentional 20% performance regression
94. Verify CI fails with benchmark regression error
95. Verify PR comment shows regression warning
96. Verify artifact contains regression details
97. Close regression test PR

### Test baseline updates
98. Merge PR to main with acceptable performance
99. Verify baseline is updated in `.benchmarks/baselines/main.json`
100. Verify commit appears in Git history
101. Verify history file created in `.benchmarks/history/<date>.json`

### Test review workflow
102. Create test PR with mixed results (some regressions, some improvements)
103. Run `/review-pr` command on the PR
104. Verify benchmark data is fetched
105. Verify formatted comparison appears in output
106. Verify review comment includes benchmark section

### Test documentation
107. Deploy docs to GitHub Pages staging
108. Verify benchmark history page loads
109. Verify charts display correctly
110. Verify data download works
111. Verify links to commits/CI runs work
112. Check page on mobile and desktop

## Phase 6: Polish and Monitoring

### Refine thresholds
113. Review historical variance in benchmark results
114. Adjust default threshold based on observed CI runner variance
115. Identify benchmarks needing stricter thresholds
116. Apply per-benchmark overrides where needed
117. Document threshold tuning rationale

### Error handling
118. Add graceful handling for missing baseline
119. Add fallback for artifact download failures
120. Add retry logic for Git operations
121. Log detailed errors for debugging

### Documentation polish
122. Add screenshots to benchmarking guide
123. Record video walkthrough of PR benchmark workflow
124. Add FAQ section addressing common questions
125. Link to real example PRs showing workflow

### Monitoring setup
126. Document metrics to track (false positive rate, etc.)
127. Set up alerts for baseline storage failures
128. Create runbook for investigating regressions
129. Schedule quarterly review of threshold effectiveness

## Phase 7: Rollout and Communication

### Announce feature
130. Write announcement message for team
131. Update CHANGELOG.md with new feature
132. Add note to PR template about benchmark checks
133. Update contributing guide with benchmark expectations

### Training
134. Walk team through example PR with benchmarks
135. Demonstrate review command with benchmark integration
136. Show how to investigate regressions using artifacts
137. Collect feedback and iterate

### Final validation
138. Run full benchmark suite on main
139. Verify all baselines are current
140. Verify all documentation is accurate
141. Mark OpenSpec change as complete