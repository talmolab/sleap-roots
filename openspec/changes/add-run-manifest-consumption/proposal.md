## Why

`trait_extractor/extractor.py`'s `extract_batch` has two correctness gaps, both tracked as the
"sleap-roots (traits)" row of `sleap-roots-pipeline`'s issue #37 ("Cross-repo correctness:
manifest-scoped processing"):

1. **Unscoped discovery (contamination risk).** `extract_batch` recursively globs `input_dir` for
   every `*.predictions.json` file with no scoping to what a given pipeline run actually requested.
   A leftover scan directory from a prior run can be silently reprocessed by a later run — one of
   two real contamination incidents that motivated this cross-repo initiative (see
   `sleap-roots-pipeline`'s `docs/superpowers/specs/2026-08-03-manifest-scoped-processing-redesign.md`).
2. **No skip-if-done at all.** Every discovered scan is unconditionally (re)computed and its
   `{scan_key}.result.json` unconditionally overwritten. Unlike `sleap-roots-predict` (which already
   has weak, existence-only skip-if-done), `trait_extractor` has none whatsoever today — confirmed
   by full read of `extractor.py`/`envelope.py`.

`sleap-roots-contracts` v0.1.0a7 already ships `RunManifest` (`pipeline_run_id` + `scan_keys`) and
`bloomctl` (bloom #653/#655, merged 2026-08-14) already writes it. This change makes `trait_extractor`
consume it: scope discovery to `scan_keys`, and add a from-scratch skip-if-done check driven by a
real `idempotency_key` comparison (not existence-only, so it doesn't repeat predict's original
design mistake).

## What Changes

- Bump `sleap-roots-contracts` from `==0.1.0a3` to `==0.1.0a7` in all three `pyproject.toml` pins
  (dev dependency-group, dev optional-dependencies mirror, `extractor` optional-dependency/container
  profile) — `RunManifest` does not exist before a7. Update the two tests that hardcode the old pin
  literal (`test_envelope.py`'s `contract_version == "0.1.0a3"` assertion and
  `test_package_boundary.py`'s exact-pin guard) to `0.1.0a7`.
- `extract_batch` looks for `RUN_MANIFEST_FILENAME` (`run_manifest.json`) at the top level of
  `input_dir`:
  - **Present:** validate it (`RunManifest.model_validate_json`), scope processing to exactly its
    `scan_keys` (a `*.predictions.json` present under `input_dir` for a `scan_key` NOT in
    `scan_keys` is silently excluded — that's the contamination this change prevents), and copy
    `run_manifest.json` forward into `output_dir` (mirroring predict's established sidecar-copy-forward
    pattern) so `write-back` can see it downstream.
  - **A manifest-declared `scan_key` has no matching `{scan_key}.predictions.json`:** record it as a
    per-scan failure in `BatchResult.failed`, consistent with how a missing sidecar is already a
    per-scan failure today.
  - **Absent:** fall back to today's exact unscoped `rglob` behavior — preserves backward
    compatibility for existing tests, local dev runs, and non-pipeline callers. (Confirmed with user.)
- Add skip-if-done: for each scan about to be processed, build its `Provenance` (and therefore its
  auto-derived `idempotency_key`) early — before the expensive `load_series`/`choose_pipeline`/
  `compute_scan_traits` steps, all of whose inputs are already available at that point. Compare
  BOTH `idempotency_key` AND `contract_version` against any pre-existing
  `output_dir/{scan_key}.result.json` (found during review: `idempotency_key`'s hash does not
  include `contract_version`, so comparing `idempotency_key` alone would let skip-if-done silently
  perpetuate a stale `contract_version` forever across this very proposal's own pin bump — see
  design.md). Both match -> skip (no recompute, no rewrite, recorded in a new `BatchResult.skipped`
  bucket). Either differs, or nothing exists yet -> compute and overwrite as today. A pre-existing
  output file that's corrupt/unparseable is treated as "not done" (never crashes the batch) and logs
  a warning naming the scan.
- Add `skipped: List[str] = field(default_factory=list)` to `BatchResult`, additive and
  non-breaking; `BatchResult.ok` semantics unchanged (skipped is not a failure). Update
  `trait_extractor/__main__.py`'s CLI output to report skipped scans distinctly from succeeded/failed
  (found during review: the CLI is the only audit trail that survives into Argo pod logs in
  production, and today's `main()` would otherwise print nothing at all for a skipped scan).
- Manifest-scoped resolution reuses the existing `seen: Dict[str, Path]` duplicate-scan_key guard
  (found during review: a naive first-match resolution could silently pick a stale leftover
  directory over the correct one for the same in-scope `scan_key` — precisely the contamination
  case this proposal exists to close).
- No CLI/argparse signature change — the entrypoint stays exactly `python -m trait_extractor
  <input_dir> <output_dir>`.

## Impact

- **Affected specs:** `result-envelope-output` (MODIFIED: "Batch driver and module CLI",
  "Provenance assembly with deterministic idempotency key"; ADDED: "Run-manifest scoped discovery
  and copy-forward", "Skip-if-done via idempotency-key comparison"); `trait-extractor-image`
  (MODIFIED: "Container image runs the trait extractor over predict outputs", "Slim contracts
  install via an extractor extra" — found during implementation: this deployed spec independently
  hardcoded the `0.1.0a3` pin literal in its own requirement text/scenarios, unrelated to
  `result-envelope-output`'s copy of the same literal; both needed the same bump or this capability
  would have gone stale relative to the actual pinned version and the updated
  `test_package_boundary.py` guard).
- **Affected code:** `pyproject.toml` (3 pin bumps) + regenerated `uv.lock`, `trait_extractor/extractor.py`
  (`extract_batch`, `extract_scan`, `BatchResult`), `trait_extractor/__main__.py` (report skipped
  scans), a new small `trait_extractor/run_manifest.py` helper module (load + copy-forward for
  `RunManifest`, kept separate from `manifest.py`'s `PredictionManifest`/`ScanMetadata` to avoid
  confusing the two unrelated manifest concepts), a small addition to `trait_extractor/envelope.py`
  (read an existing output's `idempotency_key` + `contract_version`), `tests/trait_extractor/` (new
  fixtures + test cases), `tests/trait_extractor/test_envelope.py` and `test_package_boundary.py`
  (pin-literal bumps, committed atomically with the `pyproject.toml` bump — the suite goes red
  between them otherwise), `docs/dev/trait-extractor-service.md` (stale `0.1.0a3` references +
  unscoped-discovery description), `docs/changelog.md` (`[Unreleased]` entry).
- **Explicitly out of scope:** `sleap-roots-predict`'s own `RunManifest` consumption/idempotency-key
  upgrade (separate repo, parallel handoff); `write-back`'s unscoped-glob bug (confirmed as
  `Salk-Harnessing-Plants-Initiative/bloom#678`, still open, not this repo); any manifest-write-side
  locking (bloomctl already owns that); any CLI/argparse signature change.
- **Confirmed pre-deploy blocker, NOT resolved by this proposal** (found during review, verified via
  `gh`): Bloom's live `insert_cyl_result_envelope` write-back RPC (bloom PR #399, closing bloom#393)
  hard-pins its accepted `contract_version` to exactly `0.1.0a3` (prefix-tolerant only for a leading
  `v`; its own tests assert `0.1.0a2`/`0.1.0a30`/etc. are rejected). This proposal's pin bump means
  every envelope emitted after this ships will carry `contract_version = "0.1.0a7"`, which that RPC
  will reject until a companion Bloom-side PR (mirroring #399's pattern) re-pins its accepted
  literal. Merging and testing this proposal in `sleap-roots` is unaffected (nothing here talks to
  the live RPC), but **the resulting image must not be redeployed to the pipeline until that
  Bloom-side change lands** — see design.md's Risks section. This is a deploy-sequencing decision
  for the user, not something this proposal can resolve on its own.
- **Cross-repo tracking:** `talmolab/sleap-roots-pipeline#37`. On merge, update that repo's
  `docs/bloom-integration/roadmap.md` roadmap table + status log (separate PR in that repo, not part
  of this change). Also worth cross-linking `talmolab/sleap-roots#259` (open, same files —
  Argo-ready exit semantics/SIGTERM handling — different concern, not a scope gap) in the eventual
  PR description to reduce merge-sequencing surprises.
