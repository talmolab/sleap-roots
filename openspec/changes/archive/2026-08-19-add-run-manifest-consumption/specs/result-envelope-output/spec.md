## MODIFIED Requirements

### Requirement: Batch driver and module CLI

The `trait_extractor` package SHALL provide a callable entry `python -m trait_extractor
<input_dir> <output_dir>` (exactly 2 required positional args; no CLI flag for manifest scoping is
added by this requirement) that **recursively** discovers each `{scan_key}.predictions.json` under
`input_dir` (matching predict's per-scan `out_dir/{scan_key}/` batch layout as well as a flat
layout) when no run-manifest scoping applies (see the "Run-manifest scoped discovery and
copy-forward" requirement for the scoped case), resolves each manifest's sidecar and `.slp` files
**co-located in the manifest's own directory**, and writes one `{manifest.scan_key}.result.json`
per scan to `output_dir` (a separate tree, so `*.result.json` never collides with discovery). The
manifest filename stem SHALL equal `manifest.scan_key` (raise on disagreement); the sidecar is
paired by that key. One scan's failure SHALL NOT discard the other scans' envelopes — the broad
per-scan `except` lives ONLY in the batch loop (never in the manifest guards), the failure is
reported, and the process exits non-zero if any scan failed. The CLI SHALL report skipped scans
(see "Skip-if-done via idempotency-key comparison") distinctly from succeeded and failed scans — a
skipped scan MUST appear in the CLI's output, not simply be omitted, so an operator watching
container/Argo logs can distinguish "reused from a prior run" from "never discovered."

#### Scenario: Batch run emits one envelope per scan

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs over an input tree containing
  two per-scan directories (`scan0K9E8BI/`, `scanYR39SJX/`), each with its manifest + sidecar +
  `.slp`, and no `run_manifest.json`
- **THEN** exactly one `{scan_key}.result.json` is written per scan under `output_dir`

#### Scenario: Manifest filename stem must equal manifest.scan_key

- **WHEN** a manifest's filename stem disagrees with its `scan_key` field
- **THEN** the driver raises an error identifying the disagreement for that scan

#### Scenario: One scan's failure does not abort the batch

- **WHEN** a batch tree contains a valid scan and a scan that raises (e.g. a manifest naming a
  nonexistent `.slp`)
- **THEN** the driver writes the valid scan's `{scan_key}.result.json`, reports the failed scan,
  and exits non-zero — without discarding the successful envelope

#### Scenario: Manifest without a matching sidecar

- **WHEN** a `{scan_key}.predictions.json` has no `{scan_key}.scan_metadata.json` in its directory
- **THEN** the pairing loop reports an error naming the missing sidecar for that scan and does not
  emit an envelope for it

#### Scenario: run_manifest.json absent falls back to unscoped discovery

- **WHEN** `input_dir` contains no `run_manifest.json` anywhere
- **THEN** `extract_batch` discovers scans via the same unscoped recursive glob as before this
  change, byte-identical in behavior and output to the pre-manifest implementation

#### Scenario: The CLI prints skipped scans, not just succeeded/failed

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs a second time over inputs
  that are unchanged since the first run
- **THEN** the CLI's output names the skipped scan(s) distinctly from `ok`/`FAIL` lines, and the
  summary counts include a skipped count

### Requirement: Provenance assembly with deterministic idempotency key

The `trait_extractor` package SHALL assemble a `Provenance` with: `contract_version` from
`importlib.metadata.version("sleap-roots-contracts")`; `scan_key` from the manifest; `inputs` from
the sidecar (`image_ids`, `images_checksum`); `predict_models = [artifact.model for artifact in
manifest.artifacts]`; `predict_container_digest`, `predict_code_sha`, `predict_inference_config`,
and `predict_output_params` from the manifest (by value, no coercion); `traits_sleap_roots_version`
from `sleap_roots.__version__`; `params` as the canonical `ResolvedParams`; and `traits_code_sha` /
`traits_container_digest` resolved fail-soft from arguments, then environment, then `""`. For
byte-stable re-emission, `produced_at` SHALL be left `None` (not `datetime.now()`), and the
orchestration fields `pipeline_run_id`, `worker_request_id`, `argo_workflow_uid`, `argo_node_id`
SHALL be `None` in this slice. The resulting `idempotency_key` SHALL equal
`compute_idempotency_key(...)` for the same inputs and be identical across repeated runs. This
`Provenance` (and its `idempotency_key`) SHALL be buildable using only `manifest`, `sidecar`,
`params`, and the batch-level `traits_code_sha`/`traits_container_digest` — i.e. before any
SLEAP-data-dependent step (`Series` loading, pipeline selection/compatibility, trait computation)
runs, so it can be computed and compared cheaply as part of skip-if-done.

#### Scenario: Idempotency key is deterministic and matches the contract helper

- **WHEN** a `Provenance` is assembled and `compute_idempotency_key` is called with the same
  `scan_key`, `images_checksum`, models, `param_hash`, `predict_code_sha`, `traits_code_sha`, and
  `predict_output_params`
- **THEN** `Provenance.idempotency_key` is non-empty, equals the helper's result, and re-running
  the assembly for the same inputs produces the identical key

#### Scenario: contract_version is the pinned bare package version

- **WHEN** a `Provenance` is assembled
- **THEN** `Provenance.contract_version` equals
  `importlib.metadata.version("sleap-roots-contracts")` (tracking the pin), AND additionally
  asserts the literal `== "0.1.0a7"` and `not .startswith("v")` — so a silent pin bump fails the
  test and forces conscious cross-repo coordination (`talmolab/sleap-roots-pipeline#37`); note that
  bumping the *actually-deployed* pin also requires a companion Bloom-side RPC update (bloom PR
  #399's pattern, re-pinning `insert_cyl_result_envelope`'s accepted literal) before rollout — see
  this change's `design.md` Risks section for the confirmed, not-fixable-here dependency

#### Scenario: predict_output_params passes through unchanged

- **WHEN** a manifest carries `predict_output_params = {"peak_threshold": 0.2}`
- **THEN** the assembled `Provenance.predict_output_params` is byte-equal and the idempotency key
  reflects it unchanged (no value coercion)

#### Scenario: Build identity resolves fail-soft

- **WHEN** neither `traits_code_sha`/`traits_container_digest` arguments nor their environment
  variables are set
- **THEN** those fields are `""` and assembly does not raise

#### Scenario: Provenance is computable before any expensive step

- **WHEN** `build_provenance` is called with only `manifest`, `sidecar`, and `params` (no `Series`
  loaded, no pipeline selected, no traits computed)
- **THEN** it returns a fully valid `Provenance` with a non-empty `idempotency_key`

## ADDED Requirements

### Requirement: Run-manifest scoped discovery and copy-forward

At the top of a batch run, `extract_batch` SHALL look for
`sleap_roots_contracts.run_manifest.RUN_MANIFEST_FILENAME` (`"run_manifest.json"`) at the top level
of `input_dir`.

- **If present:** it SHALL be validated as a `RunManifest`
  (`RunManifest.model_validate_json`); a present-but-invalid manifest (e.g. empty `scan_keys`,
  failing `RunManifest`'s own pydantic validation) SHALL raise before any scan is processed — this
  is a once-per-batch, top-level file, not a per-scan best-effort read, and MUST NOT be treated the
  same as a per-scan corrupt-output fallback. When valid, discovery SHALL be scoped to exactly its
  `scan_keys`: each `scan_key` SHALL be resolved to its `{scan_key}.predictions.json` under
  `input_dir` using the same recursive search as the unscoped path, and two candidate files
  resolving to the same in-scope `scan_key` SHALL be treated as a duplicate-scan_key collision
  exactly as the existing unscoped-duplicate guard already does (reported as a per-scan failure,
  not silently resolved by picking either candidate) — this prevents the scoped path from
  reintroducing the "right scan_key, stale/wrong file" contamination case even as it fixes the
  "wrong scan_key entirely" case. Two candidates whose scan_key stems differ ONLY by case SHALL
  also be treated as a collision and reported as a per-scan failure, since they would write to the
  same output filename on a case-insensitive filesystem (the default on Windows and macOS) despite
  being distinct strings that an exact-match duplicate check alone would never catch. A
  `*.predictions.json` present under `input_dir` for a `scan_key`
  NOT in `scan_keys` SHALL be silently excluded from `BatchResult` entirely (not processed, not
  reported in `succeeded`, `skipped`, or `failed`) — that exclusion is precisely the
  contamination-prevention this requirement exists for. A manifest-declared `scan_key` with no
  matching `{scan_key}.predictions.json` SHALL be recorded as a per-scan failure in
  `BatchResult.failed` naming the missing file, without aborting the rest of the batch. A
  pre-existing `{scan_key}.result.json` in `output_dir` whose `scan_key` is NOT in the current
  run's `scan_keys` (e.g. left over from a prior run whose manifest was wider) SHALL be left
  completely untouched — not reprocessed, not reported in any `BatchResult` bucket, not deleted —
  but SHALL be logged as a warning naming it, so its presence is at least traceable rather than
  silently indistinguishable from a current result. After
  processing, `run_manifest.json` SHALL be copied forward into `output_dir` (creating `output_dir`
  if missing) as a raw file copy (not a re-serialization through the `RunManifest` model), byte-identical
  to the source, so `write-back` can see it downstream. This copy-forward is **best-effort
  infrastructure for the next pipeline stage, not part of this batch's own computed result**: an
  `OSError` during the copy (e.g. a disk/permission error, or an equivalent same-file condition)
  SHALL be logged as a warning naming both directories and MUST NOT raise, crash the batch, or
  discard the already-computed `succeeded`/`skipped`/`failed` results.
- **If absent:** discovery SHALL fall back to the unscoped recursive glob described in "Batch
  driver and module CLI" — i.e. today's exact pre-manifest behavior, unchanged.

#### Scenario: run_manifest.json present scopes discovery to scan_keys

- **WHEN** `run_manifest.json` is present at the top level of `input_dir` with `scan_keys` equal
  to exactly the fixture tree's two scan keys
- **THEN** both scans are processed and the resulting `.result.json` files are byte-for-byte
  identical to the no-manifest happy-path output

#### Scenario: An out-of-scope predictions.json is silently excluded

- **WHEN** `run_manifest.json`'s `scan_keys` names only one of two scans present under
  `input_dir`
- **THEN** only the named scan appears in `BatchResult.succeeded`; the other scan's
  `.predictions.json` is left untouched and does not appear in `succeeded`, `skipped`, or `failed`

#### Scenario: A manifest-declared scan_key with no predictions.json is a per-scan failure

- **WHEN** `run_manifest.json`'s `scan_keys` includes a scan_key with no matching
  `{scan_key}.predictions.json` anywhere under `input_dir`
- **THEN** that scan_key appears in `BatchResult.failed` with a message naming the missing file,
  and other in-scope scans still process normally

#### Scenario: A duplicate in-scope scan_key is a per-scan failure, not a silent pick

- **WHEN** `run_manifest.json` names a `scan_key` that resolves to two different
  `{scan_key}.predictions.json` candidates under `input_dir` (e.g. a stale leftover directory
  alongside the correct one)
- **THEN** that scan_key is reported as a duplicate-collision failure in `BatchResult.failed`,
  exactly as the unscoped path already reports a duplicate across manifests — never silently
  resolved by taking whichever candidate is found first

#### Scenario: An invalid run_manifest.json aborts the batch

- **WHEN** `run_manifest.json` is present under `input_dir` but fails `RunManifest`'s own
  validation (e.g. `scan_keys` is empty)
- **THEN** `extract_batch` raises before processing any scan, rather than silently falling back to
  unscoped discovery or partially succeeding

#### Scenario: A case-only scan_key difference is a per-scan failure, not a silent overwrite

- **WHEN** `run_manifest.json` names a `scan_key` that resolves to two candidates whose stems
  differ only by case (e.g. `ScanYR39SJX` and `scanyr39sjx`)
- **THEN** the second-discovered candidate is reported as a case-insensitive collision failure in
  `BatchResult.failed`, never silently overwriting the first candidate's output

#### Scenario: A scan_key dropped from scope leaves its prior output untouched, but logged

- **WHEN** a prior run's wider manifest produced `{scan_key}.result.json` in `output_dir`, and the
  current run's `run_manifest.json` no longer includes that `scan_key`
- **THEN** that file is not reprocessed, not reported in `succeeded`, `skipped`, or `failed`, and
  is not deleted, but a warning naming it is logged

#### Scenario: The manifest is copied forward into output_dir

- **WHEN** `run_manifest.json` is present under `input_dir` and `extract_batch` completes
- **THEN** `output_dir/run_manifest.json` exists with content identical to the source file

#### Scenario: input_dir and output_dir resolving to the same file does not crash the batch

- **WHEN** `run_manifest.json` is present and `input_dir` and `output_dir` are the same directory
  (or otherwise resolve to the same file)
- **THEN** the copy-forward is a no-op (the file is already in place) and `extract_batch` completes
  normally, without raising

#### Scenario: A copy-forward failure does not discard the batch's computed result

- **WHEN** copying `run_manifest.json` forward raises an `OSError` (e.g. a disk or permission
  error)
- **THEN** a warning naming `input_dir`, `output_dir`, and the error is logged, and
  `extract_batch` still returns the `BatchResult` already computed by the per-scan loop above,
  rather than raising

### Requirement: Skip-if-done via idempotency-key comparison

The `trait_extractor` package SHALL avoid recomputing a scan whose expected output already exists
and matches, using a real comparison of BOTH `idempotency_key` AND `contract_version` — never an
existence-only check, and never `idempotency_key` alone (a `contract_version`-only difference, such
as a contracts pin bump with no other input change, MUST also force recomputation, since
`idempotency_key`'s hash does not include `contract_version`). For each scan about to be processed,
the orchestrator SHALL build the scan's `Provenance` early (immediately after resolving `params`,
before `Series` loading/pipeline selection/trait computation) and compare both its
`idempotency_key` and `contract_version` against the values stored in any pre-existing
`output_dir/{scan_key}.result.json`. If both match, the scan SHALL be skipped: no `Series` loading,
pipeline selection, trait computation, or file write occurs, and the scan_key is recorded in a new
`BatchResult.skipped: List[str]` field (additive; `BatchResult.ok` is `True` iff `failed` is empty,
unaffected by `skipped`). If either differs, or no valid pre-existing output exists, the scan SHALL
be computed and its output written/overwritten exactly as if skip-if-done did not exist, reusing the
already-built `Provenance` for the final envelope. A pre-existing `{scan_key}.result.json` that is
missing, unreadable, not valid JSON, or fails `ResultEnvelope` validation SHALL be treated as "not
done" (recompute) and MUST NOT raise or abort the batch. A pre-existing file that is present but
unreadable/invalid (corrupt JSON, or valid JSON failing `ResultEnvelope` validation) SHALL
additionally log a warning naming the scan_key, so that anomaly is traceable rather than silently
overwritten with zero record — a plain missing file (the ordinary first-run case) is expected and
MUST NOT log a warning.

#### Scenario: An unchanged scan is skipped, not recomputed

- **WHEN** `extract_batch` runs twice over identical inputs (same manifest, predictions, and
  sidecar) into the same `output_dir`
- **THEN** the second run's `BatchResult.skipped` contains the scan_key (not `succeeded`), and the
  output file's content is unchanged by the second run

#### Scenario: A changed identity input forces recomputation

- **WHEN** `extract_batch` runs a second time with a different `traits_code_sha` than the first run
- **THEN** the scan_key appears in `BatchResult.succeeded` (not `skipped`) on the second run, and
  the output file's `idempotency_key` changes accordingly

#### Scenario: A contract_version-only change forces recomputation even when idempotency_key matches

- **WHEN** `extract_batch` runs a second time after the installed `sleap-roots-contracts` version
  changed (so `contract_version` differs) but every other idempotency-key input is unchanged (so
  `idempotency_key` alone would still match)
- **THEN** the scan_key appears in `BatchResult.succeeded` (not `skipped`), and the output file's
  `contract_version` is updated to the new value — a stale `contract_version` is never silently
  perpetuated by skip-if-done

#### Scenario: A corrupt prior output does not crash the batch, and is logged

- **WHEN** `output_dir` already contains a `{scan_key}.result.json` that is not valid JSON (or is
  valid JSON that fails `ResultEnvelope` validation)
- **THEN** `extract_batch` treats that scan as not-yet-done, recomputes it, overwrites the file with
  a valid envelope, logs a warning naming the scan_key, and does not raise

#### Scenario: Skipping avoids the expensive computation, not just the write

- **WHEN** a scan is skipped via idempotency-key and contract_version match
- **THEN** `Series` loading, pipeline selection, pipeline-compatibility checking, and trait
  computation are not invoked for that scan
