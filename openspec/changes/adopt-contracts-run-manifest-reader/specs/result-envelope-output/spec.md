## MODIFIED Requirements

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
SHALL be `None` in this slice — in particular the run identity resolved by `extract_batch` (see
"Run-manifest scoped discovery and copy-forward") SHALL NOT be stamped into
`Provenance.pipeline_run_id`, so envelopes re-emitted by different runs over identical inputs stay
byte-identical. The resulting `idempotency_key` SHALL equal
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
  asserts the literal `== "0.1.0a9"` and `not .startswith("v")` — so a silent pin bump fails the
  test and forces conscious cross-repo coordination (`talmolab/sleap-roots-pipeline#37`); note that
  bumping the *actually-deployed* pin also requires a companion Bloom-side update so
  `insert_cyl_result_envelope` accepts the new literal (bloom PRs #399 and #766 are prior
  instances) before rollout — a confirmed cross-repo dependency this repository cannot fix itself

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

### Requirement: Batch driver and module CLI

The `trait_extractor` package SHALL provide a callable entry `python -m trait_extractor
<input_dir> <output_dir>` (exactly 2 required positional args; no CLI flag for manifest scoping or
run identity is added by this requirement — run identity comes from the environment, see the
"Run-manifest scoped discovery and copy-forward" requirement) that **recursively** discovers each
`{scan_key}.predictions.json` under `input_dir` (matching predict's per-scan `out_dir/{scan_key}/`
batch layout as well as a flat layout) when no run-manifest scoping applies (see the "Run-manifest
scoped discovery and copy-forward" requirement for the scoped case), resolves each manifest's
sidecar and `.slp` files **co-located in the manifest's own directory**, and writes one
`{manifest.scan_key}.result.json` per scan to `output_dir` (a separate tree, so `*.result.json`
never collides with discovery). The manifest filename stem SHALL equal `manifest.scan_key` (raise
on disagreement); the sidecar is paired by that key. One scan's failure SHALL NOT discard the other
scans' envelopes — the broad per-scan `except` lives ONLY in the batch loop (never in the manifest
guards), and the failure is reported. The CLI SHALL report skipped scans (see "Skip-if-done via
idempotency-key comparison") distinctly from succeeded and failed scans — a skipped scan MUST
appear in the CLI's output, not simply be omitted, so an operator watching container/Argo logs can
distinguish "reused from a prior run" from "never discovered."

The process SHALL exit with one of three driver-owned codes so an Argo-driven caller can
distinguish a fully-successful run from a partially-failed-but-completed run from a run that could
not proceed at all:
- `0` — every discovered scan succeeded or was skipped; `BatchResult.ok` is `True`.
- `3` — **partial**: the batch ran to completion but one or more scans failed inside the driver's
  own per-scan isolation boundary (`BatchResult.failed` is non-empty).
- `1` — **crash**: an exception escaped `extract_batch` entirely before it could return a
  `BatchResult` at all (e.g. an invalid run manifest, a missing run manifest for a known run
  identity, a per-run manifest naming a different run, an unusable `ARGO_WORKFLOW_NAME`, a missing
  `input_dir`, or the input-discovery guard below). For each escaping exception of type
  `RuntimeError`, `OSError`, `ValueError` (which includes `pydantic.ValidationError` and
  `UnicodeDecodeError`), `yaml.YAMLError`, or `sleap_roots_contracts.RunManifestError` (neither an
  `OSError` nor a `ValueError`), the CLI SHALL log a one-line `Batch aborted: <message>` at `ERROR`
  before re-raising it.

Scenarios in this requirement assume no `ARGO_WORKFLOW_NAME` in the environment unless they
state otherwise.

Exit code `2` is deliberately NOT part of this convention: `argparse` already exits `2` on a CLI
usage error (missing/extra positional arguments), before `extract_batch` runs at all. Reusing `2`
for "partial" would make a CLI-invocation misconfiguration indistinguishable from a completed batch
with isolated scan failures.

When no run manifest resolves (no run identity and no `run_manifest.json` at the top level of an
existing `input_dir` — unscoped mode) and zero `{scan_key}.predictions.json` files are discovered
anywhere under it, `extract_batch` SHALL raise rather than return a vacuous, all-succeeded
`BatchResult` — an empty or misconfigured input mount SHALL NOT be reported as a successful run
that simply had nothing to do. An `input_dir` that does not exist at all SHALL raise
`FileNotFoundError` naming it (raised by the contracts reader before discovery), which the CLI
reports as a crash.

#### Scenario: Batch run emits one envelope per scan

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs over an input tree containing
  two per-scan directories (`scan0K9E8BI/`, `scanYR39SJX/`), each with its manifest + sidecar +
  `.slp`, no run manifest, and no `ARGO_WORKFLOW_NAME` in the environment
- **THEN** exactly one `{scan_key}.result.json` is written per scan under `output_dir`
- **AND** the process exits `0`

#### Scenario: Manifest filename stem must equal manifest.scan_key

- **WHEN** a manifest's filename stem disagrees with its `scan_key` field
- **THEN** the driver raises an error identifying the disagreement for that scan

#### Scenario: One scan's failure does not abort the batch and exits with the partial code

- **WHEN** a batch tree contains a valid scan and a scan that raises (e.g. a manifest naming a
  nonexistent `.slp`)
- **THEN** the driver writes the valid scan's `{scan_key}.result.json`, reports the failed scan,
  and exits `3` — without discarding the successful envelope

#### Scenario: Manifest without a matching sidecar

- **WHEN** a `{scan_key}.predictions.json` has no `{scan_key}.scan_metadata.json` in its directory
- **THEN** the pairing loop reports an error naming the missing sidecar for that scan and does not
  emit an envelope for it

#### Scenario: run_manifest.json absent falls back to unscoped discovery

- **WHEN** `input_dir` contains no `run_manifest.json` anywhere and the run has no identity
  (`pipeline_run_id` resolves to `None`)
- **THEN** `extract_batch` discovers scans via the same unscoped recursive glob as before this
  change, byte-identical in behavior and output to the pre-manifest implementation, unless zero
  scans are discovered (see "Empty, unscoped input directory is not a silent success" below)

#### Scenario: The CLI prints skipped scans, not just succeeded/failed

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs a second time over inputs
  that are unchanged since the first run
- **THEN** the CLI's output names the skipped scan(s) distinctly from `ok`/`FAIL` lines, and the
  summary counts include a skipped count, and the process exits `0`

#### Scenario: Empty, unscoped input directory is not a silent success

- **WHEN** `extract_batch` runs with no run identity over an existing `input_dir` with no
  `run_manifest.json` and zero `{scan_key}.predictions.json` files anywhere under it
- **THEN** `extract_batch` raises `RuntimeError` naming `input_dir` before writing anything to
  `output_dir`
- **AND** `python -m trait_extractor <input_dir> <output_dir>` exits `1`

#### Scenario: A nonexistent input directory is a crash naming the directory

- **WHEN** `extract_batch` runs over an `input_dir` that does not exist
- **THEN** it raises `FileNotFoundError` whose message names `input_dir` before writing anything
  to `output_dir`

#### Scenario: A manifest-scoped scan_key with no matching file is already a reported failure

- **WHEN** a run manifest declares a `scan_key` for which no `{scan_key}.predictions.json`
  exists anywhere under `input_dir`
- **THEN** that `scan_key` is recorded in `BatchResult.failed` and the process exits `3` (this
  scoped case is unaffected by the unscoped empty-input guard, since `BatchResult.ok` is
  already `False`)

#### Scenario: A CLI usage error is unrelated to the partial/crash codes

- **WHEN** `python -m trait_extractor` is invoked with a missing required argument
- **THEN** the process exits `2` via `argparse`'s own pre-existing usage-error handling, before
  `extract_batch` ever runs, and this is unrelated to (does not collide in meaning with) the
  `3` = partial convention

#### Scenario: A run-manifest resolution failure is a logged crash

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs with `ARGO_WORKFLOW_NAME` set
  and `input_dir` holds neither `run_manifest.<ARGO_WORKFLOW_NAME>.json` nor `run_manifest.json`
- **THEN** the process exits `1` and its stderr contains a `Batch aborted:` line that itself names
  the run id (the logged line, not merely the traceback that follows it)
- **AND** the same holds (exit `1`, a `Batch aborted:` line) when the per-run manifest names a
  different run, when `ARGO_WORKFLOW_NAME` is unusable as a filename component (e.g. `../x`), and
  when `input_dir` does not exist

#### Scenario: SIGTERM during a run terminates promptly and leaves completed output intact

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` receives `SIGTERM` while processing
  a batch of more than one scan
- **THEN** the process exits promptly with code `143`, without waiting out an external termination
  grace period
- **AND** any `{scan_key}.result.json` already durably written before the signal remains intact and
  unmodified

### Requirement: Run-manifest scoped discovery and copy-forward

At the top of a batch run, `extract_batch` SHALL resolve its run identity and load the run manifest
**exactly once**, and SHALL use that single snapshot both to scope discovery and to forward the
manifest — it SHALL NOT re-open the manifest by name to forward it.

**Run identity.** `extract_batch` SHALL accept a keyword `pipeline_run_id`; when the caller omits
it, the identity SHALL be resolved by `sleap_roots_contracts.pipeline_run_id_from_env()` (the
stripped `ARGO_WORKFLOW_NAME`, or `None` when unset or blank) — the function the cross-repo design
requires the manifest writer to use as well — never a local re-implementation of the environment
read. An explicit `None` SHALL mean
"no run identity".

**Resolution.** The manifest SHALL be loaded by
`sleap_roots_contracts.load_run_manifest(input_dir, pipeline_run_id, allow_legacy=True)` (top level
of `input_dir` only, never recursive). Its resolution rule is owned by the contracts package; for
this call site it means:

- **With a run identity:** `run_manifest.<pipeline_run_id>.json` first, then the legacy
  `RUN_MANIFEST_FILENAME` (`"run_manifest.json"`) because `allow_legacy=True`. If neither exists,
  `RunManifestMissingError` SHALL propagate out of `extract_batch` (a crash, exit `1`) — it SHALL
  NOT fall back to unscoped discovery. A per-run-named manifest whose `pipeline_run_id` differs
  from the run identity SHALL raise `RunManifestIdentityError` (crash). An identity unusable as a
  filename component SHALL raise `ValueError` (crash). When the legacy name was read and its
  `pipeline_run_id` differs from the run identity, `extract_batch` SHALL log a `WARNING` naming the
  filename read, the manifest's `pipeline_run_id`, and the run identity, and SHALL still honor its
  `scan_keys` (the legacy fallback is explicitly allowed during the rollout;
  `talmolab/sleap-roots-pipeline#82` removes it).
- **Without a run identity:** `RUN_MANIFEST_FILENAME` is the correct name (not a fallback) and is
  the only candidate; if it is absent, discovery SHALL fall back to the unscoped recursive glob
  described in "Batch driver and module CLI" — today's exact pre-manifest behavior, unchanged.
  Per-run manifests (`run_manifest.*.json`) are never read without an identity; if any are present
  at the top level of `input_dir` when this fallback is taken, `extract_batch` SHALL log a
  `WARNING` naming them before discovering, so the widening is visible rather than silent.

Only a genuinely absent candidate advances resolution; any other failure to read a candidate —
`PermissionError`, a dangling symlink, a missing `input_dir` — SHALL propagate as an `OSError`
(crash) rather than be treated as absent.

Scenarios below that name `run_manifest.json` without stating a run identity assume
`pipeline_run_id` is `None`.

**When a manifest is loaded**, it SHALL be parsed and validated as a `RunManifest`
(`RunManifest.model_validate_json` over the bytes read); a present-but-invalid manifest (e.g.
empty `scan_keys` failing `RunManifest`'s own pydantic validation, or bytes that are not valid
UTF-8/JSON) SHALL raise `pydantic.ValidationError` before any scan is
processed — this is a once-per-batch, top-level file, not a per-scan best-effort read, and MUST NOT
be treated the same as a per-scan corrupt-output fallback. When valid, discovery SHALL be scoped to
exactly its `scan_keys`: each `scan_key` SHALL be resolved to its `{scan_key}.predictions.json`
under `input_dir` using the same recursive search as the unscoped path, and two candidate files
resolving to the same in-scope `scan_key` SHALL be treated as a duplicate-scan_key collision
exactly as the existing unscoped-duplicate guard already does (reported as a per-scan failure, not
silently resolved by picking either candidate) — this prevents the scoped path from reintroducing
the "right scan_key, stale/wrong file" contamination case even as it fixes the "wrong scan_key
entirely" case. Two candidates whose scan_key stems differ ONLY by case SHALL also be treated as a
collision and reported as a per-scan failure, since they would write to the same output filename
on a case-insensitive filesystem (the default on Windows and macOS) despite being distinct strings
that an exact-match duplicate check alone would never catch. A `*.predictions.json` present under
`input_dir` for a `scan_key` NOT in `scan_keys` SHALL be silently excluded from `BatchResult`
entirely (not processed, not reported in `succeeded`, `skipped`, or `failed`) — that exclusion is
precisely the contamination-prevention this requirement exists for. A manifest-declared `scan_key`
with no matching `{scan_key}.predictions.json` SHALL be recorded as a per-scan failure in
`BatchResult.failed` naming the missing file, without aborting the rest of the batch. A
pre-existing `{scan_key}.result.json` in `output_dir` whose `scan_key` is NOT in the current run's
`scan_keys` (e.g. left over from a prior run whose manifest was wider) SHALL be left completely
untouched — not reprocessed, not reported in any `BatchResult` bucket, not deleted — but SHALL be
logged as a warning naming it, so its presence is at least traceable rather than silently
indistinguishable from a current result.

**Copy-forward.** After processing, the loaded manifest SHALL be republished into `output_dir`
(creating `output_dir` if missing) **under the filename it was read from** (`read.filename` —
per-run or legacy), from the **bytes already read** (`read.data`, byte-identical to the file scoped
against; never a re-serialization through the `RunManifest` model), with the source's permission
bits (`read.mode`), so `write-back` can see it downstream. The publish SHALL be atomic: the bytes
are written to a uniquely-named, dot-prefixed temporary file inside `output_dir`, its mode set to
`read.mode` **before** it is moved into place with `os.replace`. On any failure after the
temporary file is created — including a `BaseException` such as the `SystemExit(143)` raised by
the `SIGTERM` handler — that temporary file SHALL be unlinked before the exception propagates out
of the forward; if the unlink itself fails, a `WARNING` naming the temporary path SHALL be logged
and the original exception SHALL propagate unmasked. (A process killed by `SIGKILL` mid-forward
may still leave a dot-prefixed orphan.) When `(input_dir / read.filename).resolve() ==
(output_dir / read.filename).resolve()` — path identity after symlink resolution, not inode
identity — the copy-forward SHALL be a no-op. This copy-forward is **best-effort infrastructure
for the next pipeline stage, not part of this batch's own computed result**: an `OSError` during
the copy (e.g. a disk/permission error, or a temporary-file name exceeding `NAME_MAX`) SHALL be
logged as a warning naming the filename read and both directories, and MUST NOT raise, crash the
batch, or discard the already-computed `succeeded`/`skipped`/`failed` results.

#### Scenario: run_manifest.json present scopes discovery to scan_keys

- **WHEN** `run_manifest.json` is present at the top level of `input_dir` with `scan_keys` equal
  to exactly the fixture tree's two scan keys, and the run has no identity
- **THEN** both scans are processed and the resulting `.result.json` files are byte-for-byte
  identical to the no-manifest happy-path output

#### Scenario: A per-run manifest scopes discovery for a run with that identity

- **WHEN** `extract_batch` runs with `pipeline_run_id="wf-a"` and `input_dir` holds
  `run_manifest.wf-a.json` (`pipeline_run_id="wf-a"`, `scan_keys` naming one fixture scan) and a
  legacy `run_manifest.json` naming both scans
- **THEN** only the scan named in `run_manifest.wf-a.json` is processed — the per-run file wins
  over the legacy one

#### Scenario: Concurrent runs sharing a directory are each scoped to their own manifest

- **WHEN** `input_dir` holds `run_manifest.wf-a.json` naming `scan0K9E8BI` and
  `run_manifest.wf-b.json` naming `scanYR39SJX`, and `extract_batch` runs once with
  `pipeline_run_id="wf-a"` and once with `pipeline_run_id="wf-b"`
- **THEN** each run processes only its own scan, and each run's `output_dir` receives only its own
  per-run manifest

#### Scenario: A known run identity with no manifest aborts instead of widening scope

- **WHEN** `extract_batch` runs with `pipeline_run_id="wf-a"` over the fixture tree, which holds
  neither `run_manifest.wf-a.json` nor `run_manifest.json`
- **THEN** it raises `sleap_roots_contracts.RunManifestMissingError` before processing any scan,
  and no `*.result.json` is written

#### Scenario: A per-run manifest naming a different run aborts the batch

- **WHEN** `extract_batch` runs with `pipeline_run_id="wf-a"` and `run_manifest.wf-a.json`
  contains `pipeline_run_id="wf-b"`
- **THEN** it raises `sleap_roots_contracts.RunManifestIdentityError` before processing any scan

#### Scenario: An unusable run identity aborts the batch

- **WHEN** `extract_batch` runs with `pipeline_run_id="../x"`
- **THEN** it raises `ValueError` (not a `pydantic.ValidationError`) before processing any scan,
  and no `*.result.json` is written

#### Scenario: Per-run manifests are ignored without a run identity, but logged

- **WHEN** `extract_batch` runs with `pipeline_run_id=None` over the fixture tree whose top level
  holds only `run_manifest.wf-a.json` (naming one scan) and no `run_manifest.json`
- **THEN** discovery is unscoped (both fixture scans are processed), and a `WARNING` naming
  `run_manifest.wf-a.json` is logged

#### Scenario: A legacy manifest naming another run is honored but logged

- **WHEN** `extract_batch` runs with `pipeline_run_id="wf-a"`, no `run_manifest.wf-a.json`
  exists, and `run_manifest.json` contains `pipeline_run_id="wf-old"`
- **THEN** discovery is scoped to that manifest's `scan_keys`, and a `WARNING` naming
  `run_manifest.json`, `wf-old`, and `wf-a` is logged
- **AND** when the legacy manifest's `pipeline_run_id` equals the run identity, no such warning is
  logged

#### Scenario: Run identity defaults to the environment

- **WHEN** `extract_batch` is called without a `pipeline_run_id` argument and `ARGO_WORKFLOW_NAME`
  is set to `wf-a` in the environment
- **THEN** it resolves `run_manifest.wf-a.json` exactly as if `pipeline_run_id="wf-a"` had been
  passed, and when `ARGO_WORKFLOW_NAME` is unset or blank it behaves as `pipeline_run_id=None`

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

- **WHEN** the run manifest resolved for this run (legacy or per-run name) is present under
  `input_dir` but fails to parse or validate as a `RunManifest` (e.g. `scan_keys` is empty, or its
  bytes are not valid UTF-8/JSON)
- **THEN** `extract_batch` raises `pydantic.ValidationError` before processing any scan, rather
  than silently falling back to unscoped discovery, to another candidate name, or partially
  succeeding

#### Scenario: A dangling-symlink manifest aborts rather than reading as absent

- **WHEN** `run_manifest.json` at the top of `input_dir` is a symlink whose target does not exist,
  and the run has no identity
- **THEN** `extract_batch` raises `FileNotFoundError` naming the link before processing any scan,
  rather than falling back to unscoped discovery

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

#### Scenario: A per-run manifest is forwarded under its own name

- **WHEN** `extract_batch` runs with `pipeline_run_id="wf-a"` and loads `run_manifest.wf-a.json`
- **THEN** `output_dir/run_manifest.wf-a.json` exists, byte-identical to the source, and no
  `output_dir/run_manifest.json` is created

#### Scenario: The forwarded bytes are the bytes scoped against

- **WHEN** `input_dir` and `output_dir` are distinct and the source manifest file is rewritten with different content after `extract_batch` has
  loaded it but before the copy-forward runs
- **THEN** `output_dir`'s forwarded manifest holds the originally loaded bytes, not the rewritten
  ones

#### Scenario: The forwarded manifest keeps the source's permission bits

- **WHEN** (POSIX only) the source manifest has mode `0644` and is forwarded
- **THEN** the forwarded file has mode `0644`, not `mkstemp`'s `0600`

#### Scenario: A failed forward leaves no temporary file behind

- **WHEN** the publish step fails after the temporary file was created (e.g. `os.replace` of the
  manifest raises `OSError`) during a batch whose scans all succeed
- **THEN** `output_dir` (empty before the batch) contains exactly the scans' `*.result.json`
  files — no temporary file and no forwarded manifest — a warning naming both directories is logged, and `extract_batch` still
  returns its computed `BatchResult` with every scan in `succeeded`

#### Scenario: A SIGTERM during the forward leaves no temporary file behind

- **WHEN** the forward is interrupted after the temporary file was created by a `SystemExit(143)`
  (the `SIGTERM` handler's exit)
- **THEN** the temporary file is removed and the `SystemExit` propagates unchanged (it is not
  swallowed by the best-effort `OSError` handling)

#### Scenario: A temp-cleanup failure does not mask the forward error

- **WHEN** the publish step fails with one `OSError` and unlinking the temporary file then fails
  with another
- **THEN** the original publish error is the one that propagates out of the forward, and a
  `WARNING` naming the temporary path is logged

#### Scenario: Concurrent runs sharing an output directory keep each other's manifests

- **WHEN** runs `wf-a` and `wf-b` run in turn over the same `input_dir` and the same `output_dir`,
  each with its own per-run manifest naming a different scan
- **THEN** `output_dir` holds both `run_manifest.wf-a.json` and `run_manifest.wf-b.json`, each
  byte-identical to its source, and neither run's `succeeded` contains the other run's scan

#### Scenario: input_dir and output_dir resolving to the same file does not crash the batch

- **WHEN** a run manifest is present and `input_dir` and `output_dir` are the same directory
  (or differently spelled paths resolving to it)
- **THEN** the copy-forward is a no-op (the file is already in place) and `extract_batch` completes
  normally, without raising

#### Scenario: A copy-forward failure does not discard the batch's computed result

- **WHEN** copying the run manifest forward raises an `OSError` (e.g. a disk or permission
  error)
- **THEN** a warning naming `input_dir`, `output_dir`, and the error is logged, and
  `extract_batch` still returns the `BatchResult` already computed by the per-scan loop above,
  rather than raising
