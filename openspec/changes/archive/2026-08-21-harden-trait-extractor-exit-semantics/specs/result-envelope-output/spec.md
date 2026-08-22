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
per-scan `except` lives ONLY in the batch loop (never in the manifest guards), and the failure is
reported. The CLI SHALL report skipped scans (see "Skip-if-done via idempotency-key comparison")
distinctly from succeeded and failed scans — a skipped scan MUST appear in the CLI's output, not
simply be omitted, so an operator watching container/Argo logs can distinguish "reused from a
prior run" from "never discovered."

The process SHALL exit with one of three driver-owned codes so an Argo-driven caller can
distinguish a fully-successful run from a partially-failed-but-completed run from a run that could
not proceed at all:
- `0` — every discovered scan succeeded or was skipped; `BatchResult.ok` is `True`.
- `3` — **partial**: the batch ran to completion but one or more scans failed inside the driver's
  own per-scan isolation boundary (`BatchResult.failed` is non-empty).
- `1` — **crash**: an exception escaped `extract_batch` entirely before it could return a
  `BatchResult` at all (e.g. an invalid `run_manifest.json`, or the input-discovery guard below).

Exit code `2` is deliberately NOT part of this convention: `argparse` already exits `2` on a CLI
usage error (missing/extra positional arguments), before `extract_batch` runs at all. Reusing `2`
for "partial" would make a CLI-invocation misconfiguration indistinguishable from a completed batch
with isolated scan failures.

When no `run_manifest.json` is present at the top level of `input_dir` (unscoped mode) and zero
`{scan_key}.predictions.json` files are discovered anywhere under it, `extract_batch` SHALL raise
rather than return a vacuous, all-succeeded `BatchResult` — an empty or misconfigured input mount
SHALL NOT be reported as a successful run that simply had nothing to do.

#### Scenario: Batch run emits one envelope per scan

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs over an input tree containing
  two per-scan directories (`scan0K9E8BI/`, `scanYR39SJX/`), each with its manifest + sidecar +
  `.slp`, and no `run_manifest.json`
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

- **WHEN** `input_dir` contains no `run_manifest.json` anywhere
- **THEN** `extract_batch` discovers scans via the same unscoped recursive glob as before this
  change, byte-identical in behavior and output to the pre-manifest implementation, unless zero
  scans are discovered (see "Empty, unscoped input directory is not a silent success" below)

#### Scenario: The CLI prints skipped scans, not just succeeded/failed

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs a second time over inputs
  that are unchanged since the first run
- **THEN** the CLI's output names the skipped scan(s) distinctly from `ok`/`FAIL` lines, and the
  summary counts include a skipped count, and the process exits `0`

#### Scenario: Empty, unscoped input directory is not a silent success

- **WHEN** `extract_batch` runs over an `input_dir` with no `run_manifest.json` and zero
  `{scan_key}.predictions.json` files anywhere under it
- **THEN** `extract_batch` raises before writing anything to `output_dir`
- **AND** `python -m trait_extractor <input_dir> <output_dir>` exits `1`

#### Scenario: A manifest-scoped scan_key with no matching file is already a reported failure

- **WHEN** a `run_manifest.json` declares a `scan_key` for which no `{scan_key}.predictions.json`
  exists anywhere under `input_dir`
- **THEN** that `scan_key` is recorded in `BatchResult.failed` and the process exits `3` (this
  scoped case is unaffected by the new unscoped empty-input guard, since `BatchResult.ok` is
  already `False`)

#### Scenario: A CLI usage error is unrelated to the partial/crash codes

- **WHEN** `python -m trait_extractor` is invoked with a missing required argument
- **THEN** the process exits `2` via `argparse`'s own pre-existing usage-error handling, before
  `extract_batch` ever runs, and this is unrelated to (does not collide in meaning with) the new
  `3` = partial convention

#### Scenario: SIGTERM during a run terminates promptly and leaves completed output intact

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` receives `SIGTERM` while processing
  a batch of more than one scan
- **THEN** the process exits promptly with code `143`, without waiting out an external termination
  grace period
- **AND** any `{scan_key}.result.json` already durably written before the signal remains intact and
  unmodified
