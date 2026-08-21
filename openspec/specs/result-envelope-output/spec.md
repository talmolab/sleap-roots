# result-envelope-output Specification

## Purpose
TBD - created by archiving change add-trait-extractor-service. Update Purpose after archive.
## Requirements
### Requirement: Prediction manifest consumption without a predict runtime dependency

The `trait_extractor` package SHALL read `sleap-roots-predict`'s per-scan
`{scan_key}.predictions.json` (`PredictionManifest`, `schema_version: "1"`) through a
consumer-side pydantic model that imports `ModelRef` and `RootType` from `sleap-roots-contracts`
(no other runtime dependency) and MUST NOT import `sleap-roots-predict` at runtime. The model
SHALL expose `schema_version`, `scan_key`, `plant_qr_code`, `artifacts` (a list of
`{root_type: RootType, model_id, model: ModelRef, slp_path, checksum, file_size}`),
`predict_inference_config`, `predict_output_params` (both free-form `dict[str, Any] | None`,
passed through by value with no coercion), `predict_code_sha`, and `predict_container_digest`.
`schema_version` SHALL be pinned (`Literal["1"]`) so an unrecognized future version raises rather
than being silently consumed under the old shape. Artifact `slp_path` values (basenames) SHALL be
resolved relative to the manifest file's directory.

#### Scenario: Manifest parses and slp paths resolve relative to the manifest

- **WHEN** `trait_extractor` loads a `{scan_key}.predictions.json` whose `artifacts` list names
  per-root `.slp` basenames that exist beside it
- **THEN** the consumer model validates, `artifacts` is a list each self-identifying via
  `root_type`, and each `slp_path` resolves to an existing file in the manifest's directory

#### Scenario: Unrecognized schema_version is rejected

- **WHEN** a manifest declares `schema_version` other than `"1"`
- **THEN** validation raises rather than consuming it under the `"1"` shape

#### Scenario: Malformed, schema-invalid, or unknown-root_type manifest is rejected

- **WHEN** a `{scan_key}.predictions.json` is not valid JSON, is missing a required field, or has
  an artifact `root_type` outside `RootType` (`primary`/`lateral`/`crown`)
- **THEN** loading raises a `pydantic.ValidationError` (or a JSON decode error) and no envelope is
  produced for that scan

#### Scenario: A manifest referencing a missing .slp file is rejected

- **WHEN** an artifact's `slp_path` basename does not exist beside the manifest
- **THEN** loading raises a `FileNotFoundError`-class error naming the missing artifact

#### Scenario: No sleap-roots-predict import at runtime

- **WHEN** the `trait_extractor` package and its manifest module are imported in an environment
  where `sleap-roots-predict` is not installed
- **THEN** the import succeeds and manifests can be read

#### Scenario: Consumer model accepts predict's real output when predict is importable

- **WHEN** `sleap_roots_predict.output_contract` is importable and produces a real manifest via
  its `model_dump_json`
- **THEN** the consumer `PredictionManifest` validates that JSON; when predict is not importable
  the cross-check test is skipped rather than failed

### Requirement: Scan-metadata sidecar with canonical params

The `trait_extractor` package SHALL define a frozen pydantic `ScanMetadata` model for a per-scan
`{scan_key}.scan_metadata.json` sidecar with typed fields `scan_key: str`, `image_ids: list[str]`,
`images_checksum: str`, and `params` (carrying `species`, `mode`, and `age`). The sidecar's
`scan_key` MUST equal the manifest's `scan_key`; a mismatch SHALL raise. To keep the idempotency
key reproducible, `ResolvedParams.values` SHALL be built once at sidecar load as the **closed set
exactly `{"species": str, "mode": str, "age": int}`** — `age` canonicalized to a Python `int`
(rejecting `bool` and non-whole values), `species`/`mode` as `str`, and any extra sidecar `params`
keys excluded from `values` (never hashed). This single canonical `ResolvedParams` feeds BOTH
pipeline selection and `Provenance.params`.

#### Scenario: Sidecar yields inputs and canonical params

- **WHEN** a `{scan_key}.scan_metadata.json` with `image_ids`, `images_checksum`, and
  `params={species, mode, age}` is read alongside a matching manifest
- **THEN** it yields an `InputRef` and a `ResolvedParams` whose `values` is exactly
  `{"species", "mode", "age"}` with `age` as an `int`

#### Scenario: age encoding does not change the idempotency key

- **WHEN** three sidecars for the same scan encode `age` as `3`, `3.0`, and `"3"` respectively
- **THEN** the assembled `Provenance.idempotency_key` is identical across all three

#### Scenario: extra sidecar params keys do not change the idempotency key

- **WHEN** two sidecars for the same scan differ only by an extra key in `params` (e.g. a
  `notes` field)
- **THEN** the assembled `Provenance.idempotency_key` is identical, because `values` is the closed
  `{species, mode, age}` set

#### Scenario: Scan-key mismatch is rejected

- **WHEN** the sidecar's `scan_key` differs from the manifest's `scan_key`
- **THEN** loading raises an error identifying the inconsistency

### Requirement: Series loading from manifest artifacts

The `trait_extractor` package SHALL construct a `sleap_roots.Series` via
`Series.load(series_name=scan_key, ...)`, passing each resolved artifact path under the keyword
matching its `root_type` (`primary_path`, `lateral_path`, `crown_path`), using real `.slp` files
(no mocks). Because `Series.load` tolerates missing paths silently (it prints and returns `None`
labels rather than raising), an empty `artifacts` list SHALL be rejected by an explicit guard
**before** `Series.load` is called, raising a `ValueError` naming the scan.

#### Scenario: A rice pipeline-output scan loads primary and crown labels

- **WHEN** a manifest for a rice scan lists `root_type` `primary` and `crown` artifacts pointing
  at real `.slp` files
- **THEN** `Series.load` returns a `Series` whose `primary_labels` and `crown_labels` are loaded
  and whose `series_name` is the `scan_key`

#### Scenario: Empty artifact list is rejected before loading

- **WHEN** a manifest's `artifacts` list is empty (predict resolved zero roots)
- **THEN** the pre-load guard raises a `ValueError` naming the scan, rather than loading an
  all-`None` `Series`

### Requirement: Pipeline selection by species, mode, and age

The `trait_extractor` package SHALL select exactly one `sleap_roots` `Pipeline` subclass via
`choose_pipeline(params, cards, override=None)`, modeled on predict's `choose_models` (which is
NOT importable here — the matcher and age-coercion are authored in-tree). `cards` is an injectable
list of a public, test-constructible `PipelineCard` type (`{species, mode, age_min, age_max,
pipeline_class}`); production cards load from a packaged `pipeline_selection.yaml`. A card matches
when `species` and `mode` are equal and `age_min <= age <= age_max` (contiguous inclusive window),
reading `age` from the already-canonicalized `params.values` (an `int`). `choose_pipeline` MUST
NOT mutate `params.values`. An explicit `override` pipeline-class name wins and bypasses matching.
Zero matches, more than one match, or an unknown `pipeline_class` name SHALL each raise a
`ValueError`.

#### Scenario: Rice age selects younger vs older monocot by window

- **WHEN** `choose_pipeline` is given canonical `params.values = {species: "rice", mode:
  "cylinder", age: 3}` and again `age: 8` against the packaged cards
- **THEN** it returns `YoungerMonocotPipeline` for age 3 and `OlderMonocotPipeline` for age 8, and
  `params.values` is unchanged by the call

#### Scenario: Arabidopsis plate resolves the plate pipeline (legacy gap fixed)

- **WHEN** `choose_pipeline` is given `{species: "arabidopsis", mode: "plate", age: 10}`
- **THEN** it returns `MultipleDicotPlatePipeline` (the class the legacy chooser table named but
  could not resolve) — selection resolves the class; scan-grain support is guarded separately

#### Scenario: Explicit override wins

- **WHEN** an `override` of `"DicotPipeline"` is supplied
- **THEN** `choose_pipeline` returns `DicotPipeline` regardless of species/mode/age

#### Scenario: Ambiguous, unmatched, or invalid selection raises

- **WHEN** no card matches, more than one card matches, or a matched/override `pipeline_class`
  name is not a known pipeline
- **THEN** `choose_pipeline` raises `ValueError`

### Requirement: Pipeline compatibility and scan-grain support

Before computing traits, the orchestrator SHALL, in this order: (1) **reject unsupported grain** —
multi-plant / plate pipelines (`MultipleDicotPipeline`, `MultipleDicotPlatePipeline`,
`MultiplePrimaryRootPipeline`) route through the base `compute_batch_traits` single-plant path and
therefore yield **empty or count-only** scan-grain output (their per-plant traits are
`include_in_csv=False`; per-plant expansion lives only in `compute_multiple_dicots_traits` /
`compute_multiple_primary_roots_traits`, which the scan-grain path never calls) — selecting one for
emission SHALL raise a clear "not supported for scan-grain emission" error; then (2) **check root
compatibility** — the pipeline's required root types MUST be a **subset** of the loaded root types
(`required ⊆ loaded`, so a crown-only pipeline accepts a primary+crown scan), else raise naming the
pipeline (`cls.__name__`) and the missing root type. Required root types come from a single
class-keyed map `PIPELINE_REQUIRED_ROOTS: dict[type, frozenset[str]]` (in the `compatibility`
module, alongside the guards that consume it); a pipeline absent from BOTH the map and the
reject-list SHALL raise a clear "not registered for scan-grain emission" error (never a bare
`KeyError`). A guard test SHALL pin each map entry to the
root-type getters that pipeline's `get_initial_frame_traits` actually calls (derived independently
via `inspect.getsource` + `ast`) and assert the map and reject-list together partition every class
`choose_pipeline` can return.

#### Scenario: Grain guard runs first and is deterministic

- **WHEN** a multi-plant / plate pipeline (e.g. `MultipleDicotPlatePipeline`) is selected for
  emission — even if its required roots would not match the loaded set
- **THEN** the orchestrator raises the "not supported for scan-grain emission" error (the grain
  guard short-circuits before any `PIPELINE_REQUIRED_ROOTS` lookup)

#### Scenario: Crown-only pipeline accepts a superset manifest (subset semantics)

- **WHEN** `OlderMonocotPipeline` (requires `crown` only) is selected for a manifest that loaded
  `primary` and `crown`
- **THEN** the compatibility check passes (`required ⊆ loaded`)

#### Scenario: Missing required root type raises

- **WHEN** a pipeline requiring `primary` + `crown` is selected for a manifest that loaded only
  `primary` + `lateral`
- **THEN** the check raises an error naming the pipeline (`cls.__name__`) and the missing `crown`
  root type

### Requirement: Scan-grain trait values

The `trait_extractor` package SHALL compute scan-grain traits by calling
`pipeline_cls().compute_batch_traits([series])` on the single loaded `Series`, taking the resulting
one-row DataFrame (first column `plant_name`) as a flat `{trait_name: value}` mapping
(`df.iloc[0].to_dict()`), and mapping it to a `list[TraitValue]` with `grain="scan"` and `scan_key`
equal to the manifest `scan_key`. The identity column (`plant_name`) MUST NOT become a
`TraitValue`. Non-finite values (`NaN`/`inf`) SHALL become `None`.

#### Scenario: Batch summary becomes scan-grain TraitValues

- **WHEN** the selected pipeline produces a one-row `{trait}_{stat}` summary for the scan
- **THEN** each summary column (excluding `plant_name`) yields one `TraitValue(name=column,
  grain="scan", scan_key=scan_key)` with a finite float or `None`

#### Scenario: Non-finite trait coerces to None

- **WHEN** a computed trait value is `NaN` or `inf`
- **THEN** the corresponding `TraitValue.value` is `None`

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

### Requirement: Per-scan ResultEnvelope emission

The `trait_extractor` package SHALL emit exactly one `ResultEnvelope` per scan
(`provenance` + `traits` + `blobs=[]`), creating the output directory if missing
(`parents=True, exist_ok=True`) and writing `{manifest.scan_key}.result.json` **atomically** (temp
file then `replace`, so a mid-write crash never leaves a partial file the batch counts as success)
using `pathlib.Path` with POSIX path strings. The written JSON SHALL round-trip back into a valid
`ResultEnvelope`, and re-emitting over identical inputs SHALL produce a **byte-identical** file. The
envelope MUST satisfy the write-back RPC's locally-checkable acceptance rules: a non-empty
`idempotency_key` and a single `scan_key` used consistently across `provenance` and every
`TraitValue`. No `BlobRef` is emitted in this capability.

#### Scenario: Emitted envelope validates and round-trips

- **WHEN** an end-to-end extraction runs for a real pipeline-output scan
- **THEN** a `ResultEnvelope` validates against `sleap-roots-contracts`, `blobs == []`, it is
  written to `{scan_key}.result.json`, and reading that file reconstructs an equal envelope

#### Scenario: Re-emission is byte-stable

- **WHEN** `extract_scan` runs twice over the same manifest + sidecar
- **THEN** the two `{scan_key}.result.json` files are byte-identical (`produced_at` is `None`, not
  a fresh timestamp)

#### Scenario: Envelope satisfies locally-checkable RPC acceptance rules

- **WHEN** the emitted envelope is inspected
- **THEN** `provenance.idempotency_key` is non-empty and the same `scan_key` appears in
  `provenance` and in every `TraitValue`

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

### Requirement: Package boundary excludes the extractor from the published wheel

The `trait_extractor` package SHALL live at the repository root as a **flat** package (no
subpackages) outside the published `sleap_roots` package, so it is not shipped in the `sleap-roots`
PyPI wheel or sdist (guaranteed by `[tool.setuptools.packages.find] include = ["sleap_roots"]`, an
allowlist that never discovers a top-level sibling), and importing `sleap_roots` SHALL NOT require
`sleap-roots-contracts`. `sleap-roots-contracts` SHALL be declared only as a development/test (and
future container) dependency, carrying a `; python_version >= '3.11'` marker.

#### Scenario: Importing the library does not require contracts

- **WHEN** `sleap_roots` is imported in an environment without `sleap-roots-contracts` installed
- **THEN** the import succeeds

#### Scenario: The library source never imports contracts (CI-enforced)

- **WHEN** a CI-runnable guard test AST-scans the `sleap_roots/` source tree for imports of
  `sleap_roots_contracts`
- **THEN** it finds none — so the boundary is caught by `pytest` even in a dev environment that
  has contracts installed

#### Scenario: The wheel and sdist do not contain the extractor

- **WHEN** the `sleap-roots` wheel and sdist are built
- **THEN** each contains the `sleap_roots` package and neither contains the top-level
  `trait_extractor` package

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

