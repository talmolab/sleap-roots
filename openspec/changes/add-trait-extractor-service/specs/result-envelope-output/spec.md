## ADDED Requirements

### Requirement: Prediction manifest consumption without a predict runtime dependency

The `trait_extractor` package SHALL read `sleap-roots-predict`'s per-scan
`{scan_key}.predictions.json` (`PredictionManifest`, `schema_version: "1"`) through a
consumer-side pydantic model that imports only `ModelRef` from `sleap-roots-contracts` and MUST
NOT import `sleap-roots-predict` at runtime. The model SHALL expose `scan_key`, `plant_qr_code`,
`artifacts` (a list of `{root_type, model_id, model: ModelRef, slp_path, checksum, file_size}`),
`predict_inference_config`, `predict_output_params`, `predict_code_sha`, and
`predict_container_digest`. Artifact `slp_path` values (basenames) SHALL be resolved relative to
the manifest file's directory.

#### Scenario: Manifest parses and slp paths resolve relative to the manifest

- **WHEN** `trait_extractor` loads a `{scan_key}.predictions.json` whose `artifacts` list names
  per-root `.slp` basenames that exist beside it
- **THEN** the consumer model validates, `artifacts` is a list each self-identifying via
  `root_type`, and each `slp_path` resolves to an existing file in the manifest's directory

#### Scenario: Malformed or schema-invalid manifest is rejected

- **WHEN** a `{scan_key}.predictions.json` is not valid JSON, or is valid JSON that fails the
  consumer model's validation (e.g. a missing required field or an unknown `root_type`)
- **THEN** loading raises a `pydantic.ValidationError` (or a JSON decode error) and no envelope
  is produced for that scan

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

### Requirement: Scan-metadata sidecar supplies idempotency inputs

The `trait_extractor` package SHALL read a per-scan `{scan_key}.scan_metadata.json` sidecar
(`ScanMetadata`) providing `scan_key`, `image_ids` (list of Bloom `cyl_images` IDs),
`images_checksum`, and `params` (a mapping containing at least `species`, `mode`, and `age`).
The sidecar's `scan_key` MUST equal the manifest's `scan_key`; a mismatch SHALL raise an error.

#### Scenario: Sidecar loads and provides inputs and params

- **WHEN** a `{scan_key}.scan_metadata.json` with `image_ids`, `images_checksum`, and
  `params={species, mode, age}` is read alongside a matching manifest
- **THEN** the sidecar validates and yields `InputRef` inputs and `ResolvedParams` params for the
  same `scan_key`

#### Scenario: Scan-key mismatch is rejected

- **WHEN** the sidecar's `scan_key` differs from the manifest's `scan_key`
- **THEN** loading raises an error identifying the inconsistency

### Requirement: Series loading from manifest artifacts

The `trait_extractor` package SHALL construct a `sleap_roots.Series` via
`Series.load(series_name=scan_key, ...)`, passing each resolved artifact path under the keyword
matching its `root_type` (`primary_path`, `lateral_path`, `crown_path`), using real `.slp`
files (no mocks). Because `Series.load` tolerates missing paths silently (it prints and returns
`None` labels rather than raising), an empty `artifacts` list SHALL be rejected by an explicit
guard **before** `Series.load` is called, raising a `ValueError` rather than producing an
all-`None` `Series`.

#### Scenario: A rice pipeline-output scan loads primary and crown labels

- **WHEN** a manifest for a rice scan lists `root_type` `primary` and `crown` artifacts pointing
  at real `.slp` files
- **THEN** `Series.load` returns a `Series` whose `primary_labels` and `crown_labels` are loaded
  and whose `series_name` is the `scan_key`

#### Scenario: Empty artifact list is rejected before loading

- **WHEN** a manifest's `artifacts` list is empty (predict resolved zero roots)
- **THEN** the pre-load guard raises a `ValueError` naming the scan, rather than loading an
  all-`None` `Series` or emitting an envelope with no traits

### Requirement: Pipeline selection by species, mode, and age

The `trait_extractor` package SHALL select exactly one `sleap_roots` `Pipeline` subclass via
`choose_pipeline(params, cards, override=None)`, modeled on predict's `choose_models` (which is
NOT importable here — the matcher and age-coercion are authored in-tree). `cards` is an
injectable list of a public, test-constructible `PipelineCard` type (`{species, mode, age_min,
age_max, pipeline_class}`); production cards load from a packaged `pipeline_selection.yaml`. A
card matches when `species` and `mode` are equal and `age_min <= age <= age_max` (contiguous
inclusive window). `age` MUST be whole-number-coercible (rejecting `bool` and non-whole floats).
An explicit `override` pipeline-class name wins and bypasses matching. Zero matches, more than
one match, or an unknown `pipeline_class` name SHALL each raise a `ValueError`.

#### Scenario: Rice age selects younger vs older monocot by window

- **WHEN** `choose_pipeline` is given `params.values = {species: "rice", mode: "cylinder", age:
  3}` and again `age: 8` against the packaged cards
- **THEN** it returns `YoungerMonocotPipeline` for age 3 and `OlderMonocotPipeline` for age 8

#### Scenario: Arabidopsis plate selects the plate pipeline (legacy gap fixed)

- **WHEN** `choose_pipeline` is given `{species: "arabidopsis", mode: "plate", age: 10}`
- **THEN** it returns `MultipleDicotPlatePipeline` (the class the legacy chooser table named but
  could not resolve)

#### Scenario: Explicit override wins

- **WHEN** an `override` of `"DicotPipeline"` is supplied
- **THEN** `choose_pipeline` returns `DicotPipeline` regardless of species/mode/age

#### Scenario: Ambiguous, unmatched, or invalid selection raises

- **WHEN** no card matches, more than one card matches, or a matched/override `pipeline_class`
  name is not a known pipeline
- **THEN** `choose_pipeline` raises `ValueError`

### Requirement: Pipeline compatibility and scan-grain support

Before computing traits, the orchestrator SHALL verify — given the loaded `Series` and the
selected pipeline class — that (a) the pipeline's required root types are a **subset** of the
loaded root types, and (b) the selected pipeline is supported for scan-grain emission. Required
root types SHALL be sourced from a single class-keyed map (`PIPELINE_REQUIRED_ROOTS: dict[type,
frozenset[str]]`) kept next to `choose_pipeline`, whose entries are pinned to each pipeline's
`get_initial_frame_traits` by a guard test. A missing required root type SHALL raise an error
naming the pipeline and the missing type. Multi-plant / plate pipelines (`MultipleDicotPipeline`,
`MultipleDicotPlatePipeline`, `MultiplePrimaryRootPipeline`) emit **per-plant** rows and are NOT
valid for one-row scan-grain emission in this capability; selecting one for emission SHALL raise
a clear "not supported for scan-grain emission" error (they remain resolvable by
`choose_pipeline` for a future per-plant grain).

#### Scenario: Crown-only pipeline accepts a superset manifest (subset semantics)

- **WHEN** `OlderMonocotPipeline` (requires `crown` only) is selected for a manifest that loaded
  `primary` and `crown`
- **THEN** the compatibility check passes (required `⊆` loaded), because extra loaded root types
  are permitted

#### Scenario: Missing required root type raises

- **WHEN** a pipeline requiring `primary` + `crown` is selected for a manifest that loaded only
  `primary` + `lateral`
- **THEN** the check raises an error naming the pipeline and the missing `crown` root type

#### Scenario: Multi-plant pipeline is rejected for scan-grain emission

- **WHEN** a multi-plant / plate pipeline (e.g. `MultipleDicotPlatePipeline`) is selected and
  emission is attempted
- **THEN** the orchestrator raises a clear "not supported for scan-grain emission" error rather
  than silently keeping only the first plant's row

### Requirement: Scan-grain trait values

The `trait_extractor` package SHALL compute scan-grain traits by calling the selected pipeline's
`compute_batch_traits` on the single loaded `Series`, taking the resulting one-row DataFrame as a
flat `{trait_name: value}` mapping (`df.iloc[0].to_dict()`), and mapping it to a
`list[TraitValue]` with `grain="scan"` and `scan_key` equal to the manifest `scan_key`. The
identity column (`plant_name`) MUST NOT become a `TraitValue`. Non-finite values (`NaN`/`inf`)
SHALL become `None`.

#### Scenario: Batch summary becomes scan-grain TraitValues

- **WHEN** the selected pipeline produces a one-row `{trait}_{stat}` summary for the scan
- **THEN** each summary column (excluding `plant_name`) yields one `TraitValue(name=column,
  grain="scan", scan_key=scan_key)` with a finite float or `None`

#### Scenario: Non-finite trait coerces to None

- **WHEN** a computed trait value is `NaN` or `inf`
- **THEN** the corresponding `TraitValue.value` is `None`

### Requirement: Provenance assembly with deterministic idempotency key

The `trait_extractor` package SHALL assemble a `Provenance` with: `contract_version` set to the
bare pinned `sleap-roots-contracts` package version (`"0.1.0a3"`); `scan_key` from the manifest;
`inputs` from the sidecar (`image_ids`, `images_checksum`); `predict_models = [artifact.model
for artifact in manifest.artifacts]`; `predict_container_digest`, `predict_code_sha`,
`predict_inference_config`, and `predict_output_params` from the manifest;
`traits_sleap_roots_version` from `sleap_roots.__version__`; `params` as `ResolvedParams` built
from the sidecar `params`; and `traits_code_sha` / `traits_container_digest` resolved fail-soft
from arguments, then environment, then `""`. The resulting `idempotency_key` SHALL equal
`compute_idempotency_key(...)` for the same inputs and be identical across repeated runs.

#### Scenario: Idempotency key is deterministic and matches the contract helper

- **WHEN** a `Provenance` is assembled for a scan and `compute_idempotency_key` is called with
  the same `scan_key`, `images_checksum`, models, `param_hash`, `predict_code_sha`,
  `traits_code_sha`, and `predict_output_params`
- **THEN** `Provenance.idempotency_key` is non-empty, equals the helper's result, and re-running
  the assembly for the same inputs produces the identical key

#### Scenario: contract_version is the pinned bare package version

- **WHEN** a `Provenance` is assembled
- **THEN** `Provenance.contract_version` equals
  `importlib.metadata.version("sleap-roots-contracts")` (so it tracks the pinned dependency), AND
  additionally asserts the literal byte `== "0.1.0a3"` and `not .startswith("v")` — so a silent
  pin bump fails the test and forces conscious Bloom-RPC coordination (bloom#393)

#### Scenario: Build identity resolves fail-soft

- **WHEN** neither `traits_code_sha`/`traits_container_digest` arguments nor their environment
  variables are set
- **THEN** those fields are `""` and assembly does not raise

### Requirement: Per-scan ResultEnvelope emission

The `trait_extractor` package SHALL emit exactly one `ResultEnvelope` per scan
(`provenance` + `traits` + `blobs=[]`), write it to `{scan_key}.result.json` under the output
directory using `pathlib.Path` with POSIX path strings (written atomically — temp file then
`replace` — so a mid-write crash never leaves a partial file the batch would count as success),
and the written JSON SHALL round-trip back into a valid `ResultEnvelope`. The envelope MUST satisfy the write-back RPC's acceptance
rules that are checkable without a live database: a non-empty `idempotency_key` and a single
`scan_key` used consistently across `provenance`, every `TraitValue`, and (when present) every
`BlobRef`. No `BlobRef` is emitted in this capability (blob locations are filled downstream at
upload).

#### Scenario: Emitted envelope validates and round-trips

- **WHEN** an end-to-end extraction runs for a real pipeline-output scan
- **THEN** a `ResultEnvelope` validates against `sleap-roots-contracts`, `blobs == []`, it is
  written to `{scan_key}.result.json`, and reading that file reconstructs an equal envelope

#### Scenario: Envelope satisfies locally-checkable RPC acceptance rules

- **WHEN** the emitted envelope is inspected
- **THEN** `provenance.idempotency_key` is non-empty and the same `scan_key` appears in
  `provenance` and in every `TraitValue`

### Requirement: Batch driver and module CLI

The `trait_extractor` package SHALL provide a callable entry `python -m trait_extractor
<input_dir> <output_dir>` that discovers each `{scan_key}.predictions.json` in `input_dir`,
pairs it with its `{scan_key}.scan_metadata.json` sidecar, and writes one
`{scan_key}.result.json` per scan to `output_dir` — the legacy `main(input_dir, output_dir)`
analog. A manifest with no matching sidecar SHALL be reported as an error for that scan. One
scan's failure SHALL NOT discard the successfully-emitted envelopes of the other scans.

#### Scenario: Batch run emits one envelope per scan

- **WHEN** `python -m trait_extractor <input_dir> <output_dir>` runs over an input directory
  containing two manifest + sidecar pairs (`scan0K9E8BI`, `scanYR39SJX`)
- **THEN** exactly one `{scan_key}.result.json` is written per scan under `output_dir`

#### Scenario: One scan's failure does not abort the batch

- **WHEN** a batch input directory contains multiple manifest + sidecar pairs and extraction of
  one scan raises
- **THEN** the driver writes `{scan_key}.result.json` for each of the other scans and reports the
  failed scan (a non-zero exit and/or logged per-scan error) without discarding the successful
  envelopes

#### Scenario: Manifest without a matching sidecar

- **WHEN** a `{scan_key}.predictions.json` has no `{scan_key}.scan_metadata.json` beside it
- **THEN** the driver reports an error naming the missing sidecar for that scan and does not emit
  an envelope for it

### Requirement: Package boundary excludes the extractor from the published wheel

The `trait_extractor` package SHALL live at the repository root outside the published
`sleap_roots` package so it is not shipped in the `sleap-roots` PyPI wheel, and importing
`sleap_roots` SHALL NOT require `sleap-roots-contracts`. `sleap-roots-contracts` SHALL be
declared only as a development/test (and future container) dependency.

#### Scenario: Importing the library does not require contracts

- **WHEN** `sleap_roots` is imported in an environment without `sleap-roots-contracts` installed
- **THEN** the import succeeds

#### Scenario: The library source never imports contracts (CI-enforced)

- **WHEN** a CI-runnable guard test scans the `sleap_roots/` source tree for
  `import sleap_roots_contracts` / `from sleap_roots_contracts`
- **THEN** it finds none — so the boundary is caught by `pytest` even in a dev environment that
  has contracts installed

#### Scenario: The wheel and sdist do not contain the extractor

- **WHEN** the `sleap-roots` wheel and sdist are built
- **THEN** each contains the `sleap_roots` package and neither contains the top-level
  `trait_extractor` package
