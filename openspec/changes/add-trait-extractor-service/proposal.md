## Why

The A4 Bloom-integration DAG needs a predict→traits handoff: nothing today turns
`sleap-roots-predict`'s per-scan output into the `sleap-roots-contracts` `ResultEnvelope` that
Bloom's `insert_cyl_result_envelope` write-back RPC ingests. This adds that emitter (porting the
legacy `salk-tm/sleap-roots-traits`). Design:
`docs/superpowers/specs/2026-07-06-a3-traits-result-envelope-emitter-design.md`; tracking issue
talmolab/sleap-roots#250.

## What Changes

- Add a **top-level `trait_extractor/` package** (repo root), deliberately **excluded from the
  PyPI wheel** so `pip install sleap-roots` stays pure; `sleap-roots-contracts` becomes a
  **dev/test/container** dependency only.
- Consume predict's `PredictionManifest` via a **consumer-side pydantic model** (importing
  `ModelRef`+`RootType` from contracts; no `sleap-roots-predict` runtime dependency; pins
  `schema_version` `Literal["1"]`); resolve each artifact's `.slp` relative to the manifest and
  load a `Series` via `Series.load`.
- Define + consume a per-scan **`ScanMetadata` sidecar** (`scan_key`, `image_ids`,
  `images_checksum`, `params={species,mode,age}`) supplying the idempotency inputs predict defers.
  For reproducibility, `ResolvedParams.values` is the **closed set `{species, mode, age}` with
  `age` canonicalized to `int`**, built once and fed to both selection and provenance (so
  `age:3`/`3.0`/`"3"` and extra sidecar keys never change the idempotency key).
- Port pipeline selection as **`choose_pipeline`** modeled on predict's `choose_models`/`ModelCard`
  (authored in-tree — `choose_models` is not importable): a packaged `pipeline_selection.yaml` of
  `PipelineCard`s `{species, mode, age_min, age_max, pipeline_class}`. Resolves the legacy
  `MultipleDicotPlatePipeline` gap (it is now selectable).
- Enforce **pipeline compatibility** in the orchestrator, in order: (1) a **grain guard** rejects
  multi-plant/plate pipelines (they yield empty/count-only scan-grain output via base
  `compute_batch_traits`), then (2) a **subset** check via a class-keyed `PIPELINE_REQUIRED_ROOTS`
  map (pinned to reality by an `ast`-based guard test) that the pipeline's required roots are a
  subset of the loaded root types (`required ⊆ loaded`).
- Compute **scan-grain** traits (`compute_batch_traits(...).iloc[0]`) and map a flat
  `{trait}_{stat}` dict to `list[TraitValue]` (grain=`scan`, NaN/inf→None).
- Assemble `Provenance` (deterministic `idempotency_key` via `compute_idempotency_key`;
  `contract_version` = bare `"0.1.0a3"` from the pinned package) and emit a per-scan
  `ResultEnvelope` as `{scan_key}.result.json` with `blobs=[]`.
- Add a `python -m trait_extractor <input_dir> <output_dir>` batch driver (legacy `main` analog)
  with per-scan **failure isolation** (one scan's error does not discard the others' envelopes).
- Add **`sleap-roots-contracts==0.1.0a3`** as a **dev/test** dependency carrying a
  `; python_version >= '3.11'` marker (contracts requires Python `>=3.11`; `sleap-roots` keeps its
  `>=3.10` floor). Regenerate and commit `uv.lock` (CI is `uv sync --frozen`).
- Wire CI: extend `ci.yml` path filters (both `pull_request` and `push`) + lint/test over
  `trait_extractor/`; set `pytest` `pythonpath = ["."]`. The wheel/sdist already exclude a
  top-level sibling via `include = ["sleap_roots"]`; add `trait_extractor*` to `exclude` as
  belt-and-suspenders. `trait_extractor` is kept **flat** (no subpackages) so the existing
  `pydocstyle match-dir` override lints every module (no `match-dir` change needed).
- Document the change: a `docs/changelog.md` `[Unreleased]` entry and a durable human-facing
  README/mkdocs description of the service + the `ScanMetadata` sidecar schema.

## Impact

- **Affected specs:** new capability `result-envelope-output`.
- **Affected code:** new flat top-level `trait_extractor/` package + `tests/trait_extractor/` +
  new JSON manifest/sidecar fixtures under `tests/data/rice_3do_pipeline_output/`; `pyproject.toml`
  (marked dev dep, `pytest` pythonpath, packaging `exclude`); `uv.lock`;
  `.github/workflows/ci.yml`; `docs/changelog.md`, `docs/dev/trait-extractor-service.md` + mkdocs
  nav; the brainstorming design doc's reconciliation appendix.
- **New dependency:** `sleap-roots-contracts==0.1.0a3` — dev/test/container only (marked
  `python_version >= '3.11'`), NOT a published runtime dependency of `sleap-roots`. It IS on
  public PyPI (pre-release), so `uv sync --frozen` resolves it once locked.
- **Cross-repo (out of scope, tracked):** the GHCR image/Dockerfile/Argo template (fast-follow);
  Bloom RPC must accept `0.1.0a3` — Salk-Harnessing-Plants-Initiative/bloom#393; future
  `PipelineCard` contract type — talmolab/sleap-roots-contracts#14.
