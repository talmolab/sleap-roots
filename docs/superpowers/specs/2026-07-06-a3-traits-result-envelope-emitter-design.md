# A3-traits ResultEnvelope emitter — design

- **Date:** 2026-07-06
- **Feature branch:** `add-traits-extractor-service`
- **Tracking issues:** talmolab/sleap-roots#250 (feature) · talmolab/sleap-roots-contracts#14 (`PipelineCard` + `contract_version`) · Salk-Harnessing-Plants-Initiative/bloom#393 (RPC re-pin)
- **Ports/redesigns:** `salk-tm/sleap-roots-traits` (legacy `main(input_dir, output_dir)` + `pipeline_chooser`)

## 1. Purpose

Consume `sleap-roots-predict`'s per-scan output contract, compute root traits with the existing
`sleap-roots` pipelines, and emit a per-scan `sleap-roots-contracts` **`ResultEnvelope`**
(Provenance + `list[TraitValue]` + blobs) to disk as JSON — the input Bloom's
`insert_cyl_result_envelope` write-back RPC will accept. This is the predict→traits handoff
in the A4 roadmap.

## 2. Context (what already exists)

- **Input** — predict ships `{scan_key}.predictions.json` (`PredictionManifest`) + named per-root
  `.slp`. `artifacts` is a **JSON array** of `PredictionArtifact` `{root_type, model_id,
  model: ModelRef, slp_path (basename), checksum, file_size}`; manifest also carries
  `predict_inference_config`, `predict_output_params`, `predict_code_sha`,
  `predict_container_digest`. Deliberately **defers** `inputs` (image_ids/images_checksum),
  `params` (ResolvedParams), `idempotency_key`, and `contract_version` to traits/upstream.
- **Trait engine** — `Series.load(series_name, primary_path=, lateral_path=, crown_path=,
  csv_path=)`; `*Pipeline` classes; scan-grain summary `compute_batch_traits([series]) →`
  1-row wide DataFrame with `{trait}_{stat}` columns (matches committed
  `tests/data/**/*.batch_traits.csv` goldens). `MultipleDicotPipeline` also exposes a native
  flat `compute_multiple_dicots_traits(...)["summary_stats"]` dict.
- **Contract types** (`sleap-roots-contracts==0.1.0a3`) — `ResultEnvelope{provenance, traits,
  blobs=[]}`; `Provenance` (free-`str` `contract_version`; `inputs: InputRef{image_ids,
  images_checksum}`; `params: ResolvedParams{values, param_hash}`; `predict_models:
  list[ModelRef]`; predict_* + traits_* build fields; `idempotency_key` auto-derived by a
  validator calling `compute_idempotency_key`); `TraitValue{name, value(float|None,
  NaN/inf→None), grain(Literal["scan","image"]="scan"), scan_key}`; `BlobRef` (validator
  rejects a location-less object). Model-selection metadata already lives here as `ModelCard`
  (`species, mode, age_min, age_max, root_type`), matched by predict's `choose_models`.
- **Absent in sleap-roots today** — no contracts dep, no envelope emitter, no trait-extraction
  CLI/driver, no Dockerfile/GHCR workflow. All net-new.

## 3. Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| Hosting | Top-level `trait_extractor/` package, **excluded from the PyPI wheel**. `sleap-roots-contracts` is a **dev/test/container** dep, never a published-library runtime dep. | The extractor is a *service* that uses the library, not part of its public API. Keeps `pip install sleap-roots` pure. Mirrors the legacy `PYTHONPATH=/workspace` container layout. |
| Manifest consumption | **Consumer-side** pydantic model of `PredictionManifest`/`PredictionArtifact` in `trait_extractor`, importing only `ModelRef` from contracts. No `sleap-roots-predict` runtime dep. | predict pulls the GPU/torch/sleap-nn stack; too heavy for a traits consumer. Guard drift with a skip-if-unimportable cross-check test against predict's real model. |
| Idempotency inputs | Define + consume a per-scan **`ScanMetadata` sidecar** (`image_ids`, `images_checksum`, `params={species,mode,age}`). | Neither `inputs` nor `params` is in predict's manifest; both feed `idempotency_key`. The sidecar is the only path to a self-consistent, TDD-testable envelope. Downloader/Bloom populates it in production. |
| Pipeline selection | Port `pipeline_chooser` but **mirror `choose_models`/`ModelCard`**: packaged `pipeline_selection.yaml` cards `{species, mode, age_min, age_max, pipeline_class}` + `choose_pipeline(params: ResolvedParams, cards, override)`. Contiguous inclusive age window; override wins; ambiguous/no-match → `ValueError`. Fix legacy `MultipleDicotPlatePipeline` gap. | One `ResolvedParams` (from the sidecar) drives selection **and** `param_hash`. Consistent with the established model-selection convention. Future: promote to a `PipelineCard` contract type (contracts#14). |
| `contract_version` | Stamp **bare `"0.1.0a3"`**, sourced from the pinned package version (`importlib.metadata.version("sleap-roots-contracts")`). | Forward-consistent with the package predict pins. **Requires** the Bloom RPC to accept `0.1.0a3` (bloom#393) — recorded as an A4 blocker. |
| Trait grain | **Scan-grain only** this slice. Map from a flat `{trait_name: value}` dict (1-row `compute_batch_traits`; native `summary_stats` for MultipleDicot as a future adapter). | Matches legacy `traits_summary.csv` + the RPC's needs. Image-grain (`compute_plant_traits`) is a noted follow-up. |
| Blobs | Emit `blobs=[]`. | predict writes only local paths + checksums; s3/box locations are filled downstream at upload (A4 flow step G, out of scope), and `BlobRef`'s validator rejects a location-less object. |
| Container/CI | Dockerfile + GHCR workflow = **fast-follow slice**. This PR: `python -m trait_extractor` CLI + `ci.yml` test wiring + contracts dev-dep. | Keeps the emitter slice focused with real TDD; matches the brief's scoping and predict's split. |

## 4. Architecture — `trait_extractor/` (repo root, not in wheel)

| Module | Responsibility |
|--------|----------------|
| `manifest.py` | Consumer-side `PredictionManifest` + `PredictionArtifact` models (read predict's `schema_version:"1"` JSON); `ScanMetadata` sidecar model; loaders that resolve each artifact's `slp_path` **relative to the manifest's directory** and bin by `root_type`. |
| `pipeline_chooser.py` | `PipelineCard` (local) + `load_pipeline_cards()` (reads packaged `pipeline_selection.yaml`) + `choose_pipeline(params, cards, override=None)` mirroring `choose_models`. Maps `pipeline_class` name → `sleap_roots` `Pipeline` subclass. |
| `envelope.py` | `traits_dict → list[TraitValue]` (grain=`scan`, NaN/inf→None); `build_provenance(...)`; `build_envelope(...) → ResultEnvelope`; `write_envelope(envelope, output_dir) → {scan_key}.result.json`. |
| `extractor.py` | Orchestrator `extract_scan(manifest_path, scan_metadata_path, output_dir, *, traits_code_sha="", traits_container_digest="") → ResultEnvelope`; `extract_batch(input_dir, output_dir, ...)` (legacy `main` analog). |
| `__main__.py` | Thin CLI: `python -m trait_extractor <input_dir> <output_dir>` → `extract_batch`. The container entry command. |
| `pipeline_selection.yaml` | Declarative selection cards. |

Build-identity fields (`traits_code_sha`, `traits_container_digest`) resolve **fail-soft**
from args → env (`SRT_TRAITS_CODE_SHA` / `SRT_TRAITS_CONTAINER_DIGEST`) → `""`, mirroring
predict's `_resolve_identity`.

## 5. Data contracts

### `ScanMetadata` sidecar (new — `{scan_key}.scan_metadata.json`)
```
scan_key: str            # must equal manifest.scan_key (consistency check)
image_ids: list[str]     # Bloom cyl_images IDs (downloader-supplied)
images_checksum: str
params: dict             # {"species": ..., "mode": ..., "age": ...}; → ResolvedParams.values
```

### `Provenance` field mapping
| Provenance field | Source |
|------------------|--------|
| `contract_version` | bare `"0.1.0a3"` from pinned package version |
| `scan_key` | `manifest.scan_key` |
| `inputs` | `InputRef(image_ids=sidecar.image_ids, images_checksum=sidecar.images_checksum)` |
| `predict_models` | `[a.model for a in manifest.artifacts]` |
| `predict_container_digest`, `predict_code_sha`, `predict_inference_config`, `predict_output_params` | from manifest |
| `traits_sleap_roots_version` | `sleap_roots.__version__` |
| `traits_container_digest`, `traits_code_sha` | fail-soft arg/env/`""` |
| `params` | `ResolvedParams(values=sidecar.params)` |
| `produced_at` | `datetime.now(UTC)` at emit |
| `idempotency_key` | auto-derived by the `Provenance` validator (`compute_idempotency_key`) |

### `TraitValue` mapping
For each `{trait}_{stat}` column of the 1-row `compute_batch_traits` result (excluding
`plant_name`): `TraitValue(name=col, value=float(v), grain="scan", scan_key=scan_key)`; the
model coerces NaN/inf → None.

## 6. Testing (real TDD, no mocks)

Fixtures: real `.slp` pipeline-output under `tests/data/rice_3do_pipeline_output/`
(`scan0K9E8BI` + `scanYR39SJX`, each primary+crown → `YoungerMonocotPipeline`), plus authored
`{scan_key}.predictions.json` manifest and `{scan_key}.scan_metadata.json` sidecar fixtures.

Test-first, each behavior before its implementation:
1. **Manifest+sidecar loader** — parse fixtures, resolve `slp_path` relative to manifest dir,
   real `Series.load` succeeds (labels loaded); `scan_key` consistency enforced.
2. **`choose_pipeline`** — rice/cylinder/3 → `YoungerMonocotPipeline`; rice/cylinder/8 →
   `OlderMonocotPipeline`; canola/cylinder/7 → `DicotPipeline`; arabidopsis/plate/10 →
   `MultipleDicotPlatePipeline` (legacy gap fixed); explicit override wins; invalid class →
   `ValueError`; no-match/ambiguous → `ValueError`; non-whole `age` → `ValueError`.
3. **Envelope emission (end-to-end)** — `scan0K9E8BI` → `ResultEnvelope` validates (pydantic);
   TraitValues finite-or-None + grain=`scan` + consistent `scan_key`; `idempotency_key` equals
   `compute_idempotency_key(...)` and is deterministic; `contract_version == "0.1.0a3"`;
   `blobs == []`; JSON round-trips; RPC acceptance rules (scan_key consistency, non-empty
   idempotency_key).
4. **NaN coercion** — a NaN trait → `TraitValue.value is None`.
5. **Golden regression** — computed batch-trait numbers match the committed `*.batch_traits.csv`
   within tolerance (commit a fixture-specific golden if the pipeline-output `.slp` diverge
   from the existing `rice_3do` golden).
6. **Manifest cross-check** (dev-only, skip-if-unimportable) — import
   `sleap_roots_predict.output_contract` if installed; assert the consumer `PredictionManifest`
   accepts predict's real `model_dump_json` output.

## 7. Container / CI organization (plan; Dockerfile is fast-follow)

- **Image identity** — distinct `ghcr.io/talmolab/sleap-roots-trait-extractor` (explicit
  `images:`; not `${{ github.repository }}`, which is the library's identity).
- **Dockerfile** (fast-follow) — root `trait-extractor.Dockerfile`, base
  `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`; `uv sync` sleap-roots + contracts, `COPY
  trait_extractor/`, `ENTRYPOINT ["python","-m","trait_extractor"]`.
- **CI, two independent concerns:**
  - *Tests* (**this PR**) — extend `ci.yml`: add `trait_extractor/**` to path filters + lint
    (`black`/`pydocstyle` over `trait_extractor`); add `sleap-roots-contracts` to
    `[dependency-groups] dev`; set `[tool.pytest.ini_options] pythonpath = ["."]`.
  - *Image build* (**fast-follow `docker-trait-extractor.yml`**) — near-copy of predict's
    `docker-build.yml`, path-filtered to `trait_extractor/**` + Dockerfile. `build.yml` (PyPI)
    untouched — library release and service image never trigger each other.
- **Extraction path** — the self-contained boundary lets the whole `trait_extractor/` + its
  Dockerfile/workflow move to `talmolab/sleap-roots-trait-extractor` later with minimal rework.

## 8. Risks / cross-repo dependencies

- **Bloom RPC re-pin (A4 blocker)** — emitting `0.1.0a3` is rejected by today's live RPC
  (expects `v0.1.0a2`) until bloom#393 lands. End-to-end write-back is blocked on it; the
  emitter + its tests are not.
- **Manifest schema drift** — consumer-side model duplicates predict's shape; mitigated by the
  cross-check test and predict's stable `schema_version:"1"`.
- **`image_ids` semantics** — the emitter faithfully passes sidecar `image_ids` through; that
  they resolve to a real `cyl_images.scan_id` is the downloader's/Bloom's responsibility (only
  checkable against live Bloom, out of scope).

## 9. Scope

- **IN** — the emitter (manifest+sidecar → select pipeline → compute traits → TraitValues +
  Provenance → per-scan `ResultEnvelope` JSON), the pipeline-chooser port, the `python -m
  trait_extractor` CLI, `ci.yml` test wiring + contracts dev-dep.
- **OUT (fast-follow)** — GHCR image / Dockerfile / `docker-trait-extractor.yml` / Argo
  template; MinIO/Box upload; the write-back RPC call; BlobRef locations; image-grain
  TraitValues.
