Each implementation task is **test-first**: write the failing test that pins the behavior, then
implement until green. Real fixtures only — no mocks. Drive real `.slp` through `Series.load`.

**Commit discipline (from review):** every commit MUST bundle test + implementation so the
committed tree is green — never commit a lone failing test. CI runs `uv sync --frozen`, so
`pyproject.toml` and `uv.lock` MUST stay byte-consistent in the same commit. `trait_extractor/` is
a **flat package** (no subpackages) so the pre-existing `[pydocstyle] match-dir = "sleap_roots"`
override never silently skips a module; author every module Black-clean and Google-docstring-
complete from the start (lint over `trait_extractor` is only wired in §10). Fold the §2 fixtures
into the §3 commit (their `Verify` step needs the §3 consumer model).

## 1. Scaffolding & dependencies
- [ ] 1.1 **Pre-flight (hard gate), explicit ordering:** FIRST edit BOTH dependency locations —
  add `"sleap-roots-contracts==0.1.0a3 ; python_version >= '3.11'"` to `[dependency-groups] dev`
  AND `[project.optional-dependencies] dev` — THEN run `uv lock`, THEN `uv sync --frozen`
  (verified: the marker makes `uv lock` resolve exit-0 despite `requires-python = ">=3.10"`;
  without it `uv lock` fails; `uv add --dev` only writes the PEP 735 group, so edit the pip extra
  by hand before locking). No `--prerelease` flag needed. Commit the regenerated `uv.lock`.
  **Verify:** `uv sync --frozen` green; `python -c "import sleap_roots"` works with contracts
  absent from the runtime deps.
- [ ] 1.2 Set `[tool.pytest.ini_options] pythonpath = ["."]`. Wheel/sdist exclusion of
  `trait_extractor` is guaranteed by the existing `include = ["sleap_roots"]`; add
  `trait_extractor*` to `exclude` as belt-and-suspenders (do NOT touch `match-dir`). **Verify:**
  `uv build`, then assert BOTH the wheel and the sdist omit `trait_extractor`.
- [ ] 1.3 Create the flat `trait_extractor/` package (`__init__.py`) and `tests/trait_extractor/`.
  **Test first:** `import trait_extractor` passes.
- [ ] 1.4 **Test first (CI-enforced boundary guard, AST-based):** a `pytest` test parses every
  `sleap_roots/**/*.py` with `ast`, walks `Import`/`ImportFrom`, and asserts no imported module
  name starts with `sleap_roots_contracts` (robust vs docstrings/strings, unlike grep). Then confirm.

## 2. Fixtures (authored, real — data only; committed with §3)
- [ ] 2.1 Author a **nested** per-scan layout under `tests/data/rice_3do_pipeline_output/` mirroring
  predict's `out_dir/{scan_key}/` batch output: `scan0K9E8BI/` and `scanYR39SJX/`, each containing
  its `{scan_key}.predictions.json` (field-ordered to predict's `model_dump_json`),
  `{scan_key}.scan_metadata.json` (species=rice, mode=cylinder, age=3), and its two `.slp`
  (relocate/copy the existing byte-identical `scan….model123.root{primary,crown}.slp` into the
  subdir so `slp_path` basenames resolve beside the manifest). Text sidecars/manifests (not LFS).
- [ ] 2.2 Golden source confirmed reusable: the pipeline-output `.slp` are **byte-identical**
  (matching sha256) to `rice_3do/0K9E8BI.*.predictions.slp`, so
  `tests/data/rice_3do/rice_3do.batch_traits.csv` reproduces the same numbers — **no new golden**
  (the golden-regression test lives in §8).

## 3. Manifest + sidecar consumption
- [ ] 3.1 **Test first:** consumer `PredictionManifest`/`PredictionArtifact` (importing `ModelRef`
  + `RootType` from contracts) parse the nested fixture; `artifacts` is a list; each `slp_path`
  resolves relative to the manifest dir; import works with `sleap-roots-predict` absent. Then
  implement `manifest.py`.
- [ ] 3.2 **Test first:** `schema_version` is pinned `Literal["1"]` — a manifest with
  `schema_version:"2"` raises; an unknown `root_type` raises `ValidationError`. Then confirm.
- [ ] 3.3 **Test first:** cross-check — if `sleap_roots_predict.output_contract` is importable, the
  consumer model accepts its real `model_dump_json`; else `pytest.skip`.
- [ ] 3.4 **Test first (negative paths; guards raise — no broad `except` in `manifest.py`):**
  malformed (non-JSON) manifest and one missing a required field each raise; a manifest naming a
  `.slp` basename absent from its directory raises a `FileNotFoundError`-class error naming it.
  Then implement.
- [ ] 3.5 **Test first (`ScanMetadata` model + canonical params — the reproducibility core):**
  define frozen `ScanMetadata` (`scan_key: str`, `image_ids: list[str]`, `images_checksum: str`,
  `params`); its `to_input_ref()` / `to_resolved_params()` build `ResolvedParams.values` as the
  **closed set `{"species": str, "mode": str, "age": int}`** (age coerced to `int`, rejecting
  `bool`/non-whole; extra `params` keys excluded). Assert: (a) sidecars with `age` `3`/`3.0`/`"3"`
  → the SAME `idempotency_key`; (b) a sidecar with an extra `params` key → the SAME key;
  (c) `values["age"]` is `int`; (d) sidecar `scan_key` ≠ manifest `scan_key` raises. Then implement.

## 4. Series loading
- [ ] 4.1 **Test first:** `Series.load(series_name=scan_key, primary_path=…, crown_path=…)` returns
  a `Series` with `primary_labels`/`crown_labels` loaded (real `.slp`). Then implement the loader
  mapping `root_type` → the matching `Series.load` keyword.
- [ ] 4.2 **Test first (empty-artifacts guard, extractor-owned, BEFORE `Series.load`):** because
  `Series.load` tolerates empty/missing paths silently (never raises), an explicit pre-load guard
  raises `ValueError` naming the scan on an empty `artifacts` list. Then implement.

## 5. Pipeline selection
- [ ] 5.1 **Test first:** a public, test-constructible `PipelineCard` type + `pipeline_selection.yaml`
  cards load; rice/cylinder/3 → `YoungerMonocotPipeline`, rice/cylinder/8 → `OlderMonocotPipeline`,
  canola/cylinder/7 → `DicotPipeline`, arabidopsis/plate/10 → `MultipleDicotPlatePipeline`
  (selection resolves the class). Then implement `choose_pipeline` + the YAML.
- [ ] 5.2 **Test first:** pass an in-memory `cards` list (DI — no YAML shadowing): override wins;
  no-match → `ValueError`; two overlapping windows → `ValueError`; unknown class → `ValueError`;
  reading `age` from the canonical `params.values` (already `int`); `choose_pipeline` does NOT
  mutate `params.values`. Then implement the in-tree matcher/coercion.

## 6. Pipeline compatibility & scan-grain support (needs BOTH §4 loader and §5 selection)
- [ ] 6.1 **Test first (grain guard FIRST):** selecting a multi-plant / plate pipeline
  (`MultipleDicotPipeline`, `MultipleDicotPlatePipeline`, `MultiplePrimaryRootPipeline`) for
  emission raises a clear "not supported for scan-grain emission" error — even when its required
  roots would not match the loaded set (short-circuits before any map lookup). Rationale is the
  TRUE one: base `compute_batch_traits` routes these through the single-plant path → **empty or
  count-only** traits (their per-plant traits are `include_in_csv=False`), NOT a dropped-row
  `df.iloc[0]` issue. Then implement the guard.
- [ ] 6.2 **Test first (subset compatibility):** `assert_pipeline_compatible(series, pipeline_cls)`
  with class-keyed `PIPELINE_REQUIRED_ROOTS: dict[type, frozenset[str]]` — subset semantics
  (`required ⊆ loaded`): `OlderMonocotPipeline` (crown-only) vs a primary+crown `Series` passes; a
  primary+crown pipeline vs a primary+lateral `Series` raises naming `cls.__name__` + missing
  `crown`; a pipeline in neither map nor reject-list raises a clear "not registered" error (not
  `KeyError`). Then implement (standalone helper; `extract_scan` calls guard-then-check in §8).
- [ ] 6.3 **Test first (guard test pins the map independently):** for every mapped pipeline,
  `PIPELINE_REQUIRED_ROOTS[cls]` equals the root-type getters its `get_initial_frame_traits`
  actually calls — derived via `inspect.getsource(cls.get_initial_frame_traits)` + `ast` (a
  different artifact than the map, so it catches drift); and the map ∪ reject-list **partition**
  every class `choose_pipeline` can return. Then confirm.

## 7. Trait → TraitValue mapping
- [ ] 7.1 **Test first:** `pipeline_cls().compute_batch_traits([series]).iloc[0].to_dict()` →
  `list[TraitValue]` with `grain="scan"`, correct `scan_key`, `plant_name` excluded. Then implement.
- [ ] 7.2 **Test first:** a `NaN`/`inf` trait → `TraitValue.value is None`. Then confirm.

## 8. Provenance + envelope emission (+ golden regression)
- [ ] 8.1 **Test first:** `Provenance` has `predict_models == [a.model for a in artifacts]`,
  predict_* from the manifest (`predict_output_params` by value — a `{"peak_threshold":0.2}`
  round-trips byte-equal), `params` = the canonical `ResolvedParams`; `contract_version` equals
  `importlib.metadata.version("sleap-roots-contracts")` AND the literal `== "0.1.0a3"` and
  `not .startswith("v")`; `produced_at is None` and orchestration fields (`pipeline_run_id`,
  `worker_request_id`, `argo_*`) are `None`. Then implement `build_provenance`.
- [ ] 8.2 **Test first:** `idempotency_key` non-empty, equals `compute_idempotency_key(...)`, stable
  across repeated assembly; `traits_code_sha`/`traits_container_digest` fail-soft (arg → env → `""`).
  Then confirm.
- [ ] 8.3 **Test first:** end-to-end `extract_scan` (creates `output_dir` `parents=True,
  exist_ok=True`; calls the §6 grain-guard-then-compat check) → `ResultEnvelope` validates,
  `blobs == []`, writes `{scan_key}.result.json` **atomically** (temp + `replace`), round-trips to
  an equal envelope, re-running is **byte-identical**; non-empty `idempotency_key`; single
  `scan_key` across `provenance` and every `TraitValue`. Then implement `envelope.py` + `extract_scan`.
- [ ] 8.4 **Test first (golden regression):** for `scan0K9E8BI`, compare an explicit shared
  trait-column list (drop `plant_name`; mirror `test_younger_monocot_pipeline`'s `batch_trait_cols`)
  against `rice_3do/rice_3do.batch_traits.csv`, selecting the golden row via
  `scan_key.removeprefix("scan")` (→ `"0K9E8BI"`), `assert_frame_equal(check_exact=False,
  atol=1e-8)`. Then confirm.

## 9. Batch driver + CLI
- [ ] 9.1 **Test first:** `extract_batch(input_dir, output_dir)` **recursively** discovers
  `{scan_key}.predictions.json` (nested per-scan dirs), asserts each manifest's filename stem ==
  `manifest.scan_key`, pairs the co-located sidecar, resolves `.slp` in the manifest's dir, and
  writes one `{scan_key}.result.json` per scan to a separate `output_dir` (both fixtures). Then
  implement `extractor.py`.
- [ ] 9.2 **Test first (failure isolation — broad `except` ONLY here):** a batch with one valid
  scan (`scanYR39SJX`) and one deterministically-failing scan (`scan0K9E8BI` manifest naming a
  nonexistent `.slp`) still writes the valid scan's envelope, reports the failure, and exits
  non-zero; a stem≠`scan_key` disagreement and a manifest with no sidecar are each reported. Then
  implement pairing + isolation.
- [ ] 9.3 **Test first:** `python -m trait_extractor <input_dir> <output_dir>` (via
  `subprocess`/`runpy`) writes the envelopes. Then implement `__main__.py`.

## 10. CI wiring & verification
- [ ] 10.1 Extend `.github/workflows/ci.yml`: add `trait_extractor/**` to path filters in BOTH the
  `pull_request` and `push` `paths:` blocks; extend lint to `black --check sleap_roots tests
  trait_extractor` and `pydocstyle --convention=google sleap_roots trait_extractor` (flat package
  → every module linted; `match-dir` unchanged).
- [ ] 10.2 Decide coverage scope: `trait_extractor` is not counted under `--cov=sleap_roots` —
  either add `--cov=trait_extractor` or record the omission as intentional. **Verify:** reflected
  in `ci.yml`.
- [ ] 10.3 Run the full local gate mirroring CI: `uv run black --check sleap_roots tests
  trait_extractor`, `uv run pydocstyle --convention=google sleap_roots trait_extractor`,
  `uv run pytest tests/`, `uv sync --frozen` clean. **Verify:** all green; `uv.lock` committed.
- [ ] 10.4 Reconcile implementation against `proposal.md` / `spec.md` / `design.md` / this file;
  record any deviation with a `### Why N instead of M?` note before committing.

## 11. Documentation
- [ ] 11.1 Add a `docs/changelog.md` `[Unreleased]` entry (repo convention): under `### Added`, the
  `trait_extractor/` service, the `ScanMetadata` sidecar contract, and the `python -m
  trait_extractor` CLI; under `### Changed`/`### Internal`, the marked
  `sleap-roots-contracts==0.1.0a3` dev/test dependency, `pytest pythonpath`, packaging `exclude`,
  and the `ci.yml` wiring. Note the consumer-side `PredictionManifest` duplicates predict's schema
  (coupled via `schema_version:"1"`, guarded by the cross-check + pin).
- [ ] 11.2 Add a durable **`docs/dev/trait-extractor-service.md`** page (in `mkdocs.yml` under
  "Developer Guide") covering the service boundary, a **field-typed `ScanMetadata` schema** for the
  external Bloom/downloader populator (`scan_key: str`, `image_ids: list[str]`, `images_checksum:
  str`, `params: {species, mode, age}` — **`age` is canonicalized to an integer for the hash**;
  only `{species, mode, age}` feed the idempotency key), and `python -m trait_extractor` usage;
  add a short pointer section in README mirroring the `sleap-roots viewer` section.
- [ ] 11.3 Record that `trait_extractor` is intentionally excluded from the generated API reference
  (`docs/gen_ref_pages.py` walks only `sleap_roots/`).
- [ ] 11.4 Update the brainstorming design doc's reconciliation appendix if any decision changed
  during implementation (keep durable docs in sync).
