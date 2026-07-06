Each implementation task is **test-first**: write the failing test that pins the behavior, then
implement until green. Real fixtures only — no mocks. Drive real `.slp` through `Series.load`.

**Commit discipline (from review):** every commit MUST bundle test + implementation so the
committed tree is green — never commit a lone failing test (it would break `pytest` collection
in CI). CI runs `uv sync --frozen`, so `pyproject.toml` and `uv.lock` MUST stay byte-consistent
in the same commit. `trait_extractor/` is a **flat package** (no subpackages) — keep it flat so
the pre-existing `[pydocstyle] match-dir = "sleap_roots"` override (which only affects subdir
recursion) never silently skips a module; author every module Black-clean and Google-docstring-
complete from the start (lint over `trait_extractor` is only wired in §10).

## 1. Scaffolding & dependencies
- [ ] 1.1 **Pre-flight (hard gate), explicit ordering:** FIRST edit BOTH dependency locations —
  add `"sleap-roots-contracts==0.1.0a3 ; python_version >= '3.11'"` to `[dependency-groups] dev`
  AND `[project.optional-dependencies] dev` — THEN run `uv lock`, THEN `uv sync --frozen`
  (verified: the marker makes `uv lock` resolve exit-0 despite `requires-python = ">=3.10"`;
  without it `uv lock` fails; `uv add --dev` alone only writes the PEP 735 group, so edit the pip
  extra by hand before locking). The exact pre-release pin resolves from public PyPI without a
  `--prerelease` flag. Commit the regenerated `uv.lock`. **Verify:** `uv sync --frozen` green;
  `python -c "import sleap_roots"` still works with contracts absent from the runtime deps.
- [ ] 1.2 Set `[tool.pytest.ini_options] pythonpath = ["."]` so `tests/` can `import
  trait_extractor`. The wheel/sdist exclusion of `trait_extractor` is guaranteed by the existing
  `[tool.setuptools.packages.find] include = ["sleap_roots"]` (an allowlist — a sibling
  top-level package is never discovered); add `trait_extractor*` to the `exclude` list as
  belt-and-suspenders for a future switch to auto-discovery (do NOT touch `match-dir`). **Verify:**
  `uv build`, then assert BOTH the wheel (`unzip -l dist/*.whl`) and the sdist (`tar tzf
  dist/*.tar.gz`) omit `trait_extractor` (both hold via `include`).
- [ ] 1.3 Create the flat `trait_extractor/` package (`__init__.py`) and a `tests/trait_extractor/`
  module. **Test first:** a smoke test `import trait_extractor` passes.
- [ ] 1.4 **Test first (CI-enforced boundary guard, AST-based):** a `pytest` test parses every
  `sleap_roots/**/*.py` with `ast` and walks `Import`/`ImportFrom`, asserting no imported module
  name starts with `sleap_roots_contracts` (robust against docstrings/strings/comments, unlike a
  plain grep). Zero hits today, so it passes. Then confirm.

## 2. Fixtures (authored, real — data only, no compute here)
- [ ] 2.1 Author `{scan_key}.predictions.json` manifest + `{scan_key}.scan_metadata.json` sidecar
  fixtures for **both** `tests/data/rice_3do_pipeline_output/` scans — `scan0K9E8BI` and
  `scanYR39SJX` (primary + crown; species=rice, mode=cylinder, age=3) — so the batch path
  exercises N>1. Field-order the manifest to match predict's `PredictionManifest.model_dump_json`.
  Text fixtures (not Git-LFS). **Verify:** JSON parses with the consumer model (added in §3).
- [ ] 2.2 Golden source is confirmed reusable: the `rice_3do_pipeline_output/*.slp` are
  **byte-identical** (matching sha256) to `rice_3do/0K9E8BI.*.predictions.slp`, so
  `tests/data/rice_3do/rice_3do.batch_traits.csv` reproduces the same numbers — **no new golden
  file.** (The golden-regression *test* lives in §8, where the compute path first exists.)

## 3. Manifest + sidecar consumption
- [ ] 3.1 **Test first:** consumer `PredictionManifest`/`PredictionArtifact` parse the fixture;
  `artifacts` is a list; each `slp_path` resolves relative to the manifest dir; import works with
  `sleap-roots-predict` absent. Then implement `manifest.py`.
- [ ] 3.2 **Test first:** cross-check test — if `sleap_roots_predict.output_contract` is
  importable, the consumer model accepts its real `model_dump_json`; else `pytest.skip`.
- [ ] 3.3 **Test first (negative paths, guards raise — no broad `except` inside `manifest.py`):**
  a malformed (non-JSON) manifest, one missing a required field, and one with an unknown
  `root_type` each raise a decode/`pydantic.ValidationError`; a manifest naming a `.slp` basename
  absent from its directory raises a `FileNotFoundError`-class error identifying the missing
  artifact. Then implement the guards.
- [ ] 3.4 **Test first:** `ScanMetadata` sidecar validates and yields `InputRef` + `ResolvedParams`;
  a `scan_key` mismatch vs the manifest raises. Then implement.

## 4. Series loading
- [ ] 4.1 **Test first:** `Series.load(series_name=scan_key, primary_path=…, crown_path=…)` from
  the manifest returns a `Series` with `primary_labels` and `crown_labels` loaded (real `.slp`).
  Then implement the loader mapping `root_type` → the matching `Series.load` keyword.
- [ ] 4.2 **Test first (empty-artifacts guard, extractor-owned):** because `Series.load` tolerates
  empty/missing paths silently (prints, returns `None` labels — never raises), an explicit guard
  BEFORE `Series.load` raises a `ValueError` naming the scan on an empty `artifacts` list. Then
  implement the pre-load guard. (Root-type/pipeline compatibility is NOT here — it needs the
  selected pipeline; see §6.)

## 5. Pipeline selection
- [ ] 5.1 **Test first:** a public, test-constructible `PipelineCard` type + `pipeline_selection.yaml`
  cards load; rice/cylinder/3 → `YoungerMonocotPipeline`, rice/cylinder/8 → `OlderMonocotPipeline`,
  canola/cylinder/7 → `DicotPipeline`, arabidopsis/plate/10 → `MultipleDicotPlatePipeline`
  (selection resolves the class; scan-grain support is guarded separately in §6). Then implement
  `choose_pipeline` + the YAML.
- [ ] 5.2 **Test first:** pass an in-memory `cards` list (dependency injection — no YAML shadowing):
  explicit override wins; no-match → `ValueError`; two overlapping windows for the same
  species/mode make a matching age raise (`>1` match → `ValueError`); unknown class name →
  `ValueError`; non-whole / `bool` `age` → `ValueError` (age-coercion authored in-tree, modeled on
  predict's `choose_models`). Then implement the guards.

## 6. Pipeline compatibility & scan-grain support (needs BOTH §4 loader and §5 selection)
- [ ] 6.1 **Test first:** a class-keyed `PIPELINE_REQUIRED_ROOTS: dict[type, frozenset[str]]`
  constant + `assert_pipeline_compatible(series, pipeline_cls)` helper — subset semantics
  (`required ⊆ loaded`): `OlderMonocotPipeline` (crown-only) against a primary+crown `Series`
  passes; a primary+crown pipeline against a primary+lateral `Series` raises naming the pipeline
  and missing `crown`. Then implement the helper (standalone unit — construct a mismatched
  `Series` + a pipeline class; `extract_scan` calls it in §8).
- [ ] 6.2 **Test first (guard-test pins the map to reality):** for every mapped pipeline,
  `PIPELINE_REQUIRED_ROOTS[cls]` matches the root-type point-getters its `get_initial_frame_traits`
  actually calls — so the map cannot silently drift from the pipeline. Then confirm.
- [ ] 6.3 **Test first (grain guard):** selecting a multi-plant / plate pipeline
  (`MultipleDicotPipeline`, `MultipleDicotPlatePipeline`, `MultiplePrimaryRootPipeline`) for
  emission raises a clear "not supported for scan-grain emission" error (they emit per-plant rows;
  `df.iloc[0]` would silently drop plants). Then implement the guard.

## 7. Trait → TraitValue mapping
- [ ] 7.1 **Test first:** the flat `{trait}_{stat}` mapping from
  `compute_batch_traits(...).iloc[0].to_dict()` → `list[TraitValue]` with `grain="scan"`, correct
  `scan_key`, `plant_name` excluded. Then implement the mapper.
- [ ] 7.2 **Test first:** a `NaN`/`inf` trait → `TraitValue.value is None`. Then confirm coercion
  (contract model normalizes; guard input casting to `float`).

## 8. Provenance + envelope emission (+ golden regression)
- [ ] 8.1 **Test first:** assembled `Provenance` has `predict_models == [a.model for a in
  artifacts]`, predict_* from the manifest, `params` from the sidecar; `contract_version` equals
  `importlib.metadata.version("sleap-roots-contracts")` AND the literal `== "0.1.0a3"` and
  `not .startswith("v")` (so a silent pin bump fails, forcing bloom#393 coordination). Then
  implement `build_provenance`.
- [ ] 8.2 **Test first:** `idempotency_key` is non-empty, equals `compute_idempotency_key(...)`
  for the same inputs, and is stable across repeated assembly; `traits_code_sha`/
  `traits_container_digest` resolve fail-soft (arg → env → `""`). Then confirm.
- [ ] 8.3 **Test first:** end-to-end `extract_scan` (which calls the §6 compatibility + grain
  guards) → `ResultEnvelope` validates, `blobs == []`, writes `{scan_key}.result.json`
  atomically (temp + `replace`), round-trips to an equal envelope; non-empty `idempotency_key`;
  single `scan_key` across `provenance` and every `TraitValue` (the model has no cross-field
  scan_key validator, so `extract_scan` enforces it). Then implement `envelope.py` + `extract_scan`.
- [ ] 8.4 **Test first (golden regression):** for `scan0K9E8BI`, compare an explicit shared
  trait-column list (drop `plant_name`; mirror the existing `test_younger_monocot_pipeline`
  `batch_trait_cols` approach) against `tests/data/rice_3do/rice_3do.batch_traits.csv`, selecting
  the golden row via `scan_key.removeprefix("scan")` (→ `"0K9E8BI"`) and using
  `assert_frame_equal(..., check_exact=False, atol=1e-8)` for cross-OS float stability. Then confirm.

## 9. Batch driver + CLI
- [ ] 9.1 **Test first:** `extract_batch(input_dir, output_dir)` discovers manifest+sidecar pairs
  and writes one `{scan_key}.result.json` per scan (both fixtures). Then implement `extractor.py`.
- [ ] 9.2 **Test first (failure isolation — broad `except` lives ONLY here):** a batch with one
  valid scan (`scanYR39SJX`) and one deterministically-failing scan (a `scan0K9E8BI` manifest
  naming a nonexistent `.slp`, raising via the §3.3 guard) still writes the valid scan's envelope
  and reports the failure with a non-zero per-batch exit; a manifest with no matching sidecar is
  reported+skipped in the pairing loop (before `extract_scan`). `extract_scan` raises; only
  `extract_batch` catches. Then implement pairing + isolation.
- [ ] 9.3 **Test first:** invoking `python -m trait_extractor <input_dir> <output_dir>` (via
  `subprocess`/`runpy`) writes the envelopes. Then implement `__main__.py`.

## 10. CI wiring & verification
- [ ] 10.1 Extend `.github/workflows/ci.yml`: add `trait_extractor/**` to the path filters in
  BOTH the `pull_request` and `push` `paths:` blocks; extend the lint job to `black --check
  sleap_roots tests trait_extractor` and `pydocstyle --convention=google sleap_roots
  trait_extractor` (`trait_extractor` is flat, so every module is linted; `match-dir` unchanged).
- [ ] 10.2 Decide coverage scope: `trait_extractor` is not counted under the existing
  `--cov=sleap_roots`; either add `--cov=trait_extractor` to the test job or record the omission
  as intentional. **Verify:** the chosen behavior is reflected in `ci.yml`.
- [ ] 10.3 Run the full local gate mirroring CI: `uv run black --check sleap_roots tests
  trait_extractor`, `uv run pydocstyle --convention=google sleap_roots trait_extractor`,
  `uv run pytest tests/`, and `uv sync --frozen` clean. **Verify:** all green; `uv.lock` committed.
- [ ] 10.4 Reconcile implementation against `proposal.md` / `spec.md` / `design.md` / this file;
  record any deviation with a `### Why N instead of M?` note before committing.

## 11. Documentation
- [ ] 11.1 Add a `docs/changelog.md` `[Unreleased]` entry (repo convention — every PR logs here):
  under `### Added`, the `trait_extractor/` service, the `{scan_key}.scan_metadata.json`
  `ScanMetadata` sidecar contract, and the `python -m trait_extractor <input_dir> <output_dir>`
  CLI; under `### Changed`/`### Internal`, the marked `sleap-roots-contracts==0.1.0a3` dev/test
  dependency, `pytest pythonpath`, the packaging `exclude`, and the `ci.yml` wiring. Note the
  consumer-side `PredictionManifest` duplicates predict's schema (coupled via `schema_version:
  "1"`, guarded by the cross-check test).
- [ ] 11.2 Add a durable **`docs/dev/trait-extractor-service.md`** page (registered in `mkdocs.yml`
  under "Developer Guide") — not a README-only blurb — covering the service boundary (why it is
  excluded from the wheel), a **field-typed `ScanMetadata` schema** for the external
  Bloom/downloader populator (`scan_key: str`, `image_ids: list[str]`, `images_checksum: str`,
  `params: {species, mode, age}` with `age` whole-number-coercible — rejects `bool`/non-whole
  floats), and `python -m trait_extractor` usage; add a short pointer section to README mirroring
  the `sleap-roots viewer` section.
- [ ] 11.3 Record that `trait_extractor` is intentionally excluded from the generated API
  reference (`docs/gen_ref_pages.py` walks only `sleap_roots/`) — no generator change needed.
- [ ] 11.4 Update the brainstorming design doc
  `docs/superpowers/specs/2026-07-06-a3-traits-result-envelope-emitter-design.md` with a
  post-review reconciliation appendix (golden reuse confirmed; Python-floor marker; batch failure
  isolation; class-keyed pipeline compatibility + multi-plant grain guard; negative-path
  validation; `contract_version` literal assertions) so the durable design doc matches the spec.
