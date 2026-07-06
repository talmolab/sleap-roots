## Context

Full design: `docs/superpowers/specs/2026-07-06-a3-traits-result-envelope-emitter-design.md`
(committed). This change realizes the A3-traits emitter — the predict→traits handoff in the A4
Bloom-integration DAG. It is cross-cutting (new top-level package, new contract dependency, a
new consumer-side model of predict's output, a new sidecar contract) so it warrants a design
note. Ports/redesigns `salk-tm/sleap-roots-traits`.

## Goals / Non-Goals

- **Goals:** consume predict's `PredictionManifest` + a new `ScanMetadata` sidecar; select a
  pipeline by species/mode/age; compute scan-grain traits; emit a per-scan `ResultEnvelope`
  JSON that validates against `sleap-roots-contracts` and satisfies the write-back RPC's
  locally-checkable acceptance rules.
- **Non-Goals (fast-follow / out of scope):** the GHCR image / `trait-extractor.Dockerfile` /
  `docker-trait-extractor.yml` / Argo template; MinIO/Box upload; the actual write-back RPC
  call; `BlobRef` locations; image-grain (`compute_plant_traits`) TraitValues.

## Decisions

- **Top-level `trait_extractor/`, excluded from the wheel.** The extractor is a service that
  uses the library, not part of its public API. `sleap-roots-contracts` is a dev/test/container
  dependency only, so `pip install sleap-roots` stays pure. `[tool.setuptools.packages.find]`
  already includes only `["sleap_roots"]`, so a sibling top-level package is excluded by
  default; tests import it via `pytest` `pythonpath = ["."]`.
- **Consumer-side manifest model (no `sleap-roots-predict` dep).** predict pulls the
  GPU/torch/sleap-nn stack — too heavy for a traits consumer. Re-declare a lightweight
  `PredictionManifest`/`PredictionArtifact` importing only `ModelRef` from contracts; guard
  drift with a skip-if-unimportable cross-check against predict's real model.
- **`ScanMetadata` sidecar for idempotency inputs.** predict's manifest lacks `inputs`
  (`image_ids`, `images_checksum`) and `params` (`ResolvedParams`), both of which feed
  `idempotency_key`. The sidecar supplies them; the downloader/Bloom populates it in
  production. One `ResolvedParams` (from the sidecar) drives both pipeline selection and the
  idempotency `param_hash`.
- **`choose_pipeline` mirrors predict's `choose_models`/`ModelCard`.** Contiguous inclusive
  `[age_min, age_max]` windows via a packaged `pipeline_selection.yaml`; override wins;
  ambiguous/no-match/unknown → `ValueError`; whole-number `age` coercion. Fixes the legacy
  `MultipleDicotPlatePipeline` gap (present in the legacy table but unresolvable there).
- **`contract_version` = bare `"0.1.0a3"`** read at runtime from
  `importlib.metadata.version("sleap-roots-contracts")` (not a frozen literal, so it tracks the
  pin). Chosen for forward-consistency with the package predict pins; requires the Bloom RPC to
  accept `0.1.0a3`.
- **Python-floor marker.** `sleap-roots-contracts` requires Python `>=3.11`, but `sleap-roots`
  supports `>=3.10`. The extractor is a service (CI runs 3.11), so the dev/test dependency carries
  a `; python_version >= '3.11'` marker rather than bumping the library's floor — keeps `uv lock`
  resolving cleanly and the published library's 3.10 support intact.
- **Batch failure isolation.** The batch driver isolates per-scan failures: one scan raising (bad
  manifest, missing sidecar, incompatible root types) MUST NOT discard the other scans' emitted
  envelopes; the failed scan is reported (non-zero exit / logged error). This is the behavior that
  lets the A4 DAG degrade safely.
- **Input validation at the boundary.** Empty `artifacts` (guarded before `Series.load`, which
  never raises), malformed/schema-invalid manifests, missing referenced `.slp`, and sidecar
  `scan_key` mismatch each raise a clear, identifying error rather than emitting a malformed or
  empty envelope.
- **Pipeline compatibility mechanism.** No pipeline exposes its required root types (they are
  implicit in each `get_initial_frame_traits`), so a class-keyed constant
  `PIPELINE_REQUIRED_ROOTS: dict[type[Pipeline], frozenset[str]]` lives next to `choose_pipeline`
  as the single source of truth, pinned to reality by a guard test. The orchestrator checks
  **subset** semantics (`required ⊆ loaded`) — so a crown-only `OlderMonocotPipeline` accepts a
  primary+crown manifest — rather than equality, which would false-reject that valid case. The
  required-roots map is keyed on the pipeline **class** (a property of the class), NOT a field on
  the `(species, mode, age)` `PipelineCard` (two cards → one class could drift).
- **Scan-grain support guard.** Multi-plant / plate pipelines (`MultipleDicotPipeline`,
  `MultipleDicotPlatePipeline`, `MultiplePrimaryRootPipeline`) emit **per-plant** rows;
  `df.iloc[0]` would silently keep only the first plant. They remain resolvable by
  `choose_pipeline` (faithful legacy port) but the orchestrator rejects them for one-row
  scan-grain emission this slice — per-plant grain is a documented follow-up.
- **`contract_version` test pins the literal.** The assembly reads
  `importlib.metadata.version("sleap-roots-contracts")`; the test asserts BOTH that equality AND
  the literal `"0.1.0a3"` / no-`v`-prefix, so a silent contracts pin bump fails the test and
  forces conscious Bloom-RPC coordination (bloom#393).

## Risks / Trade-offs

- **Bloom RPC re-pin (A4 blocker).** The live RPC pins `v0.1.0a2`
  (`supabase/migrations/20260630180000_add_cyl_writeback_rpc.sql:33`); bare `0.1.0a3` is
  rejected until Salk-Harnessing-Plants-Initiative/bloom#393 lands. → The emitter + its tests
  are independent of this; end-to-end write-back is gated on the re-pin.
- **Manifest schema drift.** Consumer-side model duplicates predict's shape. → Mitigated by the
  cross-check test and predict's stable `schema_version: "1"`.
- **`image_ids` semantics.** The emitter passes sidecar `image_ids` through faithfully; that
  they resolve to a real `cyl_images.scan_id` is the downloader's/Bloom's responsibility
  (only checkable against live Bloom; out of scope).
- **Golden reuse (confirmed).** The `rice_3do_pipeline_output/*.slp` are byte-identical (matching
  sha256) to `rice_3do/0K9E8BI.*.predictions.slp`, so `rice_3do/rice_3do.batch_traits.csv` is a
  valid golden — no new golden file. The golden test must drop `plant_name` (the extractor's
  `series_name` is `"scan0K9E8BI"` vs the golden's `"0K9E8BI"`; select the row via
  `removeprefix("scan")`) and compare with `atol=1e-8` for cross-OS float stability.
- **`uv.lock` / `--frozen`.** CI runs `uv sync --frozen`; the marked contracts pin + regenerated
  `uv.lock` MUST land in the same commit or every CI job red-fails at sync. Contracts is a
  pre-release on PyPI — the exact `==0.1.0a3` pin admits it; `uv lock` records the PyPI source.

## Migration Plan

Additive only — new package, new dev dependency, new capability. No existing `sleap_roots`
behavior changes; no published-runtime dependency added. Rollback = drop `trait_extractor/`,
the dev dependency, and the CI additions.

## Open Questions

- Canonical `contract_version` byte convention (bare vs `v`-prefixed) — settle in
  talmolab/sleap-roots-contracts#14 with the Bloom owner; the emitter tracks the package version.
- Future: promote pipeline selection to a `PipelineCard` contract type
  (talmolab/sleap-roots-contracts#14) so model- and pipeline-selection share one source.
