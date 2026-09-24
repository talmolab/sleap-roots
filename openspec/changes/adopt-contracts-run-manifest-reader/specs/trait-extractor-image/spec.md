## MODIFIED Requirements

### Requirement: Container image runs the trait extractor over predict outputs

The repository SHALL provide a `trait-extractor.Dockerfile` at the repo root that builds an
image which runs the merged `trait_extractor` service as its entry point. The built image
SHALL execute `python -m trait_extractor <input_dir> <output_dir>` such that, given a
directory of predict per-scan outputs (a `{scan_key}.predictions.json` manifest, a
`{scan_key}.scan_metadata.json` sidecar, and the referenced `.slp` artifacts), it writes one
`{scan_key}.result.json` `ResultEnvelope` per scan into the output directory. The image SHALL
contain the importable `sleap_roots` library, the `sleap_roots_contracts` package, and the
`trait_extractor` package.

The emitter's parsing, pipeline-selection, and per-scan emission behavior is owned by the
`result-envelope-output` capability; this requirement asserts only that the built image
executes that behavior end-to-end at the container boundary, including propagating the batch
driver's three-way process exit code (`0` full success, `3` partial, `1` crash — see
`result-envelope-output`'s "Batch driver and module CLI" requirement) unchanged, which Argo reads
for DAG-node success/retry decisions. Coverage of this container-boundary propagation is currently
via the driver-level subprocess tests as a proxy (the exec-form `ENTRYPOINT` has no intermediary
process to diverge from the driver's own exit code); no automated test runs the built image itself
end-to-end (a pre-existing gap, not introduced by this requirement).

#### Scenario: Real entry emits a valid envelope over the committed fixture

- **WHEN** the image is built and run as
  `docker run -v <tests/data/rice_3do_pipeline_output>:/in -v <out>:/out
  ghcr.io/talmolab/sleap-roots-trait-extractor /in /out` (no `ARGO_WORKFLOW_NAME` set)
- **THEN** a `scan0K9E8BI.result.json` and a `scanYR39SJX.result.json` are written to the
  output directory
- **AND** each file parses as a `sleap-roots-contracts` `ResultEnvelope` whose
  `provenance.scan_key` matches its manifest and whose `provenance.contract_version == "0.1.0a9"`
- **AND** the process exits `0` (all scans succeeded)

#### Scenario: Required packages are importable inside the image

- **WHEN** `python -c "import sleap_roots; import sleap_roots_contracts; import trait_extractor; import yaml"`
  is run inside the built image
- **THEN** all four imports succeed with a `0` exit code

#### Scenario: A failing scan yields the partial container exit code without discarding good envelopes

- **WHEN** the image is run over an input tree containing one valid scan and one scan whose
  manifest references a nonexistent `.slp`
- **THEN** the valid scan's `{scan_key}.result.json` is still written to the output mount
- **AND** the container exits `3` — the exec-form `ENTRYPOINT` propagates the batch driver's exit
  code unchanged (a shell-form entry would swallow it), and A4's Argo template can key off `3`
  specifically to treat this as a completed run with partial failures rather than retrying the
  whole batch (the `retryStrategy`/`continueOn` wiring itself is a separate, pipeline-repo change —
  see design.md)

#### Scenario: SIGTERM during preemption terminates the pod promptly

- **WHEN** Argo sends `SIGTERM` to the running container (preemption or cancellation) while a
  batch is in progress
- **THEN** the process exits promptly with code `143` instead of waiting out
  `terminationGracePeriodSeconds` before SIGKILL
- **AND** any `{scan_key}.result.json` already written to the output mount before the signal is
  unaffected — per-scan writes are atomic (temp→rename) and the batch is idempotent on retry

### Requirement: Slim contracts install via an extractor extra

`pyproject.toml` SHALL declare a `[project.optional-dependencies] extractor` group containing
`sleap-roots-contracts==0.1.0a9` marked `; python_version >= '3.11'`, and the image SHALL
install it with `uv sync --frozen --no-dev --extra extractor`. This SHALL install the
`sleap-roots` library (whose `sleap_roots/` source is copied into the build context so
setuptools can build it and resolve its dynamic version), its runtime dependencies, and
`sleap-roots-contracts` and `pyyaml` (declared explicitly since `trait_extractor` imports it
directly) **without** installing the
`[dependency-groups] dev` tooling (mkdocs/pytest/black/twine/…). Every `sleap-roots-contracts`
pin in `pyproject.toml` (the `[dependency-groups] dev` group, the
`[project.optional-dependencies] dev` extra, and the `extractor` extra) SHALL name the same
version. `uv.lock` SHALL be re-locked in the same commit as the `pyproject.toml` change so the
frozen sync resolves (`uv lock` exits `0`; the published `sleap-roots` runtime dependency set is
unchanged).

#### Scenario: Image installs contracts but not the dev group

- **WHEN** the image build runs `uv sync --frozen --no-dev --extra extractor`
- **THEN** `python -c "import sleap_roots_contracts, yaml"` succeeds inside the image
- **AND** the dev/docs tooling (e.g. `mkdocs`, `pytest`) is not installed in the image venv

#### Scenario: Lockfile stays resolvable with the extra

- **WHEN** `uv lock` is run after adding the `extractor` extra
- **THEN** it exits `0` and the published `sleap-roots` runtime dependency set is unchanged
  (contracts remains a dev/test/container dependency, never a published runtime dependency)

#### Scenario: The extractor extra is guarded by a CI-run test

- **WHEN** the test suite runs (`ci.yml` triggers on `pyproject.toml`/`uv.lock` changes)
- **THEN** a test asserts the `[project.optional-dependencies] extractor` group exists and
  pins `sleap-roots-contracts==0.1.0a9` with the `; python_version >= '3.11'` marker
- **AND** removing or renaming the extra fails that test (so the image's install path cannot
  be silently broken by a CI-green change)

#### Scenario: All contracts pins agree

- **WHEN** the test suite runs
- **THEN** a test asserts that every `sleap-roots-contracts` requirement in `pyproject.toml`
  (dependency group `dev`, extras `dev` and `extractor`) pins the identical `==` version, so a
  partial bump (e.g. only the `extractor` extra) fails CI
