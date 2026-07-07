## ADDED Requirements

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
executes that behavior end-to-end at the container boundary (including propagating the batch
driver's process exit code, which Argo reads for DAG-node success).

#### Scenario: Real entry emits a valid envelope over the committed fixture

- **WHEN** the image is built and run as
  `docker run -v <tests/data/rice_3do_pipeline_output>:/in -v <out>:/out
  ghcr.io/talmolab/sleap-roots-trait-extractor /in /out`
- **THEN** a `scan0K9E8BI.result.json` and a `scanYR39SJX.result.json` are written to the
  output directory
- **AND** each file parses as a `sleap-roots-contracts` `ResultEnvelope` whose
  `provenance.scan_key` matches its manifest and whose `provenance.contract_version == "0.1.0a3"`
- **AND** the process exits `0` (all scans succeeded)

#### Scenario: Required packages are importable inside the image

- **WHEN** `python -c "import sleap_roots; import sleap_roots_contracts; import trait_extractor; import yaml"`
  is run inside the built image
- **THEN** all four imports succeed with a `0` exit code

#### Scenario: A failing scan yields a non-zero container exit without discarding good envelopes

- **WHEN** the image is run over an input tree containing one valid scan and one scan whose
  manifest references a nonexistent `.slp`
- **THEN** the valid scan's `{scan_key}.result.json` is still written to the output mount
- **AND** the container exits **non-zero** — the exec-form `ENTRYPOINT` propagates the batch
  driver's exit code (a shell-form entry would swallow it). (The batch-level exit *convention*
  vs A4's `retryStrategy`/`continueOn` `partial`-run policy is reconciled in the traits-wiring
  step; this image packages the merged driver behavior unchanged — see design.md.)

### Requirement: Image stamps its build commit into trait provenance

The image SHALL bake its build commit SHA into `ENV SRT_TRAITS_CODE_SHA` (supplied via a
Docker `build-arg` that the build workflow sets to `github.sha` on the push/dispatch builds
that publish). This makes envelopes emitted by the container carry a non-empty
`provenance.traits_code_sha` identifying the traits code version — which the
`sleap-roots-contracts` idempotency key hashes, so a change to the traits code invalidates a
prior "done" result under A4's first-writer-wins dedup (a `traits_code_sha=""` default would
silently collide). The image SHALL NOT bake `SRT_TRAITS_CONTAINER_DIGEST` (the digest is not
known at build time and is not an idempotency-key input; it stays a runtime-supplied value). A
runtime environment value SHALL override the baked default so the Argo template keeps control.

#### Scenario: Emitted provenance carries the baked code SHA

- **WHEN** the image is built with `--build-arg SRT_TRAITS_CODE_SHA=<sha>` and run over the
  fixture
- **THEN** each emitted `{scan_key}.result.json` has `provenance.traits_code_sha == "<sha>"`
  (non-empty)
- **AND** a runtime `SRT_TRAITS_CODE_SHA` environment value passed to the container overrides
  the baked default

### Requirement: Distinct GHCR image identity

The trait-extractor image SHALL be published under the explicit identity
`ghcr.io/talmolab/sleap-roots-trait-extractor`, distinct from the library's identity
`ghcr.io/talmolab/sleap-roots`. The build workflow SHALL pass this identity as an explicit
`images:` literal to the image-metadata step and SHALL NOT derive it from
`${{ github.repository }}`.

#### Scenario: Workflow uses the explicit service identity

- **WHEN** the docker build workflow derives image tags
- **THEN** the metadata step's `images:` value is the literal
  `ghcr.io/talmolab/sleap-roots-trait-extractor`
- **AND** it is not `ghcr.io/${{ github.repository }}` (which would name the library package)

### Requirement: Slim contracts install via an extractor extra

`pyproject.toml` SHALL declare a `[project.optional-dependencies] extractor` group containing
`sleap-roots-contracts==0.1.0a3` marked `; python_version >= '3.11'`, and the image SHALL
install it with `uv sync --frozen --no-dev --extra extractor`. This SHALL install the
`sleap-roots` library (whose `sleap_roots/` source is copied into the build context so
setuptools can build it and resolve its dynamic version), its runtime dependencies, and
`sleap-roots-contracts` and `pyyaml` (declared explicitly since `trait_extractor` imports it
directly) **without** installing the
`[dependency-groups] dev` tooling (mkdocs/pytest/black/twine/…). `uv.lock` SHALL be re-locked
in the same commit as the `pyproject.toml` change so the frozen sync resolves (`uv lock` exits
`0`; the published `sleap-roots` runtime dependency set is unchanged).

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
  pins `sleap-roots-contracts==0.1.0a3` with the `; python_version >= '3.11'` marker
- **AND** removing or renaming the extra fails that test (so the image's install path cannot
  be silently broken by a CI-green change)

### Requirement: Install-free, headless container entry

The image SHALL run the service directly from the build-time virtual environment without any
runtime dependency resolution. The `trait_extractor` package SHALL be made available by
copying it into the image (it is excluded from the wheel and is not pip-installed) and placed
on `PYTHONPATH`; the venv interpreter SHALL be first on `PATH`; the working directory SHALL be
`/app`; and the entry point SHALL be `ENTRYPOINT ["python","-m","trait_extractor"]` in exec
form. The image SHALL set `MPLBACKEND=Agg` and a writable `MPLCONFIGDIR` so matplotlib imports
succeed headlessly on the slim base.

#### Scenario: Entry is the exec-form module run from the baked venv

- **WHEN** the built image is inspected (`docker inspect --format '{{json .Config.Entrypoint}}'`)
- **THEN** the entry point is exactly `["python","-m","trait_extractor"]` (exec form)
- **AND** `python` resolves to the `/app/.venv` interpreter and reports Python 3.12.x (the
  environment is frozen at build time; no `uv sync`/lockfile resolution runs on container start)

#### Scenario: Headless matplotlib import succeeds

- **WHEN**
  `python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure()"`
  runs inside the container
- **THEN** it exits `0`, using the `Agg` backend against the writable `MPLCONFIGDIR` (no
  writable-HOME cache error and no Tk requirement)

### Requirement: Build/push workflow gated by event and path

The repository SHALL provide `.github/workflows/docker-trait-extractor.yml` that builds the
image on pull requests (validation only, no registry login and no push) and builds **and**
pushes it on `main` and on manual dispatch. The workflow SHALL declare
`permissions: { contents: read, packages: write }` so the push to GHCR is authorized. Both the
`push` and `pull_request` triggers SHALL be path-filtered to the image's build inputs:
`sleap_roots/**` (the library is baked into the image from source, so a library-source change is
an image-content change), `trait_extractor/**`, `trait-extractor.Dockerfile`, `pyproject.toml`,
`uv.lock`, `.dockerignore`, and the workflow file itself. The workflow SHALL NOT trigger on
`release` events, so that a library PyPI *release* (handled by `build.yml`) never builds the
service image and vice versa. It SHALL build using `file: trait-extractor.Dockerfile` with
`context: .`. The PyPI `build.yml` and the test `ci.yml` SHALL remain unmodified.

#### Scenario: Pull request builds but does not push

- **WHEN** the workflow runs for a `pull_request` event
- **THEN** the image is built to validate the Dockerfile
- **AND** no GHCR login occurs and nothing is pushed to the registry

#### Scenario: Main build pushes to GHCR

- **WHEN** the workflow runs for a push to `main` (or a manual dispatch)
- **THEN** it logs in to GHCR and pushes the tagged
  `ghcr.io/talmolab/sleap-roots-trait-extractor` image (`latest` on the default branch plus an
  immutable `sha-<sha>` tag)

#### Scenario: A library PyPI release does not trigger the image build

- **WHEN** a `release: published` event fires (the PyPI library release handled by `build.yml`)
- **THEN** `docker-trait-extractor.yml` does not run (it has no `release:` trigger)
- **AND** conversely, an image rebuild never runs `build.yml`

#### Scenario: Changes outside the build inputs do not trigger the image build

- **WHEN** a push to `main` or a pull request touches only non-input paths (e.g. `docs/**` or
  `tests/**`, none of the filtered build-input paths)
- **THEN** `docker-trait-extractor.yml` does not run
- **AND** a change under `sleap_roots/**` *does* trigger it (the library is baked into the image)

### Requirement: Build-context hygiene via .dockerignore

The repository SHALL provide a `.dockerignore` that excludes VCS, virtualenv, build, and test
assets (`.git`, `.venv`, build artifacts, caches, `tests/`) from the Docker build context,
while leaving the library build inputs (`pyproject.toml`, `uv.lock`, `README.md`), the
`sleap_roots/` package source, and the `trait_extractor/` package available to `COPY`.
Verification fixtures SHALL be mounted at run time rather than baked into the image.

#### Scenario: Build inputs are present but test data is excluded

- **WHEN** the image is built with the `.dockerignore` in place
- **THEN** `sleap_roots/`, `trait_extractor/`, `pyproject.toml`, `uv.lock`, and `README.md`
  are available to the build (so `uv sync` and the `COPY`s succeed)
- **AND** the `tests/` directory is not part of the build context and is not baked into the
  image, yet the fixture under `tests/data/rice_3do_pipeline_output/` can still drive the real
  entry by being mounted with `-v` at `docker run` time
