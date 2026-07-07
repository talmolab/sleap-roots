# Tasks

This is a container/CI slice: the "test-first" discipline means defining the **verification
check** (the assertion that must pass) before writing the artifact that satisfies it. Each
task states the check first, then the implementation. The end-to-end image verification (task
5) is the primary acceptance gate and is written down before the Dockerfile is authored.

The emitter *logic* over this exact fixture is already covered cross-platform in CI by
`tests/trait_extractor/test_batch.py::test_module_cli_writes_envelopes` (which runs
`python -m trait_extractor <rice_3do_pipeline_output> <out>` on ubuntu/windows/macOS with
`lfs: true`). This slice therefore does **not** add a docker-run job to CI (heavy; would need
LFS in the image workflow); the in-container run is a documented **manual pre-merge gate**, and
CI only build-validates the image on PRs.

The packaging path (tasks 1/3) was empirically preflighted in an isolated worktree: `uv lock`
with the extra exits `0` with a minimal diff (root `provides-extras` + one mapping line, no
package/hash churn), and `uv sync --frozen --no-dev --extra extractor` installs
`sleap_roots_contracts` + `pyyaml` (editable `sleap_roots`) while excluding
mkdocs/pytest/black/twine.

## 1. Packaging — slim `extractor` extra + re-lock (atomic commit)

- [x] 1.1 **Check first:** `uv lock` exits `0` with the new extra; `git diff uv.lock` changes
  only the root project's `provides-extras` + one extra-mapping line (contracts + pyyaml are
  already locked via the dev group, so no new packages / no hash churn); `uv sync --frozen
  --no-dev --extra extractor` installs `sleap_roots_contracts` + `pyyaml` but NOT
  `mkdocs`/`pytest`.
- [x] 1.2 Add `[project.optional-dependencies] extractor = ["sleap-roots-contracts==0.1.0a3 ;
  python_version >= '3.11'"]` to `pyproject.toml` (leave the `dev` group + `optional-dependencies
  dev` intact; CI still needs contracts there).
- [x] 1.3 Run `uv lock`; commit the refreshed `uv.lock` **in the same commit** as 1.2 (else
  `ci.yml`'s `uv sync --frozen` breaks). Verify the assertions from 1.1 hold.
- [x] 1.4 **Cheap CI guard (test-first):** a pytest that parses `pyproject.toml` and asserts
  `[project.optional-dependencies] extractor` exists and pins `sleap-roots-contracts==0.1.0a3`
  with the `; python_version >= '3.11'` marker. This makes the image's install path CI-enforced,
  since `ci.yml` triggers on `pyproject.toml`/`uv.lock`.
  - **Why here instead of a new file?** Landed as `test_packaging_config_declares_the_extractor_extra`
    in the existing `tests/trait_extractor/test_package_boundary.py` (which already holds
    `test_packaging_config_excludes_the_extractor`, using the same `tomllib` + `_REPO_ROOT`
    pattern) rather than a new `test_packaging_extractor_extra.py` — DRY, co-located with the
    sibling packaging guard. Same coverage.

## 2. `.dockerignore`

- [x] 2.1 **Check first:** the build context must exclude `.git`, `.venv`, build artifacts,
  caches, and `tests/`, while KEEPING `pyproject.toml`, `uv.lock`, `README.md`, `sleap_roots/`,
  and `trait_extractor/` (excluding `sleap_roots/` would break `uv sync`; `README.md` is
  copied because the project declares a `dynamic` readme).
- [x] 2.2 Author `.dockerignore` (near-copy of `sleap-roots-predict`'s), adjusted for this repo.

## 3. `trait-extractor.Dockerfile`

- [x] 3.1 **Check first:** the image-acceptance checks (they become task 5) — `import
  sleap_roots`, `import sleap_roots_contracts`, `import trait_extractor`, `import yaml` succeed
  inside the image; the real entry emits `{scan_key}.result.json`; the emitted
  `provenance.traits_code_sha` equals the baked build-arg SHA.
- [x] 3.2 Author `trait-extractor.Dockerfile`: base
  `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`; `WORKDIR /app`; `UV_*` env
  (`UV_COMPILE_BYTECODE=1`, `UV_LINK_MODE=copy`, `UV_PYTHON_DOWNLOADS=never`); `COPY
  pyproject.toml uv.lock README.md ./`; **`COPY sleap_roots ./sleap_roots`** (required — `uv
  sync` builds the `sleap-roots` project and resolves its dynamic `attr =
  "sleap_roots.__version__"` version); `COPY trait_extractor ./trait_extractor`; `RUN uv sync
  --frozen --no-dev --extra extractor --python 3.12` (pin `--python 3.12` — an empirical
  preflight showed a `<3.11` interpreter makes `uv sync` exit `0` while **silently dropping
  `sleap-roots-contracts`**); `ENV PATH="/app/.venv/bin:$PATH"`; `ENV PYTHONPATH=/app` (the
  editable install exposes only `sleap_roots`, NOT `trait_extractor`, so the repo root must be
  on `sys.path`); `ENV MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib`; then — **after** the
  sync/COPY layers so a per-commit SHA doesn't bust the dependency cache — `ARG
  SRT_TRAITS_CODE_SHA=""` and `ENV SRT_TRAITS_CODE_SHA=${SRT_TRAITS_CODE_SHA}`; `ENTRYPOINT
  ["python","-m","trait_extractor"]`. Do **not** copy `.python-version` (a copied 3.11 pin +
  `UV_PYTHON_DOWNLOADS=never` would fail to find an interpreter).
- [x] 3.3 Start apt libs at `build-essential` only; add `libgl1`/`libglib2.0-0`/`tk` **only if**
  the in-image `import sleap_roots` check (task 5) fails, and record the final minimal set in
  a Dockerfile comment.

## 4. `.github/workflows/docker-trait-extractor.yml`

- [x] 4.1 **Check first:** workflow assertions — `images:` is the literal
  `ghcr.io/talmolab/sleap-roots-trait-extractor` (not `${{ github.repository }}`); a
  job-level `permissions: { contents: read, packages: write }` block is present; `push:` is
  gated `github.event_name != 'pull_request'` and GHCR login is gated the same way; **both**
  the `push: branches:[main]` and `pull_request` triggers carry the same `paths:` filter
  (`trait_extractor/**` + `trait-extractor.Dockerfile` + `pyproject.toml` + `uv.lock` +
  `.dockerignore` + the workflow file); there is **no `release:` trigger**; `build-push-action`
  sets `file: trait-extractor.Dockerfile`, `context: .`, and `build-args: SRT_TRAITS_CODE_SHA=${{
  github.sha }}`. Validate YAML with `actionlint` (or a parse check) — note actionlint checks
  syntax only, so the identity/gating assertions above are eyeballed against the file.
- [x] 4.2 Author the workflow as a near-copy of predict's `docker-build.yml` with: the identity
  override; the `permissions` block; the `paths:` filter on **both** triggers; **`release:`
  removed**; the `file:` field; `build-args: SRT_TRAITS_CODE_SHA=${{ github.sha }}`; tags
  `type=ref,event=branch|pr`, `type=sha`, and `type=raw,value=latest,enable={{is_default_branch}}`
  (drop the `type=semver` lines — no release drives this image); `cache-from/to: type=gha`.
  Leave `build.yml` and `ci.yml` untouched.
- [x] 4.3 On push builds, surface the pushed image digest (`build-push-action`'s `digest`
  output) to `$GITHUB_STEP_SUMMARY` so A4's Argo template can pin `@sha256:…` (A4 pins the
  digest so retries recompute an identical idempotency key).

## 5. Real end-to-end verification (primary acceptance gate — manual pre-merge)

- [x] 5.1 Ensure the LFS fixtures are materialized: `git lfs pull` (or assert each
  `tests/data/rice_3do_pipeline_output/**/*.slp` is > 1 KB real HDF5, not a ~130-byte LFS
  pointer) — the `.slp` are LFS-tracked and are only real files after a smudge/pull.
- [x] 5.2 Build with a test SHA to exercise stamping:
  `docker build -f trait-extractor.Dockerfile --build-arg SRT_TRAITS_CODE_SHA=deadbeef -t
  sleap-roots-trait-extractor:local .` succeeds.
- [x] 5.3 Inside the image: `python -c "import sleap_roots; import sleap_roots_contracts; import
  trait_extractor; import yaml"` exits `0` (drives task 3.3 apt-lib trimming); `python -c
  "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; plt.figure()"`
  exits `0` (headless matplotlib); `docker inspect --format '{{json .Config.Entrypoint}}'
  sleap-roots-trait-extractor:local` == `["python","-m","trait_extractor"]`; and `python
  --version` reports 3.12.x.
- [x] 5.4 Drive the **real** entry over the committed fixture. On Windows, pass absolute host
  paths and guard MSYS path mangling (`MSYS_NO_PATHCONV=1`, or use `${PWD}`/`C:\…` in
  PowerShell): `docker run -v <abs tests/data/rice_3do_pipeline_output>:/in -v <abs out>:/out
  sleap-roots-trait-extractor:local /in /out`; assert `scan0K9E8BI.result.json` and
  `scanYR39SJX.result.json` are written, each parses as a `ResultEnvelope` with matching
  `provenance.scan_key`, `provenance.contract_version == "0.1.0a3"`, and
  `provenance.traits_code_sha == "deadbeef"` (proving the baked SHA flows through), and the
  container exits `0`.
- [x] 5.5 **Failure-isolation at the boundary:** run over a crafted tree (one valid scan + one
  manifest referencing a nonexistent `.slp`); assert the valid envelope is still written and
  the container exits **non-zero** (exec-form propagation — a shell-form entry would swallow it).
- [x] 5.6 Confirm the image does **not** contain dev tooling and that `tests/` is not baked in.
  (Verification trap: `uv pip list`/`pip list` read the ambient `VIRTUAL_ENV`, not the image's
  `.venv` — inside the container this is moot, but any local slim-check must target
  `.venv/bin/python`. Spot-check `pip show mkdocs`/`pytest` absent.)

## 6. Docs (single source of truth = the dev doc)

- [x] 6.1 Add a `## Container image` section to `docs/dev/trait-extractor-service.md` as the
  **SSOT**: image name `ghcr.io/talmolab/sleap-roots-trait-extractor`, the
  `docker run -v <in>:/in -v <out>:/out <image>:latest /in /out` usage, and the tag scheme
  (`latest` on `main` + immutable `sha-<sha>` / `@sha256:…` digest; pin the digest for
  reproducible A4 runs). Fix the now-stale "Notes & follow-ups" line (~L97-99) that still calls
  the GHCR image a fast-follow — only the A4 Argo template + bloom#393 remain fast-follow. (The
  dev doc is already in mkdocs nav — no nav change.)
- [x] 6.2 **Update** (not add) the existing "Trait-Extractor Service" section in `README.md`
  with a one-line pointer to the dev-doc `#container-image` section (mirror the existing "See
  the guide for details" pattern; do not duplicate the full `docker run` invocation).
- [x] 6.3 Add a `docs/changelog.md` `### Added` (Unreleased) entry naming the image + the
  `extractor` extra, noting contracts is now dev/test/**container** (and reachable as an opt-in
  `pip install sleap-roots[extractor]`, updating the earlier "dev/test" framing); point to the
  dev doc rather than repeating the run command.
- [x] 6.4 Reconcile the emitter design doc
  `docs/superpowers/specs/2026-07-06-a3-traits-result-envelope-emitter-design.md` for internal
  consistency (not just §7): drop "(plan; Dockerfile is fast-follow)" from the §7 heading (L123)
  and the "(fast-follow)" markers on the §7 Dockerfile/image-build bullets (L127, L134); remove
  `GHCR image / Dockerfile / docker-trait-extractor.yml` from the §9 **OUT** list (L156-159);
  update the §3 Decisions Container/CI row (L51) so it no longer calls the image a "fast-follow
  slice"; **add** a new reconciliation-appendix bullet recording that the image shipped in
  `add-trait-extractor-image` (there is no existing appendix line to "flip"). Record the image
  name + tag scheme once (in §7) and cross-reference the dev doc.

## 7. Validate + close out

- [x] 7.1 `openspec validate add-trait-extractor-image --strict` passes.
- [x] 7.2 Reconcile implementation against this proposal (image name, `COPY sleap_roots`,
  `--python 3.12`, baked `SRT_TRAITS_CODE_SHA`, entrypoint, extra, apt-lib final set, workflow
  triggers/permissions/build-args); update proposal/design/spec if anything deviated, with a
  `### Why …` note.
- [x] 7.3 Run `/pre-merge-check`; open the PR linking tracking issue talmolab/sleap-roots#256;
  present **READY TO MERGE** and stop (author merges their own PR). Leave the post-merge roadmap
  update (`sleap-roots-pipeline`) and the OpenSpec archive as separate follow-up PRs.
