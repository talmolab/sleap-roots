## Context

The `trait_extractor/` service (merged #254) is a flat, top-level package deliberately
**excluded from the PyPI wheel** — it *uses* the `sleap-roots` library but is not part of
its public API. It runs as `python -m trait_extractor <input_dir> <output_dir>`, consuming
predict's `{scan_key}.predictions.json` manifest + a `ScanMetadata` sidecar and emitting a
per-scan `sleap-roots-contracts` `ResultEnvelope` as `{scan_key}.result.json`.

A4's Argo DAG needs to *pull* this as a container. Today its `trait-extractor` template
pulls the legacy `registry.gitlab.com/salk-tm/sleap-roots-traits:latest` (a
`mambaorg/micromamba` image driven by `python /workspace/src/main.py <in> <out>`). This
change publishes a GHCR replacement built from the current repo source.

Two sibling reference implementations frame the design:
- **`sleap-roots-predict`** (`Dockerfile` + `.github/workflows/docker-build.yml`) — the
  pattern to mirror: `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` base, `uv sync --frozen
  --extra …`, headless matplotlib env, buildx + `docker/metadata-action` tags, GHCR login +
  push gated to non-PR events, `cache-from/to: gha`.
- **`salk-tm/sleap-roots-traits`** (legacy) — the thing being reclassified: `PYTHONPATH=/workspace`,
  `MPLCONFIGDIR` set, positional `<in> <out>` entry, GitLab CI push-on-main.

## Goals / Non-Goals

- **Goals:** a reproducible, per-commit GHCR image that runs the real
  `python -m trait_extractor <in> <out>` entry; a build/push CI workflow that is safe on PRs
  (build-only) and publishes on `main`; a slim contracts install path; a self-contained
  boundary so `trait_extractor/` + its Dockerfile/workflow can later lift to its own repo.
- **Non-Goals:** the A4 Argo template wiring; the write-back RPC / MinIO / Box upload /
  `BlobRef` locations; image-grain and multi-plant-plate per-plant grain (#252);
  multi-arch builds (Argo runs linux/amd64; mirror predict's single-arch default);
  changing the published library's runtime deps or the wheel contents.

## Decisions

- **Distinct image identity `ghcr.io/talmolab/sleap-roots-trait-extractor`.** An explicit
  `images:` literal in the workflow, **not** `${{ github.repository }}` (which resolves to
  `ghcr.io/talmolab/sleap-roots` — the *library's* identity). The library (PyPI) and the
  service (image) are separate deliverables with separate lifecycles.
  - *Alternative considered:* reuse `${{ github.repository }}` with a tag suffix — rejected;
    it conflates the two artifacts under one package name.
- **Base `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`.** Mirrors predict; Python 3.12
  satisfies contracts' `>=3.11`. The repo's `.python-version` (3.11) is **not** copied into
  the build context, so `uv` resolves against the image's 3.12 (within the universal
  lockfile's `>=3.10` range). `UV_PYTHON_DOWNLOADS=never` keeps the build offline for Python.
  - *Version-skew note:* dev/CI test on 3.11; the image runs 3.12. Accepted — predict does
    the same, both are within `requires-python`, and the lockfile is universal. Recorded as a
    risk below.
  - *Harden the interpreter choice:* the `sync` command pins `--python 3.12`. An empirical
    preflight showed that if uv selects a `<3.11` interpreter, `uv sync` exits `0` but
    **silently drops `sleap-roots-contracts`** (its `python_version >= '3.11'` marker fails) —
    a no-error build that ships a broken image. Pinning `--python 3.12` (the base image's only
    interpreter, with `UV_PYTHON_DOWNLOADS=never`) forecloses that; the in-image
    `import sleap_roots_contracts` check (task 5) is the backstop that would catch a silent drop.
- **Install contracts via a slim `extractor` extra, not the dev group.** New
  `[project.optional-dependencies] extractor = ["sleap-roots-contracts==0.1.0a3 ; python_version >= '3.11'"]`;
  the image runs `uv sync --frozen --no-dev --extra extractor`. This installs the
  `sleap-roots` project (the `sleap_roots` package) + its runtime deps + contracts (+
  transitive `pyyaml`), but **not** the `[dependency-groups] dev` set (mkdocs/pytest/black/…).
  `uv.lock` is re-locked once (contracts is already resolved in the lock via the dev group, so
  the package set does not change — only the extra-mapping is added; `uv lock` stays exit-0).
  - *Alternatives considered:* (a) `uv sync --frozen` with the full dev group — heavier image,
    ships docs/test tooling to Argo; (b) ad-hoc `uv pip install sleap-roots-contracts==0.1.0a3`
    — not lockfile-pinned, weaker reproducibility. Both rejected.
- **`ENTRYPOINT ["python","-m","trait_extractor"]` — install-free, off the baked venv.**
  `WORKDIR /app`; `PATH=/app/.venv/bin:$PATH` makes `python` the venv interpreter (with
  `sleap_roots` + `sleap_roots_contracts` installed at build time); `COPY trait_extractor
  ./trait_extractor` + `PYTHONPATH=/app` makes the copied service importable. `python -m` runs
  the merged, already-tested `__main__.py` directly — no console-script (the service is not
  pip-installable) and no runtime `uv run` (which would re-resolve the environment on every
  container start, needs `uv`+project metadata in the final image, and defeats the frozen
  image). The **exec-form** `ENTRYPOINT` (JSON array) makes `python` PID 1, so the batch
  driver's `sys.exit(1)` propagates as the container exit code Argo reads. `docker run <image>
  /in /out` then passes the two dirs straight through as argv.
- **`COPY sleap_roots ./sleap_roots` is required (not just `trait_extractor`).** `uv sync`
  installs the root `sleap-roots` project, which setuptools builds from the `sleap_roots/`
  package source (`include = ["sleap_roots"]`) and whose dynamic version is
  `attr = "sleap_roots.__version__"`. Without the source in the build context the build fails
  before anything is pushed. So the Dockerfile COPYs `pyproject.toml uv.lock README.md`, then
  `sleap_roots/`, then `trait_extractor/`, before `uv sync`. The `.dockerignore` must **not**
  exclude `sleap_roots/`. (Mirrors predict, which copies `sleap_roots_predict`.)
- **Bake `SRT_TRAITS_CODE_SHA=<git sha>` at build time.** The `sleap-roots-contracts`
  idempotency key hashes `traits_code_sha` (`compute_idempotency_key`); the emitter reads it
  from the `SRT_TRAITS_CODE_SHA` env via `envelope.py`'s arg→env→`""` ladder, and
  `python -m trait_extractor <in> <out>` exposes **no CLI flag** to inject it. So if the env is
  unset, every envelope carries `traits_code_sha=""`, and a change to the traits code would
  **not** invalidate a prior "done" result under A4's first-writer-wins dedup — a silent
  staleness bug. The image knows its own build SHA, so the Dockerfile takes
  `ARG SRT_TRAITS_CODE_SHA` → `ENV SRT_TRAITS_CODE_SHA`, and the workflow passes
  `build-args: SRT_TRAITS_CODE_SHA=${{ github.sha }}` (on the push/dispatch events that publish;
  it matches the emitted `type=sha` tag). The `ARG`/`ENV` go **after** the `uv sync`/COPY layers
  so a per-commit SHA doesn't bust the dependency cache. A runtime `env:` in the Argo template
  still overrides the baked default, so A4 keeps control. `SRT_TRAITS_CONTAINER_DIGEST` is
  **not** baked — the digest isn't known until after the push (self-referential) and is
  recorded-but-not-hashed, so it stays A4's runtime responsibility.
- **Runs as root (default of the slim uv base).** No `USER` directive; matches predict's base
  and the legacy `privileged: true, runAsUser: 0` template. A4's design mandates no user/root
  policy, so dropping root/`privileged` for this CPU filesystem job is an A4-template hardening
  choice, not an image concern.
- **Headless matplotlib.** `MPLBACKEND=Agg` + a writable `MPLCONFIGDIR=/tmp/matplotlib`
  (sleap-roots' `circumnutation` code and its plotting deps import matplotlib; the slim base
  has no writable HOME cache and no Tk).
- **System apt libs — start minimal, verify empirically.** `sleap_roots` imports **no**
  `cv2`/OpenCV (verified by grep), and `MPLBACKEND=Agg` removes the Tk need, so predict's
  `tk`/`libgl1`/`libglib2.0-0` are likely unnecessary. Start from `build-essential` only and
  add libs **only if** `python -c "import sleap_roots"` fails in the built image. The exact
  final set is settled by the real build in implementation, not guessed here.
- **Separate, path-filtered workflow.** `.github/workflows/docker-trait-extractor.yml`,
  triggered on `push:[main]`, `pull_request`, and `workflow_dispatch` — with the **same
  `paths:` filter on both the `push` and `pull_request` triggers** (`trait_extractor/**`,
  `trait-extractor.Dockerfile`, `pyproject.toml`, `uv.lock`, `.dockerignore`, the workflow
  file). Build-only on PR (`push: ${{ github.event_name != 'pull_request' }}`, GHCR login
  gated the same way, `permissions: { contents: read, packages: write }` for the push);
  build+push otherwise. `build-push-action` gets `file: trait-extractor.Dockerfile`
  (non-default name) with `context: .`. Tags: `type=ref,event=branch|pr`, `type=sha`, and
  `type=raw,value=latest,enable={{is_default_branch}}`.
  - **No `release:` trigger, no `type=semver` tags.** Predict's workflow (which this mirrors)
    triggers on `release: published` and unfilters its `push`, but that would (a) rebuild the
    image on *every* merge to `main` including library-only changes and (b) fire the image
    build on the same `release: published` event that drives the PyPI `build.yml` — both
    violating the invariant that a library release and the service image never trigger each
    other. Dropping `release:` and path-filtering the `push` honors the invariant; the image's
    lifecycle is driven only by changes to `trait_extractor/`, the Dockerfile, or the locked
    deps. The PyPI `build.yml` and test `ci.yml` are untouched.

## Risks / Trade-offs

- **Python version skew (3.12 image vs 3.11 dev/CI).** → The universal `uv.lock` covers both;
  contracts + sleap-roots both support 3.12; the real end-to-end verification runs *inside*
  the 3.12 image over the committed fixture, catching any 3.12-specific import break.
- **`uv sync --frozen` staleness.** Adding the `extractor` extra without re-locking would make
  the frozen build fail. → Re-run `uv lock` and commit; a task asserts `uv lock` is exit-0 and
  `git diff uv.lock` is limited to the extra-mapping.
- **apt libs guessed too lean.** → Mitigated by the mandatory in-image
  `import sleap_roots` + real-entry verification; if it fails, add the specific lib and note it.
- **First push / package visibility.** On first push GHCR auto-creates
  `ghcr.io/talmolab/sleap-roots-trait-extractor` (private) and links it to this repo;
  `permissions: packages: write` + `GITHUB_TOKEN` suffices to publish (predict already proves
  talmolab GHCR pushes work). The new package is created **private**, so A4's Argo puller will
  get 403/404 until someone flips it to public or grants pull access — an **operational
  follow-up for the A4 wiring PR**, not a blocker for building/publishing here. (If the org
  enforced a "restrict package creation" policy the first push could 403 — unlikely given
  predict; worth a one-line pre-check at merge.)
- **Fixture availability in the build context.** The `.dockerignore` excludes `tests/`, so the
  fixture is **not** baked into the image; verification **mounts** it at `docker run` time
  (`-v <fixture>:/in`). The `rice_3do_pipeline_output/*.slp` are **Git-LFS-tracked**
  (`.gitattributes`: `*.slp filter=lfs`) — they are materialized as real HDF5 only after a
  smudge / `git lfs pull` (already true on this dev box; a fresh clone with LFS skipped would
  leave ~130-byte pointer stubs and every scan would fail). The manual verification (task 6)
  therefore `git lfs pull`s / asserts real `.slp` first. The image **build** needs no LFS
  (`tests/` is excluded), so build-only PR CI is unaffected.

## Migration Plan

- Additive only: new files + one `pyproject.toml` extra + a `uv.lock` refresh. No existing
  behavior changes; nothing to roll back beyond deleting the new files.
- **Post-merge (separate PR in `sleap-roots-pipeline`):** update
  `docs/bloom-integration/roadmap.md` — flip the A3-traits "GHCR image" fast-follow to done and
  set the A4 "traits wiring" row to "image ready; needs the Argo template rewrite + write-back".
  Record for the A4 Argo template:
  - **Image:** `ghcr.io/talmolab/sleap-roots-trait-extractor`; prefer the `@sha256:…` digest
    (A4 names the digest specifically), with the immutable `sha-<sha>` tag as a convenience —
    both surfaced by the build workflow. A4 pins the container per run so a **retry recomputes
    an identical idempotency key** (A4 design §8): the key hashes `traits_code_sha` — which the
    pinned image carries (baked as `SRT_TRAITS_CODE_SHA`, see Decisions) — plus models/params;
    the digest itself is **not** a key input, it just freezes those inputs across retries.
    `latest` is convenience-only.
  - **Args rewrite (required — not a drop-in swap):** the current `trait-extractor` template
    passes `args: ["python","/workspace/src/main.py","/workspace/input","/workspace/output"]`
    with **no `command:`**. This image's `ENTRYPOINT ["python","-m","trait_extractor"]` means the
    template must become `args: ["/workspace/input","/workspace/output"]` (drop the
    `python /workspace/src/main.py` prefix; leave `command:` unset so the ENTRYPOINT runs). The
    two mount paths (`/workspace/input` = predict output, `/workspace/output` = traits output)
    are unchanged and are consumed as the two positional args.

## Cross-repo alignment (verified against `sleap-roots-pipeline` @ `a4-request-driven-pipeline`)

Checked the pipeline roadmap (`docs/bloom-integration/roadmap.md`) and the A4 design
(`docs/superpowers/specs/2026-07-06-a4-request-driven-pipeline-design.md`) plus the committed
Argo templates. The image design matches A4's intent (filesystem-only, positional `<in> <out>`,
per-scan `ResultEnvelope`, bare `0.1.0a3`, pin-by-digest). Two handoff frictions are noted so
the A4 wiring PR (out of scope here) can resolve them; **neither changes this image**:

- **Exit-code convention vs Argo retry.** This image packages the merged driver behavior:
  `python -m trait_extractor` exits non-zero if *any* scan failed (per-scan envelopes are still
  written). A4's design wants an isolated "poison" scan to yield a **`partial`** run (continue,
  don't fail the step) while the step carries `retryStrategy: {limit: 2}`. Non-zero-on-any-failure
  would instead retry the whole batch and mark the step failed. Reconciliation (distinct exit
  codes so Argo can distinguish "some scans isolated-failed" from "pod-level crash", or an Argo
  `continueOn`/retry tweak, or a follow-up change to the driver's exit convention in the emitter
  capability) belongs to the traits-wiring step. This image intentionally does not alter
  `trait_extractor/__main__.py`.
- **Handoff not yet wired in today's DAG.** The currently-wired predictor template still pulls
  the legacy predict image (does not emit `{scan}.predictions.json`), and no committed step yet
  produces the `{scan}.scan_metadata.json` sidecar. So the image cannot run end-to-end in the
  *existing* DAG until A4 upgrades the predictor to the manifest-emitting GHCR predict image and
  wires the sidecar producer. This is expected A4 sequencing, not a defect in this image.

## Open Questions

- Which tag should A4's Argo template pin — mutable `latest` (auto-updates on `main`) or the
  immutable `@sha256:…` digest / `sha-<sha>` tag (reproducible; A4 pins the digest so a retry
  recomputes an identical idempotency key)? Deferred to the A4 wiring PR; this change publishes
  `latest` + `sha-<sha>` and surfaces the digest from the build step for the template to pin.
- The exit-code convention above — settle in the A4 traits-wiring step. Flag surfaced here for
  traceability; not resolved by this image.
