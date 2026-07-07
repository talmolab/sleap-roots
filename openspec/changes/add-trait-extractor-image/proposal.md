## Why

The A3-traits **ResultEnvelope emitter** merged in #254, but nothing publishes the
top-level `trait_extractor/` service as a pullable image. A4's Argo `trait-extractor`
template still pulls the legacy `registry.gitlab.com/salk-tm/sleap-roots-traits:latest`.
This is the first scoped-out fast-follow: package the merged service as a distributable
**GHCR image** so A4 can pull `ghcr.io/talmolab/sleap-roots-trait-extractor`.

Design: `docs/superpowers/specs/2026-07-06-a3-traits-result-envelope-emitter-design.md` §7
"Container / CI organization" + its reconciliation appendix. Tracking issue:
talmolab/sleap-roots#256 (umbrella #250 is closed — emitter done).

## What Changes

- Add a root **`trait-extractor.Dockerfile`** — base
  `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` (predict's base; satisfies contracts'
  `>=3.11`). `WORKDIR /app`; **COPYs `sleap_roots/`** (required — `uv sync` builds the
  `sleap-roots` project and resolves its dynamic `attr = "sleap_roots.__version__"` version)
  **and `trait_extractor/`** (excluded from the wheel, so copied — not pip-installed — and
  reached via `PYTHONPATH`); installs the library + `sleap-roots-contracts` via `uv sync
  --frozen --no-dev --extra extractor`; sets headless-matplotlib env (`MPLBACKEND=Agg`,
  `MPLCONFIGDIR=/tmp/matplotlib`); and `ENTRYPOINT ["python","-m","trait_extractor"]` (exec
  form, so the batch driver's exit code propagates to the container for Argo). The `uv sync`
  pins `--python 3.12` (a `<3.11` interpreter would silently drop the contracts dep).
- **Bake the build commit into `ENV SRT_TRAITS_CODE_SHA`** (Dockerfile `ARG` + workflow
  `build-args: SRT_TRAITS_CODE_SHA=${{ github.sha }}`), so emitted envelopes carry a non-empty
  `provenance.traits_code_sha`. The idempotency key hashes `traits_code_sha`, and the CLI has no
  flag to inject it — a `""` default would let a traits-code change silently collide with a
  prior "done" result under A4's dedup. (`SRT_TRAITS_CONTAINER_DIGEST` is not baked — unknown at
  build time, not a key input; stays A4's runtime concern.)
- Add a slim **`[project.optional-dependencies] extractor`** = `["sleap-roots-contracts==0.1.0a3 ; python_version >= '3.11'", "pyyaml"]`
  so the image installs contracts + `pyyaml` (declared explicitly since the pipeline chooser's
  `import yaml` uses it directly) **without** the heavy dev/docs group (mkdocs/pytest/black/twine/…).
  Re-lock and commit `uv.lock` (CI is `uv sync --frozen`).
- Add a slim `tests/trait_extractor/` **packaging guard test** that asserts the `extractor`
  extra exists and pins contracts (CI-enforced, since `ci.yml` triggers on `pyproject.toml`).
- Add **`.dockerignore`** (near-copy of predict's) so the build context excludes
  `.git`/`tests`/`.venv`/caches; the library build inputs (`pyproject.toml`, `uv.lock`,
  `README.md`), `sleap_roots/`, and `trait_extractor/` remain available to `COPY`.
- Add **`.github/workflows/docker-trait-extractor.yml`** — a near-copy of predict's
  `docker-build.yml` with: a **distinct image identity** `ghcr.io/talmolab/sleap-roots-trait-extractor`
  (explicit `images:`, **not** `${{ github.repository }}` which is the library's identity);
  a `permissions: { contents: read, packages: write }` block; the **same `paths:` filter on
  both the `push:[main]` and `pull_request` triggers** (`sleap_roots/**` — the library is baked
  into the image from source — + `trait_extractor/**` + `trait-extractor.Dockerfile` +
  `pyproject.toml` + `uv.lock` + `.dockerignore` + the workflow file); **no `release:` trigger
  and no `type=semver` tags**. Build-only on PRs (no GHCR
  login/push); build **and** push on `main`/dispatch (`latest` on `main` + immutable
  `sha-<sha>`); `file: trait-extractor.Dockerfile`, `context: .`.
- Update **docs** (single source of truth = the dev doc): add a `## Container image` section to
  `docs/dev/trait-extractor-service.md` (image name + `docker run` + tag scheme) and fix its
  now-stale "GHCR image is a fast-follow" note; point the existing README service section at it;
  add a `docs/changelog.md` `### Added` entry; and reconcile the emitter design doc's §7/§9/§3
  + appendix for internal consistency (§9 currently lists the image as OUT).

Explicitly **untouched**: the PyPI `build.yml` and the test `ci.yml` (which already
lint/test `trait_extractor/` from #254). Because the new workflow drops the `release:` trigger
and path-filters its `push`, a library PyPI release and the service image never trigger each
other. No change to the published `sleap-roots` **runtime** dependencies (`pip install
sleap-roots` stays pure); contracts stays a dev/test/**container** dependency, now also
reachable as an opt-in `pip install sleap-roots[extractor]` extra.

## Impact

- **Affected specs:** new capability `trait-extractor-image`. (No change to
  `result-envelope-output`, which owns the emitter behavior.)
- **Affected code:** new `trait-extractor.Dockerfile`, `.dockerignore`,
  `.github/workflows/docker-trait-extractor.yml`; a new `tests/trait_extractor/` packaging
  guard test; `pyproject.toml` (add the `extractor` extra); `uv.lock` (re-locked, same commit);
  `docs/dev/trait-extractor-service.md` (SSOT `## Container image` section + stale-note fix);
  `README.md` (one-line pointer); `docs/changelog.md`; the emitter design doc's §3/§7/§9 +
  reconciliation appendix.
- **No runtime dependency change** to the published library and **no change to the
  `sleap_roots` package or the wheel/sdist** — the image installs the same locked deps CI
  already uses, plus the contracts extra.
- **Out of scope (tracked elsewhere):** the A4 Argo template that pulls the image
  (`sleap-roots-pipeline`, "traits wiring" row); the write-back RPC call + MinIO/Box blob
  upload + `BlobRef` locations; image-grain / multi-plant-plate per-plant grain (#252). A4
  write-back is separately blocked on bloom#393 (RPC must accept bare `0.1.0a3`) — **not** a
  dependency of building this image.
