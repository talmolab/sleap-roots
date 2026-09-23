# Trait-extractor service

`trait_extractor/` is the **A3-traits trait-extractor**: a service that consumes
[`sleap-roots-predict`](https://github.com/talmolab/sleap-roots-predict)'s per-scan output,
computes root traits with the `sleap_roots` pipelines, and emits a per-scan
[`sleap-roots-contracts`](https://github.com/talmolab/sleap-roots-contracts) `ResultEnvelope`
as JSON for Bloom write-back.

## Service boundary (why it is not in the wheel)

The extractor is an *application* that uses the `sleap_roots` library — not part of the
library's public API. It lives at the **repository root** (`trait_extractor/`, a flat package)
and is deliberately **excluded from the published `sleap-roots` wheel**
(`[tool.setuptools.packages.find] include = ["sleap_roots"]` never discovers a top-level
sibling). Consequently:

- `pip install sleap-roots` stays pure — `sleap-roots-contracts` is a **dev/test/container**
  dependency only, never a runtime dependency of the library. A CI-enforced AST guard test
  asserts the `sleap_roots` library source never imports `sleap_roots_contracts`.
- Tests import it via the repo root on `sys.path` (`[tool.pytest.ini_options] pythonpath = ["."]`).
- It is intentionally absent from the generated API reference (`docs/gen_ref_pages.py` walks
  only `sleap_roots/`).

## Inputs

Per scan, discovered recursively under an input directory (mirroring predict's per-scan
`out_dir/{scan_key}/` batch layout — manifest, sidecar, and `.slp` co-located):

1. **`{scan_key}.predictions.json`** — predict's `PredictionManifest` (`schema_version: "1"`):
   `scan_key`, `plant_qr_code`, a list of `artifacts` (each `{root_type, model_id, model:
   ModelRef, slp_path, checksum, file_size}`), `predict_inference_config`,
   `predict_output_params`, `predict_code_sha`, `predict_container_digest`. Consumed via a
   lightweight consumer-side model (no `sleap-roots-predict` runtime dependency).
2. **`{scan_key}.scan_metadata.json`** — the `ScanMetadata` sidecar (defined by this service),
   supplying the idempotency inputs predict defers.

### `ScanMetadata` sidecar schema (for the Bloom/downloader populator)

```json
{
  "scan_key": "scan0K9E8BI",
  "image_ids": ["cyl_img_0001", "cyl_img_0002"],
  "images_checksum": "sha256:...",
  "params": { "species": "rice", "mode": "cylinder", "age": 3 }
}
```

| Field | Type | Notes |
|-------|------|-------|
| `scan_key` | `str` | Must equal the manifest's `scan_key` (and its filename stem). |
| `image_ids` | `list[str]` | Bloom `cyl_images` IDs. Carried in `Provenance.inputs`; validated against `cyl_images` by Bloom's write-back RPC (not by the idempotency key). |
| `images_checksum` | `str` | Feeds the idempotency key; its stability is the downloader's responsibility. |
| `params.species` | `str` | Selection + `param_hash`. |
| `params.mode` | `str` | Selection + `param_hash`. |
| `params.age` | int-coercible | **Canonicalized to an integer** for the hash — `3`, `3.0`, and `"3"` are equivalent; `3.5`/`"abc"`/`true` are rejected. |

Only `{species, mode, age}` feed the idempotency key: `ResolvedParams.values` is built as that
**closed set** with `age` coerced to `int`, so a differently-encoded age or an extra `params`
key never changes the key (which would otherwise break Bloom's first-writer-wins dedup).

> **Operational requirement — build identity in the idempotency key.** The key hashes
> `predict_code_sha` and `traits_code_sha`, both of which resolve **fail-soft to `""`** when
> unset. So in a production/container run, set `SRT_TRAITS_CODE_SHA` (traits build) and ensure
> predict stamps `predict_code_sha` — otherwise a `sleap-roots` version bump that changes trait
> numbers would produce a byte-different envelope under the **same** idempotency key, and Bloom
> would dedup two scientifically-distinct results as one run. (`traits_sleap_roots_version` is
> recorded in `Provenance` for audit but is not itself a key input — see
> [sleap-roots-contracts#14](https://github.com/talmolab/sleap-roots-contracts/issues/14).)

## Output

One `{scan_key}.result.json` per scan — a `ResultEnvelope` = `Provenance` + `list[TraitValue]`
(`grain="scan"`; `NaN`/`inf` → `None`) + `blobs=[]` (blob locations are filled downstream at
upload). `Provenance.contract_version` is the pinned bare `sleap-roots-contracts` version
(`0.1.0a9`); `produced_at` is left `None` so re-emitting over identical inputs is byte-stable.
The run identity (below) is deliberately **not** stamped into `Provenance.pipeline_run_id`, so
envelopes stay byte-identical across runs.

## Usage

```bash
python -m trait_extractor <input_dir> <output_dir>
```

**Run identity and the run manifest.** A run manifest (`sleap_roots_contracts.RunManifest`,
written by `bloomctl` and copied forward by each pipeline stage — see
[talmolab/sleap-roots-pipeline#37](https://github.com/talmolab/sleap-roots-pipeline/issues/37)
and [#71](https://github.com/talmolab/sleap-roots-pipeline/issues/71)) scopes discovery to exactly
its `scan_keys`. It is read once, from the top level of `input_dir`, by contracts'
`load_run_manifest`. Which file is used depends on the run identity, taken from
`ARGO_WORKFLOW_NAME` via contracts' `pipeline_run_id_from_env()` (stripped; unset or blank means
no identity). The cluster template sets it; `local-WSL2-*` templates and plain local runs do not.

| Run identity | Manifest used | If none exists |
|---|---|---|
| none | `run_manifest.json` | unscoped discovery (below) |
| `<id>` | `run_manifest.<id>.json`, else the legacy `run_manifest.json` (`allow_legacy=True` until [pipeline#82](https://github.com/talmolab/sleap-roots-pipeline/issues/82)) | **crash** (`RunManifestMissingError`, exit `1`) — a run that knows its identity never widens to the whole tree |

A per-run manifest whose `pipeline_run_id` names a different run is a crash
(`RunManifestIdentityError`), and so is an `ARGO_WORKFLOW_NAME` unusable as a filename. Two
cases are honored but logged as a `WARNING`:
- a legacy manifest read under a known identity that names another run (a stale file, or a later
  concurrent chunk's merge);
- per-run manifests present in a run with no identity, which ignores them and discovers
  everything.

With a manifest, a `.predictions.json` present but out of scope is silently ignored — this is the
contamination-prevention this manifest exists for — and a scan whose output already matches (both
`idempotency_key` and `contract_version`) is skipped rather than recomputed. After the batch, the
manifest is republished into `output_dir` **under the name it was read from**, byte-identical to
what was scoped against and with the source's permissions. The publish is atomic (a dot-prefixed
temp file, removed on failure) and best-effort: a failure is logged and never costs the batch its
results.

With no manifest, discovery falls back to recursively finding every `{scan_key}.predictions.json`
under `input_dir` (the original, pre-manifest behavior — used by local/non-pipeline runs); if that
unscoped fallback discovers **zero** manifests, the driver raises rather than reporting an empty
run as a silent success (an empty or misconfigured input mount is an operator error, not a
no-op). A missing `input_dir` raises `FileNotFoundError` naming it. In both cases, each manifest's
sidecar is paired and one envelope is written per scan to `output_dir`. Per-scan failures (bad
manifest, missing sidecar, a manifest-declared scan_key with no matching predictions.json,
incompatible/unsupported pipeline) are isolated and reported without discarding the successful or
skipped envelopes.

**Exit codes** (Argo-ready, per [sleap-roots#259](https://github.com/talmolab/sleap-roots/issues/259)):

| Code | Meaning |
|---|---|
| `0` | Full success — every discovered scan succeeded or was skipped. |
| `3` | **Partial** — the batch ran to completion but one or more scans isolated-failed (per-scan failures caught inside the driver's own loop). An Argo caller should treat this as a completed run with partial failures, not retry the whole batch. |
| `1` | **Crash** — an exception escaped the batch entirely before it could return a result at all: an invalid run manifest, no manifest for a known run identity, a per-run manifest naming another run, an unusable `ARGO_WORKFLOW_NAME`, a missing `input_dir`, the empty-input guard above, or any other bug. A `Batch aborted: …` line is logged first. A real pod-level failure; Argo's `retryStrategy` retries it, though the resolution failures are deterministic and fail again. |
| `2` | *(not used by this driver)* — reserved: `argparse` already exits `2` on a CLI usage error, before the batch ever runs. |
| `143` | `SIGTERM` received (Argo preemption/cancellation) — the process exits promptly (`128 + SIGTERM`) rather than waiting out `terminationGracePeriodSeconds`. Per-scan writes are already atomic (temp→rename) and the batch is idempotent on retry, so this loses no completed envelope. |

## Container image

Published to GHCR as `ghcr.io/talmolab/sleap-roots-trait-extractor` — an identity distinct
from the library image `ghcr.io/talmolab/sleap-roots`. Built from the root
`trait-extractor.Dockerfile` (base `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`); it installs
the library + `sleap-roots-contracts` via the slim `extractor` extra
(`uv sync --frozen --no-dev --extra extractor`), copies `trait_extractor/` in (it is not
pip-installable), and runs headless (`MPLBACKEND=Agg`). The `ENTRYPOINT` is
`python -m trait_extractor`, so run it with two positional args:

```bash
docker run --rm \
  -v /abs/path/to/predict_output:/in \
  -v /abs/path/to/results:/out \
  ghcr.io/talmolab/sleap-roots-trait-extractor:latest /in /out
```

Passing `-e ARGO_WORKFLOW_NAME=<id>` gives the run an identity. It then needs
`run_manifest.<id>.json` (or the legacy `run_manifest.json`) in `/in`, or it exits `1` (see
[Usage](#usage)). Note that `latest` emits `contract_version = "0.1.0a9"` envelopes, which Bloom's
write-back rejects until it accepts that version (see Downstream below).

On Windows in Git Bash, prefix the command with `MSYS_NO_PATHCONV=1` (or use PowerShell with
`C:\…` absolute host paths) so the `/in` and `/out` container paths aren't rewritten to host
paths by MSYS.

Tags: `latest` tracks `main`; every pushed build also publishes an immutable `sha-<commit>`
tag, and the workflow surfaces the `@sha256:…` digest in its run summary — pin the digest (or
the `sha-` tag) for reproducible downstream runs. The image bakes its build commit into
`SRT_TRAITS_CODE_SHA` so emitted envelopes carry a non-empty `provenance.traits_code_sha`; a
runtime `SRT_TRAITS_CODE_SHA` env value overrides the baked default.

Built and pushed by [`.github/workflows/docker-trait-extractor.yml`](https://github.com/talmolab/sleap-roots/blob/main/.github/workflows/docker-trait-extractor.yml),
path-filtered to the image's inputs and independent of the PyPI release (`build.yml`):
build-only on PRs, build + push on `main`.

## Notes & follow-ups

- **Consumer/predict coupling** — the consumer `PredictionManifest` duplicates predict's shape
  (pinned to `schema_version: "1"`); a skip-if-unimportable cross-check test validates it
  against predict's real output when predict is installed.
- **Pipeline compatibility** — a class-keyed `PIPELINE_REQUIRED_ROOTS` map (a workaround for a
  missing public pipeline API, [#251](https://github.com/talmolab/sleap-roots/issues/251))
  checks `required ⊆ loaded`; multi-plant / plate pipelines are rejected for scan-grain
  emission ([#252](https://github.com/talmolab/sleap-roots/issues/252)).
- **Downstream** — the trait-extractor ships as the GHCR image
  `ghcr.io/talmolab/sleap-roots-trait-extractor` (see [Container image](#container-image)).
  Bloom's write-back RPC (`insert_cyl_result_envelope`) originally required
  `contract_version == "0.1.0a3"` ([bloom#393](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/393),
  closed by [bloom PR #399](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/pull/399)),
  and was re-pinned to exactly `0.1.0a7` by
  [bloom PR #766](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/pull/766)
  ([bloom#685](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/685), closed
  2026-09-10). It accepts a single literal, so this repo's bump to `0.1.0a9` reopens the same
  class of blocker: every a9 envelope is rejected until Bloom accepts `0.1.0a9`. **An image built
  from `0.1.0a9` must not be applied to the cluster's trait-extractor template until that Bloom
  change is applied to the database the cluster write-back targets.** See the Deploy gate in
  `openspec/changes/adopt-contracts-run-manifest-reader/proposal.md`.
