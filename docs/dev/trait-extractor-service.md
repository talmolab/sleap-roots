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
(`0.1.0a3`); `produced_at` is left `None` so re-emitting over identical inputs is byte-stable.

## Usage

```bash
python -m trait_extractor <input_dir> <output_dir>
```

Discovers each `{scan_key}.predictions.json` under `input_dir`, pairs its sidecar, and writes
one envelope per scan to `output_dir`. Per-scan failures (bad manifest, missing sidecar,
incompatible/unsupported pipeline) are isolated and reported; the process exits non-zero if any
scan failed, without discarding the successful envelopes.

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
  Still fast-follow: A4's Argo template that pulls it (must rewrite the step's `args:` to the
  two positional dirs and pin the image), and Bloom's RPC accepting
  `contract_version == "0.1.0a3"`
  ([bloom#393](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/393)).
