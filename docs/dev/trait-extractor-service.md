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

## Notes & follow-ups

- **Consumer/predict coupling** — the consumer `PredictionManifest` duplicates predict's shape
  (pinned to `schema_version: "1"`); a skip-if-unimportable cross-check test validates it
  against predict's real output when predict is installed.
- **Pipeline compatibility** — a class-keyed `PIPELINE_REQUIRED_ROOTS` map (a workaround for a
  missing public pipeline API, [#251](https://github.com/talmolab/sleap-roots/issues/251))
  checks `required ⊆ loaded`; multi-plant / plate pipelines are rejected for scan-grain
  emission ([#252](https://github.com/talmolab/sleap-roots/issues/252)).
- **Downstream** — the GHCR image + Argo template are a fast-follow slice; Bloom's RPC must
  accept `contract_version == "0.1.0a3"`
  ([bloom#393](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/393)).
