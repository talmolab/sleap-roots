"""Tests for provenance assembly, envelope emission, and golden regression."""

import importlib.metadata
from pathlib import Path

import pandas as pd
import sleap_roots
from sleap_roots.trait_pipelines import YoungerMonocotPipeline
from sleap_roots_contracts import ResultEnvelope
from sleap_roots_contracts.identity import compute_idempotency_key

from trait_extractor.envelope import build_provenance
from trait_extractor.extractor import extract_scan
from trait_extractor.loading import load_series
from trait_extractor.manifest import load_manifest, load_scan_metadata

_RICE_DIR = Path("tests/data/rice_3do_pipeline_output/scan0K9E8BI")
_MANIFEST = _RICE_DIR / "scan0K9E8BI.predictions.json"
_SIDECAR = _RICE_DIR / "scan0K9E8BI.scan_metadata.json"
_GOLDEN = Path("tests/data/rice_3do/rice_3do.batch_traits.csv")


def _manifest_and_sidecar():
    manifest = load_manifest(_MANIFEST)
    sidecar = load_scan_metadata(_SIDECAR, manifest.scan_key)
    return manifest, sidecar


def test_provenance_fields_and_contract_version():
    """Provenance carries predict_models, canonical params, and the pinned version."""
    manifest, sidecar = _manifest_and_sidecar()
    prov = build_provenance(manifest, sidecar)

    assert prov.predict_models == [a.model for a in manifest.artifacts]
    assert prov.predict_code_sha == manifest.predict_code_sha
    assert prov.predict_output_params == {"peak_threshold": 0.2}
    assert prov.params.values == {"species": "rice", "mode": "cylinder", "age": 3}
    assert prov.traits_sleap_roots_version == sleap_roots.__version__
    assert prov.contract_version == importlib.metadata.version(
        "sleap-roots-contracts"
    )
    assert prov.contract_version == "0.1.0a3"
    assert not prov.contract_version.startswith("v")
    assert prov.produced_at is None
    assert prov.pipeline_run_id is None
    assert prov.worker_request_id is None
    assert prov.argo_workflow_uid is None


def test_build_identity_fail_soft(monkeypatch):
    """traits_code_sha/container_digest resolve to '' with no arg and no env."""
    monkeypatch.delenv("SRT_TRAITS_CODE_SHA", raising=False)
    monkeypatch.delenv("SRT_TRAITS_CONTAINER_DIGEST", raising=False)
    manifest, sidecar = _manifest_and_sidecar()
    prov = build_provenance(manifest, sidecar)
    assert prov.traits_code_sha == ""
    assert prov.traits_container_digest == ""


def test_idempotency_key_matches_helper_and_is_stable():
    """idempotency_key is non-empty, matches the helper, and is stable."""
    manifest, sidecar = _manifest_and_sidecar()
    prov = build_provenance(manifest, sidecar)
    expected = compute_idempotency_key(
        scan_key=prov.scan_key,
        images_checksum=prov.inputs.images_checksum,
        models=[
            (m.registry_id, m.version, m.weights_checksum) for m in prov.predict_models
        ],
        param_hash=prov.params.param_hash,
        predict_code_sha=prov.predict_code_sha,
        traits_code_sha=prov.traits_code_sha,
        predict_output_params=prov.predict_output_params,
    )
    assert prov.idempotency_key
    assert prov.idempotency_key == expected
    prov2 = build_provenance(manifest, sidecar)
    assert prov2.idempotency_key == prov.idempotency_key


def test_extract_scan_emits_valid_byte_stable_envelope(tmp_path):
    """extract_scan writes a valid envelope that round-trips and is byte-stable."""
    envelope = extract_scan(_MANIFEST, _SIDECAR, tmp_path)
    assert isinstance(envelope, ResultEnvelope)
    assert envelope.blobs == []

    out = tmp_path / "scan0K9E8BI.result.json"
    assert out.exists()
    reloaded = ResultEnvelope.model_validate_json(out.read_text())
    assert reloaded == envelope

    # RPC acceptance rules checkable locally.
    assert envelope.provenance.idempotency_key
    scan_keys = {tv.scan_key for tv in envelope.traits} | {
        envelope.provenance.scan_key
    }
    assert scan_keys == {"scan0K9E8BI"}

    # Byte-stable re-emission.
    first = out.read_bytes()
    extract_scan(_MANIFEST, _SIDECAR, tmp_path)
    assert out.read_bytes() == first


def test_golden_regression(tmp_path):
    """Computed traits match the committed rice_3do golden (trait columns only)."""
    manifest = load_manifest(_MANIFEST)
    series = load_series(manifest, _RICE_DIR)
    computed = YoungerMonocotPipeline().compute_batch_traits([series])

    golden = pd.read_csv(_GOLDEN)
    golden_key = manifest.scan_key.removeprefix("scan")  # "0K9E8BI"
    golden_row = golden[golden["plant_name"] == golden_key].reset_index(drop=True)
    assert len(golden_row) == 1

    trait_cols = [c for c in computed.columns if c != "plant_name"]
    pd.testing.assert_frame_equal(
        computed[trait_cols].reset_index(drop=True),
        golden_row[trait_cols].reset_index(drop=True),
        check_exact=False,
        atol=1e-7,
    )
