"""Tests for provenance assembly, envelope emission, and golden regression."""

import importlib.metadata
import json
from pathlib import Path

import pandas as pd
import sleap_roots
from sleap_roots.trait_pipelines import YoungerMonocotPipeline
from sleap_roots_contracts import ResultEnvelope
from sleap_roots_contracts.identity import compute_idempotency_key

from trait_extractor.envelope import build_provenance, read_existing_identity
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
    prov = build_provenance(manifest, sidecar, sidecar.to_resolved_params())

    assert prov.predict_models == [a.model for a in manifest.artifacts]
    assert prov.predict_code_sha == manifest.predict_code_sha
    assert prov.predict_output_params == {"peak_threshold": 0.2}
    assert prov.params.values == {"species": "rice", "mode": "cylinder", "age": 3}
    assert prov.traits_sleap_roots_version == sleap_roots.__version__
    assert prov.contract_version == importlib.metadata.version("sleap-roots-contracts")
    assert prov.contract_version == "0.1.0a7"
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
    prov = build_provenance(manifest, sidecar, sidecar.to_resolved_params())
    assert prov.traits_code_sha == ""
    assert prov.traits_container_digest == ""


def test_idempotency_key_matches_helper_and_is_stable():
    """idempotency_key is non-empty, matches the helper, and is stable."""
    manifest, sidecar = _manifest_and_sidecar()
    prov = build_provenance(manifest, sidecar, sidecar.to_resolved_params())
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
    prov2 = build_provenance(manifest, sidecar, sidecar.to_resolved_params())
    assert prov2.idempotency_key == prov.idempotency_key


def test_age_encoding_does_not_change_idempotency_key():
    """age 3 vs "3" yield the same full Provenance.idempotency_key (not just param_hash)."""
    from trait_extractor.manifest import ScanMetadata

    manifest = load_manifest(_MANIFEST)

    def _key(age):
        sidecar = ScanMetadata(
            scan_key=manifest.scan_key,
            image_ids=["i"],
            images_checksum="c",
            params={"species": "rice", "mode": "cylinder", "age": age},
        )
        prov = build_provenance(manifest, sidecar, sidecar.to_resolved_params())
        return prov.idempotency_key

    assert _key(3) == _key("3") == _key(3.0)
    assert _key(3)


def test_extract_scan_emits_valid_envelope(tmp_path):
    """extract_scan writes a valid envelope that round-trips."""
    envelope = extract_scan(_MANIFEST, _SIDECAR, tmp_path)
    assert isinstance(envelope, ResultEnvelope)
    assert envelope.blobs == []

    out = tmp_path / "scan0K9E8BI.result.json"
    assert out.exists()
    reloaded = ResultEnvelope.model_validate_json(out.read_text())
    assert reloaded == envelope

    # RPC acceptance rules checkable locally.
    assert envelope.provenance.idempotency_key
    scan_keys = {tv.scan_key for tv in envelope.traits} | {envelope.provenance.scan_key}
    assert scan_keys == {"scan0K9E8BI"}


def test_extract_scan_recompute_is_byte_stable(tmp_path):
    """Two independent (non-skipped) recomputes over identical inputs are byte-stable.

    Each call targets its own fresh output_dir, so neither has a pre-existing envelope
    to skip against -- both are genuine recomputes (skip-if-done, added later, would
    otherwise turn a same-output_dir second call into a skip rather than a recompute,
    which would prove nothing about recompute-determinism).
    """
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    extract_scan(_MANIFEST, _SIDECAR, first_dir)
    extract_scan(_MANIFEST, _SIDECAR, second_dir)

    first = (first_dir / "scan0K9E8BI.result.json").read_bytes()
    second = (second_dir / "scan0K9E8BI.result.json").read_bytes()
    # LF-only so it is byte-identical across OSes.
    assert b"\r\n" not in first
    assert second == first


def test_read_existing_identity_returns_none_when_missing(tmp_path):
    """No {scan_key}.result.json in output_dir -> None."""
    assert read_existing_identity(tmp_path, "scan0K9E8BI") is None


def test_read_existing_identity_returns_key_and_version_when_present(tmp_path):
    """A valid pre-written envelope -> its exact (idempotency_key, contract_version)."""
    envelope = extract_scan(_MANIFEST, _SIDECAR, tmp_path)
    identity = read_existing_identity(tmp_path, "scan0K9E8BI")
    assert identity == (
        envelope.provenance.idempotency_key,
        envelope.provenance.contract_version,
    )


def test_read_existing_identity_returns_none_on_invalid_json(tmp_path, caplog):
    """A {scan_key}.result.json with invalid JSON -> None, not an exception."""
    out = tmp_path / "scanBAD.result.json"
    out.write_text("{not valid json", encoding="utf-8")
    assert read_existing_identity(tmp_path, "scanBAD") is None


def test_read_existing_identity_returns_none_on_schema_invalid_json(tmp_path):
    """Valid JSON that fails ResultEnvelope validation -> None, not an exception."""
    out = tmp_path / "scanBAD.result.json"
    out.write_text(json.dumps({"not": "an envelope"}), encoding="utf-8")
    assert read_existing_identity(tmp_path, "scanBAD") is None


def test_read_existing_identity_logs_warning_on_corrupt_file(tmp_path, caplog):
    """A corrupt pre-existing file logs a warning naming the scan_key and the error."""
    out = tmp_path / "scanBAD.result.json"
    out.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        read_existing_identity(tmp_path, "scanBAD")
    assert "scanBAD" in caplog.text
    # The underlying exception is included, not just a generic message -- this is what
    # lets an operator distinguish "this file's content is corrupt" from a genuine I/O
    # failure (see test_read_existing_identity_returns_none_on_read_error).
    assert "Invalid JSON" in caplog.text


def test_read_existing_identity_does_not_warn_when_missing(tmp_path, caplog):
    """A plain missing file (ordinary first run) does not log a warning."""
    with caplog.at_level("WARNING"):
        read_existing_identity(tmp_path, "scanNEW")
    assert caplog.text == ""


def test_read_existing_identity_returns_none_on_invalid_utf8(tmp_path, caplog):
    """A {scan_key}.result.json with invalid UTF-8 bytes -> None, logs a warning."""
    out = tmp_path / "scanBAD.result.json"
    out.write_bytes(b"\xff\xfe not valid utf-8")
    with caplog.at_level("WARNING"):
        assert read_existing_identity(tmp_path, "scanBAD") is None
    assert "scanBAD" in caplog.text


def test_read_existing_identity_returns_none_on_read_error(
    tmp_path, monkeypatch, caplog
):
    """An OSError while reading a present file -> None, not a crash, logs a warning.

    Covers a read-time anomaly (e.g. a permissions issue, or a TOCTOU race where the
    file is removed between the is_file() check and the read) on a file that briefly
    existed -- without this, such an error would misclassify the scan as FAILED instead
    of "not done", per trait_extractor.envelope.read_existing_identity's documented
    contract.

    The pre-seeded file is a genuinely valid envelope (via a real extract_scan call),
    not schema-invalid content -- so a `None` result can only be explained by the
    injected PermissionError, not by an independent, unrelated validation failure (a
    schema-invalid fixture like `"{}"` would make this test pass even if the
    monkeypatch silently failed to apply).
    """
    extract_scan(_MANIFEST, _SIDECAR, tmp_path)
    out = tmp_path / "scan0K9E8BI.result.json"
    assert out.is_file()

    def _boom(self, *args, **kwargs):
        raise PermissionError("simulated read failure")

    monkeypatch.setattr(Path, "read_text", _boom)
    with caplog.at_level("WARNING"):
        assert read_existing_identity(tmp_path, "scan0K9E8BI") is None
    assert "scan0K9E8BI" in caplog.text
    assert "simulated read failure" in caplog.text


def test_read_existing_identity_returns_none_on_is_file_error(
    tmp_path, monkeypatch, caplog
):
    """A PermissionError from is_file() itself -> None, not a crash, logs a warning.

    Distinct from test_read_existing_identity_returns_none_on_read_error, which
    patches read_text() -- that test alone does NOT prove is_file() is covered by the
    same error handling: is_file() swallows ENOENT/ENOTDIR internally but a
    PermissionError from its own stat() call is not one of those and would propagate
    uncaught if is_file() sat outside the try/except (verified: this exact code shape
    regressed once, silently, before this test existed).
    """
    extract_scan(_MANIFEST, _SIDECAR, tmp_path)

    def _boom(self, *args, **kwargs):
        raise PermissionError("simulated stat failure")

    monkeypatch.setattr(Path, "is_file", _boom)
    with caplog.at_level("WARNING"):
        assert read_existing_identity(tmp_path, "scan0K9E8BI") is None
    assert "scan0K9E8BI" in caplog.text
    assert "simulated stat failure" in caplog.text


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
    # atol=1e-7 with pandas' default rtol=1e-5 (effective tol 1e-7 + 1e-5*|value|),
    # mirroring the existing test_younger_monocot_pipeline convention.
    pd.testing.assert_frame_equal(
        computed[trait_cols].reset_index(drop=True),
        golden_row[trait_cols].reset_index(drop=True),
        check_exact=False,
        atol=1e-7,
    )
