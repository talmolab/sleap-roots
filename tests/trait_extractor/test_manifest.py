"""Tests for consuming predict's manifest + the ScanMetadata sidecar."""

import json
from pathlib import Path

import pytest

from trait_extractor.manifest import (
    PredictionManifest,
    ScanMetadata,
    load_manifest,
    load_scan_metadata,
    resolve_artifact_paths,
)

_FIXTURE_DIR = Path("tests/data/rice_3do_pipeline_output/scan0K9E8BI")
_MANIFEST = _FIXTURE_DIR / "scan0K9E8BI.predictions.json"
_SIDECAR = _FIXTURE_DIR / "scan0K9E8BI.scan_metadata.json"


def test_manifest_parses_and_paths_resolve():
    """The manifest validates, artifacts is a list, and slp paths resolve."""
    manifest = load_manifest(_MANIFEST)
    assert isinstance(manifest, PredictionManifest)
    assert manifest.scan_key == "scan0K9E8BI"
    assert [a.root_type for a in manifest.artifacts] == ["primary", "crown"]
    paths = resolve_artifact_paths(manifest, _FIXTURE_DIR)
    assert set(paths) == {"primary", "crown"}
    for p in paths.values():
        assert p.exists()


def test_manifest_preserves_full_model_ref():
    """Each artifact carries the full ModelRef (feeds the idempotency key)."""
    manifest = load_manifest(_MANIFEST)
    primary = next(a for a in manifest.artifacts if a.root_type == "primary")
    assert primary.model.registry_id == "rice-primary"
    assert primary.model.version == "123"
    assert primary.model.sleap_nn_version == "0.3.0"


def test_unrecognized_schema_version_rejected(tmp_path):
    """A manifest with schema_version != '1' is rejected."""
    data = json.loads(_MANIFEST.read_text())
    data["schema_version"] = "2"
    bad = tmp_path / "bad.predictions.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(Exception):
        load_manifest(bad)


def test_unknown_root_type_rejected(tmp_path):
    """An artifact root_type outside RootType is rejected."""
    data = json.loads(_MANIFEST.read_text())
    data["artifacts"][0]["root_type"] = "tuber"
    bad = tmp_path / "bad.predictions.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(Exception):
        load_manifest(bad)


def test_malformed_json_rejected(tmp_path):
    """Non-JSON content raises."""
    bad = tmp_path / "bad.predictions.json"
    bad.write_text("{not json")
    with pytest.raises(Exception):
        load_manifest(bad)


def test_missing_required_field_rejected(tmp_path):
    """A manifest missing a required field raises."""
    data = json.loads(_MANIFEST.read_text())
    del data["scan_key"]
    bad = tmp_path / "bad.predictions.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(Exception):
        load_manifest(bad)


def test_missing_slp_file_rejected(tmp_path):
    """A manifest naming a .slp absent from its directory raises FileNotFoundError."""
    manifest = load_manifest(_MANIFEST)
    with pytest.raises(FileNotFoundError):
        resolve_artifact_paths(manifest, tmp_path)


def test_sidecar_yields_inputs_and_canonical_params():
    """The sidecar yields an InputRef and a closed {species, mode, age:int} params."""
    meta = load_scan_metadata(_SIDECAR, expected_scan_key="scan0K9E8BI")
    assert isinstance(meta, ScanMetadata)
    inputs = meta.to_input_ref()
    assert inputs.image_ids == ["cyl_img_0001", "cyl_img_0002"]
    params = meta.to_resolved_params()
    assert set(params.values) == {"species", "mode", "age"}
    assert params.values["age"] == 3
    assert isinstance(params.values["age"], int)


def test_scan_key_mismatch_rejected():
    """A sidecar whose scan_key disagrees with the manifest raises."""
    with pytest.raises(ValueError):
        load_scan_metadata(_SIDECAR, expected_scan_key="scanOTHER")


def _resolved_key(age):
    """Build the idempotency key contribution of a sidecar age via ResolvedParams."""
    meta = ScanMetadata(
        scan_key="s",
        image_ids=["i"],
        images_checksum="c",
        params={"species": "rice", "mode": "cylinder", "age": age},
    )
    return meta.to_resolved_params().param_hash


@pytest.mark.parametrize("age", [3, 3.0, "3"])
def test_age_encoding_does_not_change_param_hash(age):
    """age encoded as 3, 3.0, or '3' produces the same param_hash."""
    assert _resolved_key(age) == _resolved_key(3)


def test_extra_params_key_does_not_change_param_hash():
    """An extra sidecar params key does not change the param_hash (closed set)."""
    with_extra = ScanMetadata(
        scan_key="s",
        image_ids=["i"],
        images_checksum="c",
        params={"species": "rice", "mode": "cylinder", "age": 3, "notes": "hi"},
    )
    assert with_extra.to_resolved_params().param_hash == _resolved_key(3)


@pytest.mark.parametrize("bad_age", [True, 3.5, "3.5", "abc"])
def test_non_integer_age_rejected(bad_age):
    """bool and non-whole ages are rejected during canonicalization."""
    meta = ScanMetadata(
        scan_key="s",
        image_ids=["i"],
        images_checksum="c",
        params={"species": "rice", "mode": "cylinder", "age": bad_age},
    )
    with pytest.raises(ValueError):
        meta.to_resolved_params()
