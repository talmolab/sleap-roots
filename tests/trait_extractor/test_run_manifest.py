"""Tests for loading + copying forward the run-scoping ``RunManifest``."""

import json
from pathlib import Path

import pydantic
import pytest
from sleap_roots_contracts import RUN_MANIFEST_FILENAME, RunManifest

from trait_extractor.run_manifest import copy_run_manifest_forward, load_run_manifest


def _write_manifest(directory: Path, **overrides) -> Path:
    payload = {
        "pipeline_run_id": "local-abc123",
        "scan_keys": ["scan0K9E8BI", "scanYR39SJX"],
    }
    payload.update(overrides)
    path = directory / RUN_MANIFEST_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_run_manifest_returns_none_when_absent(tmp_path):
    """No run_manifest.json under input_dir -> None, not an exception."""
    assert load_run_manifest(tmp_path) is None


def test_load_run_manifest_parses_valid_file(tmp_path):
    """A valid run_manifest.json loads with its exact fields."""
    _write_manifest(tmp_path)
    manifest = load_run_manifest(tmp_path)
    assert isinstance(manifest, RunManifest)
    assert manifest.pipeline_run_id == "local-abc123"
    assert manifest.scan_keys == ["scan0K9E8BI", "scanYR39SJX"]


def test_load_run_manifest_raises_on_invalid_manifest(tmp_path):
    """A present-but-invalid manifest (empty scan_keys) raises loudly at load time."""
    _write_manifest(tmp_path, scan_keys=[])
    with pytest.raises(pydantic.ValidationError):
        load_run_manifest(tmp_path)


def test_copy_manifest_forward_writes_into_output_dir(tmp_path):
    """The manifest is copied forward byte-identical, creating output_dir if missing."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    source = _write_manifest(in_dir)

    copy_run_manifest_forward(in_dir, out_dir)

    dest = out_dir / RUN_MANIFEST_FILENAME
    assert dest.exists()
    assert dest.read_bytes() == source.read_bytes()


def test_copy_manifest_forward_is_a_noop_when_manifest_absent(tmp_path):
    """No run_manifest.json under input_dir -> copy-forward is a no-op, no error."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    copy_run_manifest_forward(in_dir, out_dir)

    assert not (out_dir / RUN_MANIFEST_FILENAME).exists()


def test_copy_manifest_forward_overwrites_a_different_prior_manifest(tmp_path):
    """A different pre-existing manifest in output_dir is overwritten, not merged."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    source = _write_manifest(in_dir)
    _write_manifest(out_dir, pipeline_run_id="stale-run")

    copy_run_manifest_forward(in_dir, out_dir)

    dest = out_dir / RUN_MANIFEST_FILENAME
    assert dest.read_bytes() == source.read_bytes()
    assert RunManifest.model_validate_json(dest.read_text()).pipeline_run_id == (
        "local-abc123"
    )


def test_copy_manifest_forward_is_a_noop_when_input_and_output_are_the_same(tmp_path):
    """input_dir == output_dir does not raise shutil.SameFileError."""
    _write_manifest(tmp_path)

    copy_run_manifest_forward(tmp_path, tmp_path)  # must not raise

    assert (tmp_path / RUN_MANIFEST_FILENAME).is_file()
