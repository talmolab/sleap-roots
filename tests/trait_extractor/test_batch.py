"""Tests for the batch driver, failure isolation, and the module CLI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from sleap_roots_contracts import ResultEnvelope

from trait_extractor.extractor import extract_batch

_FIXTURE_TREE = Path("tests/data/rice_3do_pipeline_output")


def test_batch_emits_one_envelope_per_scan(tmp_path):
    """extract_batch discovers both nested scans and writes one envelope each."""
    result = extract_batch(_FIXTURE_TREE, tmp_path)
    assert result.ok
    assert set(result.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    for scan_key in ("scan0K9E8BI", "scanYR39SJX"):
        out = tmp_path / f"{scan_key}.result.json"
        assert out.exists()
        ResultEnvelope.model_validate_json(out.read_text())


def _make_bad_scan_missing_slp(dest: Path):
    """Create a per-scan dir whose manifest names a nonexistent .slp."""
    good = _FIXTURE_TREE / "scanYR39SJX"
    dest.mkdir()
    manifest = json.loads((good / "scanYR39SJX.predictions.json").read_text())
    manifest["scan_key"] = "scanBAD"
    manifest["artifacts"] = manifest["artifacts"][:1]
    manifest["artifacts"][0]["slp_path"] = "does_not_exist.slp"
    (dest / "scanBAD.predictions.json").write_text(json.dumps(manifest))
    sidecar = json.loads((good / "scanYR39SJX.scan_metadata.json").read_text())
    sidecar["scan_key"] = "scanBAD"
    (dest / "scanBAD.scan_metadata.json").write_text(json.dumps(sidecar))


def test_one_scan_failure_does_not_abort_batch(tmp_path):
    """A failing scan is reported but the valid scan's envelope is still written."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    shutil.copytree(_FIXTURE_TREE / "scanYR39SJX", in_dir / "scanYR39SJX")
    _make_bad_scan_missing_slp(in_dir / "scanBAD")

    result = extract_batch(in_dir, out_dir)
    assert not result.ok
    assert result.succeeded == ["scanYR39SJX"]
    assert [k for k, _ in result.failed] == ["scanBAD"]
    assert (out_dir / "scanYR39SJX.result.json").exists()
    assert not (out_dir / "scanBAD.result.json").exists()


def test_missing_sidecar_reported(tmp_path):
    """A manifest with no co-located sidecar is reported and skipped."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    dest = in_dir / "scanYR39SJX"
    dest.mkdir(parents=True)
    good = _FIXTURE_TREE / "scanYR39SJX"
    for name in (
        "scanYR39SJX.predictions.json",
        "scanYR39SJX.model123.rootprimary.slp",
        "scanYR39SJX.model123.rootcrown.slp",
    ):
        shutil.copy(good / name, dest / name)
    # No sidecar copied.
    result = extract_batch(in_dir, out_dir)
    assert not result.ok
    assert [k for k, _ in result.failed] == ["scanYR39SJX"]
    assert "sidecar" in result.failed[0][1]


def test_stem_scan_key_mismatch_reported(tmp_path):
    """A manifest filename stem that disagrees with scan_key is reported."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    dest = in_dir / "scanWRONG"
    dest.mkdir(parents=True)
    good = _FIXTURE_TREE / "scanYR39SJX"
    manifest = json.loads((good / "scanYR39SJX.predictions.json").read_text())
    # filename stem is scanWRONG but scan_key stays scanYR39SJX
    (dest / "scanWRONG.predictions.json").write_text(json.dumps(manifest))
    (dest / "scanWRONG.scan_metadata.json").write_text(
        (good / "scanYR39SJX.scan_metadata.json").read_text()
    )
    result = extract_batch(in_dir, out_dir)
    assert not result.ok
    assert "scan_key" in result.failed[0][1]


def test_module_cli_writes_envelopes(tmp_path):
    """`python -m trait_extractor <in> <out>` writes the envelopes and exits 0."""
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "out"
    fixture_tree = repo_root / _FIXTURE_TREE
    proc = subprocess.run(
        [sys.executable, "-m", "trait_extractor", str(fixture_tree), str(out_dir)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "scan0K9E8BI.result.json").exists()
    assert (out_dir / "scanYR39SJX.result.json").exists()
