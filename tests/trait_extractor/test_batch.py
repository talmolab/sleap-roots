"""Tests for the batch driver, failure isolation, and the module CLI."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import pydantic
import pytest
from sleap_roots_contracts import RUN_MANIFEST_FILENAME, ResultEnvelope

from trait_extractor.extractor import extract_batch

_FIXTURE_TREE = Path("tests/data/rice_3do_pipeline_output")


def _write_run_manifest(
    directory: Path, scan_keys: Iterable[str], pipeline_run_id: str = "local-abc123"
) -> None:
    """Write a run_manifest.json into ``directory`` scoping to ``scan_keys``."""
    payload = {"pipeline_run_id": pipeline_run_id, "scan_keys": list(scan_keys)}
    (directory / RUN_MANIFEST_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )


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


def test_duplicate_scan_key_across_manifests_reported(tmp_path):
    """Two manifests declaring the same scan_key are refused, not silently clobbered."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    # Two copies of the same valid scan in sibling directories -> same scan_key.
    shutil.copytree(_FIXTURE_TREE / "scanYR39SJX", in_dir / "a" / "scanYR39SJX")
    shutil.copytree(_FIXTURE_TREE / "scanYR39SJX", in_dir / "b" / "scanYR39SJX")

    result = extract_batch(in_dir, out_dir)
    assert not result.ok
    # First occurrence succeeds; the collision is reported, not written over silently.
    assert result.succeeded == ["scanYR39SJX"]
    assert [k for k, _ in result.failed] == ["scanYR39SJX"]
    assert "duplicate scan_key" in result.failed[0][1]
    assert (out_dir / "scanYR39SJX.result.json").exists()


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


def test_no_manifest_falls_back_to_unscoped_rglob(tmp_path):
    """No run_manifest.json anywhere -> both fixture scans process, as before this change."""
    result = extract_batch(_FIXTURE_TREE, tmp_path)
    assert result.ok
    assert set(result.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    assert not (tmp_path / RUN_MANIFEST_FILENAME).exists()


def test_manifest_scoping_both_scans_in_scope_matches_current_output(tmp_path):
    """A manifest scoping to exactly the two fixture scans matches the no-manifest output."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"])

    baseline_dir = tmp_path / "baseline"
    baseline = extract_batch(_FIXTURE_TREE, baseline_dir)
    result = extract_batch(in_dir, out_dir)

    assert result.ok
    assert set(result.succeeded) == set(baseline.succeeded)
    for scan_key in result.succeeded:
        assert (out_dir / f"{scan_key}.result.json").read_bytes() == (
            baseline_dir / f"{scan_key}.result.json"
        ).read_bytes()


def test_manifest_scoping_excludes_out_of_scope_scan(tmp_path):
    """A manifest naming only one scan leaves the other completely untouched."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI"])

    result = extract_batch(in_dir, out_dir)

    assert result.ok
    assert result.succeeded == ["scan0K9E8BI"]
    assert result.skipped == []
    assert [k for k, _ in result.failed] == []
    assert (out_dir / "scan0K9E8BI.result.json").exists()
    assert not (out_dir / "scanYR39SJX.result.json").exists()


def test_manifest_declares_scan_key_with_no_predictions_json(tmp_path):
    """A manifest-declared scan_key with no matching predictions.json is a failure."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanMISSING"])

    result = extract_batch(in_dir, out_dir)

    assert not result.ok
    assert "scan0K9E8BI" in result.succeeded
    assert [k for k, _ in result.failed] == ["scanMISSING"]
    assert "scanMISSING" in result.failed[0][1]


def test_manifest_scoping_duplicate_in_scope_scan_key_is_a_failure(tmp_path):
    """Two candidates for the same in-scope scan_key are a failure, not a silent pick."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    shutil.copytree(_FIXTURE_TREE / "scanYR39SJX", in_dir / "a" / "scanYR39SJX")
    shutil.copytree(_FIXTURE_TREE / "scanYR39SJX", in_dir / "b" / "scanYR39SJX")
    _write_run_manifest(in_dir, ["scanYR39SJX"])

    result = extract_batch(in_dir, out_dir)

    assert not result.ok
    assert result.succeeded == ["scanYR39SJX"]
    assert [k for k, _ in result.failed] == ["scanYR39SJX"]
    assert "duplicate scan_key" in result.failed[0][1]


def test_invalid_manifest_aborts_batch(tmp_path):
    """A present-but-invalid run_manifest.json raises before any scan is processed."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, [])  # empty scan_keys is invalid

    with pytest.raises(pydantic.ValidationError):
        extract_batch(in_dir, out_dir)
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_manifest_present_input_dir_equals_output_dir_does_not_crash(tmp_path, caplog):
    """input_dir == output_dir does not crash the batch (copy-forward same-file case).

    Also asserts NO warning was logged: the same-file no-op guard in
    copy_run_manifest_forward should fire cleanly here, not extract_batch's separate
    `except OSError` safety net (which would also prevent a crash, but via a
    shutil.SameFileError caught after the fact, logging a warning) -- this distinguishes
    which of the two defense layers actually handled this specific case.
    """
    shutil.copytree(_FIXTURE_TREE, tmp_path, dirs_exist_ok=True)
    _write_run_manifest(tmp_path, ["scan0K9E8BI", "scanYR39SJX"])

    with caplog.at_level("WARNING"):
        result = extract_batch(tmp_path, tmp_path)

    assert result.ok
    assert set(result.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    assert caplog.text == ""


def test_copy_forward_failure_does_not_discard_already_computed_result(
    tmp_path, monkeypatch, caplog
):
    """A copy-forward OSError logs a warning but doesn't discard the batch result.

    Copy-forward is best-effort infrastructure for write-back, not part of this
    batch's own computed result -- a disk/permission error there must not crash the
    batch or discard the already-computed (and already durably written) results.
    """
    import trait_extractor.extractor

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"])

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trait_extractor.extractor, "copy_run_manifest_forward", _boom)

    with caplog.at_level("WARNING"):
        result = extract_batch(in_dir, out_dir)

    assert result.ok
    assert set(result.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    assert (out_dir / "scan0K9E8BI.result.json").exists()
    assert "failed to copy run_manifest.json" in caplog.text
    assert str(in_dir.as_posix()) in caplog.text
    assert str(out_dir.as_posix()) in caplog.text


def test_manifest_copied_forward_into_output_dir(tmp_path):
    """The manifest is copied forward into output_dir after a successful batch."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"])

    extract_batch(in_dir, out_dir)

    dest = out_dir / RUN_MANIFEST_FILENAME
    assert dest.exists()
    assert dest.read_bytes() == (in_dir / RUN_MANIFEST_FILENAME).read_bytes()


def test_manifest_scoped_scan_is_also_skipped_on_second_run(tmp_path):
    """Scoping and skip-if-done compose: a scoped scan is skipped on a second run."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"])

    first = extract_batch(in_dir, out_dir)
    assert first.ok
    assert set(first.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}

    second = extract_batch(in_dir, out_dir)
    assert second.ok
    assert set(second.skipped) == {"scan0K9E8BI", "scanYR39SJX"}


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


def test_module_cli_reports_skipped_scans_on_second_run(tmp_path):
    """A second CLI invocation over unchanged inputs reports skips, not just ok/FAIL."""
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = tmp_path / "out"
    fixture_tree = repo_root / _FIXTURE_TREE

    def _run():
        return subprocess.run(
            [sys.executable, "-m", "trait_extractor", str(fixture_tree), str(out_dir)],
            cwd=repo_root,
            env={**os.environ, "PYTHONPATH": str(repo_root)},
            capture_output=True,
            text=True,
        )

    first = _run()
    assert first.returncode == 0, first.stderr
    assert "0 skipped" in first.stderr
    assert "skip  scan0K9E8BI" not in first.stdout
    assert "skip  scanYR39SJX" not in first.stdout

    second = _run()
    assert second.returncode == 0, second.stderr
    assert "skip  scan0K9E8BI" in second.stdout
    assert "skip  scanYR39SJX" in second.stdout
    assert "2 skipped" in second.stderr
