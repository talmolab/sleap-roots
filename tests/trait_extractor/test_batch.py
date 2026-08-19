"""Tests for the batch driver, failure isolation, and the module CLI."""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
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


def test_empty_unscoped_input_dir_raises(tmp_path):
    """An empty, unscoped input_dir raises rather than silently succeeding.

    No run_manifest.json and zero *.predictions.json anywhere -> extract_batch must
    not return a vacuous BatchResult(ok=True); a misconfigured/empty mount is an
    operator error, not a successful no-op run.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    with pytest.raises(RuntimeError, match=re.escape(in_dir.as_posix())):
        extract_batch(in_dir, out_dir)
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_scoped_input_with_no_matching_files_still_reports_per_scan_failure(tmp_path):
    """A scoped run_manifest.json with zero matching files is unaffected by the new guard.

    Regression pin: when a run_manifest.json IS present, an in-scope scan_key with no
    matching file was already recorded as a per-scan failure before this change (see
    test_manifest_declares_scan_key_with_no_predictions_json) -- the new unscoped
    empty-input guard must not change this scoped path's behavior.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    _write_run_manifest(in_dir, ["scanA", "scanB"])

    result = extract_batch(in_dir, out_dir)

    assert not result.ok
    assert set(k for k, _ in result.failed) == {"scanA", "scanB"}


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


def test_shrinking_scope_orphans_prior_result_and_logs_a_warning(tmp_path, caplog):
    """A scan_key dropped from a later, narrower manifest is not touched, but logged.

    Round-3 review found this case completely unaddressed: run 1 scopes to both
    fixture scans; run 2's manifest narrows to just one. The dropped scan's prior
    {scan_key}.result.json is left exactly as run 1 wrote it -- not reprocessed, not
    reported in any BatchResult bucket -- but now logged as an orphan so it's at least
    traceable, rather than silently indistinguishable from a current result.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"])

    first = extract_batch(in_dir, out_dir)
    assert first.ok
    assert set(first.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    orphan_bytes_before = (out_dir / "scanYR39SJX.result.json").read_bytes()

    # Run 2's manifest narrows scope to just one of the two previously-in-scope scans.
    _write_run_manifest(in_dir, ["scan0K9E8BI"])

    with caplog.at_level("WARNING"):
        second = extract_batch(in_dir, out_dir)

    assert second.ok
    assert second.skipped == ["scan0K9E8BI"]
    # The dropped scan's prior output is untouched: not reprocessed, not reported.
    assert (out_dir / "scanYR39SJX.result.json").read_bytes() == orphan_bytes_before
    assert "scanYR39SJX" not in second.succeeded
    assert "scanYR39SJX" not in second.skipped
    assert not any(k == "scanYR39SJX" for k, _ in second.failed)
    # But it IS now traceable via a warning.
    assert "scanYR39SJX" in caplog.text
    assert "outside this run's scope" in caplog.text


def test_case_insensitive_scan_key_collision_reported(tmp_path):
    """Two scan_keys differing only by case are refused, not silently clobbered.

    On a case-insensitive filesystem (default on Windows/macOS, this repo's dev
    platform), "ScanYR39SJX" and "scanyr39sjx" would both write to the same
    {scan_key}.result.json filename despite being different strings -- `seen`'s
    exact-string keys alone would never detect this. Refused the same way an exact
    duplicate is refused.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    good = _FIXTURE_TREE / "scanYR39SJX"
    shutil.copytree(good, in_dir / "a" / "scanYR39SJX")

    # A second candidate whose scan_key differs from the first ONLY by case.
    dest = in_dir / "b" / "scanyr39sjx"
    dest.mkdir(parents=True)
    manifest = json.loads((good / "scanYR39SJX.predictions.json").read_text())
    manifest["scan_key"] = "scanyr39sjx"
    (dest / "scanyr39sjx.predictions.json").write_text(json.dumps(manifest))
    sidecar = json.loads((good / "scanYR39SJX.scan_metadata.json").read_text())
    sidecar["scan_key"] = "scanyr39sjx"
    (dest / "scanyr39sjx.scan_metadata.json").write_text(json.dumps(sidecar))

    result = extract_batch(in_dir, out_dir)

    assert not result.ok
    assert result.succeeded == ["scanYR39SJX"]
    assert [k for k, _ in result.failed] == ["scanyr39sjx"]
    assert "collides case-insensitively" in result.failed[0][1]


def test_manifest_scoping_duplicate_of_an_already_skipped_scan_key_is_a_failure(
    tmp_path,
):
    """A duplicate scan_key can appear in BOTH `skipped` and `failed` simultaneously.

    The same accepted trade-off `test_manifest_scoping_duplicate_in_scope_scan_key_is_a_failure`
    exercises for `succeeded` -- the first-discovered candidate's own outcome is recorded,
    then the collision itself is ALSO recorded as a failure -- applies identically when
    that first candidate is a skip (not a fresh success). Both buckets are populated for
    the same scan_key; `BatchResult.ok` is still correctly False either way.
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE / "scanYR39SJX", in_dir / "a" / "scanYR39SJX")
    _write_run_manifest(in_dir, ["scanYR39SJX"])

    first = extract_batch(in_dir, out_dir)
    assert first.ok
    assert first.succeeded == ["scanYR39SJX"]

    # A second, duplicate candidate appears (e.g. a stale leftover directory) alongside
    # the first, unchanged one -- "a" sorts before "b", so "a" is still discovered first
    # and, since nothing about it changed, skips; "b" is then a duplicate collision.
    shutil.copytree(_FIXTURE_TREE / "scanYR39SJX", in_dir / "b" / "scanYR39SJX")

    result = extract_batch(in_dir, out_dir)

    assert not result.ok
    assert result.skipped == ["scanYR39SJX"]
    assert result.succeeded == []
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


def test_copy_forward_failure_does_not_discard_missing_scan_key_failures(
    tmp_path, monkeypatch, caplog
):
    """A copy-forward OSError doesn't discard missing-scan_key failures either.

    The missing-scan_key bookkeeping loop and the copy-forward call are both inside
    the same `if scope is not None:` block, with the failures appended strictly before
    the copy-forward call -- this test makes that ordering guarantee independently
    verifiable (both failure sources present in the same run) rather than only
    inferable from reading the source.
    """
    import trait_extractor.extractor

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanMISSING"])

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(trait_extractor.extractor, "copy_run_manifest_forward", _boom)

    with caplog.at_level("WARNING"):
        result = extract_batch(in_dir, out_dir)

    assert not result.ok
    assert result.succeeded == ["scan0K9E8BI"]
    assert [k for k, _ in result.failed] == ["scanMISSING"]
    assert "failed to copy run_manifest.json" in caplog.text


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


def _run_module_cli(repo_root, in_dir, out_dir):
    """Invoke `python -m trait_extractor <in> <out>` as a subprocess."""
    return subprocess.run(
        [sys.executable, "-m", "trait_extractor", str(in_dir), str(out_dir)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )


def test_module_cli_exits_partial_code_on_isolated_scan_failure(tmp_path):
    """`python -m trait_extractor <in> <out>` exits 3 when a scan isolated-fails.

    Round-3 review found the exit-code logic (`return 0 if result.ok else 1` in
    `__main__.main()`) had NO test enforcing it -- hardcoding `main()` to always
    `return 0` left the entire suite green, since every other subprocess-level CLI
    test only exercises the all-succeeding/all-skipped happy path. This is exactly
    the kind of bug that would make a broken Argo pod look successful. Exit code 3
    (not 1) distinguishes "isolated per-scan failure, batch completed" from "crash".
    """
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    fixture_tree = repo_root / _FIXTURE_TREE
    shutil.copytree(fixture_tree / "scanYR39SJX", in_dir / "scanYR39SJX")
    _make_bad_scan_missing_slp(in_dir / "scanBAD")

    proc = _run_module_cli(repo_root, in_dir, out_dir)

    assert proc.returncode == 3
    assert "FAIL" in proc.stderr
    assert (out_dir / "scanYR39SJX.result.json").exists()


def test_module_cli_exits_partial_code_on_scoped_missing_scan_key(tmp_path):
    """A manifest-scoped scan_key with no matching file also exits 3, not just extract_batch."""
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    fixture_tree = repo_root / _FIXTURE_TREE
    shutil.copytree(fixture_tree, in_dir)
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanMISSING"])

    proc = _run_module_cli(repo_root, in_dir, out_dir)

    assert proc.returncode == 3
    assert "scanMISSING" in proc.stderr


def test_module_cli_exits_crash_code_on_empty_input(tmp_path):
    """An empty, unscoped input_dir exits 1 (crash), with a clean logged message.

    The exception still propagates after logging (matching Python's default
    uncaught-exception exit code), so a traceback is also present -- the
    log-quality fix adds a clean one-line message ahead of it, it doesn't
    suppress the traceback entirely.
    """
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    proc = _run_module_cli(repo_root, in_dir, out_dir)

    assert proc.returncode == 1
    assert "Batch aborted:" in proc.stderr
    assert in_dir.as_posix() in proc.stderr


def test_module_cli_exits_crash_code_on_invalid_run_manifest(tmp_path):
    """An invalid run_manifest.json exits 1 (crash), with a clean logged message.

    This already crashed today via pydantic.ValidationError; this test pins the
    exit code explicitly for the first time and asserts a clean logged line now
    precedes the (still-present) traceback.
    """
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    fixture_tree = repo_root / _FIXTURE_TREE
    shutil.copytree(fixture_tree, in_dir)
    _write_run_manifest(in_dir, [])  # empty scan_keys is invalid

    proc = _run_module_cli(repo_root, in_dir, out_dir)

    assert proc.returncode == 1
    assert "Batch aborted:" in proc.stderr


def test_module_cli_usage_error_exits_two_unrelated_to_partial_code(tmp_path):
    """A CLI usage error exits 2 via argparse, unrelated to the 0/1/3 convention."""
    repo_root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, "-m", "trait_extractor"],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2


def test_handle_sigterm_raises_systemexit_143():
    """The SIGTERM handler exits 143, called directly (no subprocess, no timing)."""
    import signal as signal_module

    from trait_extractor.__main__ import _handle_sigterm

    with pytest.raises(SystemExit) as exc_info:
        _handle_sigterm(signal_module.SIGTERM, None)
    assert exc_info.value.code == 143


def _duplicate_scan(source_dir: Path, dest_dir: Path, new_scan_key: str) -> None:
    """Copy a valid fixture scan into ``dest_dir`` under a new, distinct scan_key.

    The .slp file(s) are copied verbatim (basenames unchanged -- slp_path is
    resolved as a basename relative to the manifest's own directory, so it need
    not match the new scan_key). Both the manifest's and sidecar's scan_key
    fields are rewritten consistently, matching the new filename stem.
    """
    dest_dir.mkdir(parents=True)
    orig_stem = source_dir.name
    manifest = json.loads((source_dir / f"{orig_stem}.predictions.json").read_text())
    sidecar = json.loads((source_dir / f"{orig_stem}.scan_metadata.json").read_text())
    for artifact in manifest["artifacts"]:
        slp_name = artifact["slp_path"]
        shutil.copy(source_dir / slp_name, dest_dir / slp_name)
    manifest["scan_key"] = new_scan_key
    sidecar["scan_key"] = new_scan_key
    (dest_dir / f"{new_scan_key}.predictions.json").write_text(json.dumps(manifest))
    (dest_dir / f"{new_scan_key}.scan_metadata.json").write_text(json.dumps(sidecar))


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGTERM delivery to a subprocess is not POSIX-equivalent on Windows",
)
def test_module_cli_sigterm_exits_promptly_and_preserves_completed_output(tmp_path):
    """SIGTERM during a multi-scan batch exits 143 and leaves completed output intact."""
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    fixture_tree = repo_root / _FIXTURE_TREE

    for i in range(20):
        for orig in ("scan0K9E8BI", "scanYR39SJX"):
            new_key = f"{orig}_{i:03d}"
            _duplicate_scan(fixture_tree / orig, in_dir / new_key, new_key)

    proc = subprocess.Popen(
        [sys.executable, "-m", "trait_extractor", str(in_dir), str(out_dir)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if out_dir.exists() and list(out_dir.glob("*.result.json")):
                break
            time.sleep(0.1)
        else:
            pytest.fail("no *.result.json appeared within the poll bound")

        proc.send_signal(signal.SIGTERM)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 143, stderr
    result_files = list(out_dir.glob("*.result.json"))
    assert result_files
    for result_file in result_files:
        ResultEnvelope.model_validate_json(result_file.read_text())


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
