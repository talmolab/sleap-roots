"""Tests for the batch driver, failure isolation, and the module CLI."""

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import pydantic
import pytest
from sleap_roots_contracts import (
    RUN_MANIFEST_FILENAME,
    ResultEnvelope,
    RunManifestIdentityError,
    RunManifestMissingError,
    run_manifest_filename,
)

from trait_extractor.extractor import extract_batch

_FIXTURE_TREE = Path("tests/data/rice_3do_pipeline_output")


def _write_run_manifest(
    directory: Path,
    scan_keys: Iterable[str],
    *,
    pipeline_run_id: str = "local-abc123",
    filename: Optional[str] = None,
) -> Path:
    """Write a run manifest into ``directory`` scoping to ``scan_keys``.

    Args:
        directory: Directory to write the manifest into.
        scan_keys: The manifest's ``scan_keys``.
        pipeline_run_id: The ``pipeline_run_id`` recorded *inside* the manifest.
        filename: The manifest's filename. Defaults to ``RUN_MANIFEST_FILENAME`` (the
            legacy name). Pass ``run_manifest_filename(id)`` for a per-run name; the name
            and the content's ``pipeline_run_id`` are independent so identity-mismatch
            cases can be expressed.

    Returns:
        The path written.
    """
    payload = {"pipeline_run_id": pipeline_run_id, "scan_keys": list(scan_keys)}
    path = directory / (filename or RUN_MANIFEST_FILENAME)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_per_run_manifest(
    directory: Path, scan_keys: Iterable[str], run_id: str
) -> Path:
    """Write ``run_manifest.<run_id>.json`` naming ``run_id`` inside it too."""
    return _write_run_manifest(
        directory,
        scan_keys,
        pipeline_run_id=run_id,
        filename=run_manifest_filename(run_id),
    )


def _warnings_from_extractor(caplog) -> list:
    """The ``WARNING``-or-above records logged by ``trait_extractor.extractor``."""
    return [
        r
        for r in caplog.records
        if r.name == "trait_extractor.extractor" and r.levelno >= logging.WARNING
    ]


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


def test_nonexistent_unscoped_input_dir_raises(tmp_path):
    """A totally missing (not just empty) input_dir raises, naming the directory.

    Round-4/PR review found this variant was only verified manually, never pinned by a
    test. Since contracts 0.1.0a9 the run-manifest reader itself raises
    FileNotFoundError("run manifest directory does not exist") before discovery runs --
    a mis-mounted input is reported as such, rather than reaching the empty-input
    RuntimeError guard as it did before.
    """
    in_dir = tmp_path / "does_not_exist"
    out_dir = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match=re.escape(in_dir.as_posix())):
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


@pytest.mark.parametrize("run_id", [None, "wf-a"], ids=["legacy", "per-run"])
def test_invalid_manifest_aborts_batch(tmp_path, run_id):
    """A present-but-invalid run manifest raises before any scan is processed.

    Covers both the legacy name (no run identity) and the per-run name (identity known).
    """
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    shutil.copytree(_FIXTURE_TREE, in_dir)
    # empty scan_keys is invalid
    if run_id is None:
        _write_run_manifest(in_dir, [])
    else:
        _write_per_run_manifest(in_dir, [], run_id)

    with pytest.raises(pydantic.ValidationError):
        extract_batch(in_dir, out_dir, pipeline_run_id=run_id)
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_manifest_present_input_dir_equals_output_dir_does_not_crash(tmp_path, caplog):
    """input_dir == output_dir does not crash the batch (copy-forward same-file case).

    Also asserts NO warning was logged: the same-path no-op guard in
    copy_run_manifest_forward should fire cleanly here, not extract_batch's separate
    `except OSError` safety net (which would also prevent a crash, but only by logging a
    warning after a failed publish) -- this distinguishes which of the two defense
    layers actually handled this specific case.
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


# --- Run identity + per-run manifest resolution (sleap-roots-pipeline#71) -------------


def _copy_fixture(tmp_path: Path, name: str = "in") -> Path:
    """Copy the two-scan fixture tree to ``tmp_path / name`` and return it."""
    in_dir = tmp_path / name
    shutil.copytree(_FIXTURE_TREE, in_dir)
    return in_dir


def test_argo_workflow_name_is_cleared_for_tests():
    """The autouse conftest fixture removes any inherited run identity.

    Vacuous wherever the variable is never exported (CI); its red step is manual --
    run with ``ARGO_WORKFLOW_NAME`` exported and the fixture removed.
    """
    assert "ARGO_WORKFLOW_NAME" not in os.environ


def test_per_run_manifest_wins_over_legacy(tmp_path):
    """With an identity, run_manifest.<id>.json is used over a legacy run_manifest.json."""
    in_dir = _copy_fixture(tmp_path)
    _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"], pipeline_run_id="wf-a")

    result = extract_batch(in_dir, tmp_path / "out", pipeline_run_id="wf-a")

    assert result.ok
    assert result.succeeded == ["scan0K9E8BI"]


def test_concurrent_runs_are_each_scoped_to_their_own_manifest(tmp_path):
    """Two runs sharing one input tree are each scoped to their own per-run manifest."""
    in_dir = _copy_fixture(tmp_path)
    _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    _write_per_run_manifest(in_dir, ["scanYR39SJX"], "wf-b")

    a = extract_batch(in_dir, tmp_path / "out-a", pipeline_run_id="wf-a")
    b = extract_batch(in_dir, tmp_path / "out-b", pipeline_run_id="wf-b")

    assert a.succeeded == ["scan0K9E8BI"]
    assert b.succeeded == ["scanYR39SJX"]


def test_known_identity_without_manifest_raises_missing(tmp_path):
    """A run that knows its identity but finds no manifest fails loud, not unscoped."""
    in_dir = _copy_fixture(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(RunManifestMissingError, match="wf-a"):
        extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_known_identity_with_empty_input_raises_missing_not_runtime_error(tmp_path):
    """The empty-unscoped-input RuntimeError guard is reachable only without an identity."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()

    with pytest.raises(RunManifestMissingError):
        extract_batch(in_dir, tmp_path / "out", pipeline_run_id="wf-a")


def test_per_run_manifest_naming_another_run_raises_identity_error(tmp_path):
    """run_manifest.wf-a.json whose content names wf-b is someone else's manifest."""
    in_dir = _copy_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _write_run_manifest(
        in_dir,
        ["scan0K9E8BI"],
        pipeline_run_id="wf-b",
        filename=run_manifest_filename("wf-a"),
    )

    with pytest.raises(RunManifestIdentityError, match="wf-b"):
        extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_unusable_run_id_raises_value_error(tmp_path):
    """An id that can't be a filename component raises a bare ValueError."""
    in_dir = _copy_fixture(tmp_path)
    out_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="not usable as a filename component") as info:
        extract_batch(in_dir, out_dir, pipeline_run_id="../x")
    # Not merely a ValueError subclass such as pydantic.ValidationError.
    assert type(info.value) is ValueError
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_invalid_per_run_manifest_does_not_fall_back_to_valid_legacy(tmp_path):
    """An invalid per-run manifest raises; it never falls through to the legacy file."""
    in_dir = _copy_fixture(tmp_path)
    out_dir = tmp_path / "out"
    _write_per_run_manifest(in_dir, [], "wf-a")
    _write_run_manifest(in_dir, ["scan0K9E8BI"], pipeline_run_id="wf-a")

    with pytest.raises(pydantic.ValidationError):
        extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_legacy_manifest_naming_another_run_is_honored_and_warned(tmp_path, caplog):
    """A stale legacy manifest under a known identity is honored, but logged (D5)."""
    in_dir = _copy_fixture(tmp_path)
    _write_run_manifest(in_dir, ["scan0K9E8BI"], pipeline_run_id="wf-old")

    with caplog.at_level("WARNING"):
        result = extract_batch(in_dir, tmp_path / "out", pipeline_run_id="wf-a")

    assert result.succeeded == ["scan0K9E8BI"]
    matching = [
        r
        for r in _warnings_from_extractor(caplog)
        if all(s in r.getMessage() for s in (RUN_MANIFEST_FILENAME, "wf-old", "wf-a"))
    ]
    assert len(matching) == 1


def test_legacy_manifest_naming_this_run_is_not_warned(tmp_path, caplog):
    """A legacy manifest naming this very run triggers no stale-manifest warning.

    Regression guard: passes against a stub that never warns, by design.
    """
    in_dir = _copy_fixture(tmp_path)
    _write_run_manifest(in_dir, ["scan0K9E8BI"], pipeline_run_id="wf-a")

    with caplog.at_level("WARNING"):
        result = extract_batch(in_dir, tmp_path / "out", pipeline_run_id="wf-a")

    assert result.succeeded == ["scan0K9E8BI"]
    assert _warnings_from_extractor(caplog) == []


def test_per_run_manifests_without_identity_are_ignored_but_warned(tmp_path, caplog):
    """No identity: per-run manifests are never read, and the widening is logged (D7)."""
    in_dir = _copy_fixture(tmp_path)
    _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")

    with caplog.at_level("WARNING"):
        result = extract_batch(in_dir, tmp_path / "out", pipeline_run_id=None)

    assert set(result.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    matching = [
        r
        for r in _warnings_from_extractor(caplog)
        if "run_manifest.wf-a.json" in r.getMessage()
    ]
    assert len(matching) == 1


def test_run_identity_defaults_to_environment(tmp_path, monkeypatch):
    """Omitting pipeline_run_id resolves it via pipeline_run_id_from_env(), once."""
    import trait_extractor.extractor as extractor_module

    calls = []
    real = extractor_module.pipeline_run_id_from_env

    def _spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(extractor_module, "pipeline_run_id_from_env", _spy)

    # Whitespace-padded identity is stripped and selects the per-run manifest.
    per_run_in = _copy_fixture(tmp_path, "in-per-run")
    _write_per_run_manifest(per_run_in, ["scan0K9E8BI"], "wf-a")
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", " wf-a\n")
    calls.clear()
    result = extract_batch(per_run_in, tmp_path / "out-1")
    assert result.succeeded == ["scan0K9E8BI"]
    assert len(calls) == 1

    # A blank identity is no identity: only the legacy name is read.
    legacy_in = _copy_fixture(tmp_path, "in-legacy")
    _write_run_manifest(legacy_in, ["scanYR39SJX"])
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "   ")
    calls.clear()
    result = extract_batch(legacy_in, tmp_path / "out-2")
    assert result.succeeded == ["scanYR39SJX"]
    assert len(calls) == 1

    # An explicit argument (None or an id) never consults the environment.
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")
    calls.clear()
    extract_batch(legacy_in, tmp_path / "out-3", pipeline_run_id=None)
    extract_batch(per_run_in, tmp_path / "out-4", pipeline_run_id="wf-a")
    assert calls == []


def test_explicit_none_ignores_environment(tmp_path, monkeypatch):
    """pipeline_run_id=None opts out of the environment's identity.

    Regression guard: passes against a stub that never reads the environment, by
    design; its red is manual (resolve the environment on None and watch it raise
    RunManifestMissingError).
    """
    monkeypatch.setenv("ARGO_WORKFLOW_NAME", "wf-a")

    result = extract_batch(_FIXTURE_TREE, tmp_path / "out", pipeline_run_id=None)

    assert result.ok
    assert set(result.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}


def test_run_identity_is_not_stamped_into_envelopes(tmp_path):
    """Envelopes are byte-identical across run identities (spec: Provenance; design D8)."""
    in_dir = _copy_fixture(tmp_path)
    _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-b")
    legacy_in = _copy_fixture(tmp_path, "in-legacy")
    _write_run_manifest(legacy_in, ["scan0K9E8BI"])

    extract_batch(in_dir, tmp_path / "out-a", pipeline_run_id="wf-a")
    extract_batch(in_dir, tmp_path / "out-b", pipeline_run_id="wf-b")
    extract_batch(legacy_in, tmp_path / "out-none", pipeline_run_id=None)

    name = "scan0K9E8BI.result.json"
    a = (tmp_path / "out-a" / name).read_bytes()
    assert a == (tmp_path / "out-b" / name).read_bytes()
    assert a == (tmp_path / "out-none" / name).read_bytes()
    assert ResultEnvelope.model_validate_json(a).provenance.pipeline_run_id is None

    # A different run over the first run's output reuses it (skip-if-done still holds).
    again = extract_batch(in_dir, tmp_path / "out-a", pipeline_run_id="wf-b")
    assert again.skipped == ["scan0K9E8BI"]


def test_dangling_symlink_manifest_raises(tmp_path):
    """A dangling-symlink manifest is a broken tree, not an absent manifest."""
    in_dir = _copy_fixture(tmp_path)
    out_dir = tmp_path / "out"
    link = in_dir / RUN_MANIFEST_FILENAME
    try:
        os.symlink(in_dir / "does-not-exist.json", link)
    except OSError as exc:  # e.g. Windows without the symlink privilege
        pytest.skip(f"cannot create a symlink here: {exc}")

    with pytest.raises(FileNotFoundError, match=re.escape(link.as_posix())):
        extract_batch(in_dir, out_dir, pipeline_run_id=None)
    assert not out_dir.exists() or not list(out_dir.glob("*.result.json"))


def test_manifest_is_loaded_once_with_allow_legacy_true(tmp_path, monkeypatch):
    """extract_batch calls contracts' load_run_manifest once, with allow_legacy=True.

    Pins the call shape (and that the env sentinel never leaks through to contracts).
    The "one snapshot" guarantee itself is pinned by
    test_batch_forwards_loaded_snapshot_not_rewritten_source: a call count of 1 alone
    was already true before this change.
    """
    import trait_extractor.extractor as extractor_module

    calls = []
    real = extractor_module.load_run_manifest

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(extractor_module, "load_run_manifest", _spy)
    in_dir = _copy_fixture(tmp_path)
    _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    _write_run_manifest(in_dir, ["scan0K9E8BI"])

    extract_batch(in_dir, tmp_path / "out-1", pipeline_run_id="wf-a")
    assert calls == [((in_dir, "wf-a"), {"allow_legacy": True})]

    calls.clear()
    extract_batch(in_dir, tmp_path / "out-2")  # environment unset -> None
    assert calls == [((in_dir, None), {"allow_legacy": True})]


# --- Forwarding the loaded snapshot -------------------------------------------------


def test_batch_forwards_loaded_snapshot_not_rewritten_source(tmp_path, monkeypatch):
    """The forwarded manifest is the snapshot scoped against, not a later re-read."""
    import trait_extractor.extractor as extractor_module

    in_dir = _copy_fixture(tmp_path)
    source = _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    original = source.read_bytes()
    loaded_reads = []
    forwarded_reads = []
    real_load = extractor_module.load_run_manifest
    real_forward = extractor_module.copy_run_manifest_forward

    def load_then_rewrite(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        loaded_reads.append(loaded.read)
        # Another valid manifest lands at the same path after it was read.
        _write_per_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"], "wf-a")
        return loaded

    def spy_forward(read, *args, **kwargs):
        forwarded_reads.append(read)
        return real_forward(read, *args, **kwargs)

    monkeypatch.setattr(extractor_module, "load_run_manifest", load_then_rewrite)
    monkeypatch.setattr(extractor_module, "copy_run_manifest_forward", spy_forward)

    out_dir = tmp_path / "out"
    result = extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")

    assert result.succeeded == ["scan0K9E8BI"]
    assert source.read_bytes() != original
    assert (out_dir / "run_manifest.wf-a.json").read_bytes() == original
    assert len(forwarded_reads) == 1
    assert forwarded_reads[0] is loaded_reads[0]


def test_per_run_manifest_forwarded_under_its_own_name(tmp_path, caplog):
    """A per-run read is forwarded as run_manifest.<id>.json, never as the legacy name."""
    in_dir = _copy_fixture(tmp_path)
    source = _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    _write_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"], pipeline_run_id="wf-a")
    out_dir = tmp_path / "out"

    with caplog.at_level("WARNING"):
        extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")

    assert (out_dir / "run_manifest.wf-a.json").read_bytes() == source.read_bytes()
    assert not (out_dir / RUN_MANIFEST_FILENAME).exists()
    assert _warnings_from_extractor(caplog) == []


def test_concurrent_runs_forward_only_their_own_manifest(tmp_path):
    """Runs sharing an input tree each forward only their own per-run manifest."""
    in_dir = _copy_fixture(tmp_path)
    _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    _write_per_run_manifest(in_dir, ["scanYR39SJX"], "wf-b")

    extract_batch(in_dir, tmp_path / "out-a", pipeline_run_id="wf-a")
    extract_batch(in_dir, tmp_path / "out-b", pipeline_run_id="wf-b")

    assert sorted(p.name for p in (tmp_path / "out-a").glob("run_manifest*")) == [
        "run_manifest.wf-a.json"
    ]
    assert sorted(p.name for p in (tmp_path / "out-b").glob("run_manifest*")) == [
        "run_manifest.wf-b.json"
    ]


def test_concurrent_runs_sharing_output_dir_do_not_clobber_each_others_manifest(
    tmp_path, caplog
):
    """Two runs writing into one output_dir keep both per-run manifests intact."""
    in_dir = _copy_fixture(tmp_path)
    source_a = _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    source_b = _write_per_run_manifest(in_dir, ["scanYR39SJX"], "wf-b")
    out_dir = tmp_path / "out"

    a = extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")
    with caplog.at_level("WARNING"):
        b = extract_batch(in_dir, out_dir, pipeline_run_id="wf-b")

    assert a.succeeded == ["scan0K9E8BI"]
    assert b.succeeded == ["scanYR39SJX"]
    assert (out_dir / "run_manifest.wf-a.json").read_bytes() == source_a.read_bytes()
    assert (out_dir / "run_manifest.wf-b.json").read_bytes() == source_b.read_bytes()
    # Expected, and pinned so it is not a surprise: from run wf-b's point of view the
    # first run's envelope is outside its scope.
    assert any(
        "scan0K9E8BI" in r.getMessage() and "outside this run's scope" in r.getMessage()
        for r in _warnings_from_extractor(caplog)
    )


def test_batch_forward_failure_leaves_no_temp_file(tmp_path, monkeypatch, caplog):
    """A failed forward leaves only the envelopes, and never costs a scan its result."""
    in_dir = _copy_fixture(tmp_path)
    _write_per_run_manifest(in_dir, ["scan0K9E8BI", "scanYR39SJX"], "wf-a")
    out_dir = tmp_path / "out"
    real_replace = os.replace

    def fake_replace(src, dst, *args, **kwargs):
        # Only the manifest publish fails; per-scan write_envelope also uses os.replace.
        if Path(dst).name.startswith("run_manifest"):
            raise OSError("replace")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fake_replace)

    with caplog.at_level("WARNING"):
        result = extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")

    assert set(result.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "scan0K9E8BI.result.json",
        "scanYR39SJX.result.json",
    ]
    assert any(
        all(
            s in r.getMessage()
            for s in (
                "failed to copy run_manifest.wf-a.json",
                in_dir.as_posix(),
                out_dir.as_posix(),
            )
        )
        for r in _warnings_from_extractor(caplog)
    )


def test_per_run_manifest_rerun_skips_and_reforwards(tmp_path):
    """A re-run skips the scan and republishes the manifest over the existing copy."""
    in_dir = _copy_fixture(tmp_path)
    source = _write_per_run_manifest(in_dir, ["scan0K9E8BI"], "wf-a")
    out_dir = tmp_path / "out"

    first = extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")
    assert first.succeeded == ["scan0K9E8BI"]
    (out_dir / "run_manifest.wf-a.json").write_text("junk", encoding="utf-8")

    second = extract_batch(in_dir, out_dir, pipeline_run_id="wf-a")

    assert second.skipped == ["scan0K9E8BI"]
    assert (out_dir / "run_manifest.wf-a.json").read_bytes() == source.read_bytes()


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


def _run_module_cli(
    repo_root: Path,
    in_dir: Path,
    out_dir: Path,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Invoke `python -m trait_extractor <in> <out>` as a subprocess.

    ``extra_env`` is merged over ``os.environ`` (which the autouse conftest fixture has
    already cleared of ``ARGO_WORKFLOW_NAME``).
    """
    return subprocess.run(
        [sys.executable, "-m", "trait_extractor", str(in_dir), str(out_dir)],
        cwd=repo_root,
        env={**os.environ, "PYTHONPATH": str(repo_root), **(extra_env or {})},
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


def test_module_cli_exits_crash_code_on_non_utf8_run_manifest(tmp_path):
    """A run_manifest.json with invalid UTF-8 bytes exits 1 with a clean logged message.

    Originally added to exercise the UnicodeDecodeError branch of main()'s except
    tuple. Since contracts 0.1.0a9 the manifest is parsed from bytes, so invalid UTF-8
    surfaces as a pydantic ValidationError (``json_invalid``) instead -- still caught,
    still exit 1.
    """
    from sleap_roots_contracts import RUN_MANIFEST_FILENAME

    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    fixture_tree = repo_root / _FIXTURE_TREE
    shutil.copytree(fixture_tree, in_dir)
    (in_dir / RUN_MANIFEST_FILENAME).write_bytes(b"\xff\xfe not valid utf-8")

    proc = _run_module_cli(repo_root, in_dir, out_dir)

    assert proc.returncode == 1
    assert "Batch aborted:" in proc.stderr


def test_main_logs_clean_message_on_os_error_from_run_manifest(
    tmp_path, monkeypatch, caplog
):
    """An OSError from load_run_manifest is logged cleanly before propagating.

    Fresh PR review found this except-tuple branch (OSError) was only ever
    exercised via extract_batch's own copy-forward path (caught INSIDE
    extract_batch, never escaping to main()) -- unlike UnicodeDecodeError, this
    gap wasn't previously flagged as an accepted one. A real permissions error
    isn't reliably reproducible cross-platform (Windows ACLs differ from POSIX
    chmod), so this tests main()'s wrapper directly, in-process, via monkeypatch.
    """
    import trait_extractor.extractor as extractor_module
    from trait_extractor.__main__ import main

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()

    def _boom(*args, **kwargs):
        raise OSError("permission denied (simulated)")

    monkeypatch.setattr(extractor_module, "load_run_manifest", _boom)

    # main() registers a real process-wide SIGTERM handler; this is the only
    # test in this file that calls main() in-process (every other main()
    # exercise goes through subprocess, where the registration dies with the
    # child) -- restore the prior handler so it doesn't leak into later tests
    # in this pytest session.
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        with caplog.at_level("ERROR"):
            with pytest.raises(OSError):
                main([str(in_dir), str(out_dir)])
    finally:
        signal.signal(signal.SIGTERM, previous_handler)

    assert "Batch aborted:" in caplog.text


def _assert_logged_abort(proc: subprocess.CompletedProcess, token: str) -> None:
    """Exit 1 with a ``Batch aborted:`` log line that itself names ``token``."""
    assert proc.returncode == 1, proc.stderr
    assert re.search(r"Batch aborted: .*" + re.escape(token), proc.stderr), proc.stderr


def test_module_cli_exits_crash_code_on_missing_manifest_for_known_run(tmp_path):
    """ARGO_WORKFLOW_NAME set + no manifest -> a logged crash naming the run id."""
    repo_root = Path(__file__).resolve().parents[2]
    proc = _run_module_cli(
        repo_root,
        repo_root / _FIXTURE_TREE,
        tmp_path / "out",
        extra_env={"ARGO_WORKFLOW_NAME": "wf-a"},
    )
    _assert_logged_abort(proc, "wf-a")


def test_module_cli_exits_crash_code_on_identity_mismatch(tmp_path):
    """A per-run manifest naming another run -> a logged crash naming that run."""
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = _copy_fixture(tmp_path)
    _write_run_manifest(
        in_dir,
        ["scan0K9E8BI"],
        pipeline_run_id="wf-b",
        filename=run_manifest_filename("wf-a"),
    )
    proc = _run_module_cli(
        repo_root, in_dir, tmp_path / "out", extra_env={"ARGO_WORKFLOW_NAME": "wf-a"}
    )
    _assert_logged_abort(proc, "wf-b")


def test_module_cli_exits_crash_code_on_unusable_run_id(tmp_path):
    """An ARGO_WORKFLOW_NAME unusable as a filename component -> a logged crash."""
    repo_root = Path(__file__).resolve().parents[2]
    proc = _run_module_cli(
        repo_root,
        repo_root / _FIXTURE_TREE,
        tmp_path / "out",
        extra_env={"ARGO_WORKFLOW_NAME": "../x"},
    )
    _assert_logged_abort(proc, "../x")


def test_module_cli_exits_crash_code_on_nonexistent_input_dir(tmp_path):
    """A missing input_dir -> a logged crash naming the directory (regression guard)."""
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "does_not_exist"
    proc = _run_module_cli(repo_root, in_dir, tmp_path / "out")
    _assert_logged_abort(proc, in_dir.as_posix())


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


def test_test_only_scan_delay_reads_env_var(monkeypatch):
    """The test-only delay hook sleeps for exactly the env-var-specified duration.

    Fresh PR review suggested this: previously the hook's own correctness (right
    env var name, float() not int()) relied entirely on the SIGTERM test noticing
    a wrong exit code/file count as a symptom, not a direct assertion.
    """
    import trait_extractor.extractor as extractor_module

    calls = []
    monkeypatch.setattr(extractor_module.time, "sleep", lambda s: calls.append(s))
    monkeypatch.setenv(extractor_module._TEST_SCAN_DELAY_ENV, "0.5")

    extractor_module._test_only_scan_delay()

    assert calls == [0.5]


def test_test_only_scan_delay_is_a_noop_by_default(monkeypatch):
    """The test-only delay hook does nothing unless the env var is explicitly set."""
    import trait_extractor.extractor as extractor_module

    monkeypatch.delenv(extractor_module._TEST_SCAN_DELAY_ENV, raising=False)
    calls = []
    monkeypatch.setattr(extractor_module.time, "sleep", lambda s: calls.append(s))

    extractor_module._test_only_scan_delay()

    assert calls == []


def test_handle_sigterm_raises_systemexit_143():
    """The SIGTERM handler exits 143, called directly (no subprocess, no timing)."""
    from trait_extractor.__main__ import _handle_sigterm

    with pytest.raises(SystemExit) as exc_info:
        _handle_sigterm(signal.SIGTERM, None)
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
    """SIGTERM during a multi-scan batch exits 143 and leaves completed output intact.

    Revised after a real CI failure on a fast runner (macos-14): the original design
    duplicated 40 real scans and raced "poll for the first result, then SIGTERM"
    against real per-scan compute time. That race is fundamentally not fixable by
    adding more scans -- the margin that matters is "time from first result to full
    batch completion," which scales with per-scan cost (single-digit ms on a fast
    runner), not scan count. On a fast enough runner, all remaining scans finished
    before the signal could take effect, and the batch exited 3 (partial) instead of
    143. Fixed by using `SRT_TRAIT_EXTRACTOR_TEST_SCAN_DELAY_S` (a deterministic,
    env-var-gated, no-op-by-default per-scan delay hook in extract_batch) to make the
    race margin explicit and runner-speed-independent, instead of real compute time.
    """
    repo_root = Path(__file__).resolve().parents[2]
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    fixture_tree = repo_root / _FIXTURE_TREE

    for i in range(3):
        for orig in ("scan0K9E8BI", "scanYR39SJX"):
            new_key = f"{orig}_{i:03d}"
            _duplicate_scan(fixture_tree / orig, in_dir / new_key, new_key)

    proc = subprocess.Popen(
        [sys.executable, "-m", "trait_extractor", str(in_dir), str(out_dir)],
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONPATH": str(repo_root),
            "SRT_TRAIT_EXTRACTOR_TEST_SCAN_DELAY_S": "1",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if out_dir.exists() and list(out_dir.glob("*.result.json")):
                break
            time.sleep(0.05)
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
    assert len(result_files) < 6, (
        "all scans finished before SIGTERM landed -- the delay hook isn't slowing "
        "the batch down as expected"
    )
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
