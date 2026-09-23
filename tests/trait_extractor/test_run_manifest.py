"""Tests for copying the run-scoping ``RunManifest`` forward.

Loading the manifest is contracts' ``load_run_manifest`` (sleap-roots-contracts 0.1.0a9);
its resolution, parsing, and identity cross-check are exercised through
``extract_batch`` in ``test_batch.py``. Every ``read`` here comes from contracts'
``read_run_manifest`` over a real file, so its ``filename``/``data``/``mode`` semantics
are contracts' own rather than hand-built.
"""

import json
import logging
import os
import pathlib
import sys
from pathlib import Path

import pytest
from sleap_roots_contracts import (
    RUN_MANIFEST_FILENAME,
    RunManifest,
    read_run_manifest,
    run_manifest_filename,
)

import trait_extractor.run_manifest as run_manifest_module
from trait_extractor.run_manifest import copy_run_manifest_forward


def _write_manifest(
    directory: Path, filename: str = RUN_MANIFEST_FILENAME, **overrides
):
    payload = {
        "pipeline_run_id": "local-abc123",
        "scan_keys": ["scan0K9E8BI", "scanYR39SJX"],
    }
    payload.update(overrides)
    path = directory / filename
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _listing(directory: Path) -> list:
    """Every entry in ``directory`` (dotfiles included), so any temp file is caught."""
    return sorted(p.name for p in directory.iterdir())


def _fail_replace_for_manifests(monkeypatch, exc: BaseException) -> None:
    """Make ``os.replace`` raise ``exc`` only when publishing a run manifest.

    ``os.replace`` is global: patching it blindly would also break every per-scan
    ``write_envelope`` (``Path.replace`` calls ``os.replace``). Delegate everything else.
    """
    real_replace = os.replace

    def fake_replace(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("run_manifest"):
            raise exc
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", fake_replace)


@pytest.fixture
def per_run(tmp_path):
    """A per-run manifest for ``wf-a`` in ``in/``, read the way extract_batch reads it."""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    source = _write_manifest(
        in_dir, run_manifest_filename("wf-a"), pipeline_run_id="wf-a"
    )
    read = read_run_manifest(in_dir, "wf-a", allow_legacy=True)
    return in_dir, tmp_path / "out", source, read


def test_forward_publishes_read_bytes_under_read_filename(per_run):
    """A per-run read is republished under its own name, byte-identical to the read."""
    in_dir, out_dir, source, read = per_run

    copy_run_manifest_forward(read, in_dir, out_dir)

    assert _listing(out_dir) == ["run_manifest.wf-a.json"]
    assert (out_dir / "run_manifest.wf-a.json").read_bytes() == read.data
    assert read.data == source.read_bytes()


def test_forward_publishes_snapshot_not_current_source(per_run):
    """The forwarded bytes are the ones read, even if the source changes afterwards."""
    in_dir, out_dir, source, read = per_run
    original = read.data
    _write_manifest(
        in_dir,
        run_manifest_filename("wf-a"),
        pipeline_run_id="wf-a",
        scan_keys=["scanOTHER"],
    )
    assert source.read_bytes() != original

    copy_run_manifest_forward(read, in_dir, out_dir)

    assert (out_dir / "run_manifest.wf-a.json").read_bytes() == original


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
@pytest.mark.parametrize("mode", [0o644, 0o640])
def test_forward_preserves_source_mode(tmp_path, monkeypatch, mode):
    """The forwarded file gets the source's mode, set BEFORE it is published.

    ``0o640`` is neither the umask default nor ``mkstemp``'s ``0o600``, so it
    distinguishes "copied the mode" from both "default permissions" and "forgot chmod".
    """
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    source = _write_manifest(in_dir)
    os.chmod(source, mode)  # before the read: read.mode comes from fstat at read time
    read = read_run_manifest(in_dir, None, allow_legacy=True)
    assert read.mode == mode

    modes_at_replace = []
    real_replace = os.replace

    def spy_replace(src, dst, *args, **kwargs):
        if Path(dst).name.startswith("run_manifest"):
            modes_at_replace.append(os.stat(src).st_mode & 0o777)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy_replace)

    copy_run_manifest_forward(read, in_dir, tmp_path / "out")

    assert modes_at_replace == [mode]
    assert (tmp_path / "out" / RUN_MANIFEST_FILENAME).stat().st_mode & 0o777 == mode


def test_forward_failure_removes_temp_file(per_run, monkeypatch):
    """A failed publish leaves neither a temp file nor a forwarded manifest."""
    in_dir, out_dir, _, read = per_run
    _fail_replace_for_manifests(monkeypatch, OSError("replace"))

    with pytest.raises(OSError, match="^replace$"):
        copy_run_manifest_forward(read, in_dir, out_dir)

    assert _listing(out_dir) == []


def test_forward_failure_during_chmod_removes_temp_file(per_run, monkeypatch):
    """A failure setting the temp file's mode also cleans the temp file up."""
    in_dir, out_dir, _, read = per_run
    real_chmod = os.chmod

    def fake_chmod(path, mode, *args, **kwargs):
        if not isinstance(path, int) and Path(path).name.startswith(".run_manifest"):
            raise OSError("chmod")
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(os, "chmod", fake_chmod)

    with pytest.raises(OSError, match="^chmod$"):
        copy_run_manifest_forward(read, in_dir, out_dir)

    assert _listing(out_dir) == []


def test_forward_systemexit_removes_temp_file_and_propagates(per_run, monkeypatch):
    """SIGTERM's SystemExit(143) mid-forward still cleans up, and is not swallowed."""
    in_dir, out_dir, _, read = per_run
    _fail_replace_for_manifests(monkeypatch, SystemExit(143))

    with pytest.raises(SystemExit) as info:
        copy_run_manifest_forward(read, in_dir, out_dir)

    assert info.value.code == 143
    assert _listing(out_dir) == []


def test_forward_cleanup_failure_does_not_mask_original_error(
    per_run, monkeypatch, caplog
):
    """If removing the temp file also fails, the publish error still propagates."""
    in_dir, out_dir, _, read = per_run
    _fail_replace_for_manifests(monkeypatch, OSError("replace"))
    real_unlink = pathlib.Path.unlink

    def fake_unlink(self, missing_ok=False):
        if self.name.startswith(".run_manifest"):
            raise OSError("unlink")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(pathlib.Path, "unlink", fake_unlink)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(OSError, match="^replace$"):
            copy_run_manifest_forward(read, in_dir, out_dir)

    [leftover] = [p for p in out_dir.iterdir() if p.name.startswith(".run_manifest")]
    assert any(
        leftover.name in r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
    )


def test_forward_temp_file_is_dot_prefixed_inside_output_dir(per_run, monkeypatch):
    """The temp file lives in output_dir (no cross-device replace) and is hidden."""
    in_dir, out_dir, _, read = per_run
    calls = []
    real_mkstemp = run_manifest_module.tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        calls.append((args, kwargs))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(run_manifest_module.tempfile, "mkstemp", spy_mkstemp)

    copy_run_manifest_forward(read, in_dir, out_dir)

    [(args, kwargs)] = calls
    # mkstemp(suffix=None, prefix=None, dir=None, text=False)
    positional = dict(zip(("suffix", "prefix", "dir"), args))
    directory = kwargs.get("dir", positional.get("dir"))
    prefix = kwargs.get("prefix", positional.get("prefix"))
    assert Path(directory).resolve() == out_dir.resolve()
    assert prefix.startswith(".")


def test_copy_manifest_forward_writes_into_output_dir(tmp_path):
    """The manifest is copied forward byte-identical, creating output_dir if missing."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    source = _write_manifest(in_dir)
    read = read_run_manifest(in_dir, None, allow_legacy=True)

    copy_run_manifest_forward(read, in_dir, out_dir)

    dest = out_dir / RUN_MANIFEST_FILENAME
    assert dest.exists()
    assert dest.read_bytes() == source.read_bytes()
    assert _listing(out_dir) == [RUN_MANIFEST_FILENAME]


def test_copy_manifest_forward_overwrites_a_different_prior_manifest(tmp_path):
    """A different pre-existing manifest in output_dir is overwritten, not merged."""
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    source = _write_manifest(in_dir)
    _write_manifest(out_dir, pipeline_run_id="stale-run")
    read = read_run_manifest(in_dir, None, allow_legacy=True)

    copy_run_manifest_forward(read, in_dir, out_dir)

    dest = out_dir / RUN_MANIFEST_FILENAME
    assert dest.read_bytes() == source.read_bytes()
    assert RunManifest.model_validate_json(dest.read_text()).pipeline_run_id == (
        "local-abc123"
    )


@pytest.fixture
def mkstemp_calls(monkeypatch):
    """Record every temp-file creation the forward starts.

    A same-path no-op must not start a publish at all: publishing over the same file
    would also leave identical bytes and no temp file, so only this proves the guard
    fired.
    """
    calls = []
    real_mkstemp = run_manifest_module.tempfile.mkstemp

    def spy_mkstemp(*args, **kwargs):
        # tempfile is global; count only the forward's dot-prefixed run-manifest temps.
        if str(kwargs.get("prefix", "")).startswith(".run_manifest"):
            calls.append((args, kwargs))
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(run_manifest_module.tempfile, "mkstemp", spy_mkstemp)
    return calls


def test_copy_manifest_forward_is_a_noop_when_input_and_output_are_the_same(
    tmp_path, mkstemp_calls
):
    """input_dir == output_dir is already satisfied: nothing is written, nothing raised."""
    source = _write_manifest(tmp_path)
    before = source.read_bytes()
    read = read_run_manifest(tmp_path, None, allow_legacy=True)

    copy_run_manifest_forward(read, tmp_path, tmp_path)  # must not raise

    assert source.read_bytes() == before
    assert _listing(tmp_path) == [RUN_MANIFEST_FILENAME]
    assert mkstemp_calls == []


def test_copy_manifest_forward_noop_for_same_dir_per_run_name(tmp_path, mkstemp_calls):
    """The same-path no-op keys on read.filename, and leaves no temp file behind."""
    _write_manifest(tmp_path, run_manifest_filename("wf-a"), pipeline_run_id="wf-a")
    read = read_run_manifest(tmp_path, "wf-a", allow_legacy=True)

    copy_run_manifest_forward(read, tmp_path, tmp_path)

    assert _listing(tmp_path) == ["run_manifest.wf-a.json"]
    assert mkstemp_calls == []


def test_copy_manifest_forward_is_a_noop_for_differently_spelled_same_path(
    tmp_path, mkstemp_calls
):
    """Two textually-different paths that resolve to the same directory also no-op.

    Guards against a regression that swaps the resolve()-based equality check for a
    plain `==` on the raw Path/string arguments -- that would pass the (already
    covered) identical-argument case but miss this one, since `tmp_path` and
    `tmp_path/"sub"/".."` are unequal as raw paths but resolve to the same directory.
    """
    (tmp_path / "sub").mkdir()
    _write_manifest(tmp_path)
    read = read_run_manifest(tmp_path, None, allow_legacy=True)
    other_spelling = tmp_path / "sub" / ".."
    assert other_spelling != tmp_path  # different Path objects/strings
    assert other_spelling.resolve() == tmp_path.resolve()  # same resolved directory

    copy_run_manifest_forward(read, tmp_path, other_spelling)  # must not raise

    assert (tmp_path / RUN_MANIFEST_FILENAME).is_file()
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".run_manifest")]
    assert mkstemp_calls == []
