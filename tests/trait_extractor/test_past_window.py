"""Past-window scans through extract_scan and the batch CLI (bloom#971 phase 1)."""

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from sleap_roots_contracts import ResolvedParams

from trait_extractor.envelope import build_provenance
from trait_extractor.extractor import extract_scan
from trait_extractor.manifest import load_manifest, load_scan_metadata
from trait_extractor.pipeline_chooser import PipelineCard

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RICE_DIR = _REPO_ROOT / "tests/data/rice_3do_pipeline_output/scan0K9E8BI"
_MANIFEST = _RICE_DIR / "scan0K9E8BI.predictions.json"
_SIDECAR = _RICE_DIR / "scan0K9E8BI.scan_metadata.json"
_EXTRACTOR_LOGGER = "trait_extractor.extractor"


@pytest.fixture(autouse=True)
def _no_identity_env(monkeypatch):
    """Keep provenance comparisons hermetic: no code sha or digest from the env."""
    monkeypatch.delenv("SRT_TRAITS_CODE_SHA", raising=False)
    monkeypatch.delenv("SRT_TRAITS_CONTAINER_DIGEST", raising=False)


def _write_sidecar(directory, **params):
    """Write a copy of the rice sidecar with ``params`` overridden; return its path."""
    data = json.loads(_SIDECAR.read_text(encoding="utf-8"))
    data["params"] = {**data["params"], **params}
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / _SIDECAR.name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _clamp_warnings(caplog):
    """WARNING records from the extractor's clamp warning."""
    return [
        r
        for r in caplog.records
        if r.name == _EXTRACTOR_LOGGER
        and r.levelno == logging.WARNING
        and r.getMessage().startswith("past-window age:")
    ]


def test_past_window_envelope_keeps_real_age(tmp_path):
    """An age-18 rice scan emits the age-8 traits under its own age-18 provenance."""
    sidecar8 = _write_sidecar(tmp_path / "sc8", age=8)
    sidecar18 = _write_sidecar(tmp_path / "sc18", age=18)

    env8 = extract_scan(_MANIFEST, sidecar8, tmp_path / "out8")
    env18 = extract_scan(_MANIFEST, sidecar18, tmp_path / "out18")

    manifest = load_manifest(_MANIFEST)
    loaded18 = load_scan_metadata(sidecar18, manifest.scan_key)
    params18 = loaded18.to_resolved_params()
    assert env18.provenance == build_provenance(manifest, loaded18, params18)
    assert env18.provenance.params.values["age"] == 18
    params10 = ResolvedParams(values={**params18.values, "age": 10})
    assert (
        env18.provenance.idempotency_key
        != build_provenance(manifest, loaded18, params10).idempotency_key
    )

    assert env18.traits == env8.traits
    assert any(tv.value is not None for tv in env18.traits)


def test_past_window_rerun_is_skipped_without_warning(tmp_path, caplog):
    """A second run of a clamped scan is skipped before selection, so it never warns."""
    sidecar18 = _write_sidecar(tmp_path / "sc18", age=18)
    out = tmp_path / "out18"
    extract_scan(_MANIFEST, sidecar18, out)
    first = (out / "scan0K9E8BI.result.json").read_bytes()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER):
        assert extract_scan(_MANIFEST, sidecar18, out) is None
    assert (out / "scan0K9E8BI.result.json").read_bytes() == first
    assert _clamp_warnings(caplog) == []


def test_past_window_warning_names_scan_and_ages(tmp_path, caplog):
    """An age-18 rice scan logs exactly one clamp warning with every field."""
    sidecar18 = _write_sidecar(tmp_path / "sc18", age=18)
    with caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER):
        extract_scan(_MANIFEST, sidecar18, tmp_path / "out18")
    warnings = _clamp_warnings(caplog)
    assert [w.getMessage() for w in warnings] == [
        "past-window age: scan_key=scan0K9E8BI species='rice' mode='cylinder' age=18 "
        "matched as age=10 -> OlderMonocotPipeline"
    ]


def test_in_window_scan_logs_no_clamp_warning(tmp_path, caplog):
    """An in-window (age 8) scan logs no clamp warning."""
    sidecar8 = _write_sidecar(tmp_path / "sc8", age=8)
    with caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER):
        extract_scan(_MANIFEST, sidecar8, tmp_path / "out8")
    assert _clamp_warnings(caplog) == []


@pytest.mark.parametrize(
    "mode, age, pipeline_class",
    [
        pytest.param(
            "multiplant cylinder", 28, "MultipleDicotPipeline", id="multiplant"
        ),
        pytest.param("plate", 20, "MultipleDicotPlatePipeline", id="plate"),
    ],
)
def test_past_window_multi_plant_warns_then_is_rejected(
    tmp_path, caplog, mode, age, pipeline_class
):
    """A clamped multiplant/plate scan logs its warning, then fails the grain guard."""
    sidecar = _write_sidecar(
        tmp_path / "multi", species="arabidopsis", mode=mode, age=age
    )
    with caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER):
        with pytest.raises(ValueError, match="not supported for scan-grain emission"):
            extract_scan(_MANIFEST, sidecar, tmp_path / "out")
    assert [w.getMessage() for w in _clamp_warnings(caplog)] == [
        f"past-window age: scan_key=scan0K9E8BI species='arabidopsis' "
        f"mode={mode!r} age={age} matched as age=14 -> {pipeline_class}"
    ]


def test_warning_uses_the_cards_passed_in(tmp_path, caplog):
    """Selection and the warning share the caller's cards, not the packaged ones."""
    sidecar = _write_sidecar(tmp_path / "sc18", age=18)
    cards = [
        PipelineCard(
            species="rice",
            mode="cylinder",
            age_min=2,
            age_max=12,
            pipeline_class="OlderMonocotPipeline",
        )
    ]
    with caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER):
        extract_scan(_MANIFEST, sidecar, tmp_path / "out", cards=cards)
    assert [w.getMessage() for w in _clamp_warnings(caplog)] == [
        "past-window age: scan_key=scan0K9E8BI species='rice' mode='cylinder' age=18 "
        "matched as age=12 -> OlderMonocotPipeline"
    ]


def test_unmatched_scan_raises_without_warning(tmp_path, caplog):
    """A no-card species raises 'No pipeline matches' and logs no clamp warning."""
    sidecar = _write_sidecar(tmp_path / "sorghum", species="sorghum", age=30)
    with caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER):
        with pytest.raises(ValueError, match="^No pipeline matches"):
            extract_scan(_MANIFEST, sidecar, tmp_path / "out")
    assert _clamp_warnings(caplog) == []


def test_clamp_ending_in_unknown_class_logs_no_warning(tmp_path, caplog):
    """The warning is logged only after choose_pipeline returns."""
    sidecar = _write_sidecar(tmp_path / "sc12", age=12)
    cards = [
        PipelineCard(
            species="rice",
            mode="cylinder",
            age_min=2,
            age_max=10,
            pipeline_class="NopePipeline",
        )
    ]
    with caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER):
        with pytest.raises(ValueError, match="^Unknown pipeline class"):
            extract_scan(_MANIFEST, sidecar, tmp_path / "out", cards=cards)
    assert _clamp_warnings(caplog) == []


def _run_cli(in_dir, out_dir):
    """Run ``python -m trait_extractor <in> <out>`` from the repo root."""
    return subprocess.run(
        [sys.executable, "-m", "trait_extractor", str(in_dir), str(out_dir)],
        cwd=_REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
    )


def test_past_window_scan_succeeds_through_batch_cli(tmp_path):
    """The batch CLI emits a past-window scan and prints its warning to stderr."""
    scan_dir = tmp_path / "in" / "scan0K9E8BI"
    shutil.copytree(_RICE_DIR, scan_dir)
    sidecar = scan_dir / _SIDECAR.name
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["params"]["age"] = 18
    sidecar.write_text(json.dumps(data), encoding="utf-8")

    result = _run_cli(tmp_path / "in", tmp_path / "out")

    assert result.returncode == 0, result.stderr
    assert "ok    scan0K9E8BI" in result.stdout
    assert "past-window age: scan_key=scan0K9E8BI" in result.stderr
