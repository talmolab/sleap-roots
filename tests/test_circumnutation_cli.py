"""Tests for the `sleap-roots circumnutation analyze` CLI (PR #17).

Section 1 provides shared synthetic-`.slp` builders (a tracked `.slp` round-trip
through the real ``Series.load`` → ``get_tracked_tips`` path, so the adapter + CLI
happy paths are covered on every OS without the Git-LFS fixtures) plus a
metadata-CSV fixture builder. The adapter tests import these helpers.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio
from click.testing import CliRunner

from sleap_roots.circumnutation import synthetic


# ---------------------------------------------------------------------------
# Section 1 — synthetic tracked-`.slp` + metadata-CSV builders
# ---------------------------------------------------------------------------


def _make_synthetic_tracked_slp(
    tmp_path,
    *,
    series_name: str = "plate_xyz",
    n_tracks: int = 2,
    n_frames: int = 64,
    noise_sigma_px: float = 2.0,
    predicted_shadow_at=None,
) -> Path:
    """Build a synthetic tracked `.slp` and return its path.

    Each track's tip trajectory is produced by the pipeline's own
    :func:`sleap_roots.circumnutation.synthetic.generate_trajectory` (distinct
    phase + seed per track) and serialized to a tracked `.slp` via the house
    idiom (PIL TIFF → ``sio.Video`` → single-node ``"tip"`` skeleton → one
    ``sio.Track`` per track → ``sio.Instance.from_numpy`` → ``sio.Labels`` →
    ``sio.save_slp``). The file round-trips through the real ``Series.load`` /
    ``get_tracked_tips`` path.

    ``n_frames >= 64`` clears every tier the full pipeline runs (Tier 1 CWT min
    9, Tier 2 psi_g min 24, Tier 3 midline/spatial); ``noise_sigma_px > 0``
    avoids degenerate QC/ridge behavior (theory.md §8).

    Args:
        tmp_path: pytest ``tmp_path``.
        series_name: Series identifier; also the `.slp` filename stem.
        n_tracks: Number of distinct tracks (plants).
        n_frames: Frames per track.
        noise_sigma_px: Trajectory noise sigma (must be > 0).
        predicted_shadow_at: Optional ``(frame_idx, track_name)`` at which to add
            a tracker ``PredictedInstance`` shadowed by the user ``Instance``, so
            ``get_tracked_tips``'s predicted-vs-user dedup is exercised.

    Returns:
        The written `.slp` :class:`pathlib.Path`.
    """
    from PIL import Image

    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    image_h, image_w = 400, 400
    tif_path = tmp_path / f"{series_name}.tif"
    Image.fromarray(np.zeros((image_h, image_w), dtype=np.uint8)).save(
        tif_path.as_posix(), dpi=(72, 72)
    )
    video = sio.Video.from_filename(tif_path.as_posix())
    skeleton = sio.Skeleton(nodes=[sio.Node("tip")])

    track_positions = {}
    for i in range(n_tracks):
        df = synthetic.generate_trajectory(
            n_frames=n_frames,
            noise_sigma_px=noise_sigma_px,
            x0_px=50.0 + 40.0 * i,
            y0_px=50.0,
            initial_phase_rad=0.3 * i,
            random_state=i,
            track_id=i,
            plant_id=i,
        ).sort_values("frame")
        track_positions[f"track_{i}"] = list(
            zip(df["tip_x"].to_numpy(), df["tip_y"].to_numpy())
        )

    tracks_by_name = {name: sio.Track(name=name) for name in track_positions}
    labeled_frames = []
    for frame_idx in range(n_frames):
        instances = []
        for name, positions in track_positions.items():
            x, y = positions[frame_idx]
            instances.append(
                sio.Instance.from_numpy(
                    np.array([[x, y]], dtype=float),
                    skeleton=skeleton,
                    track=tracks_by_name[name],
                )
            )
            if predicted_shadow_at == (frame_idx, name):
                instances.append(
                    sio.PredictedInstance.from_numpy(
                        np.array([[x + 99.0, y + 99.0]], dtype=float),
                        skeleton=skeleton,
                        track=tracks_by_name[name],
                    )
                )
        labeled_frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )

    labels = sio.Labels(
        labeled_frames=labeled_frames,
        skeletons=[skeleton],
        videos=[video],
        tracks=list(tracks_by_name.values()),
    )
    slp_path = tmp_path / f"{series_name}.primary.tracked.slp"
    sio.save_slp(labels, slp_path.as_posix())
    return slp_path


def _write_metadata_csv(
    tmp_path,
    *,
    qr: str = "plate_001",
    genotype: str = "Nipponbare",
    treatment: str = "MOCK",
    timepoint=0,
    name: str = "fixture_metadata.csv",
) -> Path:
    """Write a ``build_metadata_csv``-shaped metadata CSV and return its path.

    Blank cells (``genotype=""`` etc.) are written verbatim so the
    metadata-precedence "blank cell" path can be exercised.
    """
    csv_path = Path(tmp_path) / name
    csv_path.write_text(
        "plant_qr_code,genotype,treatment,number_of_plants_cylinder,timepoint\n"
        f"{qr},{genotype},{treatment},6,{timepoint}\n",
        encoding="utf-8",
    )
    return csv_path


# ---------------------------------------------------------------------------
# Section 1 tests — the builders themselves
# ---------------------------------------------------------------------------


def test_synthetic_tracked_slp_loads_via_series(tmp_path):
    """The synthetic `.slp` round-trips through Series.load → get_tracked_tips."""
    from sleap_roots.series import Series

    slp = _make_synthetic_tracked_slp(tmp_path, n_tracks=2, n_frames=64)
    series = Series.load(series_name="plate_xyz", primary_path=str(slp))
    df = series.get_tracked_tips()
    assert set(df["track_id"]) == {"track_0", "track_1"}
    assert len(df) == 2 * 64
    assert list(df.columns) == ["track_id", "frame", "tip_x", "tip_y"]


def test_synthetic_slp_predicted_shadow_dedups(tmp_path):
    """A PredictedInstance shadowed by a user Instance dedups to one row."""
    from sleap_roots.series import Series

    slp = _make_synthetic_tracked_slp(
        tmp_path, n_tracks=1, n_frames=64, predicted_shadow_at=(0, "track_0")
    )
    series = Series.load(series_name="plate_xyz", primary_path=str(slp))
    df = series.get_tracked_tips()
    # One row per (track, frame) despite the duplicate predicted instance at frame 0.
    assert len(df) == 64
    assert int((df["frame"] == 0).sum()) == 1
    # The kept value is the user instance (NOT the +99 shadow).
    f0 = df[df["frame"] == 0].iloc[0]
    assert f0["tip_x"] < 200 and f0["tip_y"] < 200


# ---------------------------------------------------------------------------
# Section 3 — the `analyze` command
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    """A shared synthetic tracked `.slp` + a metadata CSV (built once)."""
    d = tmp_path_factory.mktemp("synth_cli")
    slp = _make_synthetic_tracked_slp(
        d, series_name="plate_xyz", n_tracks=2, n_frames=64
    )
    csv = _write_metadata_csv(d)  # plate_001, Nipponbare, MOCK, timepoint=0
    return slp, csv


def _analyze(args, **kw):
    """Invoke `circumnutation analyze` via CliRunner."""
    from sleap_roots.circumnutation.cli import circumnutation

    runner = CliRunner(**kw)
    return runner.invoke(circumnutation, ["analyze"] + args)


def test_group_and_analyze_help_exit0():
    """`circumnutation --help` and `analyze --help` exit 0; CC-3 help text present."""
    from sleap_roots.circumnutation.cli import circumnutation

    r = CliRunner().invoke(circumnutation, ["--help"])
    assert r.exit_code == 0
    r = CliRunner().invoke(circumnutation, ["analyze", "--help"])
    assert r.exit_code == 0
    assert "--cadence-s" in r.output and "--sample-uid" in r.output
    # CC-3: no calibration option; points to convert_to_mm; documents conventions.
    assert "--px-per-mm" not in r.output
    assert "convert_to_mm" in r.output
    assert "y" in r.output.lower()  # coordinate convention note present
    assert "timepoint" in r.output


def test_missing_cadence_exit2(synth):
    slp, _ = synth
    r = _analyze([str(slp), "--sample-uid", "plate_001", "--genotype", "WT"])
    assert r.exit_code == 2
    assert "cadence" in r.output.lower()


def test_missing_sample_uid_exit2(synth):
    slp, _ = synth
    r = _analyze([str(slp), "--cadence-s", "300", "--genotype", "WT"])
    assert r.exit_code == 2
    assert "sample-uid" in r.output.lower() or "sample_uid" in r.output.lower()


def test_nonexistent_slp_exit2():
    r = _analyze(["does_not_exist.slp", "--cadence-s", "300", "--sample-uid", "x"])
    assert r.exit_code == 2


def test_non_numeric_cadence_exit2(synth):
    slp, _ = synth
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "abc",
            "--sample-uid",
            "plate_001",
            "--genotype",
            "WT",
        ]
    )
    assert r.exit_code == 2
    assert "Traceback" not in r.output


def test_nonpositive_cadence_exit1(synth, tmp_path):
    """cadence 0 → CircumnutationInputs ValueError re-raised as ClickException (exit 1)."""
    slp, _ = synth
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "0",
            "--sample-uid",
            "plate_001",
            "--genotype",
            "WT",
            "-o",
            str(tmp_path / "out"),
        ]
    )
    assert r.exit_code == 1
    assert "Traceback" not in r.output
    assert "cadence" in r.output.lower()


def test_happy_path_full_tree_and_provenance(synth, tmp_path):
    """Full pipeline (with --metadata-csv) writes the tree + complete provenance."""
    slp, csv = synth
    out = tmp_path / "out"
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--metadata-csv",
            str(csv),
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output

    # Output tree.
    top_meta = out / "run_metadata.json"
    assert top_meta.exists()
    assert (out / "per_plant" / "traits_per_plant.csv").exists()
    assert (out / "per_plant" / "traits_per_plant.units.json").exists()
    assert (out / "per_plant" / "run_metadata.json").exists()
    assert (out / "per_genotype" / "traits_per_genotype.csv").exists()
    assert (out / "per_genotype" / "run_metadata.json").exists()
    pngs = list((out / "plots").glob("*.png"))
    assert len(pngs) >= 1

    # Per-plant row count == n tracks; genotype from CSV.
    per_plant = pd.read_csv(out / "per_plant" / "traits_per_plant.csv")
    assert len(per_plant) == 2
    assert (per_plant["genotype"] == "Nipponbare").all()
    assert (per_plant["treatment"] == "MOCK").all()

    # Plots run_metadata_ref resolves to the top-level run_metadata.json.
    plots_meta = json.loads((out / "plots" / "plots_metadata.json").read_text())
    ref = (out / "plots" / plots_meta["run_metadata_ref"]).resolve()
    assert ref == top_meta.resolve()

    # Provenance: identical across all three sidecars; identity provenance present.
    metas = [
        json.loads(top_meta.read_text()),
        json.loads((out / "per_plant" / "run_metadata.json").read_text()),
        json.loads((out / "per_genotype" / "run_metadata.json").read_text()),
    ]
    assert metas[0]["input_path"] == str(Path(slp).resolve())
    assert metas[0]["metadata_csv_path"] == str(Path(csv).resolve())
    assert len(metas[0]["metadata_csv_sha256"]) == 64  # sha256 hex digest
    assert metas[0]["identity_source"]["genotype"] == "metadata_csv"
    for key in (
        "input_path",
        "metadata_csv_path",
        "metadata_csv_sha256",
        "identity_source",
        "cadence_s",
        "R_px",
        "run_id",
        "timestamp",
    ):
        assert metas[0][key] == metas[1][key] == metas[2][key]
    assert metas[0]["cadence_s"] == 300.0
    assert metas[0]["R_px"] is None


def test_genotype_unresolved_hard_error(synth, tmp_path):
    """Bare run (no genotype/CSV/--no-aggregate) → exit 1, no output tree."""
    slp, _ = synth
    out = tmp_path / "out"
    r = _analyze(
        [str(slp), "--cadence-s", "300", "--sample-uid", "plate_001", "-o", str(out)]
    )
    assert r.exit_code == 1
    assert "Traceback" not in r.output
    for flag in ("--genotype", "--metadata-csv", "--no-aggregate"):
        assert flag in r.output
    # No output tree left behind.
    assert not (out / "per_plant").exists()
    assert not (out / "run_metadata.json").exists()


def test_blank_genotype_triggers_hard_error(synth, tmp_path):
    """A whitespace --genotype is treated as unresolved → aggregation hard-errors."""
    slp, _ = synth
    out = tmp_path / "out"
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--genotype",
            "   ",
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 1
    assert "--genotype" in r.output
    assert not (out / "per_plant").exists()


def test_no_aggregate_without_genotype(synth, tmp_path):
    """--no-aggregate runs per-plant + plots without genotype; no per_genotype/."""
    slp, _ = synth
    out = tmp_path / "out"
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--no-aggregate",
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output
    assert (out / "per_plant" / "traits_per_plant.csv").exists()
    assert (out / "plots").exists()
    assert not (out / "per_genotype").exists()
    # No-CSV run: null provenance, genotype absent.
    meta = json.loads((out / "per_plant" / "run_metadata.json").read_text())
    assert meta["metadata_csv_path"] is None
    assert meta["metadata_csv_sha256"] is None
    assert "metadata_csv" not in meta["identity_source"].values()
    assert meta["identity_source"]["genotype"] == "absent"


def test_no_plots(synth, tmp_path):
    slp, _ = synth
    out = tmp_path / "out"
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--genotype",
            "WT",
            "--no-plots",
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output
    assert (out / "per_plant").exists()
    assert not (out / "plots").exists()
    # Default verbosity (no -v) → WARNING level → no INFO progress lines on stderr.
    assert "Computing circumnutation traits" not in r.stderr


def test_no_plots_and_no_aggregate(synth, tmp_path):
    """Both off → per_plant only."""
    slp, _ = synth
    out = tmp_path / "out"
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--no-plots",
            "--no-aggregate",
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output
    assert (out / "per_plant").exists()
    assert not (out / "per_genotype").exists()
    assert not (out / "plots").exists()


def test_genotype_flag_overrides_csv_logged_on_stderr(synth, tmp_path):
    """--genotype overrides the CSV value; the override is logged at INFO on stderr."""
    slp, csv = synth
    out = tmp_path / "out"
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--metadata-csv",
            str(csv),
            "--genotype",
            "KitaakeX",
            "--no-plots",
            "-v",
            "-o",
            str(out),
        ],
    )
    assert r.exit_code == 0, r.output
    per_plant = pd.read_csv(out / "per_plant" / "traits_per_plant.csv")
    assert (per_plant["genotype"] == "KitaakeX").all()
    # INFO override notice goes to stderr; the result summary goes to stdout.
    # (click 8.3: result.stderr is stderr-only; result.output is the combined stream.)
    assert "Nipponbare" in r.stderr
    assert "Analysis complete" not in r.stderr


def test_logging_state_restored_after_run(synth, tmp_path):
    """The CLI restores the `sleap_roots` logger level/handlers after the command.

    Guards against the global logging config leaking into a host process / the test
    session (an unrestored level would gate later bare `caplog.at_level(...)` tests).
    """
    import logging

    pkg = logging.getLogger("sleap_roots")
    level_before = pkg.level
    tagged_before = [h for h in pkg.handlers if getattr(h, "_tag", None)]
    slp, _ = synth
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--genotype",
            "WT",
            "--no-plots",
            "--no-aggregate",
            "-vv",
            "-o",
            str(tmp_path / "out"),
        ]
    )
    assert r.exit_code == 0, r.output
    assert pkg.level == level_before  # level restored (not left at DEBUG)
    tagged_after = [h for h in pkg.handlers if getattr(h, "_tag", None)]
    assert len(tagged_after) == len(tagged_before)  # our handler removed


def test_vv_enables_debug_on_stderr(synth, tmp_path):
    """-vv sets DEBUG: the adapter's debug line appears on stderr."""
    slp, _ = synth
    out = tmp_path / "out"
    r = _analyze(
        [
            str(slp),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--genotype",
            "WT",
            "--no-plots",
            "--no-aggregate",
            "-vv",
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output
    # `series_to_inputs` emits a DEBUG line; only visible at -vv.
    assert "series_to_inputs" in r.stderr


def test_rerun_overwrites(synth, tmp_path):
    slp, _ = synth
    out = tmp_path / "out"
    base = [
        str(slp),
        "--cadence-s",
        "300",
        "--sample-uid",
        "plate_001",
        "--genotype",
        "WT",
        "--no-plots",
        "-o",
        str(out),
    ]
    r1 = _analyze(base)
    assert r1.exit_code == 0, r1.output
    r2 = _analyze(base)
    assert r2.exit_code == 0, r2.output
    per_plant = pd.read_csv(out / "per_plant" / "traits_per_plant.csv")
    assert len(per_plant) == 2  # overwritten in place, not appended


def test_default_output_dir_derives_from_series_name(tmp_path):
    """No -o → outputs under ./<series_name>_circumnutation/ in the cwd."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path) as fs:
        slp = _make_synthetic_tracked_slp(
            Path(fs), series_name="plate_xyz", n_tracks=2, n_frames=64
        )
        from sleap_roots.circumnutation.cli import circumnutation

        r = runner.invoke(
            circumnutation,
            [
                "analyze",
                str(slp),
                "--cadence-s",
                "300",
                "--sample-uid",
                "plate_001",
                "--no-plots",
                "--no-aggregate",
            ],
        )
        assert r.exit_code == 0, r.output
        # Default dir derives from the .slp stem (Path.stem strips only ".slp").
        assert (Path(fs) / f"{slp.stem}_circumnutation" / "per_plant").exists()


# ---------------------------------------------------------------------------
# Section 4 — registration on the root CLI
# ---------------------------------------------------------------------------


def test_circumnutation_registered_on_root_main():
    """`circumnutation` is a registered command on `sleap_roots.cli:main`."""
    from sleap_roots.cli import main

    assert "circumnutation" in main.commands
    r = CliRunner().invoke(main, ["circumnutation", "analyze", "--help"])
    assert r.exit_code == 0
    assert "--cadence-s" in r.output


# ---------------------------------------------------------------------------
# Section 6 — real plate-001 end-to-end (skipif-guarded on the Git-LFS fixture)
# ---------------------------------------------------------------------------

_PLATE001_DIR = Path(__file__).parent / "data" / "circumnutation_nipponbare_plate_001"
_PLATE001_SLP = _PLATE001_DIR / "plate_001_greyscale.tracked_proofread.slp"
_PLATE001_CSV = _PLATE001_DIR / "fixture_metadata.csv"


@pytest.mark.skipif(
    not _PLATE001_SLP.exists(),
    reason=f"Git-LFS proofread fixture not present: {_PLATE001_SLP}",
)
def test_real_plate001_end_to_end(tmp_path):
    """analyze on the real plate-001 .slp + metadata CSV writes the full tree."""
    out = tmp_path / "out"
    r = _analyze(
        [
            str(_PLATE001_SLP),
            "--cadence-s",
            "300",
            "--sample-uid",
            "plate_001",
            "--metadata-csv",
            str(_PLATE001_CSV),
            "-o",
            str(out),
        ]
    )
    assert r.exit_code == 0, r.output
    per_plant = pd.read_csv(out / "per_plant" / "traits_per_plant.csv")
    assert len(per_plant) == 6  # 6 plants on plate-001
    assert (per_plant["genotype"] == "Nipponbare").all()
    assert (per_plant["treatment"] == "MOCK").all()
    assert (out / "per_genotype" / "traits_per_genotype.csv").exists()
    pngs = list((out / "plots").glob("*.png"))
    assert pngs and all(p.stat().st_size > 0 for p in pngs)  # no pixel baselines
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["input_path"] == str(_PLATE001_SLP.resolve())
    assert meta["metadata_csv_path"] == str(_PLATE001_CSV.resolve())
    assert meta["identity_source"]["genotype"] == "metadata_csv"
