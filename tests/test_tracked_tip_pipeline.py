"""Tests for TrackedTipPipeline (issue #129, OpenSpec change add-tracked-tip-pipeline).

Section 6 (pipeline DAG composition), Section 8 (file emission), Section 10
(batch method), Section 12 (real-fixture integration) of the change's
``tasks.md``.

Reuses the synthetic-tracked-`.slp` builder ``_build_tracked_slp`` from
``tests.test_series`` to avoid duplication.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio

from sleap_roots import Series, TrackedTipPipeline
from tests.test_series import _build_tracked_slp


# ---------------------------------------------------------------------------
# §6 — DAG composition tests (in-memory result dict, no file I/O)
# ---------------------------------------------------------------------------


def test_compute_tracked_tip_traits_returns_expected_dict_keys(tmp_path):
    """§6.2: result dict has exactly the expected top-level keys."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=5,
        track_positions={
            "track_a": [(float(i), 0.0) for i in range(5)],
            "track_b": [(float(i), 10.0) for i in range(5)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert set(result.keys()) == {
        "schema_version",
        "pipeline",
        "units",
        "series",
        "sample_uid",
        "timepoint",
        "tracks",
        "trajectories",
    }


def test_compute_tracked_tip_traits_schema_version_is_1(tmp_path):
    """§6.3: schema_version is 1 for the initial release."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["schema_version"] == 1


def test_compute_tracked_tip_traits_pipeline_name(tmp_path):
    """Pipeline string identifier."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["pipeline"] == "TrackedTipPipeline"


def test_compute_tracked_tip_traits_units_dict(tmp_path):
    """§6.4: units dict has exactly the expected keys/values."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["units"] == {
        "lengths": "pixels",
        "ratios": "dimensionless",
        "counts": "dimensionless",
        "time": "unspecified",
    }


def test_compute_tracked_tip_traits_tracks_table_one_row_per_track_id(tmp_path):
    """§6.5: 2-track .slp → exactly 2 rows in tracks; row keys are exact set."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={
            "track_a": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            "track_b": [(10.0, 10.0), (11.0, 10.0), (12.0, 10.0)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert len(result["tracks"]) == 2
    expected_keys = {
        "track_id",
        "n_frames_tracked",
        "n_frames_total",
        "tracking_coverage",
        "tip_trajectory_length",
        "tip_displacement_net",
    }
    for row in result["tracks"]:
        assert set(row.keys()) == expected_keys


def test_compute_tracked_tip_traits_trajectories_table_one_row_per_track_frame(
    tmp_path,
):
    """§6.6: 5×2 fully-tracked → 10 trajectory rows."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=5,
        track_positions={
            "track_a": [(float(i), 0.0) for i in range(5)],
            "track_b": [(float(i), 10.0) for i in range(5)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert len(result["trajectories"]) == 10
    # Per-row keys are exact set; top-level scalars NOT repeated in rows.
    for row in result["trajectories"]:
        assert set(row.keys()) == {"track_id", "frame", "tip_x", "tip_y"}


def test_tip_trajectory_length_straight_line(tmp_path):
    """§6.7: straight-line trajectory (0,0)→(3,0)→(6,0)→(10,0) → length 10."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=4,
        track_positions={
            "t": [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (10.0, 0.0)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    row = result["tracks"][0]
    assert row["tip_trajectory_length"] == pytest.approx(10.0)


def test_tip_displacement_net_straight_line(tmp_path):
    """§6.8: same straight-line trajectory → displacement_net == 10.0."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=4,
        track_positions={
            "t": [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0), (10.0, 0.0)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    row = result["tracks"][0]
    assert row["tip_displacement_net"] == pytest.approx(10.0)


def test_tip_displacement_net_round_trip_returns_to_origin(tmp_path):
    """§6.9: round-trip (0,0)→(5,0)→(0,0) → displacement=0, length=10. Validates the geometric distinction."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={
            "t": [(0.0, 0.0), (5.0, 0.0), (0.0, 0.0)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    row = result["tracks"][0]
    assert row["tip_displacement_net"] == pytest.approx(0.0)
    assert row["tip_trajectory_length"] == pytest.approx(10.0)


def test_tip_displacement_net_right_angle_path(tmp_path):
    """§6.10: 3-4-5 triangle path (0,0)→(3,0)→(3,4) → length=7, displacement=5."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={
            "t": [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    row = result["tracks"][0]
    assert row["tip_trajectory_length"] == pytest.approx(7.0)
    assert row["tip_displacement_net"] == pytest.approx(5.0)


def test_tracking_coverage_full(tmp_path):
    """§6.11: every frame tracked → coverage == 1.0."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=4,
        track_positions={"t": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]},
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    row = result["tracks"][0]
    assert row["tracking_coverage"] == pytest.approx(1.0)
    assert row["n_frames_tracked"] == 4
    assert row["n_frames_total"] == 4


def test_tracking_coverage_partial(tmp_path):
    """§6.12: 3 of 5 frames tracked → coverage == 0.6."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=5,
        track_positions={
            "t": [(0.0, 0.0), (1.0, 0.0), None, (3.0, 0.0), None],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    row = result["tracks"][0]
    assert row["tracking_coverage"] == pytest.approx(0.6)
    assert row["n_frames_tracked"] == 3
    assert row["n_frames_total"] == 5


def test_compute_tracked_tip_traits_single_frame_track_zero_displacement_nan_length(
    tmp_path,
):
    """§6.13: single-frame track → displacement=0.0, trajectory_length=NaN. Codifies asymmetry."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=10,
        track_positions={
            "t": [None] * 4 + [(7.0, 7.0)] + [None] * 5,  # only frame 4 tracked
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    row = result["tracks"][0]
    assert row["tip_displacement_net"] == pytest.approx(0.0)
    assert np.isnan(row["tip_trajectory_length"])
    assert row["n_frames_tracked"] == 1
    assert row["n_frames_total"] == 10


def test_compute_tracked_tip_traits_zero_tracks(tmp_path):
    """§6.14: empty .slp (no tracked instances anywhere) → empty tracks/trajectories arrays, no crash."""
    # Build an .slp with frames but ZERO instances per frame.
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    image_h, image_w = 200, 200
    img_array = np.zeros((image_h, image_w), dtype=np.uint8)
    tif_path = tmp_path / "s.tif"
    from PIL import Image

    Image.fromarray(img_array).save(tif_path.as_posix(), dpi=(72, 72))
    video = sio.Video.from_filename(tif_path.as_posix())
    skeleton = sio.Skeleton(nodes=[sio.Node("r0")])
    labeled_frames = [
        sio.LabeledFrame(video=video, frame_idx=i, instances=[]) for i in range(3)
    ]
    labels = sio.Labels(
        labeled_frames=labeled_frames,
        skeletons=[skeleton],
        videos=[video],
        tracks=[],
    )
    slp_path = tmp_path / "s.primary.tracked.slp"
    sio.save_slp(labels, slp_path.as_posix())
    series = Series.load(series_name="s", primary_path=slp_path.as_posix())
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["tracks"] == []
    assert result["trajectories"] == []


def test_tracking_coverage_bounded_when_track_has_duplicate_frame(tmp_path):
    """Duplicate (track_id, frame) MUST NOT inflate tracking_coverage above 1.0.

    Surfaced by /review-pr on PR #190: pathological tracker output (e.g.
    over-eager track merging) can produce two instances with the same
    `inst.track.name` in the same frame. The previous implementation used
    `n_frames_tracked = len(group)` which counted instances, not unique
    frames — so a 1-frame .slp with 2 same-track instances yielded
    `tracking_coverage = 2.0`, violating the spec contract that
    `tracking_coverage ∈ [0.0, 1.0]`.

    Fix: `n_frames_tracked = int(group["frame"].nunique())`.
    """
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    image_h, image_w = 200, 200
    img_array = np.zeros((image_h, image_w), dtype=np.uint8)
    tif_path = tmp_path / "s.tif"
    from PIL import Image

    Image.fromarray(img_array).save(tif_path.as_posix(), dpi=(72, 72))
    video = sio.Video.from_filename(tif_path.as_posix())
    skeleton = sio.Skeleton(nodes=[sio.Node("r0")])
    track = sio.Track(name="t")
    # ONE frame, TWO instances both with the SAME track — pathological.
    pts_a = np.array([[10.0, 20.0]])
    pts_b = np.array([[15.0, 25.0]])
    inst_a = sio.Instance.from_numpy(pts_a, skeleton=skeleton, track=track)
    inst_b = sio.Instance.from_numpy(pts_b, skeleton=skeleton, track=track)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst_a, inst_b])
    labels = sio.Labels(
        labeled_frames=[lf], skeletons=[skeleton], videos=[video], tracks=[track]
    )
    slp_path = tmp_path / "s.primary.tracked.slp"
    sio.save_slp(labels, slp_path.as_posix())

    series = Series.load(series_name="s", primary_path=slp_path.as_posix())
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)

    assert len(result["tracks"]) == 1
    row = result["tracks"][0]
    assert row["track_id"] == "t"
    assert (
        row["n_frames_tracked"] == 1
    ), f"n_frames_tracked must be unique-frame count, got {row['n_frames_tracked']}"
    assert row["n_frames_total"] == 1
    assert (
        row["tracking_coverage"] == 1.0
    ), f"tracking_coverage must be bounded to [0,1], got {row['tracking_coverage']}"

    # Trajectory rows still record every tracked instance — duplicates
    # remain visible in the per-frame table for downstream debugging.
    # Only the per-track summary deduplicates.
    assert len(result["trajectories"]) == 2


def test_compute_tracked_tip_traits_emits_sample_uid_and_timepoint_from_csv(tmp_path):
    """§6.15: with CSV → sample_uid and timepoint resolved from CSV."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
        csv_content="plant_qr_code,timepoint\nplate_X,2.5\n",
        sample_uid="plate_X",
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["sample_uid"] == "plate_X"
    assert result["timepoint"] == 2.5


def test_compute_tracked_tip_traits_no_csv_defaults_sample_uid_to_series_name(tmp_path):
    """§6.16: no CSV → sample_uid defaults to series_name, timepoint NaN."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["sample_uid"] == "s"
    assert np.isnan(result["timepoint"])


def test_track_id_values_match_inst_track_name(tmp_path):
    """§6.17: custom track names → output track_ids match exactly."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={
            "alpha": [(0.0, 0.0), (1.0, 1.0)],
            "beta": [(2.0, 2.0), (3.0, 3.0)],
        },
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert {r["track_id"] for r in result["tracks"]} == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# §8 — file emission tests
# ---------------------------------------------------------------------------


def test_compute_tracked_tip_traits_writes_summary_csv(tmp_path):
    """§8.1: summary CSV exists with expected columns."""
    series = _build_tracked_slp(
        tmp_path,
        "plate_001",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series, write_csv=True, output_dir=out_dir.as_posix()
    )
    summary_path = out_dir / "plate_001.tracked_tip_traits.csv"
    assert summary_path.exists()
    df = pd.read_csv(summary_path)
    assert list(df.columns) == [
        "series",
        "sample_uid",
        "timepoint",
        "track_id",
        "n_frames_tracked",
        "n_frames_total",
        "tracking_coverage",
        "tip_trajectory_length",
        "tip_displacement_net",
    ]


def test_compute_tracked_tip_traits_writes_trajectory_csv(tmp_path):
    """§8.2: trajectory CSV exists with expected columns."""
    series = _build_tracked_slp(
        tmp_path,
        "plate_001",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series, write_csv=True, output_dir=out_dir.as_posix()
    )
    traj_path = out_dir / "plate_001.tracked_tip_trajectories.csv"
    assert traj_path.exists()
    df = pd.read_csv(traj_path)
    assert list(df.columns) == [
        "series",
        "sample_uid",
        "timepoint",
        "track_id",
        "frame",
        "tip_x",
        "tip_y",
    ]


def test_compute_tracked_tip_traits_writes_json_with_both_tables(tmp_path):
    """§8.3: JSON exists, parses, has both tables at top level."""
    series = _build_tracked_slp(
        tmp_path,
        "plate_001",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series, write_json=True, output_dir=out_dir.as_posix()
    )
    json_path = out_dir / "plate_001.tracked_tip_traits.json"
    assert json_path.exists()
    parsed = json.loads(json_path.read_text())
    assert "tracks" in parsed
    assert "trajectories" in parsed
    assert isinstance(parsed["tracks"], list)
    assert isinstance(parsed["trajectories"], list)


def test_compute_tracked_tip_traits_emit_trajectories_false_skips_trajectory_csv(
    tmp_path,
):
    """§8.4: emit_trajectories=False → no trajectory CSV written."""
    series = _build_tracked_slp(
        tmp_path,
        "plate_001",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series,
        write_csv=True,
        emit_trajectories=False,
        output_dir=out_dir.as_posix(),
    )
    assert (out_dir / "plate_001.tracked_tip_traits.csv").exists()
    assert not (out_dir / "plate_001.tracked_tip_trajectories.csv").exists()


def test_compute_tracked_tip_traits_emit_trajectories_false_omits_trajectories_in_json(
    tmp_path,
):
    """§8.5: emit_trajectories=False with JSON → trajectories array is empty."""
    series = _build_tracked_slp(
        tmp_path,
        "plate_001",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series,
        write_json=True,
        emit_trajectories=False,
        output_dir=out_dir.as_posix(),
    )
    parsed = json.loads((out_dir / "plate_001.tracked_tip_traits.json").read_text())
    assert parsed["trajectories"] == []
    assert len(parsed["tracks"]) == 1


def test_compute_tracked_tip_traits_csv_repeats_top_level_scalars_per_row(tmp_path):
    """§8.6: every CSV row carries series, sample_uid, timepoint."""
    series = _build_tracked_slp(
        tmp_path,
        "plate_001",
        n_frames=3,
        track_positions={
            "ta": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            "tb": [(10.0, 10.0), (11.0, 10.0), (12.0, 10.0)],
        },
        csv_content="plant_qr_code,timepoint\nplate_001,5\n",
        sample_uid="plate_001",
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series, write_csv=True, output_dir=out_dir.as_posix()
    )
    df_summary = pd.read_csv(out_dir / "plate_001.tracked_tip_traits.csv")
    assert (df_summary["series"] == "plate_001").all()
    assert (df_summary["sample_uid"] == "plate_001").all()
    assert (df_summary["timepoint"] == 5.0).all()
    df_traj = pd.read_csv(out_dir / "plate_001.tracked_tip_trajectories.csv")
    assert (df_traj["series"] == "plate_001").all()
    assert (df_traj["sample_uid"] == "plate_001").all()
    assert (df_traj["timepoint"] == 5.0).all()


def test_compute_tracked_tip_traits_json_top_level_scalars_not_in_per_row(tmp_path):
    """§8.7: JSON tracks/trajectories rows do NOT carry series/sample_uid/timepoint."""
    series = _build_tracked_slp(
        tmp_path,
        "plate_001",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series, write_json=True, output_dir=out_dir.as_posix()
    )
    parsed = json.loads((out_dir / "plate_001.tracked_tip_traits.json").read_text())
    for row in parsed["tracks"]:
        assert "series" not in row
        assert "sample_uid" not in row
        assert "timepoint" not in row
    for row in parsed["trajectories"]:
        assert "series" not in row
        assert "sample_uid" not in row
        assert "timepoint" not in row


def test_compute_tracked_tip_traits_json_nan_to_null(tmp_path):
    """§8.8: NaN trajectory_length (single-frame) serializes to JSON null."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=10,
        track_positions={"t": [None] * 4 + [(7.0, 7.0)] + [None] * 5},
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_tracked_tip_traits(
        series, write_json=True, output_dir=out_dir.as_posix()
    )
    parsed = json.loads((out_dir / "s.tracked_tip_traits.json").read_text())
    row = parsed["tracks"][0]
    assert row["tip_trajectory_length"] is None  # JSON null
    assert row["tip_displacement_net"] == 0.0


# ---------------------------------------------------------------------------
# §10 — batch method tests
# ---------------------------------------------------------------------------


def test_compute_batch_tracked_tip_traits_concatenates_summary(tmp_path):
    """§10.1: 3 series × 2 tracks each → 6 summary rows in concatenated CSV."""
    serieses = [
        _build_tracked_slp(
            tmp_path / f"d{i}",
            f"s{i}",
            n_frames=3,
            track_positions={
                "ta": [(0.0, 0.0)] * 3,
                "tb": [(10.0, 10.0)] * 3,
            },
        )
        for i in range(3)
    ]
    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_batch_tracked_tip_traits(
        serieses, write_csv=True, output_dir=out_dir.as_posix()
    )
    df = pd.read_csv(out_dir / "tracked_tip_batch_traits.csv")
    assert len(df) == 6


def test_compute_batch_tracked_tip_traits_concatenates_trajectories(tmp_path):
    """§10.2: 3 series × 5 frames × 2 tracks → 30 trajectory rows."""
    serieses = [
        _build_tracked_slp(
            tmp_path / f"d{i}",
            f"s{i}",
            n_frames=5,
            track_positions={
                "ta": [(float(j), 0.0) for j in range(5)],
                "tb": [(float(j), 10.0) for j in range(5)],
            },
        )
        for i in range(3)
    ]
    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_batch_tracked_tip_traits(
        serieses, write_csv=True, output_dir=out_dir.as_posix()
    )
    df = pd.read_csv(out_dir / "tracked_tip_batch_trajectories.csv")
    assert len(df) == 30


def test_compute_batch_tracked_tip_traits_writes_json_list(tmp_path):
    """§10.3: batch JSON parses as list of per-series dicts."""
    serieses = [
        _build_tracked_slp(
            tmp_path / f"d{i}",
            f"s{i}",
            n_frames=2,
            track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
        )
        for i in range(3)
    ]
    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_batch_tracked_tip_traits(
        serieses, write_json=True, output_dir=out_dir.as_posix()
    )
    parsed = json.loads((out_dir / "tracked_tip_batch_traits.json").read_text())
    assert isinstance(parsed, list)
    assert len(parsed) == 3
    for entry in parsed:
        assert "tracks" in entry
        assert "trajectories" in entry
        assert "schema_version" in entry


def test_compute_batch_tracked_tip_traits_emit_trajectories_false(tmp_path):
    """§10.4: batch with emit_trajectories=False → no batch trajectory CSV."""
    serieses = [
        _build_tracked_slp(
            tmp_path / f"d{i}",
            f"s{i}",
            n_frames=2,
            track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
        )
        for i in range(2)
    ]
    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_batch_tracked_tip_traits(
        serieses,
        write_csv=True,
        emit_trajectories=False,
        output_dir=out_dir.as_posix(),
    )
    assert (out_dir / "tracked_tip_batch_traits.csv").exists()
    assert not (out_dir / "tracked_tip_batch_trajectories.csv").exists()


def test_compute_batch_tracked_tip_traits_empty_input(tmp_path):
    """§10.5: empty list → empty outputs, no crash."""
    out_dir = tmp_path / "batch_out"
    out_dir.mkdir()
    TrackedTipPipeline().compute_batch_tracked_tip_traits(
        [], write_json=True, output_dir=out_dir.as_posix()
    )
    parsed = json.loads((out_dir / "tracked_tip_batch_traits.json").read_text())
    assert parsed == []


# ---------------------------------------------------------------------------
# §12 — real-fixture integration tests
# ---------------------------------------------------------------------------

REAL_FIXTURE_SLP = "tests/data/circumnutation_plate/plate_001_greyscale.tracked.slp"
REAL_FIXTURE_CSV = "tests/data/circumnutation_plate/fixture_metadata.csv"


@pytest.mark.skipif(
    not Path(REAL_FIXTURE_SLP).exists(),
    reason=f"Real fixture not present: {REAL_FIXTURE_SLP}",
)
def test_real_fixture_with_csv_metadata():
    """§12.1: real fixture + CSV → sample_uid, timepoint, 6 tracks, 1866 trajectory rows."""
    series = Series.load(
        series_name="plate_001",
        primary_path=REAL_FIXTURE_SLP,
        csv_path=REAL_FIXTURE_CSV,
        sample_uid="plate_001",
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["sample_uid"] == "plate_001"
    assert result["timepoint"] == 0.0
    assert len(result["tracks"]) == 6
    assert len(result["trajectories"]) == 1866  # 311 frames × 6 tracked instances
    for row in result["tracks"]:
        assert 0.0 <= row["tracking_coverage"] <= 1.0


@pytest.mark.skipif(
    not Path(REAL_FIXTURE_SLP).exists(),
    reason=f"Real fixture not present: {REAL_FIXTURE_SLP}",
)
def test_real_fixture_no_csv_metadata():
    """§12.2: real fixture without CSV → sample_uid defaults, timepoint NaN."""
    series = Series.load(
        series_name="plate_001",
        primary_path=REAL_FIXTURE_SLP,
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert result["sample_uid"] == "plate_001"
    assert np.isnan(result["timepoint"])
    assert len(result["tracks"]) == 6
    assert len(result["trajectories"]) == 1866


@pytest.mark.skipif(
    not Path(REAL_FIXTURE_SLP).exists(),
    reason=f"Real fixture not present: {REAL_FIXTURE_SLP}",
)
def test_real_fixture_track_id_values():
    """§12.3: track names are track_0 .. track_5 (verified during brainstorm)."""
    series = Series.load(
        series_name="plate_001",
        primary_path=REAL_FIXTURE_SLP,
    )
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    track_ids = {row["track_id"] for row in result["tracks"]}
    assert track_ids == {
        "track_0",
        "track_1",
        "track_2",
        "track_3",
        "track_4",
        "track_5",
    }


@pytest.mark.skipif(
    not Path(REAL_FIXTURE_SLP).exists(),
    reason=f"Real fixture not present: {REAL_FIXTURE_SLP}",
)
def test_real_fixture_track_order_non_determinism_handled():
    """§12.4: positional order differs across frames in source — pipeline still groups correctly."""
    labels = sio.load_slp(REAL_FIXTURE_SLP)
    # Confirm the brainstorm-verified non-determinism: frame 0 vs frame 1
    # have different positional ordering of track names.
    f0_order = [inst.track.name for inst in labels.labeled_frames[0].instances]
    f1_order = [inst.track.name for inst in labels.labeled_frames[1].instances]
    assert f0_order != f1_order, (
        "Real fixture should have non-deterministic track-positional order "
        "across frames; if this assertion fails, the fixture data has "
        "changed."
    )

    # The pipeline groups correctly regardless.
    series = Series.load(series_name="plate_001", primary_path=REAL_FIXTURE_SLP)
    result = TrackedTipPipeline().compute_tracked_tip_traits(series)
    assert {row["track_id"] for row in result["tracks"]} == {
        "track_0",
        "track_1",
        "track_2",
        "track_3",
        "track_4",
        "track_5",
    }
