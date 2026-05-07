import sleap_io as sio
import numpy as np
import pandas as pd
import pytest
from sleap_roots.series import (
    Series,
    find_all_slp_paths,
    find_all_h5_paths,
    load_series_from_h5s,
    load_series_from_slps,
)
from pathlib import Path
from typing import Literal
from contextlib import redirect_stdout
import io
import matplotlib.figure


@pytest.fixture
def series_instance():
    # Create a Series instance with dummy data
    return Series(
        series_name="dummy",
        h5_path="dummy.h5",
        primary_path="dummy.model1.rootprimary.slp",
        lateral_path="dummy.model1.rootlateral.slp",
    )


@pytest.fixture
def dummy_video_path(tmp_path):
    video_path = tmp_path / "dummy.h5"
    video_path.write_text("This is a dummy video file.")
    return str(video_path)


@pytest.fixture(params=["primary", "lateral", "crown"])
def label_type(request):
    """Yields root types for tests, one by one."""
    return request.param


@pytest.fixture
def dummy_labels_path(tmp_path, label_type):
    labels_path = tmp_path / f"dummy.model1.root{label_type}.slp"
    # Simulate the structure of a SLEAP labels file.
    labels_path.write_text("Dummy SLEAP labels content.")
    return str(labels_path)


@pytest.fixture
def dummy_series(dummy_video_path, label_type, dummy_labels_path):
    # Assuming dummy_labels_path names are formatted as
    # "dummy.model1.root{label_type}.slp"

    # Construct the keyword argument for Series.load()
    kwargs = {
        "series_name": "dummy",
        "h5_path": dummy_video_path,
        f"{label_type}_path": dummy_labels_path,
    }
    return Series.load(**kwargs)


@pytest.fixture
def csv_path(tmp_path):
    # Create a dummy CSV file
    csv_path = tmp_path / "dummy.csv"
    csv_path.write_text(
        "plant_qr_code,number_of_plants_cylinder,genotype,qc_cylinder\ndummy,10,1100,0\nseries2,15,Kitaake-X,1\n"
    )
    return csv_path


def test_primary_prediction_not_found(tmp_path):
    dummy_video_path = Path(tmp_path) / "dummy_video.mp4"
    dummy_video_path.write_text("This is a dummy video file.")

    # Create a dummy Series instance with a non-existent primary prediction file
    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(
            series_name="dummy_video",
            h5_path=dummy_video_path,
            primary_path="dummy_video.model1.rootprimary.slp",
        )

    # format file path string for assert statement
    new_file_path = Path(dummy_video_path).with_suffix("").as_posix()
    print(new_file_path)

    assert (
        output.getvalue()
        == f"Primary prediction file not found: dummy_video.model1.rootprimary.slp\n"
    )


def test_lateral_prediction_not_found(tmp_path):
    dummy_video_path = Path(tmp_path) / "dummy_video.mp4"
    dummy_video_path.write_text("This is a dummy video file.")

    # Create a dummy Series instance with a non-existent primary prediction file
    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(
            series_name="dummy_video",
            h5_path=dummy_video_path,
            lateral_path="dummy_video.model1.rootlateral.slp",
        )

    # format file path string for assert statement
    new_file_path = Path(dummy_video_path).with_suffix("").as_posix()

    assert (
        output.getvalue()
        == f"Lateral prediction file not found: dummy_video.model1.rootlateral.slp\n"
    )


def test_crown_prediction_not_found(tmp_path):
    dummy_video_path = Path(tmp_path) / "dummy_video.mp4"
    dummy_video_path.write_text("This is a dummy video file.")

    # Create a dummy Series instance with a non-existent primary prediction file
    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(
            series_name="dummy_video",
            h5_path=dummy_video_path,
            crown_path="dummy_video.model1.rootcrown.slp",
        )

    # format file path string for assert statement
    new_file_path = Path(dummy_video_path).with_suffix("").as_posix()

    assert (
        output.getvalue()
        == f"Crown prediction file not found: dummy_video.model1.rootcrown.slp\n"
    )


def test_video_loading_error(tmp_path):
    # Create a dummy Series instance with an invalid video file path
    invalid_video_path = Path(tmp_path) / "invalid_video.mp4"

    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(series_name="invalid_video", h5_path=invalid_video_path)

    # Check if the correct error message is output
    assert output.getvalue() == f"Video file not found: {invalid_video_path}\n"


def test_get_frame(dummy_series):
    frame_idx = 0
    frames = dummy_series.get_frame(frame_idx)
    assert isinstance(frames, dict)
    assert "primary" in frames
    assert "lateral" in frames
    assert "crown" in frames


def test_series_name(series_instance):
    assert series_instance.series_name == "dummy"


def test_expected_count(series_instance, csv_path):
    series_instance.csv_path = csv_path
    assert series_instance.expected_count == 10


def test_expected_count_error(series_instance, tmp_path):
    series_instance.csv_path = tmp_path / "invalid"

    output = io.StringIO()
    with redirect_stdout(output):
        series_instance.expected_count
    # Check if the correct error message is output
    assert output.getvalue() == "CSV path is not set or the file does not exist.\n"


def test_qc_cylinder(series_instance, csv_path):
    series_instance.csv_path = csv_path
    assert series_instance.qc_fail == 0


def test_group(series_instance, csv_path):
    """`Series.group` returns the genotype value via the get_metadata wrapper."""
    series_instance.csv_path = csv_path
    # The csv_path fixture has a mixed-dtype genotype column ("1100", "Kitaake-X"),
    # so pandas infers object dtype and the value comes back as a string.
    assert str(series_instance.group) == "1100"


def test_wrappers_use_sample_uid_not_series_name(tmp_path):
    """Regression: expected_count/group/qc_fail key on sample_uid, not series_name.

    Pre-refactor the wrappers used `df["plant_qr_code"] == self.series_name`. Post-
    refactor they call `get_metadata` which keys on `self.sample_uid`. With sample_uid
    defaulting to series_name the byte-for-byte default behavior is preserved. This
    test pins the new (sample_uid-keyed) lookup explicitly.
    """
    csv = tmp_path / "m.csv"
    csv.write_text(
        "plant_qr_code,number_of_plants_cylinder,genotype,qc_cylinder\n"
        "X,7,GENO_X,0\n"
    )
    series = Series.load(
        series_name="X_day0",
        sample_uid="X",
        csv_path=str(csv),
    )
    assert series.expected_count == 7
    assert series.group == "GENO_X"
    assert series.qc_fail == 0


def test_series_get_metadata_csv_path_set_but_file_missing(tmp_path):
    """csv_path set but file missing → `get_metadata` returns NaN (no raise)."""
    series = Series(series_name="x", csv_path=str(tmp_path / "nonexistent.csv"))
    assert np.isnan(series.get_metadata("genotype"))


def test_series_get_metadata_rejects_bool_plant_id(tmp_path):
    """Booleans collide with int 0/1 in pandas equality — reject them upfront."""
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,plant_id,genotype\nplant1,0,A\nplant1,1,B\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    with pytest.raises(TypeError) as excinfo:
        series.get_metadata("genotype", plant_id=False)
    assert "bool" in str(excinfo.value).lower()
    with pytest.raises(TypeError):
        series.get_metadata("genotype", plant_id=True)


def test_series_timepoint_rejects_infinity(tmp_path):
    """Non-finite timepoints would silently produce nonsensical deltas — reject them."""
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,timepoint\nplant1,inf\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    with pytest.raises(ValueError) as excinfo:
        series.timepoint
    msg = str(excinfo.value).lower()
    assert "non-finite" in msg or "infinite" in msg or "inf" in msg
    assert "plant1" in str(excinfo.value)


def test_series_get_metadata_csv_missing_plant_qr_code_column(tmp_path, caplog):
    """CSV without `plant_qr_code` lookup-key column → NaN + WARNING (no KeyError).

    Regression for the fail-soft contract: if a user supplies a misconfigured CSV
    (no `plant_qr_code` column at all), the method must NOT raise `KeyError`.
    Instead it logs a WARNING and returns NaN, so wrapper properties (timepoint,
    expected_count, etc.) keep working on malformed metadata files.
    """
    csv = _write_csv(
        tmp_path / "no_lookup_key.csv",
        "id,genotype,timepoint\nplant1,MK22,3\n",  # `id` not `plant_qr_code`
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    with caplog.at_level("WARNING", logger="sleap_roots.series"):
        result = series.get_metadata("genotype")
    assert pd.isna(result)
    warnings = [
        r
        for r in caplog.records
        if r.name == "sleap_roots.series" and r.levelname == "WARNING"
    ]
    assert len(warnings) >= 1
    assert "plant_qr_code" in warnings[0].message
    # Wrapper properties must also stay fail-soft.
    assert pd.isna(series.timepoint)
    assert pd.isna(series.expected_count)


def test_series_timepoint_nan_cell_in_existing_row(tmp_path):
    """Empty-string timepoint in a matching row → NaN (collapses with no-row case)."""
    csv = tmp_path / "m.csv"
    csv.write_text("plant_qr_code,timepoint\nplant1,\n")
    series = Series(series_name="plant1", csv_path=str(csv))
    assert np.isnan(series.timepoint)


def test_len_video():
    series = Series(series_name="test_video", video=["frame1", "frame2"])
    assert len(series) == 2


def test_len_no_video():
    series = Series(series_name="test_no_video", video=None)
    assert len(series) == 0


def test_series_load_canola(
    canola_h5: Literal["tests/data/canola_7do/919QDUH.h5"],
    canola_primary_slp: Literal[
        "tests/data/canola_7do/919QDUH.primary.predictions.slp"
    ],
):
    series = Series.load(
        series_name="919QDUH", h5_path=canola_h5, primary_path=canola_primary_slp
    )
    assert len(series) == 72


def test_find_all_series_from_slps(
    sleap_roots_pipeline_output_folder: Literal[
        "/tests/data/sleap-roots-pipeline-output"
    ],
):
    slp_paths = find_all_slp_paths(sleap_roots_pipeline_output_folder)
    all_series = load_series_from_slps(slp_paths)
    assert len(slp_paths) == 2
    assert len(all_series) == 1

    # Test the first series
    series = all_series[0]
    assert series.series_name == "scan6791737"
    assert series.h5_path == None
    assert (
        series.primary_path
        == Path(
            "tests/data/sleap-roots-pipeline-outputs/scan6791737.model230104_182346.multi_instance.n=720.rootprimary.slp"
        ).as_posix()
    )
    assert series.lateral_path == None
    assert (
        series.crown_path
        == Path(
            "tests/data/sleap-roots-pipeline-outputs/scan6791737.model220821_163331.multi_instance.n=867.rootcrown.slp"
        ).as_posix()
    )

    # Test the first frame of the first series
    frame_index = 0
    labeled_frames = series.get_frame(frame_index)
    primary_points = series.get_primary_points(frame_index)
    crown_points = series.get_crown_points(frame_index)
    assert len(labeled_frames) == 3
    assert "primary" in labeled_frames
    assert labeled_frames["lateral"] == None
    assert "crown" in labeled_frames
    assert primary_points.shape == (1, 6, 2)
    assert crown_points.shape == (1, 6, 2)


def test_load_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
    rice_main_10do_slp: Literal["tests/data/rice_10do/0K9E8BI.crown.predictions.slp"],
):
    series = Series.load(
        series_name="0K9E8BI", h5_path=rice_main_10do_h5, crown_path=rice_main_10do_slp
    )
    expected_video = sio.Video.from_filename(rice_main_10do_h5)

    assert len(series) == 72
    assert series.h5_path == rice_main_10do_h5
    assert series.video.filename == expected_video.filename


def test_get_frame_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
    rice_main_10do_slp: Literal["tests/data/rice_10do/0K9E8BI.crown.predictions.slp"],
):
    # Set the frame index to 0
    frame_idx = 0

    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(rice_main_10do_slp)
    # Get the first labeled frame
    expected_labeled_frame = expected_labels[0]

    # Load the series
    series = Series.load(
        series_name="0K9E8BI", h5_path=rice_main_10do_h5, crown_path=rice_main_10do_slp
    )
    # Retrieve all available frames
    frames = series.get_frame(frame_idx)
    # Get the crown labeled frame
    crown_lf = frames.get("crown")

    # Compare instances
    for i in range(len(expected_labeled_frame.instances)):

        crown_instance = crown_lf.instances[i]
        expected_labeled_frame_instance = expected_labeled_frame.instances[i]

        assert np.allclose(
            crown_instance.numpy(),
            expected_labeled_frame_instance.numpy(),
            atol=1e-7,
            equal_nan=True,
        )

        assert (
            (crown_instance.track is None)
            and (expected_labeled_frame_instance.track is None)
        ) or (crown_instance.track is expected_labeled_frame_instance.track)

        assert np.isclose(
            crown_instance.score,
            expected_labeled_frame_instance.score,
            atol=1e-7,
            equal_nan=True,
        )
        assert np.isclose(
            crown_instance.tracking_score,
            expected_labeled_frame_instance.tracking_score,
            atol=1e-7,
            equal_nan=True,
        )

    # Compare the attributes of the labeled frames
    assert crown_lf.frame_idx == expected_labeled_frame.frame_idx
    assert crown_lf.video.filename == expected_labeled_frame.video.filename
    assert crown_lf.video.shape == expected_labeled_frame.video.shape
    assert crown_lf.video.backend == expected_labeled_frame.video.backend
    assert series.series_name == "0K9E8BI"


def test_get_frame_rice_10do_no_video(
    rice_main_10do_slp: Literal["tests/data/rice_10do/0K9E8BI.crown.predictions.slp"],
):
    # Set the frame index to 0
    frame_idx = 0

    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(rice_main_10do_slp)
    # Get the first labeled frame
    expected_labeled_frame = expected_labels[frame_idx]

    # Load the series
    series = Series.load(series_name="0K9E8BI", crown_path=rice_main_10do_slp)
    # Retrieve all available frames
    frames = series.get_frame(frame_idx)
    # Get the crown labeled frame
    crown_lf = frames.get("crown")

    # Compare instances
    for i in range(len(expected_labeled_frame.instances)):

        crown_instance = crown_lf.instances[i]
        expected_labeled_frame_instance = expected_labeled_frame.instances[i]

        assert np.allclose(
            crown_instance.numpy(),
            expected_labeled_frame_instance.numpy(),
            atol=1e-7,
            equal_nan=True,
        )

        assert (
            (crown_instance.track is None)
            and (expected_labeled_frame_instance.track is None)
        ) or (crown_instance.track is expected_labeled_frame_instance.track)

        assert np.isclose(
            crown_instance.score,
            expected_labeled_frame_instance.score,
            atol=1e-7,
            equal_nan=True,
        )
        assert np.isclose(
            crown_instance.tracking_score,
            expected_labeled_frame_instance.tracking_score,
            atol=1e-7,
            equal_nan=True,
        )

    # Compare the attributes of the labeled frames
    assert crown_lf.frame_idx == expected_labeled_frame.frame_idx
    assert crown_lf.video.filename == expected_labeled_frame.video.filename
    assert crown_lf.video.shape == expected_labeled_frame.video.shape
    assert crown_lf.video.backend == expected_labeled_frame.video.backend
    assert series.series_name == "0K9E8BI"


def test_find_all_series_from_h5s_rice_10do(
    rice_10do_folder: Literal["tests/data/rice_10do"],
):
    """Test finding and loading all `Series` from a folder with h5 files.

    To load the `Series`, the files must be named with the following convention:
    h5_path: '/path/to/scan/series_name.h5'
    primary_path: '/path/to/scan/series_name.model{model_id}.rootprimary.slp'
    lateral_path: '/path/to/scan/series_name.model{model_id}.rootlateral.slp'
    crown_path: '/path/to/scan/series_name.model{model_id}.rootcrown.slp'
    """
    series_h5_path = Path(rice_10do_folder) / "0K9E8BI.h5"
    h5_paths = find_all_h5_paths(rice_10do_folder)
    all_series = load_series_from_h5s(h5_paths, model_id="123")

    assert len(h5_paths) == 1
    assert len(all_series) == 1
    assert all_series[0].h5_path == series_h5_path.as_posix()


def test_series_plot(
    rice_h5,
    rice_long_slp,
    rice_main_slp,
    canola_h5,
    canola_primary_slp,
    canola_lateral_slp,
):
    """Test return type of the Series `plot` method."""
    # Younger Monocot, rice_3do, YR39SJX
    rice_3do = Series.load(
        series_name="rice",
        primary_path=rice_long_slp,
        crown_path=rice_main_slp,
        h5_path=rice_h5,
    )

    # Dicot, canola_7do, 919QDUH
    canola = rice_3do = Series.load(
        series_name="canola",
        primary_path=canola_primary_slp,
        lateral_path=canola_lateral_slp,
        h5_path=canola_h5,
    )

    series_examples = [rice_3do, canola]

    for series in series_examples:
        for frame_idx in range(len(series)):
            fig = series.plot(frame_idx)
            assert isinstance(fig, matplotlib.figure.Figure)


# Section 1: sample_uid tests (TDD red phase 1)


def test_series_sample_uid_defaults_to_series_name():
    series = Series.load(series_name="plant1")
    assert series.sample_uid == "plant1"


def test_series_sample_uid_explicit_kwarg():
    series = Series.load(series_name="plant1_day0", sample_uid="plant1")
    assert series.sample_uid == "plant1"


def test_series_sample_uid_empty_string_falls_through():
    series = Series.load(series_name="plant1_day0", sample_uid="")
    assert series.sample_uid == "plant1_day0"


def test_series_sample_uid_shared_across_series():
    a = Series.load(series_name="plant1_day0", sample_uid="plant1")
    b = Series.load(series_name="plant1_day1", sample_uid="plant1")
    assert a.series_name != b.series_name
    assert a.sample_uid == b.sample_uid == "plant1"


def test_series_sample_uid_direct_construction_defaults():
    series = Series(series_name="test_video")
    assert series.sample_uid == "test_video"


def test_series_sample_uid_str_coercion():
    series = Series(series_name="x", sample_uid=1002)
    assert series.sample_uid == "1002"
    assert isinstance(series.sample_uid, str)


# Section 3: get_metadata tests (TDD red phase 2)


def _write_csv(path, text):
    path.write_text(text)
    return path


def test_series_get_metadata_no_csv():
    series = Series(series_name="plant1")
    assert np.isnan(series.get_metadata("number_of_plants_cylinder"))


def test_series_get_metadata_missing_column(tmp_path):
    csv = _write_csv(tmp_path / "m.csv", "plant_qr_code,genotype\nplant1,MK22\n")
    series = Series(series_name="plant1", csv_path=str(csv))
    assert np.isnan(series.get_metadata("timepoint"))


def test_series_get_metadata_no_matching_row(tmp_path):
    csv = _write_csv(tmp_path / "m.csv", "plant_qr_code,genotype\na,MK22\nb,MK23\n")
    series = Series(series_name="c", csv_path=str(csv))
    assert pd.isna(series.get_metadata("genotype"))


def test_series_get_metadata_matches_row(tmp_path):
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,genotype,timepoint\nplant1,MK22,3\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    assert series.get_metadata("genotype") == "MK22"
    assert series.get_metadata("timepoint") == 3


def test_series_get_metadata_plant_id_composite_lookup(tmp_path):
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,plant_id,genotype\nplant1,0,A\nplant1,1,B\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    assert series.get_metadata("genotype", plant_id=0) == "A"
    assert series.get_metadata("genotype", plant_id=1) == "B"


def test_series_get_metadata_plant_id_ignored_when_no_column_emits_warning(
    tmp_path, caplog
):
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,genotype,timepoint\nplant1,MK22,3\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    with caplog.at_level("WARNING", logger="sleap_roots.series"):
        result = series.get_metadata("genotype", plant_id=99)
    assert result == "MK22"
    warnings = [
        r
        for r in caplog.records
        if r.name == "sleap_roots.series" and r.levelname == "WARNING"
    ]
    assert len(warnings) == 1
    assert "plant_id" in warnings[0].message
    assert str(csv) in warnings[0].message or "m.csv" in warnings[0].message

    # Second call: no additional warning (one-shot dedup)
    caplog.clear()
    with caplog.at_level("WARNING", logger="sleap_roots.series"):
        series.get_metadata("timepoint", plant_id=99)
    second = [
        r
        for r in caplog.records
        if r.name == "sleap_roots.series" and r.levelname == "WARNING"
    ]
    assert len(second) == 0


def test_series_get_metadata_plant_id_none_equivalent_to_omitted(tmp_path):
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,genotype\nplant1,MK22\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    assert series.get_metadata("genotype", plant_id=None) == series.get_metadata(
        "genotype"
    )


def test_series_get_metadata_multiple_matches_first_row(tmp_path):
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,genotype\nplant1,A\nplant1,B\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    assert series.get_metadata("genotype") == "A"


# Section 5: timepoint tests (TDD red phase 3)


def test_series_timepoint_from_csv_numeric(tmp_path):
    csv = _write_csv(tmp_path / "m.csv", "plant_qr_code,timepoint\nplant1,2\n")
    series = Series(series_name="plant1", csv_path=str(csv))
    assert series.timepoint == 2.0
    assert isinstance(series.timepoint, float)


def test_series_timepoint_no_csv():
    series = Series(series_name="plant1")
    assert np.isnan(series.timepoint)


def test_series_timepoint_string_float_parses(tmp_path):
    csv = _write_csv(tmp_path / "m.csv", "plant_qr_code,timepoint\nplant1,3.5\n")
    series = Series(series_name="plant1", csv_path=str(csv))
    assert series.timepoint == 3.5


def test_series_timepoint_raises_on_non_numeric(tmp_path):
    csv = _write_csv(
        tmp_path / "m.csv",
        "plant_qr_code,timepoint\nplant1,2024-03-15\n",
    )
    series = Series(series_name="plant1", csv_path=str(csv))
    with pytest.raises(ValueError) as excinfo:
        series.timepoint
    msg = str(excinfo.value)
    assert "plant1" in msg
    assert "timepoint" in msg
    assert "2024-03-15" in msg


# ---------------------------------------------------------------------------
# Synthetic tracked-`.slp` builder + tests for Series.get_tracked_tips
# (Section 2 of openspec/changes/add-tracked-tip-pipeline/tasks.md)
# ---------------------------------------------------------------------------


def _build_tracked_slp(
    tmp_path,
    series_name: str,
    n_frames: int,
    track_positions,
    *,
    root_type: str = "primary",
    skeleton_node_names=("r0",),
    instance_order_per_frame=None,
    untracked_at=None,
    empty_track_name_at=None,
    other_root_paths=None,
    csv_content: str = None,
    sample_uid: str = None,
):
    """Build a synthetic tracked .slp and return a `Series`.

    Args:
        tmp_path: pytest tmp_path fixture.
        series_name: Series identifier.
        n_frames: Number of frames in the .slp.
        track_positions: dict mapping track_name (str) → list of (x, y) tuples
            of length n_frames. A `None` entry in the list means the track has
            no instance in that frame (gap).
        root_type: "primary", "lateral", or "crown" — determines which `*_path`
            kwarg is set on Series.load.
        skeleton_node_names: Tuple of node names. The LAST node is treated as
            the tip by skeleton convention. Default `("r0",)` is the
            single-node circumnutation skeleton.
        instance_order_per_frame: Optional list of length `n_frames`, each
            element a list of track names specifying instance positional order
            within that frame. Defaults to the dict iteration order. Use this
            to test the brainstorm-verified track-order non-determinism.
        untracked_at: Optional list of (frame_idx, track_name) tuples. For each,
            the corresponding instance is created with track=None instead of a
            real Track.
        empty_track_name_at: Optional list of (frame_idx, track_name) tuples.
            Each gets a Track with name="" (empty string) — should trip the
            same untracked-instance error path.
        other_root_paths: Optional dict mapping root-type strings ("primary"/
            "lateral"/"crown") to None or another tracked .slp path, for
            testing multi-path-populated cases. None means leave kwarg unset
            on Series.load.
        csv_content: Optional CSV text to write to a sidecar.
        sample_uid: Optional sample_uid kwarg for Series.load.

    Returns:
        A `Series` loaded via `Series.load`.
    """
    Path(tmp_path).mkdir(parents=True, exist_ok=True)
    image_h, image_w = 200, 200
    img_array = np.zeros((image_h, image_w), dtype=np.uint8)
    tif_path = tmp_path / f"{series_name}.tif"
    from PIL import Image

    Image.fromarray(img_array).save(tif_path.as_posix(), dpi=(72, 72))
    video = sio.Video.from_filename(tif_path.as_posix())

    skeleton = sio.Skeleton(nodes=[sio.Node(n) for n in skeleton_node_names])
    n_nodes = len(skeleton_node_names)
    untracked_set = set(untracked_at or [])
    empty_name_set = set(empty_track_name_at or [])
    tracks_by_name = {name: sio.Track(name=name) for name in track_positions.keys()}

    labeled_frames = []
    for frame_idx in range(n_frames):
        if instance_order_per_frame is not None:
            order = instance_order_per_frame[frame_idx]
        else:
            order = list(track_positions.keys())

        instances = []
        for track_name in order:
            xy = track_positions[track_name][frame_idx]
            if xy is None:
                continue  # track has a gap in this frame
            # Build the (n_nodes, 2) point array — replicate the (x, y) at
            # every node for multi-node skeletons (the [-1] tip convention
            # picks up the last node identically regardless).
            pts = np.tile(np.asarray(xy, dtype=float), (n_nodes, 1))

            if (frame_idx, track_name) in untracked_set:
                track_arg = None
            elif (frame_idx, track_name) in empty_name_set:
                track_arg = sio.Track(name="")
            else:
                track_arg = tracks_by_name[track_name]

            inst = sio.Instance.from_numpy(pts, skeleton=skeleton, track=track_arg)
            instances.append(inst)

        labeled_frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=instances)
        )

    labels = sio.Labels(
        labeled_frames=labeled_frames,
        skeletons=[skeleton],
        videos=[video],
        tracks=list(tracks_by_name.values()),
    )

    slp_path = tmp_path / f"{series_name}.{root_type}.tracked.slp"
    sio.save_slp(labels, slp_path.as_posix())

    csv_path = None
    if csv_content is not None:
        csv_path = tmp_path / f"{series_name}.csv"
        csv_path.write_text(csv_content)

    load_kwargs = {
        "series_name": series_name,
        f"{root_type}_path": slp_path.as_posix(),
        "csv_path": csv_path.as_posix() if csv_path else None,
        "sample_uid": sample_uid,
    }
    if other_root_paths:
        for k, v in other_root_paths.items():
            load_kwargs[f"{k}_path"] = v
    return Series.load(**load_kwargs)


def test_get_tracked_tips_returns_long_dataframe(tmp_path):
    """§2.2: synthetic 3-frame, 2-track .slp → DataFrame with expected cols + 6 rows."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={
            "track_a": [(10.0, 20.0), (11.0, 21.0), (12.0, 22.0)],
            "track_b": [(30.0, 40.0), (31.0, 41.0), (32.0, 42.0)],
        },
    )
    df = series.get_tracked_tips()
    assert list(df.columns) == ["track_id", "frame", "tip_x", "tip_y"]
    assert len(df) == 6


def test_get_tracked_tips_sorted_by_track_then_frame(tmp_path):
    """§2.3: instance positional order randomized per frame → output is frame-sorted within each track."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={
            "track_a": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            "track_b": [(10.0, 10.0), (11.0, 10.0), (12.0, 10.0)],
        },
        instance_order_per_frame=[
            ["track_a", "track_b"],
            ["track_b", "track_a"],
            ["track_b", "track_a"],
        ],
    )
    df = series.get_tracked_tips()
    # Within each track group, frame is monotonically increasing.
    for track_id, group in df.groupby("track_id"):
        assert list(group["frame"]) == sorted(group["frame"])
    # Output equals its (track_id, frame)-sorted version.
    expected = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(df, expected)


def test_get_tracked_tips_auto_detects_root_type_from_populated_path(tmp_path):
    """§2.4: only primary_path populated → no root_type kwarg needed.

    Strengthened per /review-pr feedback: assert specific xy values to
    verify the impl actually read the primary path. The previous version
    only checked `len(df) == 2`, which would also pass if the impl
    accidentally read a different (empty) path.
    """
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(7.5, 8.5), (9.5, 10.5)]},
        root_type="primary",
    )
    df = series.get_tracked_tips()  # no root_type kwarg
    assert len(df) == 2
    # Verify the values came from the primary path, not silently from
    # somewhere else.
    assert df.iloc[0]["tip_x"] == 7.5
    assert df.iloc[0]["tip_y"] == 8.5
    assert df.iloc[1]["tip_x"] == 9.5
    assert df.iloc[1]["tip_y"] == 10.5


def test_get_tracked_tips_raises_when_multiple_paths_populated_no_root_type(tmp_path):
    """§2.5: primary + lateral populated, no root_type → ValueError mentioning root_type."""
    # Build two separate tracked .slp files and load them as a single Series.
    s_primary = _build_tracked_slp(
        tmp_path / "p",
        "s",
        n_frames=1,
        track_positions={"t": [(0.0, 0.0)]},
        root_type="primary",
    )
    s_lateral = _build_tracked_slp(
        tmp_path / "l",
        "s",
        n_frames=1,
        track_positions={"t": [(0.0, 0.0)]},
        root_type="lateral",
    )
    series = Series.load(
        series_name="s",
        primary_path=s_primary.primary_path,
        lateral_path=s_lateral.lateral_path,
    )
    with pytest.raises(ValueError) as excinfo:
        series.get_tracked_tips()
    assert "root_type" in str(excinfo.value)


def test_get_tracked_tips_raises_when_zero_paths_populated():
    """§2.6: no paths populated → ValueError mentioning root_type."""
    series = Series(series_name="s")
    with pytest.raises(ValueError) as excinfo:
        series.get_tracked_tips()
    assert "root_type" in str(excinfo.value)


def test_get_tracked_tips_raises_on_untracked_instance(tmp_path):
    """§2.7: instance with track=None → ValueError mentioning frame index + sleap.ai/tracking URL."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={
            "track_a": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            "track_b": [(10.0, 10.0), (11.0, 10.0), (12.0, 10.0)],
        },
        untracked_at=[(1, "track_b")],  # frame 1, track_b is untracked
    )
    with pytest.raises(ValueError) as excinfo:
        series.get_tracked_tips()
    msg = str(excinfo.value)
    assert "1" in msg
    assert "sleap.ai" in msg


def test_get_tracked_tips_raises_on_empty_track_name(tmp_path):
    """§2.8: instance with empty Track.name → same ValueError as track=None."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
        empty_track_name_at=[(0, "t")],  # frame 0, track t has empty name
    )
    with pytest.raises(ValueError) as excinfo:
        series.get_tracked_tips()
    assert "0" in str(excinfo.value)


def test_get_tracked_tips_single_node_skeleton(tmp_path):
    """§2.9: single-node skeleton ['r0'] — tip_x/tip_y are the node coordinates."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(5.5, 7.5), (10.5, 20.5)]},
        skeleton_node_names=("r0",),  # 1-node skeleton
    )
    df = series.get_tracked_tips()
    assert df.iloc[0]["tip_x"] == 5.5
    assert df.iloc[0]["tip_y"] == 7.5
    assert df.iloc[1]["tip_x"] == 10.5
    assert df.iloc[1]["tip_y"] == 20.5


def test_get_tracked_tips_multi_node_skeleton(tmp_path):
    """§2.10: multi-node skeleton ['base','mid','tip'] — tip_x/tip_y are LAST node coords."""
    image_h, image_w = 200, 200
    img_array = np.zeros((image_h, image_w), dtype=np.uint8)
    tif_path = tmp_path / "s.tif"
    from PIL import Image

    Image.fromarray(img_array).save(tif_path.as_posix(), dpi=(72, 72))
    video = sio.Video.from_filename(tif_path.as_posix())
    skeleton = sio.Skeleton(nodes=[sio.Node("base"), sio.Node("mid"), sio.Node("tip")])
    track = sio.Track(name="t")

    # Build one frame with one track. Skeleton: base=(1,1), mid=(2,2), tip=(99,99).
    pts = np.array([[1.0, 1.0], [2.0, 2.0], [99.0, 99.0]])
    inst = sio.Instance.from_numpy(pts, skeleton=skeleton, track=track)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = sio.Labels(
        labeled_frames=[lf], skeletons=[skeleton], videos=[video], tracks=[track]
    )
    slp_path = tmp_path / "s.primary.tracked.slp"
    sio.save_slp(labels, slp_path.as_posix())

    series = Series.load(series_name="s", primary_path=slp_path.as_posix())
    df = series.get_tracked_tips()
    # The LAST node (tip=99,99), NOT base or mid.
    assert df.iloc[0]["tip_x"] == 99.0
    assert df.iloc[0]["tip_y"] == 99.0


# ---------------------------------------------------------------------------
# Tests for validate_tracked_slp + validate_series_for_tracked_tip
# (Section 4 of openspec/changes/add-tracked-tip-pipeline/tasks.md)
# ---------------------------------------------------------------------------


def test_validate_tracked_slp_passes_on_fully_tracked(tmp_path):
    """§4.1: every instance tracked → returns None, no raise."""
    from sleap_roots.series import validate_tracked_slp

    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]},
    )
    assert validate_tracked_slp(series.primary_path) is None


def test_validate_tracked_slp_raises_on_untracked_instance(tmp_path):
    """§4.2: one untracked instance → ValueError mentioning frame index + sleap.ai/tracking URL."""
    from sleap_roots.series import validate_tracked_slp

    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=3,
        track_positions={
            "ta": [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
            "tb": [(10.0, 10.0), (11.0, 10.0), (12.0, 10.0)],
        },
        untracked_at=[(2, "tb")],
    )
    with pytest.raises(ValueError) as excinfo:
        validate_tracked_slp(series.primary_path)
    msg = str(excinfo.value)
    assert "2" in msg
    assert "sleap.ai" in msg


def test_validate_tracked_slp_lists_all_offending_frames(tmp_path):
    """§4.3: 3 untracked instances on 3 frames → error message lists all 3."""
    from sleap_roots.series import validate_tracked_slp

    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=20,
        track_positions={"t": [(float(i), float(i)) for i in range(20)]},
        untracked_at=[(2, "t"), (7, "t"), (15, "t")],
    )
    with pytest.raises(ValueError) as excinfo:
        validate_tracked_slp(series.primary_path)
    msg = str(excinfo.value)
    assert "2" in msg
    assert "7" in msg
    assert "15" in msg


def test_validate_series_for_tracked_tip_resolves_root_type(tmp_path):
    """§4.4: only primary_path populated → no root_type kwarg needed."""
    from sleap_roots.series import validate_series_for_tracked_tip

    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    assert validate_series_for_tracked_tip(series) is None


def test_validate_series_for_tracked_tip_explicit_root_type(tmp_path):
    """§4.5: primary + lateral both valid; explicit root_type validates the chosen path."""
    from sleap_roots.series import validate_series_for_tracked_tip

    s_primary = _build_tracked_slp(
        tmp_path / "p",
        "s",
        n_frames=1,
        track_positions={"t": [(0.0, 0.0)]},
        root_type="primary",
    )
    s_lateral = _build_tracked_slp(
        tmp_path / "l",
        "s",
        n_frames=1,
        track_positions={"t": [(0.0, 0.0)]},
        root_type="lateral",
    )
    series = Series.load(
        series_name="s",
        primary_path=s_primary.primary_path,
        lateral_path=s_lateral.lateral_path,
    )
    assert validate_series_for_tracked_tip(series, root_type="primary") is None
    assert validate_series_for_tracked_tip(series, root_type="lateral") is None


def test_validate_series_for_tracked_tip_raises_on_zero_paths_no_root_type():
    """§4.6 part 1: zero paths populated → ValueError mentioning root_type."""
    from sleap_roots.series import validate_series_for_tracked_tip

    series = Series(series_name="s")
    with pytest.raises(ValueError) as excinfo:
        validate_series_for_tracked_tip(series)
    assert "root_type" in str(excinfo.value)


def test_get_tracked_tips_raises_on_invalid_root_type(tmp_path):
    """Invalid root_type string (e.g. 'foo') → user-facing ValueError, not KeyError."""
    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    with pytest.raises(ValueError) as excinfo:
        series.get_tracked_tips(root_type="foo")
    msg = str(excinfo.value)
    assert "foo" in msg
    assert "primary" in msg  # error lists valid options


def test_validate_series_for_tracked_tip_raises_on_invalid_root_type(tmp_path):
    """Invalid root_type string (e.g. 'foo') → user-facing ValueError, not KeyError."""
    from sleap_roots.series import validate_series_for_tracked_tip

    series = _build_tracked_slp(
        tmp_path,
        "s",
        n_frames=2,
        track_positions={"t": [(0.0, 0.0), (1.0, 1.0)]},
    )
    with pytest.raises(ValueError) as excinfo:
        validate_series_for_tracked_tip(series, root_type="foo")
    msg = str(excinfo.value)
    assert "foo" in msg
    assert "primary" in msg


def test_validate_series_for_tracked_tip_raises_on_multiple_paths_no_root_type(
    tmp_path,
):
    """§4.6 part 2: multiple paths populated, no root_type → ValueError mentioning root_type."""
    from sleap_roots.series import validate_series_for_tracked_tip

    s_primary = _build_tracked_slp(
        tmp_path / "p",
        "s",
        n_frames=1,
        track_positions={"t": [(0.0, 0.0)]},
        root_type="primary",
    )
    s_lateral = _build_tracked_slp(
        tmp_path / "l",
        "s",
        n_frames=1,
        track_positions={"t": [(0.0, 0.0)]},
        root_type="lateral",
    )
    series = Series.load(
        series_name="s",
        primary_path=s_primary.primary_path,
        lateral_path=s_lateral.lateral_path,
    )
    with pytest.raises(ValueError) as excinfo:
        validate_series_for_tracked_tip(series)
    assert "root_type" in str(excinfo.value)
