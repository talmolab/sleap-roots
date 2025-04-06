import sleap_io as sio
import numpy as np
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
            plt = series.plot(frame_idx)
            assert isinstance(plt, matplotlib.figure.Figure)
