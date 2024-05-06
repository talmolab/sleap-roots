import sleap_io as sio
import numpy as np
import pytest
from sleap_roots.series import Series, find_all_series
from pathlib import Path
from typing import Literal
from contextlib import redirect_stdout
import io


@pytest.fixture
def series_instance():
    # Create a Series instance with dummy data
    return Series(h5_path="dummy.h5")


@pytest.fixture
def dummy_video_path(tmp_path):
    video_path = tmp_path / "dummy_video.mp4"
    video_path.write_text("This is a dummy video file.")
    return str(video_path)


@pytest.fixture(params=["primary", "lateral", "crown"])
def label_type(request):
    """Yields label types for tests, one by one."""
    return request.param


@pytest.fixture
def dummy_labels_path(tmp_path, label_type):
    labels_path = tmp_path / f"dummy.{label_type}.predictions.slp"
    # Simulate the structure of a SLEAP labels file.
    labels_path.write_text("Dummy SLEAP labels content.")
    return str(labels_path)


@pytest.fixture
def dummy_series(dummy_video_path, dummy_labels_path):
    # Assuming dummy_labels_path names are formatted as "{label_type}.predictions.slp"
    # Extract the label type (primary, lateral, crown) from the filename
    label_type = Path(dummy_labels_path).stem.split(".")[1]

    # Construct the keyword argument for Series.load()
    kwargs = {
        "h5_path": dummy_video_path,
        f"{label_type}_name": dummy_labels_path,
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
    dummy_video_path = tmp_path / "dummy_video.mp4"
    dummy_video_path.write_text("This is a dummy video file.")

    # Create a dummy Series instance with a non-existent primary prediction file
    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(h5_path=str(dummy_video_path), primary_name="nonexistent")

    # format file path string for assert statement
    new_file_path = Path(dummy_video_path).with_suffix('')

    assert (
        output.getvalue()
        == f"Primary prediction file not found: {new_file_path}.nonexistent.predictions.slp\n"
    )


def test_lateral_prediction_not_found(tmp_path):
    dummy_video_path = tmp_path / "dummy_video.mp4"
    dummy_video_path.write_text("This is a dummy video file.")

    # Create a dummy Series instance with a non-existent primary prediction file
    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(h5_path=str(dummy_video_path), lateral_name="nonexistent")

    # format file path string for assert statement
    new_file_path = Path(dummy_video_path).with_suffix('')

    assert (
        output.getvalue()
        == f"Lateral prediction file not found: {new_file_path}.nonexistent.predictions.slp\n"
    )


def test_crown_prediction_not_found(tmp_path):
    dummy_video_path = tmp_path / "dummy_video.mp4"
    dummy_video_path.write_text("This is a dummy video file.")

    # Create a dummy Series instance with a non-existent primary prediction file
    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(h5_path=str(dummy_video_path), crown_name="nonexistent")

    # format file path string for assert statement
    new_file_path = Path(dummy_video_path).with_suffix('')

    assert (
        output.getvalue()
        == f"Crown prediction file not found: {new_file_path}.nonexistent.predictions.slp\n"
    )


def test_video_loading_error(tmp_path):
    # Create a dummy Series instance with an invalid video file path
    invalid_video_path = tmp_path / "invalid_video.mp4"

    output = io.StringIO()
    with redirect_stdout(output):
        Series.load(h5_path=str(invalid_video_path))

    # Check if the correct error message is output
    assert (
        output.getvalue()
        == f"Error loading video file {invalid_video_path}: File not found\n"
    )


def test_series_name(dummy_series):
    expected_name = "dummy_video"  # Based on the dummy_video_path fixture
    assert dummy_series.series_name == expected_name


def test_get_frame(dummy_series):
    frame_idx = 0
    frames = dummy_series.get_frame(frame_idx)
    assert isinstance(frames, dict)
    assert "primary" in frames
    assert "lateral" in frames
    assert "crown" in frames


def test_series_name_property():
    series = Series(h5_path="mock_path/file_name.h5")
    assert series.series_name == "file_name"


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


def test_len():
    series = Series(video=["frame1", "frame2"])
    assert len(series) == 2


def test_series_load_canola(canola_h5: Literal["tests/data/canola_7do/919QDUH.h5"]):
    series = Series.load(canola_h5, primary_name="primary", lateral_name="lateral")
    assert len(series) == 72


def test_find_all_series_canola(canola_folder: Literal["tests/data/canola_7do"]):
    all_series_files = find_all_series(canola_folder)
    assert len(all_series_files) == 1


def test_load_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
):
    series = Series.load(rice_main_10do_h5, crown_name="crown")
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
    series = Series.load(rice_main_10do_h5, crown_name="crown")
    # Retrieve all available frames
    frames = series.get_frame(frame_idx)
    # Get the crown labeled frame
    crown_lf = frames.get("crown")
    assert crown_lf == expected_labeled_frame
    # Check the series name property
    assert series.series_name == "0K9E8BI"


def test_find_all_series_rice_10do(rice_10do_folder: Literal["tests/data/rice_10do"]):
    series_h5_path = Path(rice_10do_folder) / "0K9E8BI.h5"
    all_series_files = find_all_series(rice_10do_folder)
    assert len(all_series_files) == 1
    assert series_h5_path.as_posix() == "tests/data/rice_10do/0K9E8BI.h5"


def test_find_all_series_rice(rice_folder: Literal["tests/data/rice_3do"]):
    all_series_files = find_all_series(rice_folder)
    assert len(all_series_files) == 2
