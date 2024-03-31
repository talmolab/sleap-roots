import sleap_io as sio
import numpy as np
import pytest
from sleap_roots.series import Series, find_all_series
from pathlib import Path
from typing import Literal


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
        "plant_qr_code,number_of_plants_cylinder,genotype\ndummy,10,1100\nseries2,15,Kitaake-X\n"
    )
    return csv_path


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
