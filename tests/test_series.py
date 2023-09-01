import sleap_io as sio
import numpy as np
from sleap_roots.series import Series, find_all_series
from pathlib import Path
from typing import Literal


def test_series_load_canola(canola_h5: Literal["tests/data/canola_7do/919QDUH.h5"]):
    series = Series.load(canola_h5, ["primary_multi_day", "lateral_3_nodes"])
    assert len(series) == 72


def test_find_all_series_canola(canola_folder: Literal["tests/data/canola_7do"]):
    all_series_files = find_all_series(canola_folder)
    assert len(all_series_files) == 1


def test_find_all_series_rice_10do(rice_10do_folder: Literal["tests/data/rice_10do"]):
    all_series_files = find_all_series(rice_10do_folder)
    assert len(all_series_files) == 1


def test_load_rice_10do(rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"]):
    series = Series.load(rice_main_10do_h5, ["main_10do_6nodes"])
    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(
        Path(rice_main_10do_h5)
        .with_suffix(".main_10do_6nodes.predictions.slp")
        .as_posix()
    )
    expected_video = sio.Video.from_filename(rice_main_10do_h5)
    assert len(series) == 72
    assert series.h5_path == rice_main_10do_h5
    assert series.labels_dict == {"main_10do_6nodes": expected_labels}
    assert series.video.filename == expected_video.filename


def test_series_name_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
):
    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(
        Path(rice_main_10do_h5)
        .with_suffix(".main_10do_6nodes.predictions.slp")
        .as_posix()
    )
    series = Series(rice_main_10do_h5, {"main_10do_6nodes": expected_labels})
    assert series.series_name == "0K9E8BI"


def test_get_frame_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
):
    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(
        Path(rice_main_10do_h5)
        .with_suffix(".main_10do_6nodes.predictions.slp")
        .as_posix()
    )
    series = Series(rice_main_10do_h5, {"main_10do_6nodes": expected_labels})
    expected_labeled_frame = expected_labels[0]
    assert series.get_frame(0)[0] == expected_labeled_frame


def test_get_points_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
):
    series = Series.load(rice_main_10do_h5, ["main_10do_6nodes"])
    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(
        Path(rice_main_10do_h5)
        .with_suffix(".main_10do_6nodes.predictions.slp")
        .as_posix()
    )
    lf = expected_labels.find(expected_labels.video, 0, return_new=True)[0]
    gt_instances = lf.user_instances + lf.unused_predictions
    points = np.stack([inst.numpy() for inst in gt_instances], axis=0)
    np.testing.assert_array_equal(series.get_points(0, "main_10do_6nodes"), points)


def test_series_name_property():
    series = Series(h5_path="mock_path/file_name.h5")
    assert series.series_name == "file_name"


def test_len():
    series = Series(video=["frame1", "frame2"])
    assert len(series) == 2


def test_get_item_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
):
    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(
        Path(rice_main_10do_h5)
        .with_suffix(".main_10do_6nodes.predictions.slp")
        .as_posix()
    )
    series = Series(rice_main_10do_h5, {"main_10do_6nodes": expected_labels})
    expected_labeled_frame = expected_labels[0]
    assert series[0] == (expected_labeled_frame,)


def test_get_item_load_rice_10do(
    rice_main_10do_h5: Literal["tests/data/rice_10do/0K9E8BI.h5"],
):
    # Load the expected Labels object for comparison
    expected_labels = sio.load_slp(
        Path(rice_main_10do_h5)
        .with_suffix(".main_10do_6nodes.predictions.slp")
        .as_posix()
    )
    series = Series.load(rice_main_10do_h5, ["main_10do_6nodes"])
    expected_labeled_frame = expected_labels[0]
    assert series[0] == (expected_labeled_frame,)
