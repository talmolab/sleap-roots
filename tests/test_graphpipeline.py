from sleap_roots.graphpipeline import (
    get_traits_value_frame,
    get_traits_value_plant,
    get_traits_value_plant_summary,
)
import pytest
import numpy as np


@pytest.fixture
def primary_pts():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [850.17755127, 472.83520508],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ]
        ]
    )


@pytest.fixture
def lateral_pts():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, np.nan],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
            [
                [852.17755127, 216.95648193],
                [844.45300293, 472.83520508],
                [837.03405762, 588.5123291],
                [828.87963867, 692.72009277],
                [816.71142578, 808.12585449],
            ],
        ]
    )


def test_get_traits_value_frame(primary_pts, lateral_pts):
    lateral_only = False
    pts_all_array = (
        primary_pts.reshape(-1, 2)
        if lateral_only
        else np.concatenate(
            (primary_pts.reshape(-1, 2), lateral_pts.reshape(-1, 2)), axis=0
        )
    )
    pts_all_array = pts_all_array.reshape(
        (1, pts_all_array.shape[0], pts_all_array.shape[1])
    )
    pts_all_list = (
        primary_pts if lateral_only else primary_pts.tolist() + lateral_pts.tolist()
    )

    data_df = get_traits_value_frame(
        primary_pts, lateral_pts, pts_all_array, pts_all_list, lateral_only
    )
    assert len(data_df) == 1
    assert data_df.shape[1] == 44


def test_get_traits_value_plant(canola_h5):
    lateral_only = False

    data_plant = get_traits_value_plant(
        canola_h5,
        lateral_only,
        primary_name="primary_multi_day",
        lateral_name="lateral_3_nodes",
        stem_width_tolerance=0.02,
        n_line=50,
        network_fraction=2 / 3,
        write_csv=False,
        csv_name="plant_original_traits.csv",
    )
    assert data_plant.shape[0] == 72
    assert data_plant.shape[1] == 46


def test_get_traits_value_plant_summary(canola_h5):
    lateral_only = False
    data_plant_summary = get_traits_value_plant_summary(
        canola_h5,
        lateral_only,
        primary_name="primary_multi_day",
        lateral_name="lateral_3_nodes",
        stem_width_tolerance=0.02,
        n_line=50,
        network_fraction=2 / 3,
        write_csv=False,
        csv_name="plant_original_traits.csv",
        write_summary_csv=False,
        summary_csv_name="plant_summary_traits.csv",
    )
    assert data_plant_summary.shape[0] == 1
    assert data_plant_summary.shape[1] == 1036
