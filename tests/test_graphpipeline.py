from sleap_roots.graphpipeline import (
    get_traits_value_frame,
    get_traits_value_plant,
    get_traits_value_plant_summary,
    get_all_plants_traits,
)
import pytest
import numpy as np
import pandas as pd


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
    monocots = False
    pts_all_array = (
        primary_pts.reshape(-1, 2)
        if monocots
        else np.concatenate(
            (primary_pts.reshape(-1, 2), lateral_pts.reshape(-1, 2)), axis=0
        )
    )
    pts_all_array = pts_all_array.reshape(
        (1, pts_all_array.shape[0], pts_all_array.shape[1])
    )
    pts_all_list = (
        primary_pts if monocots else primary_pts.tolist() + lateral_pts.tolist()
    )

    data = get_traits_value_frame(
        primary_pts, lateral_pts, pts_all_array, pts_all_list, monocots
    )
    assert len(data) == 43


def test_get_traits_value_plant(canola_h5):
    monocots = False

    data_plant, data_plant_df = get_traits_value_plant(
        canola_h5,
        monocots,
        primary_name="primary_multi_day",
        lateral_name="lateral_3_nodes",
        stem_width_tolerance=0.02,
        n_line=50,
        network_fraction=2 / 3,
        write_csv=False,
        csv_name="plant_original_traits.csv",
    )
    assert len(data_plant) == 72
    assert data_plant_df.shape[1] == 45


def test_get_traits_value_plant_summary(canola_h5):
    monocots = False
    data_plant_summary = get_traits_value_plant_summary(
        canola_h5,
        monocots,
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
    np.testing.assert_almost_equal(data_plant_summary.iloc[0, 5], 16.643764612148875)


def test_get_all_plants_traits_dicot(
    data_folders=[r"E:\repositories\sleap-roots\tests\data\canola_7do"],
    primary_name="primary_multi_day",
    lateral_name="lateral_3_nodes",
    write_per_plant_details=True,
    write_per_plant_summary=True,
):
    all_traits_df = get_all_plants_traits(
        data_folders=data_folders,
        primary_name=primary_name,
        lateral_name=lateral_name,
        write_per_plant_details=write_per_plant_details,
        write_per_plant_summary=write_per_plant_summary,
    )
    assert all_traits_df.shape == (1, 1037)
    np.testing.assert_almost_equal(all_traits_df.iloc[0, 5], 16.643764612148875)
