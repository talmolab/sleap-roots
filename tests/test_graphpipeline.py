from sleap_roots.graphpipeline import get_traits_value_frame
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
    rice = False
    pts_all_array = (
        primary_pts.reshape(-1, 2)
        if rice
        else np.concatenate(
            (primary_pts.reshape(-1, 2), lateral_pts.reshape(-1, 2)), axis=0
        )
    )
    pts_all_array = pts_all_array.reshape(
        (1, pts_all_array.shape[0], pts_all_array.shape[1])
    )
    pts_all_list = primary_pts if rice else primary_pts.tolist() + lateral_pts.tolist()

    data = get_traits_value_frame(primary_pts, lateral_pts, pts_all_array, pts_all_list)
    assert len(data) == 44
