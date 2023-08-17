from sleap_roots import Series
from sleap_roots.lengths import get_max_length_pts, get_root_lengths
from sleap_roots.points import (
    get_all_pts_array,
)


# test get_all_pts_array function
def test_get_all_pts_array(canola_h5):
    plant = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    monocots = False
    pts_all_array = get_all_pts_array(
        primary_max_length_pts, lateral_pts, monocots=monocots
    )
    assert pts_all_array.shape[0] == 21


# test get_all_pts_array function
def test_get_all_pts_array_rice(rice_h5):
    plant = Series.load(
        rice_h5, primary_name="longest_3do_6nodes", lateral_name="main_3do_6nodes"
    )
    primary, lateral = plant[0]
    primary_pts = primary.numpy()
    # get primary length
    primary_max_length_pts = get_max_length_pts(primary_pts)
    # get lateral_lengths
    lateral_pts = lateral.numpy()
    monocots = True
    pts_all_array = get_all_pts_array(
        primary_max_length_pts, lateral_pts, monocots=monocots
    )
    assert pts_all_array.shape[0] == 12
