import pytest
import numpy as np
from sleap_roots.pipelining import (
    get_statistics,
    get_pts_pr_lr,
    get_pts_all,
    get_traits_frame,
    get_traits_plant,
)


@pytest.fixture
def traits_array_random():
    np.random.seed(0)
    return np.random.randint(100, size=(20))


@pytest.fixture
def pts_3roots_with_nan():
    return np.array(
        [
            [
                [920.48862368, 267.57325711],
                [908.88587777, 285.13679716],
                [906.19049368, 308.02657426],
                [900.85484007, 332.35722827],
                [894.77714477, 353.2618793],
                [np.nan, np.nan],
            ],
            [
                [918.0094082, 248.52049295],
                [875.89084055, 312.34001093],
                [886.19983474, 408.7826485],
                [892.15722656, 492.16012042],
                [899.53514073, 576.43033348],
                [908.02496338, 668.82440186],
            ],
            [
                [939.49111908, 291.1798956],
                [938.01029766, 309.0299704],
                [938.39169586, 324.6796079],
                [939.44596587, 339.22885535],
                [939.13551705, 355.82854929],
                [938.88545817, 371.64891802],
            ],
        ]
    )


@pytest.fixture
def pts_nan3():
    return np.array(
        [
            [
                [852.17755127, 216.95648193],
                [np.nan, 472.83520508],
                [844.45300293, np.nan],
            ]
        ]
    )


def test_get_statistics(traits_array_random):
    trait_max, trait_min, trait_mean, trait_std, trait_median = get_statistics(
        traits_array_random
    )
    np.testing.assert_almost_equal(trait_max, [88], decimal=7)
    np.testing.assert_almost_equal(trait_min, [9], decimal=7)
    np.testing.assert_almost_equal(trait_mean, [58.3], decimal=7)
    np.testing.assert_almost_equal(trait_std, [25.088045], decimal=7)
    np.testing.assert_almost_equal(trait_median, [64.5], decimal=7)


def test_get_pts_pr_lr(canola_h5):
    pts_pr = get_pts_pr_lr(canola_h5, rice=False, frame=0, primaryroot=True)
    pts_lr = get_pts_pr_lr(canola_h5, rice=False, frame=0, primaryroot=False)
    assert pts_pr.shape == (1, 6, 2)
    np.testing.assert_almost_equal(
        pts_pr,
        [
            [
                [1016.78442383, 144.41915894],
                [1207.99304199, 304.11700439],
                [1208.89245605, 472.43710327],
                [1192.0501709, 656.82409668],
                [1160.87573242, 848.52990723],
                [1136.09692383, 1020.98138428],
            ]
        ],
        decimal=7,
    )
    assert pts_lr.shape == (5, 3, 2)
    np.testing.assert_almost_equal(
        pts_lr,
        [
            [
                [1140.24816895, 212.87785339],
                [1156.17358398, 200.56602478],
                [np.nan, np.nan],
            ],
            [
                [1112.55065918, 228.09667969],
                [1100.2980957, 244.82826233],
                [1072.66101074, 276.51275635],
            ],
            [
                [1148.3215332, 228.33711243],
                [1200.88842773, 224.38265991],
                [1228.0637207, 228.92666626],
            ],
            [
                [1204.5032959, 296.57699585],
                [1184.70300293, 300.33679199],
                [1172.21728516, 308.23007202],
            ],
            [
                [np.nan, np.nan],
                [1204.88378906, 364.62561035],
                [1204.66918945, 368.51693726],
            ],
        ],
        decimal=7,
    )


def test_get_pts_pr_lr_rice(rice_h5):
    pts_pr = get_pts_pr_lr(rice_h5, rice=True, frame=0, primaryroot=True)
    pts_lr = get_pts_pr_lr(rice_h5, rice=True, frame=0, primaryroot=False)
    assert pts_pr.shape == (1, 6, 2)
    np.testing.assert_almost_equal(
        pts_pr,
        [
            [
                [860.68414307, 244.26528931],
                [852.24182129, 384.95001221],
                [852.26263428, 532.47363281],
                [848.44805908, 668.50561523],
                [856.80651855, 820.35565186],
                [856.53277588, 964.13537598],
            ]
        ],
        decimal=7,
    )
    assert pts_lr.shape == (2, 6, 2)
    np.testing.assert_almost_equal(
        pts_lr,
        [
            [
                [860.60223389, 248.60780334],
                [844.44207764, 300.43447876],
                [832.61334229, 316.18081665],
                [820.62634277, 332.58840942],
                [808.90344238, 348.73110962],
                [796.26116943, 368.55148315],
            ],
            [
                [848.09118652, 304.25735474],
                [852.44714355, 380.51464844],
                [852.52502441, 528.70410156],
                [848.33361816, 652.51721191],
                [860.19628906, 784.0904541],
                [856.54071045, 964.30279541],
            ],
        ],
        decimal=7,
    )


def test_get_pts_all(canola_h5):
    pts_all = get_pts_all(canola_h5, rice=False, frame=0)
    assert pts_all.shape == (12, 2)
    np.testing.assert_almost_equal(
        pts_all,
        [
            [1016.78442383, 144.41915894],
            [1207.99304199, 304.11700439],
            [1208.89245605, 472.43710327],
            [1192.0501709, 656.82409668],
            [1160.87573242, 848.52990723],
            [1136.09692383, 1020.98138428],
            [1016.78442383, 144.41915894],
            [1207.99304199, 304.11700439],
            [1208.89245605, 472.43710327],
            [1192.0501709, 656.82409668],
            [1160.87573242, 848.52990723],
            [1136.09692383, 1020.98138428],
        ],
        decimal=7,
    )


def test_get_pts_all_rice(rice_h5):
    pts_all = get_pts_all(rice_h5, rice=True, frame=0)
    assert pts_all.shape == (6, 2)
    np.testing.assert_almost_equal(
        pts_all,
        [
            [860.68414307, 244.26528931],
            [852.24182129, 384.95001221],
            [852.26263428, 532.47363281],
            [848.44805908, 668.50561523],
            [856.80651855, 820.35565186],
            [856.53277588, 964.13537598],
        ],
        decimal=7,
    )


def test_get_traits_frame(canola_folder, canola_h5):
    df = get_traits_frame(
        canola_folder,
        canola_h5,
        rice=False,
        frame=0,
        tolerance=0.02,
        fraction=2 / 3,
        depth=1080,
        width=2048,
        n_line=50,
    )
    assert len(df) == 1
    assert len(df.columns) == 47
    np.testing.assert_almost_equal(
        df.primary_angles_proximal[0], 50.13129559736394, decimal=7
    )
    np.testing.assert_almost_equal(
        df.lateral_length_mean[0], 40.400556356407456, decimal=7
    )
    np.testing.assert_almost_equal(df.stem_widths_median[0], 1, decimal=7)
    np.testing.assert_almost_equal(df.conv_areas[0], 93255.32153574759, decimal=7)
    np.testing.assert_almost_equal(df.ellipse_ratio[0], 0.2889459577340097, decimal=7)
    np.testing.assert_almost_equal(
        df.network_solidity[0], 0.025467588561201876, decimal=7
    )
    np.testing.assert_almost_equal(df.scanline_start[0], 6, decimal=7)


def test_get_traits_plant(canola_folder, canola_h5):
    df = get_traits_plant(
        canola_folder,
        canola_h5,
        rice=False,
        tolerance=0.02,
        fraction=2 / 3,
        depth=1080,
        width=2048,
        n_line=50,
        write_csv=False,
    )
    assert len(df) == 1
    assert len(df.columns) == 226

    np.testing.assert_almost_equal(
        df.primary_angles_proximal_max[0], 50.13129559736394, decimal=7
    )
    np.testing.assert_almost_equal(
        df.lateral_length_mean_min[0], 20.29789164406169, decimal=7
    )
    np.testing.assert_almost_equal(df.stem_widths_median_median[0], 1, decimal=7)
    np.testing.assert_almost_equal(df.conv_areas_max[0], 93255.32153574759, decimal=7)
    np.testing.assert_almost_equal(
        df.ellipse_ratio_min[0], 0.2889459577340097, decimal=7
    )
    np.testing.assert_almost_equal(
        df.network_solidity_std[0], 0.007457101693169275, decimal=7
    )
    np.testing.assert_almost_equal(
        df.scanline_start_mean[0], 5.866666666666666, decimal=7
    )
