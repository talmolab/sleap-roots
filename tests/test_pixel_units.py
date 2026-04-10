"""Regression tests for pixel unit invariance.

Guards against silent DPI-to-mm auto-conversion in trait calculations.
If a dependency (sleap-io, PIL, h5py) ever converts pixel coordinates based
on image DPI metadata, these tests will fail.

See: https://github.com/talmolab/sleap-roots/issues/146
"""

import numpy as np
import pytest
import sleap_io as sio
from PIL import Image

from sleap_roots import PrimaryRootPipeline, Series
from sleap_roots.bases import get_base_tip_dist
from sleap_roots.lengths import get_root_lengths


# DPI conversion that should NEVER be applied
DPI_WRONG_VALUE = 100.0 / 1200 * 25.4  # ~2.117mm at 1200 DPI


@pytest.fixture
def synthetic_series(tmp_path):
    """Create a Series with a synthetic 1200 DPI TIFF and known pixel coordinates.

    The TIFF is 200x400 with 1200 DPI metadata. A 6-node primary root skeleton
    spans 100px vertically (nodes evenly spaced 20px apart from y=50 to y=150).
    """
    # Create synthetic TIFF with 1200 DPI metadata
    img_array = np.zeros((400, 200), dtype=np.uint8)
    img = Image.fromarray(img_array)
    tif_path = str(tmp_path / "test_1200dpi.tif")
    img.save(tif_path, dpi=(1200, 1200))

    # Create sleap-io objects with known pixel coordinates
    skeleton = sio.Skeleton(nodes=[sio.Node(f"node_{i}") for i in range(6)])
    pts = np.array(
        [
            [100.0, 50.0],
            [100.0, 70.0],
            [100.0, 90.0],
            [100.0, 110.0],
            [100.0, 130.0],
            [100.0, 150.0],
        ]
    )
    inst = sio.Instance.from_numpy(pts, skeleton=skeleton)
    video = sio.Video(tif_path, open_backend=False)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
    labels = sio.Labels(labeled_frames=[lf], skeletons=[skeleton], videos=[video])

    return Series(series_name="synthetic_dpi_test", primary_labels=labels)


class TestPipelinePixelUnits:
    """Integration test: full pipeline with 1200 DPI TIFF metadata."""

    def test_primary_length_in_pixels(self, synthetic_series):
        """Primary root length through PrimaryRootPipeline must be in pixels."""
        pipeline = PrimaryRootPipeline()
        traits = pipeline.compute_plant_traits(synthetic_series)

        primary_length = traits["primary_length"].values[0]
        assert primary_length == pytest.approx(100.0)
        assert primary_length != pytest.approx(DPI_WRONG_VALUE)

    def test_primary_base_tip_dist_in_pixels(self, synthetic_series):
        """Base-to-tip distance through PrimaryRootPipeline must be in pixels."""
        pipeline = PrimaryRootPipeline()
        traits = pipeline.compute_plant_traits(synthetic_series)

        base_tip_dist = traits["primary_base_tip_dist"].values[0]
        assert base_tip_dist == pytest.approx(100.0)
        assert base_tip_dist != pytest.approx(DPI_WRONG_VALUE)


class TestTraitFunctionPixelUnits:
    """Unit-level contract tests: trait functions return pixel values."""

    def test_root_length_straight(self):
        """Straight 2-node root of 100px returns 100.0."""
        pts = np.array([[[100.0, 50.0], [100.0, 150.0]]])
        result = get_root_lengths(pts)
        assert result == pytest.approx(100.0)

    def test_root_length_polyline(self):
        """L-shaped 3-node root returns sum of segment lengths in pixels."""
        # Vertical segment (60px) + horizontal segment (80px) = 140px
        pts = np.array([[[0.0, 0.0], [0.0, 60.0], [80.0, 60.0]]])
        result = get_root_lengths(pts)
        assert result == pytest.approx(140.0)

    def test_base_tip_dist_in_pixels(self):
        """Base-to-tip Euclidean distance returns pixel value."""
        base_pts = np.array([100.0, 50.0])
        tip_pts = np.array([100.0, 150.0])
        result = get_base_tip_dist(base_pts, tip_pts)
        assert result == pytest.approx(100.0)

    def test_multi_instance_lengths(self):
        """Multiple root instances all return pixel-unit lengths."""
        pts = np.array(
            [
                [[0.0, 0.0], [0.0, 100.0]],
                [[0.0, 0.0], [0.0, 50.0]],
                [[0.0, 0.0], [0.0, 75.0]],
            ]
        )
        result = get_root_lengths(pts)
        np.testing.assert_array_almost_equal(result, [100.0, 50.0, 75.0])
