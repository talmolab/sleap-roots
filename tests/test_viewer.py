"""Tests for the HTML prediction viewer."""

import base64
from pathlib import Path

# Use non-interactive backend before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from sleap_roots.series import Series
from sleap_roots.viewer.renderer import (
    render_frame_root_type,
    render_frame_confidence,
    figure_to_base64,
)
from sleap_roots.viewer.generator import ViewerGenerator, ScanInfo


class TestFrameRenderingRootType:
    """Tests for frame rendering with root type overlay."""

    def test_render_frame_root_type_returns_figure(
        self, canola_h5, canola_primary_slp, canola_lateral_slp
    ):
        """Test that render_frame_root_type returns a matplotlib Figure."""
        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
            lateral_path=canola_lateral_slp,
        )
        fig = render_frame_root_type(series, frame_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_frame_root_type_with_primary_only(
        self, canola_h5, canola_primary_slp
    ):
        """Test rendering with only primary root predictions."""
        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )
        fig = render_frame_root_type(series, frame_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_frame_root_type_with_crown_roots(
        self, rice_h5, rice_long_slp, rice_main_slp
    ):
        """Test rendering with crown root predictions (monocot)."""
        series = Series.load(
            series_name="test",
            h5_path=rice_h5,
            primary_path=rice_long_slp,
            crown_path=rice_main_slp,
        )
        fig = render_frame_root_type(series, frame_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_frame_root_type_invalid_frame_raises(
        self, canola_h5, canola_primary_slp
    ):
        """Test that invalid frame index raises an error."""
        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )
        with pytest.raises(IndexError):
            render_frame_root_type(series, frame_idx=9999)


class TestFrameRenderingConfidence:
    """Tests for frame rendering with confidence colormap overlay."""

    def test_render_frame_confidence_returns_figure(
        self, canola_h5, canola_primary_slp, canola_lateral_slp
    ):
        """Test that render_frame_confidence returns a matplotlib Figure."""
        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
            lateral_path=canola_lateral_slp,
        )
        fig = render_frame_confidence(series, frame_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_frame_confidence_uses_colormap(self, canola_h5, canola_primary_slp):
        """Test that confidence rendering uses a colormap."""
        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )
        # Default colormap should be viridis
        fig = render_frame_confidence(series, frame_idx=0, colormap="viridis")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_frame_confidence_custom_colormap(
        self, canola_h5, canola_primary_slp
    ):
        """Test that custom colormap can be specified."""
        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )
        fig = render_frame_confidence(series, frame_idx=0, colormap="plasma")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestRendererEdgeCases:
    """Tests for edge cases in renderer module."""

    def test_get_confidence_range_no_scores(self):
        """Test get_confidence_range returns defaults when no scores available."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.renderer import get_confidence_range

        series = MagicMock()
        # Mock empty frames dict
        series.get_frame.return_value = {}

        min_conf, max_conf = get_confidence_range(series, 0)

        assert min_conf == 0.0
        assert max_conf == 1.0

    def test_get_confidence_range_none_labeled_frame(self):
        """Test get_confidence_range handles None labeled frames."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.renderer import get_confidence_range

        series = MagicMock()
        # Mock frames with None labeled frame
        series.get_frame.return_value = {"primary": None, "lateral": None}

        min_conf, max_conf = get_confidence_range(series, 0)

        assert min_conf == 0.0
        assert max_conf == 1.0

    def test_render_frame_confidence_no_video_raises(self):
        """Test render_frame_confidence raises when video is None."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.renderer import render_frame_confidence

        series = MagicMock()
        series.video = None
        series.__len__ = MagicMock(return_value=10)

        with pytest.raises(ValueError, match="Video is not available"):
            render_frame_confidence(series, 0)

    def test_render_frame_confidence_out_of_range_raises(self):
        """Test render_frame_confidence raises for invalid frame index."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.renderer import render_frame_confidence

        series = MagicMock()
        series.__len__ = MagicMock(return_value=5)

        with pytest.raises(IndexError, match="out of range"):
            render_frame_confidence(series, 100)

    def test_render_frame_confidence_with_empty_frames(
        self, canola_h5, canola_primary_slp
    ):
        """Test render_frame_confidence handles frames with no predictions."""
        from sleap_roots.viewer.renderer import render_frame_confidence

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        # This tests rendering a frame (even if it has predictions).
        # The key is that the function should work without crashing.
        fig = render_frame_confidence(series, frame_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_render_frame_confidence_no_skeleton_labels(
        self, canola_h5, canola_primary_slp
    ):
        """Test render_frame_confidence when skeleton labels are None."""
        from sleap_roots.viewer.renderer import render_frame_confidence

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        # Explicitly set labels to None (simulate missing labels)
        original_lateral_labels = series.lateral_labels
        series.lateral_labels = None

        fig = render_frame_confidence(series, frame_idx=0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

        # Restore
        series.lateral_labels = original_lateral_labels


class TestConfidenceNormalization:
    """Tests for confidence score normalization."""

    def test_normalize_confidence_basic(self):
        """Test that normalize_confidence returns values in 0-1 range."""
        from sleap_roots.viewer.renderer import normalize_confidence

        # Scores in normal 0-1 range should stay the same
        assert normalize_confidence(0.5, 0.0, 1.0) == 0.5
        assert normalize_confidence(0.0, 0.0, 1.0) == 0.0
        assert normalize_confidence(1.0, 0.0, 1.0) == 1.0

    def test_normalize_confidence_high_scores(self):
        """Test that confidence scores > 1 are normalized correctly."""
        from sleap_roots.viewer.renderer import normalize_confidence

        # Scores > 1 should be normalized to 0-1 range
        assert normalize_confidence(2.0, 0.0, 4.0) == 0.5
        assert normalize_confidence(4.0, 0.0, 4.0) == 1.0
        assert normalize_confidence(0.0, 0.0, 4.0) == 0.0

    def test_normalize_confidence_negative_scores(self):
        """Test that negative confidence scores are handled."""
        from sleap_roots.viewer.renderer import normalize_confidence

        # Negative minimum should work
        assert normalize_confidence(0.0, -1.0, 1.0) == 0.5
        assert normalize_confidence(-1.0, -1.0, 1.0) == 0.0
        assert normalize_confidence(1.0, -1.0, 1.0) == 1.0

    def test_normalize_confidence_same_min_max(self):
        """Test handling when all scores are the same."""
        from sleap_roots.viewer.renderer import normalize_confidence

        # When min == max, should return 0.5 (middle of range)
        assert normalize_confidence(5.0, 5.0, 5.0) == 0.5

    def test_get_confidence_range_returns_tuple(
        self, canola_h5, canola_primary_slp, canola_lateral_slp
    ):
        """Test that get_confidence_range returns min/max tuple."""
        from sleap_roots.viewer.renderer import get_confidence_range

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
            lateral_path=canola_lateral_slp,
        )
        min_conf, max_conf = get_confidence_range(series, frame_idx=0)

        # Should return numeric values
        assert isinstance(min_conf, (int, float))
        assert isinstance(max_conf, (int, float))
        assert min_conf <= max_conf


class TestBase64Encoding:
    """Tests for base64 image encoding."""

    def test_figure_to_base64_returns_string(self):
        """Test that figure_to_base64 returns a base64 string."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = figure_to_base64(fig)
        assert isinstance(result, str)
        plt.close(fig)

    def test_figure_to_base64_is_valid_base64(self):
        """Test that the returned string is valid base64."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = figure_to_base64(fig)
        # Should be able to decode without error
        decoded = base64.b64decode(result)
        assert len(decoded) > 0
        plt.close(fig)

    def test_figure_to_base64_is_png(self):
        """Test that the encoded image is a PNG."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        result = figure_to_base64(fig)
        decoded = base64.b64decode(result)
        # PNG magic number
        assert decoded[:8] == b"\x89PNG\r\n\x1a\n"
        plt.close(fig)

    def test_figure_to_base64_closes_figure(self):
        """Test that the figure is optionally closed after encoding."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        figure_to_base64(fig, close=True)
        # Figure should be closed
        assert not plt.fignum_exists(fig.number)


class TestScanDiscovery:
    """Tests for scan discovery from predictions directory."""

    def test_discover_scans_returns_list(self, canola_folder):
        """Test that discover_scans returns a list of ScanInfo."""
        generator = ViewerGenerator(Path(canola_folder))
        scans = generator.discover_scans()
        assert isinstance(scans, list)

    def test_discover_scans_finds_canola_scan(self, canola_folder):
        """Test that canola scan is discovered."""
        generator = ViewerGenerator(Path(canola_folder))
        scans = generator.discover_scans()
        assert len(scans) >= 1
        # Check that at least one scan has the expected name pattern
        scan_names = [scan.name for scan in scans]
        assert any("919QDUH" in name for name in scan_names)

    def test_scan_info_has_required_attributes(self, canola_folder):
        """Test that ScanInfo has required attributes."""
        generator = ViewerGenerator(Path(canola_folder))
        scans = generator.discover_scans()
        assert len(scans) >= 1
        scan = scans[0]
        # Check required attributes
        assert hasattr(scan, "name")
        assert hasattr(scan, "primary_path")
        assert hasattr(scan, "frame_count")

    def test_discover_scans_with_images_dir(self, canola_folder):
        """Test discovery with separate images directory."""
        generator = ViewerGenerator(
            Path(canola_folder),
            images_dir=Path(canola_folder),
        )
        scans = generator.discover_scans()
        assert len(scans) >= 1
        # Check that h5_path is set when images_dir provided
        scan = scans[0]
        assert scan.h5_path is not None

    def test_discover_scans_rice_monocot(self, rice_folder):
        """Test scan discovery with rice (monocot) predictions."""
        generator = ViewerGenerator(Path(rice_folder))
        scans = generator.discover_scans()
        assert len(scans) >= 1

    def test_discover_scans_empty_directory(self, tmp_path):
        """Test that empty directory returns empty list."""
        generator = ViewerGenerator(tmp_path)
        scans = generator.discover_scans()
        assert scans == []

    def test_discover_scans_invalid_directory(self, tmp_path):
        """Test that non-existent directory raises error."""
        invalid_path = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            generator = ViewerGenerator(invalid_path)
            generator.discover_scans()


class TestNestedDirectoryDiscovery:
    """Tests for recursive scan discovery with nested directories."""

    def test_discover_scans_recursive_finds_nested_slp(self, tmp_path):
        """Test that recursive discovery finds .slp files in subdirectories."""
        # Create nested directory structure mimicking pipeline output
        scan1_dir = tmp_path / "scan1"
        scan1_dir.mkdir()
        scan2_dir = tmp_path / "scan2"
        scan2_dir.mkdir()

        # Create dummy .slp files (just empty files for discovery test)
        (scan1_dir / "scan1.model123.rootprimary.slp").touch()
        (scan2_dir / "scan2.model123.rootprimary.slp").touch()

        generator = ViewerGenerator(tmp_path)
        scans = generator.discover_scans(recursive=True)

        # Should find both scans in subdirectories
        scan_names = [scan.name for scan in scans]
        assert "scan1" in scan_names
        assert "scan2" in scan_names

    def test_discover_scans_non_recursive_skips_nested(self, tmp_path):
        """Test that non-recursive discovery skips files in subdirectories."""
        # Create nested directory structure
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()

        # Create .slp file only in subdirectory
        (nested_dir / "scan1.model123.rootprimary.slp").touch()

        generator = ViewerGenerator(tmp_path)
        scans = generator.discover_scans(recursive=False)

        # Should NOT find the nested scan
        assert len(scans) == 0

    def test_discover_scans_mixed_depth(self, tmp_path):
        """Test discovery with files at both root and nested levels."""
        # Create files at root level
        (tmp_path / "root_scan.model123.rootprimary.slp").touch()

        # Create files in subdirectory
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        (nested_dir / "nested_scan.model123.rootprimary.slp").touch()

        generator = ViewerGenerator(tmp_path)
        scans = generator.discover_scans(recursive=True)

        scan_names = [scan.name for scan in scans]
        assert "root_scan" in scan_names
        assert "nested_scan" in scan_names

    def test_discover_scans_pipeline_filename_pattern(self, tmp_path):
        """Test that pipeline output filename patterns are correctly parsed."""
        # Create files with pipeline naming convention
        (tmp_path / "scan_ABC123.model_230104_182346.rootprimary.slp").touch()
        (tmp_path / "scan_ABC123.model_230104_182346.rootlateral.slp").touch()

        generator = ViewerGenerator(tmp_path)
        scans = generator.discover_scans()

        # Should group into a single scan with both root types
        assert len(scans) == 1
        scan = scans[0]
        assert scan.name == "scan_ABC123"
        assert scan.primary_path is not None
        assert scan.lateral_path is not None

    def test_discover_scans_deeply_nested(self, tmp_path):
        """Test recursive discovery with deeply nested directories."""
        # Create deeply nested structure
        deep_dir = tmp_path / "level1" / "level2" / "level3"
        deep_dir.mkdir(parents=True)
        (deep_dir / "deep_scan.model123.rootprimary.slp").touch()

        generator = ViewerGenerator(tmp_path)
        scans = generator.discover_scans(recursive=True)

        scan_names = [scan.name for scan in scans]
        assert "deep_scan" in scan_names


class TestPipelineOutputDiscovery:
    """Tests for pipeline output with image directories (not h5 files)."""

    def test_discover_scans_pipeline_output_no_h5(
        self, rice_10do_pipeline_output_folder
    ):
        """Test that pipeline output with image directories is discovered."""
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        scans = generator.discover_scans()

        # Should find scans even without h5 files
        assert len(scans) >= 1

    def test_discover_scans_pipeline_output_has_frame_count(
        self, rice_10do_pipeline_output_folder
    ):
        """Test that scans from pipeline output have correct frame count."""
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        scans = generator.discover_scans()

        # At least one scan should have frames (from embedded image paths)
        scans_with_frames = [s for s in scans if s.frame_count > 0]
        assert len(scans_with_frames) >= 1

    def test_generate_pipeline_output_creates_html(
        self, rice_10do_pipeline_output_folder, tmp_path
    ):
        """Test that HTML generation works with pipeline output (no h5)."""
        output_path = tmp_path / "pipeline_viewer.html"
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        generator.generate(output_path)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        # Should have scan data, not "No scans available"
        assert "scansData" in content

    def test_generate_pipeline_output_includes_images(
        self, rice_10do_pipeline_output_folder, tmp_path
    ):
        """Test that HTML generation includes rendered images from pipeline output."""
        output_path = tmp_path / "pipeline_viewer.html"
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        generator.generate(output_path)

        content = output_path.read_text(encoding="utf-8")
        # Should include base64 encoded images (proves video was loaded)
        assert "data:image/png;base64," in content


class TestHTMLGeneration:
    """Tests for HTML viewer generation."""

    def test_generate_creates_file(self, canola_folder, tmp_path):
        """Test that generate() creates an HTML file."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path)
        assert output_path.exists()

    def test_generate_creates_valid_html(self, canola_folder, tmp_path):
        """Test that generated file is valid HTML."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "</html>" in content

    def test_generate_includes_scan_data(self, canola_folder, tmp_path):
        """Test that generated HTML includes scan information."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path)

        content = output_path.read_text(encoding="utf-8")
        # Should include the scan name somewhere in the HTML
        assert "919QDUH" in content

    def test_generate_includes_images(self, canola_folder, tmp_path):
        """Test that generated HTML includes base64 images."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path)

        content = output_path.read_text(encoding="utf-8")
        # Should include base64 encoded images
        assert "data:image/png;base64," in content

    def test_generate_includes_navigation_js(self, canola_folder, tmp_path):
        """Test that generated HTML includes JavaScript for navigation."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path)

        content = output_path.read_text(encoding="utf-8")
        # Should include JavaScript for keyboard navigation
        assert "<script>" in content
        assert "keydown" in content.lower() or "keyboard" in content.lower()

    def test_generate_self_contained(self, canola_folder, tmp_path):
        """Test that generated HTML is self-contained (no external resources)."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path)

        content = output_path.read_text(encoding="utf-8")
        # Should not have external CSS or JS references
        assert 'href="http' not in content
        assert 'src="http' not in content


class TestFrameSampling:
    """Tests for frame sampling functionality."""

    def test_select_frame_indices_all_frames_when_under_limit(self):
        """Test that all frames are selected when under the limit."""
        from sleap_roots.viewer.generator import select_frame_indices

        # 5 frames, max 10 -> all frames
        indices = select_frame_indices(total_frames=5, max_frames=10)
        assert indices == [0, 1, 2, 3, 4]

    def test_select_frame_indices_samples_evenly(self):
        """Test that frames are sampled evenly across the scan."""
        from sleap_roots.viewer.generator import select_frame_indices

        # 72 frames, max 10 -> 10 evenly spaced
        indices = select_frame_indices(total_frames=72, max_frames=10)
        assert len(indices) == 10
        assert indices[0] == 0  # First frame
        assert indices[-1] == 71  # Last frame
        # Check roughly even spacing
        assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))

    def test_select_frame_indices_includes_first_and_last(self):
        """Test that first and last frames are always included."""
        from sleap_roots.viewer.generator import select_frame_indices

        indices = select_frame_indices(total_frames=100, max_frames=5)
        assert 0 in indices
        assert 99 in indices

    def test_select_frame_indices_zero_means_all(self):
        """Test that max_frames=0 returns all frames."""
        from sleap_roots.viewer.generator import select_frame_indices

        indices = select_frame_indices(total_frames=50, max_frames=0)
        assert indices == list(range(50))

    def test_select_frame_indices_single_frame(self):
        """Test handling of single frame scan."""
        from sleap_roots.viewer.generator import select_frame_indices

        indices = select_frame_indices(total_frames=1, max_frames=10)
        assert indices == [0]

    def test_select_frame_indices_max_frames_equals_total(self):
        """Test when max_frames equals total_frames."""
        from sleap_roots.viewer.generator import select_frame_indices

        indices = select_frame_indices(total_frames=10, max_frames=10)
        assert indices == list(range(10))

    def test_generate_with_max_frames_limits_output(self, canola_folder, tmp_path):
        """Test that max_frames limits the number of rendered frames."""
        import json

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=3)

        content = output_path.read_text(encoding="utf-8")
        # Extract scans_json from HTML
        start = content.find("const scansData = ") + len("const scansData = ")
        end = content.find(";\n", start)
        scans_data = json.loads(content[start:end])

        # Each scan should have at most 3 frames
        for scan in scans_data:
            assert len(scan["frames_root_type"]) <= 3

    def test_generate_default_max_frames_is_10(self, canola_folder, tmp_path):
        """Test that default max_frames is 10."""
        import json

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path)  # No max_frames specified

        content = output_path.read_text(encoding="utf-8")
        start = content.find("const scansData = ") + len("const scansData = ")
        end = content.find(";\n", start)
        scans_data = json.loads(content[start:end])

        # Each scan should have at most 10 frames (default)
        for scan in scans_data:
            assert len(scan["frames_root_type"]) <= 10


class TestFrameLimits:
    """Tests for frame limit warnings and errors."""

    def test_generate_warns_over_100_frames(self, tmp_path):
        """Test that warning is shown when total frames exceed 100."""
        import warnings
        from unittest.mock import patch

        generator = ViewerGenerator(tmp_path)

        # Mock discover_scans to return scans with enough frames to trigger warning
        mock_scans = [
            ScanInfo(name=f"scan_{i}", frame_count=30, primary_path=tmp_path / "f.slp")
            for i in range(5)
        ]  # 5 scans * 30 frames = 150 frames > 100

        with patch.object(generator, "discover_scans", return_value=mock_scans):
            with patch.object(generator, "_load_series", return_value=None):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    generator.generate(
                        tmp_path / "output.html", max_frames=0, no_limit=True
                    )
                    # Should have warned about large frame count
                    warning_messages = [str(warning.message) for warning in w]
                    assert any(
                        "total frames" in msg.lower() for msg in warning_messages
                    )

    def test_generate_errors_over_1000_frames_without_no_limit(self, tmp_path):
        """Test that error is raised when >1000 frames without --no-limit."""
        from unittest.mock import patch

        from sleap_roots.viewer.generator import FrameLimitExceededError

        generator = ViewerGenerator(tmp_path)

        # Mock discover_scans to return scans with large frame counts
        mock_scans = [
            ScanInfo(name=f"scan_{i}", frame_count=200, primary_path=tmp_path / "f.slp")
            for i in range(10)
        ]

        with patch.object(generator, "discover_scans", return_value=mock_scans):
            with pytest.raises(FrameLimitExceededError):
                generator.generate(tmp_path / "output.html", max_frames=0)

    def test_generate_allows_over_1000_frames_with_no_limit(
        self, canola_folder, tmp_path
    ):
        """Test that --no-limit allows unlimited frames."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        # Should not raise even with max_frames=0
        generator.generate(output_path, max_frames=0, no_limit=True)
        assert output_path.exists()


class TestProgressCallback:
    """Tests for progress callback functionality."""

    def test_generate_accepts_progress_callback(self, canola_folder, tmp_path):
        """Test that generate() accepts a progress_callback parameter."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))

        callback_calls = []

        def progress_callback(scan_name: str, frames_done: int, total_frames: int):
            callback_calls.append((scan_name, frames_done, total_frames))

        generator.generate(output_path, progress_callback=progress_callback)
        assert output_path.exists()
        # Callback should have been called at least once
        assert len(callback_calls) > 0

    def test_progress_callback_receives_correct_arguments(
        self, canola_folder, tmp_path
    ):
        """Test that callback receives scan name and frame counts."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))

        callback_calls = []

        def progress_callback(scan_name: str, frames_done: int, total_frames: int):
            callback_calls.append((scan_name, frames_done, total_frames))

        generator.generate(
            output_path, max_frames=3, progress_callback=progress_callback
        )

        # Check that callbacks have valid data
        for scan_name, frames_done, total_frames in callback_calls:
            assert isinstance(scan_name, str)
            assert len(scan_name) > 0
            assert isinstance(frames_done, int)
            assert isinstance(total_frames, int)
            assert frames_done >= 0
            assert frames_done <= total_frames

    def test_progress_callback_called_for_each_frame(self, canola_folder, tmp_path):
        """Test that callback is called once per frame rendered."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))

        callback_calls = []

        def progress_callback(scan_name: str, frames_done: int, total_frames: int):
            callback_calls.append((scan_name, frames_done, total_frames))

        generator.generate(
            output_path, max_frames=5, progress_callback=progress_callback
        )

        # Should have multiple calls (one per frame)
        assert len(callback_calls) >= 1

        # Last call should have frames_done == total_frames for each scan
        # Group by scan
        scans_seen = set(c[0] for c in callback_calls)
        for scan_name in scans_seen:
            scan_calls = [c for c in callback_calls if c[0] == scan_name]
            last_call = scan_calls[-1]
            assert last_call[1] == last_call[2]  # frames_done == total_frames


class TestNormalizedConfidenceBadge:
    """Tests for normalized confidence badge display."""

    def test_confidence_to_hex_returns_valid_hex(self):
        """Test that confidence_to_hex returns a valid hex color string."""
        from sleap_roots.viewer.generator import confidence_to_hex

        hex_color = confidence_to_hex(0.5)
        assert isinstance(hex_color, str)
        assert hex_color.startswith("#")
        assert len(hex_color) == 7  # #RRGGBB

    def test_confidence_to_hex_uses_viridis_range(self):
        """Test that low and high values produce different viridis colors."""
        from sleap_roots.viewer.generator import confidence_to_hex

        low_color = confidence_to_hex(0.0)
        high_color = confidence_to_hex(1.0)
        # Viridis goes from dark purple to yellow
        assert low_color != high_color

    def test_scan_confidence_normalized_to_0_1(self, canola_folder, tmp_path):
        """Test that scan-level mean_confidence is normalized to 0-1 range."""
        import json

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=3)

        content = output_path.read_text(encoding="utf-8")
        start = content.find("const scansData = ") + len("const scansData = ")
        end = content.find(";\n", start)
        scans_data = json.loads(content[start:end])

        for scan in scans_data:
            for frame in scan["frames_root_type"]:
                if frame["mean_confidence"] is not None:
                    assert 0.0 <= frame["mean_confidence"] <= 1.0

    def test_badge_shows_score_label(self, canola_folder, tmp_path):
        """Test that the confidence badge displays 'Score:' label."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=3)

        content = output_path.read_text(encoding="utf-8")
        assert "Score:" in content

    def test_badge_includes_tooltip(self, canola_folder, tmp_path):
        """Test that the confidence badge has an explanatory tooltip."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=3)

        content = output_path.read_text(encoding="utf-8")
        assert "Normalized prediction confidence" in content

    def test_badge_uses_viridis_hex_color(self, canola_folder, tmp_path):
        """Test that the badge background uses a viridis hex color."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=3)

        content = output_path.read_text(encoding="utf-8")
        # Badge should have inline style with hex background color
        assert "background:" in content and "#" in content


class TestCaseInsensitiveExtensions:
    """Tests for case-insensitive image extension discovery."""

    def test_uppercase_jpg_discovered(self, tmp_path):
        """Test that uppercase .JPG files are discovered in image directories."""
        img_dir = tmp_path / "scan_images"
        img_dir.mkdir()
        (img_dir / "1.JPG").touch()
        (img_dir / "2.JPG").touch()

        # Test using the same case-insensitive logic as production code
        # Production uses: [p for p in dir.iterdir() if p.suffix.lower() in image_exts]
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        found = [f for f in img_dir.iterdir() if f.suffix.lower() in image_exts]

        assert len(found) >= 2

    def test_mixed_case_extensions_discovered(self, tmp_path):
        """Test that mixed case extensions like .Jpg are discovered."""
        img_dir = tmp_path / "scan_images"
        img_dir.mkdir()
        (img_dir / "1.jpg").touch()
        (img_dir / "2.PNG").touch()
        (img_dir / "3.Tiff").touch()

        # Collect with case-insensitive approach
        all_images = list(img_dir.iterdir())
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        found = [f for f in all_images if f.suffix.lower() in image_exts]
        assert len(found) == 3


class TestSortKeyNonNumeric:
    """Tests for sort_key logic used in _find_and_remap_video.

    These tests validate the sorting algorithm used in production code.
    The sort_key function prefers numeric stems (sorted by value) and
    falls back to alphabetic sorting for non-numeric filenames.
    """

    def _sort_key(self, p):
        """Sort key matching production code in _find_and_remap_video."""
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)

    def test_numeric_filenames_sort_numerically(self):
        """Test that numeric filenames are sorted by numeric value."""
        from pathlib import Path

        paths = [Path("10.jpg"), Path("2.jpg"), Path("1.jpg"), Path("20.jpg")]
        sorted_paths = sorted(paths, key=self._sort_key)
        assert [p.name for p in sorted_paths] == [
            "1.jpg",
            "2.jpg",
            "10.jpg",
            "20.jpg",
        ]

    def test_non_numeric_filenames_sort_alphabetically(self):
        """Test that non-numeric filenames sort alphabetically as fallback."""
        from pathlib import Path

        paths = [Path("c.jpg"), Path("a.jpg"), Path("b.jpg")]
        sorted_paths = sorted(paths, key=self._sort_key)
        assert [p.name for p in sorted_paths] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_mixed_numeric_and_alpha_filenames(self):
        """Test that numeric filenames sort before non-numeric."""
        from pathlib import Path

        paths = [Path("b.jpg"), Path("2.jpg"), Path("a.jpg"), Path("1.jpg")]
        sorted_paths = sorted(paths, key=self._sort_key)
        names = [p.name for p in sorted_paths]
        # Numeric first, then alphabetic
        assert names == ["1.jpg", "2.jpg", "a.jpg", "b.jpg"]


class TestVideoRemapWarning:
    """Tests for warning on video existence check failure."""

    def test_find_and_remap_warns_on_check_failure(self, tmp_path):
        """Test that _find_and_remap_video warns when video.exists() raises."""
        from unittest.mock import MagicMock

        generator = ViewerGenerator(tmp_path)
        series = MagicMock()

        # Mock labels with a video that raises on exists()
        mock_video = MagicMock()
        mock_video.exists.side_effect = Exception("test error")
        mock_video.filename = ["/nonexistent/path/img_dir/1.jpg"]

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]

        series.primary_labels = mock_labels
        series.lateral_labels = None
        series.crown_labels = None

        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            success, plant_name, group = generator._find_and_remap_video(series)

            # Should have warned about the failure
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "failed to check video" in msg.lower() for msg in warning_messages
            ), f"Expected warning about video check failure, got: {warning_messages}"
            assert not success


class TestMultiTimepointVideoRemapping:
    """Tests for video remapping with multi-timepoint datasets.

    When the same plant (e.g., Fado_1) exists across multiple timepoints (Day0, Day3),
    the video remapping should correctly match the image directory based on the full
    path context, not just the leaf directory name.
    """

    def test_remap_finds_correct_timepoint_directory(self, tmp_path):
        """Test that remapping finds the correct day's directory for a plant."""
        from unittest.mock import MagicMock

        # Create multi-timepoint directory structure
        # Day0/Fado_1/ and Day3/Fado_1/ should be distinguishable
        day0_dir = tmp_path / "Wave1" / "Day0_2025-11-27" / "Fado_1"
        day3_dir = tmp_path / "Wave1" / "Day3_2025-11-30" / "Fado_1"
        day0_dir.mkdir(parents=True)
        day3_dir.mkdir(parents=True)

        # Create images in both directories
        (day0_dir / "1.jpg").touch()
        (day0_dir / "2.jpg").touch()
        (day3_dir / "1.jpg").touch()
        (day3_dir / "2.jpg").touch()

        generator = ViewerGenerator(tmp_path)
        series = MagicMock()

        # Mock embedded path pointing to Day0 (the full pipeline path)
        mock_video = MagicMock()
        mock_video.exists.return_value = False
        # Embedded path should include distinguishing components
        mock_video.filename = [
            "/workspace/images_input/images/Wave1/Day0_2025-11-27/Fado_1/1.jpg",
            "/workspace/images_input/images/Wave1/Day0_2025-11-27/Fado_1/2.jpg",
        ]

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]

        series.primary_labels = mock_labels
        series.lateral_labels = None
        series.crown_labels = None

        success, plant_name, group = generator._find_and_remap_video(series)

        assert success is True
        assert plant_name == "Fado_1"
        assert "Day0" in group
        # Verify the video was remapped to Day0's directory, not Day3's
        mock_video.replace_filename.assert_called_once()
        remapped_paths = mock_video.replace_filename.call_args[0][0]
        assert "Day0" in remapped_paths[0]
        assert "Day3" not in remapped_paths[0]

    def test_remap_distinguishes_same_plant_across_days(self, tmp_path):
        """Test that the same plant name in different days is correctly distinguished."""
        from unittest.mock import MagicMock

        # Create structure with same plant in multiple timepoints
        for day in ["Day0", "Day3", "Day5", "Day7"]:
            plant_dir = tmp_path / day / "Fado_1"
            plant_dir.mkdir(parents=True)
            (plant_dir / "1.jpg").touch()

        generator = ViewerGenerator(tmp_path)

        # Test each day individually
        for target_day in ["Day0", "Day3", "Day5", "Day7"]:
            series = MagicMock()
            mock_video = MagicMock()
            mock_video.exists.return_value = False
            mock_video.filename = [f"/workspace/images/{target_day}/Fado_1/1.jpg"]

            mock_labels = MagicMock()
            mock_labels.videos = [mock_video]

            series.primary_labels = mock_labels
            series.lateral_labels = None
            series.crown_labels = None

            success, plant_name, group = generator._find_and_remap_video(series)

            assert success is True, f"Failed for {target_day}"
            assert plant_name == "Fado_1"
            assert group == target_day
            remapped_paths = mock_video.replace_filename.call_args[0][0]
            assert target_day in remapped_paths[0], (
                f"Expected {target_day} in remapped path, got {remapped_paths[0]}"
            )

    def test_remap_backwards_compatible_single_directory(self, tmp_path):
        """Test that simple datasets with unique plant names still work."""
        from unittest.mock import MagicMock

        # Create simple structure without timepoints
        plant_dir = tmp_path / "scan_12345"
        plant_dir.mkdir()
        (plant_dir / "1.jpg").touch()

        generator = ViewerGenerator(tmp_path)
        series = MagicMock()

        mock_video = MagicMock()
        mock_video.exists.return_value = False
        mock_video.filename = ["/some/path/scan_12345/1.jpg"]

        mock_labels = MagicMock()
        mock_labels.videos = [mock_video]

        series.primary_labels = mock_labels
        series.lateral_labels = None
        series.crown_labels = None

        success, plant_name, group = generator._find_and_remap_video(series)

        assert success is True
        assert plant_name == "scan_12345"
        remapped_paths = mock_video.replace_filename.call_args[0][0]
        assert "scan_12345" in remapped_paths[0]


class TestTimepointFiltering:
    """Tests for timepoint filtering with flat prediction directories.

    When predictions are in a flat directory (scan_123.slp) but images are organized
    by timepoint (Day0/Fado_1/, Day3/Fado_1/), the filter should work based on the
    discovered group from video remapping, not the scan path.
    """

    def test_filter_scans_by_group_basic(self):
        """Test that _filter_scans_by_timepoint filters using group field."""
        from sleap_roots.viewer.generator import ViewerGenerator, _filter_scans_by_timepoint

        # Mock scans_data with different groups
        scans_data = [
            {"name": "scan1", "group": "Day0_2025-11-27", "display_name": "plant1"},
            {"name": "scan2", "group": "Day3_2025-11-30", "display_name": "plant2"},
            {"name": "scan3", "group": "Day5_2025-12-02", "display_name": "plant3"},
        ]
        scans_template = [
            {"name": "scan1"},
            {"name": "scan2"},
            {"name": "scan3"},
        ]

        # Filter for Day0 only
        filtered_data, filtered_template = _filter_scans_by_timepoint(
            scans_data, scans_template, ["Day0*"]
        )

        assert len(filtered_data) == 1
        assert filtered_data[0]["name"] == "scan1"
        assert len(filtered_template) == 1
        assert filtered_template[0]["name"] == "scan1"

    def test_filter_scans_by_group_multiple_patterns_or_logic(self):
        """Test that multiple --timepoint flags match ANY pattern (OR logic)."""
        from sleap_roots.viewer.generator import _filter_scans_by_timepoint

        # Mock scans_data with different groups
        scans_data = [
            {"name": "scan1", "group": "Day0", "display_name": "plant1"},
            {"name": "scan2", "group": "Day3", "display_name": "plant2"},
            {"name": "scan3", "group": "Day5", "display_name": "plant3"},
            {"name": "scan4", "group": "Day7", "display_name": "plant4"},
        ]
        scans_template = [{"name": s["name"]} for s in scans_data]

        # Filter for Day0 OR Day5
        filtered_data, filtered_template = _filter_scans_by_timepoint(
            scans_data, scans_template, ["Day0*", "Day5*"]
        )

        assert len(filtered_data) == 2
        assert filtered_data[0]["group"] == "Day0"
        assert filtered_data[1]["group"] == "Day5"
        assert len(filtered_template) == 2

    def test_filter_scans_warns_when_no_matches(self):
        """Test that filtering warns when no scans match the pattern."""
        import warnings
        from sleap_roots.viewer.generator import _filter_scans_by_timepoint

        scans_data = [
            {"name": "scan1", "group": "Day0", "display_name": "plant1"},
            {"name": "scan2", "group": "Day3", "display_name": "plant2"},
        ]
        scans_template = [{"name": s["name"]} for s in scans_data]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            filtered_data, filtered_template = _filter_scans_by_timepoint(
                scans_data, scans_template, ["NonexistentPattern*"]
            )

            # Should warn about no matching scans
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "no scans" in msg.lower() or "pattern" in msg.lower()
                for msg in warning_messages
            ), f"Expected warning about no matching scans, got: {warning_messages}"

        # Should return empty lists
        assert len(filtered_data) == 0
        assert len(filtered_template) == 0

    def test_filter_scans_case_insensitive(self):
        """Test that 'day0' matches 'Day0' (case insensitive)."""
        from sleap_roots.viewer.generator import _filter_scans_by_timepoint

        scans_data = [
            {"name": "scan1", "group": "Day0_2025-11-27", "display_name": "plant1"},
            {"name": "scan2", "group": "Day3_2025-11-30", "display_name": "plant2"},
        ]
        scans_template = [{"name": s["name"]} for s in scans_data]

        # Use lowercase pattern to match uppercase group
        filtered_data, filtered_template = _filter_scans_by_timepoint(
            scans_data, scans_template, ["day0*"]  # lowercase
        )

        assert len(filtered_data) == 1
        assert filtered_data[0]["group"] == "Day0_2025-11-27"

    def test_filter_scans_excludes_none_group(self):
        """Test that scans with group=None are excluded when filtering."""
        from sleap_roots.viewer.generator import _filter_scans_by_timepoint

        scans_data = [
            {"name": "scan1", "group": "Day0", "display_name": "plant1"},
            {"name": "scan2", "group": None, "display_name": "plant2"},  # Failed remap
            {"name": "scan3", "group": "Day3", "display_name": "plant3"},
        ]
        scans_template = [{"name": s["name"]} for s in scans_data]

        # Filter should exclude None groups
        filtered_data, filtered_template = _filter_scans_by_timepoint(
            scans_data, scans_template, ["Day*"]
        )

        assert len(filtered_data) == 2
        assert all(s["group"] is not None for s in filtered_data)
        assert "scan2" not in [s["name"] for s in filtered_data]

    def test_filter_scans_keeps_data_and_template_in_sync(self):
        """Test that scans_data and scans_template have same scans after filtering."""
        from sleap_roots.viewer.generator import _filter_scans_by_timepoint

        scans_data = [
            {"name": "scan1", "group": "Day0", "display_name": "plant1", "frames": []},
            {"name": "scan2", "group": "Day3", "display_name": "plant2", "frames": []},
            {"name": "scan3", "group": "Day0", "display_name": "plant3", "frames": []},
            {"name": "scan4", "group": "Day5", "display_name": "plant4", "frames": []},
        ]
        scans_template = [
            {"name": "scan1", "other_field": "a"},
            {"name": "scan2", "other_field": "b"},
            {"name": "scan3", "other_field": "c"},
            {"name": "scan4", "other_field": "d"},
        ]

        # Filter for Day0
        filtered_data, filtered_template = _filter_scans_by_timepoint(
            scans_data, scans_template, ["Day0*"]
        )

        # Both should have same length and same scan names
        assert len(filtered_data) == len(filtered_template)
        assert len(filtered_data) == 2

        data_names = [s["name"] for s in filtered_data]
        template_names = [s["name"] for s in filtered_template]
        assert data_names == template_names
        assert data_names == ["scan1", "scan3"]


class TestPredictionSerialization:
    """Tests for prediction data serialization for client-side rendering."""

    def test_serialize_instance_extracts_points(self):
        """Test that serialize_instance extracts points as [[x, y], ...] list."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import serialize_instance

        # Create mock instance with points
        mock_instance = MagicMock()
        mock_instance.numpy.return_value = np.array([[10.0, 20.0], [30.0, 40.0]])

        # Create mock skeleton
        mock_skeleton = MagicMock()
        mock_skeleton.edge_inds = [(0, 1)]

        result = serialize_instance(mock_instance, mock_skeleton, "primary")

        assert "points" in result
        assert result["points"] == [[10.0, 20.0], [30.0, 40.0]]
        mock_instance.numpy.assert_called_once_with(invisible_as_nan=True)

    def test_serialize_instance_extracts_edges(self):
        """Test that serialize_instance extracts edges as [[i, j], ...] pairs."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import serialize_instance

        mock_instance = MagicMock()
        mock_instance.numpy.return_value = np.array([[0, 0], [10, 10], [20, 20]])

        mock_skeleton = MagicMock()
        mock_skeleton.edge_inds = [(0, 1), (1, 2)]

        result = serialize_instance(mock_instance, mock_skeleton, "lateral")

        assert "edges" in result
        assert result["edges"] == [[0, 1], [1, 2]]

    def test_serialize_instance_includes_score(self):
        """Test that serialize_instance includes instance score."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import serialize_instance

        mock_instance = MagicMock()
        mock_instance.numpy.return_value = np.array([[0, 0]])
        mock_instance.score = 0.95

        mock_skeleton = MagicMock()
        mock_skeleton.edge_inds = []

        result = serialize_instance(mock_instance, mock_skeleton, "crown")

        assert "score" in result
        assert result["score"] == 0.95

    def test_serialize_instance_includes_root_type(self):
        """Test that serialize_instance includes root_type."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import serialize_instance

        mock_instance = MagicMock()
        mock_instance.numpy.return_value = np.array([[0, 0]])

        mock_skeleton = MagicMock()
        mock_skeleton.edge_inds = []

        for root_type in ["primary", "lateral", "crown"]:
            result = serialize_instance(mock_instance, mock_skeleton, root_type)
            assert "root_type" in result
            assert result["root_type"] == root_type

    def test_serialize_instance_handles_none_score(self):
        """Test that serialize_instance handles missing/None scores gracefully."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import serialize_instance

        mock_instance = MagicMock()
        mock_instance.numpy.return_value = np.array([[0, 0]])
        mock_instance.score = None

        mock_skeleton = MagicMock()
        mock_skeleton.edge_inds = []

        result = serialize_instance(mock_instance, mock_skeleton, "primary")

        assert result["score"] is None

    def test_serialize_instance_handles_nan_points(self):
        """Test that serialize_instance handles NaN points (invisible nodes)."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import serialize_instance

        mock_instance = MagicMock()
        # First point visible, second point invisible (NaN)
        mock_instance.numpy.return_value = np.array([[10.0, 20.0], [np.nan, np.nan]])

        mock_skeleton = MagicMock()
        mock_skeleton.edge_inds = [(0, 1)]

        result = serialize_instance(mock_instance, mock_skeleton, "primary")

        # Points should include NaN values (JavaScript can handle them)
        assert len(result["points"]) == 2
        # NaN should be converted to None for JSON serialization
        assert result["points"][0] == [10.0, 20.0]
        assert result["points"][1] == [None, None]


class TestFramePredictionsSerialization:
    """Tests for frame-level prediction serialization."""

    def test_serialize_frame_predictions_returns_all_instances(
        self, canola_h5, canola_primary_slp, canola_lateral_slp, tmp_path
    ):
        """Test that serialize_frame_predictions returns all instances for a frame."""
        from sleap_roots.viewer.serializer import serialize_frame_predictions

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
            lateral_path=canola_lateral_slp,
        )

        html_path = tmp_path / "viewer.html"
        result = serialize_frame_predictions(series, frame_idx=0, html_path=html_path)

        assert "instances" in result
        assert isinstance(result["instances"], list)
        # Should have at least one instance
        assert len(result["instances"]) > 0

    def test_serialize_frame_predictions_includes_image_path(
        self, canola_h5, canola_primary_slp, tmp_path
    ):
        """Test that serialize_frame_predictions includes image_path relative to HTML."""
        from sleap_roots.viewer.serializer import serialize_frame_predictions

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        html_path = tmp_path / "subdir" / "viewer.html"
        result = serialize_frame_predictions(series, frame_idx=0, html_path=html_path)

        assert "image_path" in result
        # image_path should be a string (relative or absolute path to source image)
        assert isinstance(result["image_path"], str)

    def test_serialize_frame_predictions_instance_structure(
        self, canola_h5, canola_primary_slp, tmp_path
    ):
        """Test that each serialized instance has required fields."""
        from sleap_roots.viewer.serializer import serialize_frame_predictions

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        html_path = tmp_path / "viewer.html"
        result = serialize_frame_predictions(series, frame_idx=0, html_path=html_path)

        for instance in result["instances"]:
            assert "points" in instance
            assert "edges" in instance
            assert "score" in instance
            assert "root_type" in instance


class TestScanPredictionsSerialization:
    """Tests for scan-level prediction serialization."""

    def test_serialize_scan_predictions_returns_all_frames(
        self, canola_h5, canola_primary_slp, tmp_path
    ):
        """Test that serialize_scan_predictions returns predictions for sampled frames."""
        from sleap_roots.viewer.serializer import serialize_scan_predictions

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        html_path = tmp_path / "viewer.html"
        frame_indices = [0, 1, 2]  # Request 3 frames
        result = serialize_scan_predictions(series, frame_indices, html_path)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_serialize_scan_predictions_frame_structure(
        self, canola_h5, canola_primary_slp, tmp_path
    ):
        """Test that each frame in scan predictions has correct structure."""
        from sleap_roots.viewer.serializer import serialize_scan_predictions

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        html_path = tmp_path / "viewer.html"
        frame_indices = [0]
        result = serialize_scan_predictions(series, frame_indices, html_path)

        assert len(result) == 1
        frame_data = result[0]
        assert "instances" in frame_data
        assert "image_path" in frame_data


class TestClientRenderMode:
    """Tests for client-render mode (default) in ViewerGenerator."""

    def test_generate_client_render_embeds_predictions_json(
        self, canola_folder, tmp_path
    ):
        """Test that client-render mode embeds predictions as JSON in HTML."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        # Default mode (no render=True, no embed=True) should be client-render
        generator.generate(output_path, max_frames=2)

        content = output_path.read_text(encoding="utf-8")

        # Should have prediction data embedded as JSON
        assert "predictionsData" in content or "scansData" in content

    def test_generate_client_render_has_canvas_rendering_js(
        self, canola_folder, tmp_path
    ):
        """Test that client-render HTML contains Canvas drawing JavaScript."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2)

        content = output_path.read_text(encoding="utf-8")

        # Should have Canvas-related JavaScript
        assert "<script>" in content
        assert "drawPredictions" in content
        assert "overlayCanvas" in content
        assert "viridisColor" in content
        assert "ROOT_TYPE_COLORS" in content

    def test_generate_client_render_normalizes_confidence_scores(
        self, canola_folder, tmp_path
    ):
        """Test that client-render JS normalizes raw scores before viridis colormap."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2)

        content = output_path.read_text(encoding="utf-8")

        # Should have score normalization logic (min/max calculation)
        assert "minScore" in content
        assert "maxScore" in content
        # Should normalize before passing to viridis
        assert "normalizedScore" in content or "normalized" in content.lower()

    def test_generate_embed_mode_produces_base64_images(self, canola_folder, tmp_path):
        """Test that embed=True produces base64-embedded HTML (current behavior)."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, embed=True)

        content = output_path.read_text(encoding="utf-8")

        # Should have base64 images
        assert "data:image/png;base64," in content

    def test_generate_embed_mode_no_images_directory(self, canola_folder, tmp_path):
        """Test that embed=True does not create images directory."""
        output_path = tmp_path / "viewer.html"
        images_dir = tmp_path / "viewer_images"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, embed=True)

        # No images directory should be created
        assert not images_dir.exists()

    def test_generate_render_and_embed_mutually_exclusive(
        self, canola_folder, tmp_path
    ):
        """Test that render=True and embed=True raises error."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))

        with pytest.raises(ValueError, match="mutually exclusive"):
            generator.generate(output_path, render=True, embed=True)

    def test_generate_client_render_no_matplotlib_calls(
        self, rice_10do_pipeline_output_folder, tmp_path
    ):
        """Test that default mode does NOT call matplotlib render functions."""
        from unittest.mock import patch

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))

        # Mock matplotlib rendering functions to track calls
        with patch(
            "sleap_roots.viewer.renderer.render_frame_root_type"
        ) as mock_root_type, patch(
            "sleap_roots.viewer.renderer.render_frame_confidence"
        ) as mock_confidence:
            generator.generate(output_path, max_frames=2)

            # In client-render mode, matplotlib render functions should NOT be called
            mock_root_type.assert_not_called()
            mock_confidence.assert_not_called()

    def test_generate_client_render_contains_relative_image_paths(
        self, rice_10do_pipeline_output_folder, tmp_path
    ):
        """Test that default mode HTML contains relative image paths."""
        import json

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        generator.generate(output_path, max_frames=2)

        content = output_path.read_text(encoding="utf-8")

        # Extract scans data
        start = content.find("const scansData = ") + len("const scansData = ")
        end = content.find(";\n", start)
        scans_data = json.loads(content[start:end])

        # Check that frames have relative image paths (not base64, not absolute)
        for scan in scans_data:
            for frame in scan.get("frames_root_type", []):
                image_path = frame.get("image", "")
                # Should not be base64
                assert not image_path.startswith("data:image/")
                # Should be a path (relative or absolute, but not empty)
                if image_path:
                    assert ".jpg" in image_path.lower() or ".png" in image_path.lower()

    def test_generate_embed_mode_matches_render_mode_format(
        self, canola_folder, tmp_path
    ):
        """Test that embed=True produces HTML with same structure as old format."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, embed=True)

        content = output_path.read_text(encoding="utf-8")

        # Should have base64-embedded images
        assert "data:image/png;base64," in content
        # Should have scansData with frames
        assert "const scansData = " in content
        # Should NOT have predictions in frames (pre-rendered images show predictions)
        # Check that renderMode is 'embedded' (tojson outputs double quotes)
        assert 'const renderMode = "embedded"' in content


class TestSerializerEdgeCases:
    """Tests for edge cases in serializer module."""

    def test_frame_to_base64_grayscale_image(self):
        """Test frame_to_base64 handles 2D grayscale images."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import frame_to_base64

        series = MagicMock()
        # Create a 2D grayscale image
        series.video.__getitem__ = MagicMock(
            return_value=np.ones((100, 100), dtype=np.uint8) * 128
        )

        result = frame_to_base64(series, frame_idx=0)

        assert result.startswith("data:image/jpeg;base64,")

    def test_frame_to_base64_grayscale_with_channel(self):
        """Test frame_to_base64 handles 3D grayscale images with channel dim."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import frame_to_base64

        series = MagicMock()
        # Create a 3D grayscale image with channel dimension (H, W, 1)
        series.video.__getitem__ = MagicMock(
            return_value=np.ones((100, 100, 1), dtype=np.uint8) * 128
        )

        result = frame_to_base64(series, frame_idx=0)

        assert result.startswith("data:image/jpeg;base64,")

    def test_frame_to_base64_rgba_image(self):
        """Test frame_to_base64 handles RGBA images."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import frame_to_base64

        series = MagicMock()
        # Create a 4-channel RGBA image
        series.video.__getitem__ = MagicMock(
            return_value=np.ones((100, 100, 4), dtype=np.uint8) * 128
        )

        result = frame_to_base64(series, frame_idx=0, image_format="png")

        assert result.startswith("data:image/png;base64,")

    def test_frame_to_base64_unknown_format_returns_empty(self):
        """Test frame_to_base64 returns empty string for unknown image format."""
        from unittest.mock import MagicMock
        import numpy as np
        from sleap_roots.viewer.serializer import frame_to_base64

        series = MagicMock()
        # Create an image with invalid dimensions (e.g., 5 channels)
        series.video.__getitem__ = MagicMock(
            return_value=np.ones((100, 100, 5), dtype=np.uint8) * 128
        )

        result = frame_to_base64(series, frame_idx=0)

        assert result == ""

    def test_frame_to_base64_exception_returns_empty(self):
        """Test frame_to_base64 returns empty string on exception."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.serializer import frame_to_base64

        series = MagicMock()
        # Make video access raise an exception
        series.video.__getitem__ = MagicMock(side_effect=Exception("Video error"))

        result = frame_to_base64(series, frame_idx=0)

        assert result == ""

    def test_is_h5_video_exception_returns_false(self):
        """Test is_h5_video returns False on exception."""
        from unittest.mock import MagicMock, PropertyMock
        from sleap_roots.viewer.serializer import is_h5_video

        series = MagicMock()
        # Make filename access raise an exception
        type(series.video).filename = PropertyMock(side_effect=Exception("Error"))

        result = is_h5_video(series)

        assert result is False

    def test_serialize_frame_predictions_skips_none_skeleton(self):
        """Test serialize_frame_predictions skips instances with None skeleton."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.serializer import serialize_frame_predictions

        series = MagicMock()

        # Create mock instance with None skeleton
        mock_instance = MagicMock()
        mock_instance.skeleton = None

        mock_labeled_frame = MagicMock()
        mock_labeled_frame.instances = [mock_instance]

        series.get_frame.return_value = {"primary": mock_labeled_frame}
        series.video = None

        result = serialize_frame_predictions(
            series, frame_idx=0, html_path=Path("/tmp/viewer.html")
        )

        # Instance with None skeleton should be skipped
        assert len(result["instances"]) == 0

    def test_get_frame_image_path_no_video(self):
        """Test _get_frame_image_path returns empty string when no video."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.serializer import _get_frame_image_path

        series = MagicMock()
        series.video = None

        result = _get_frame_image_path(series, 0, Path("/tmp/viewer.html"))

        assert result == ""

    def test_get_frame_image_path_frame_out_of_range(self):
        """Test _get_frame_image_path returns empty for out of range frame."""
        from unittest.mock import MagicMock, PropertyMock
        from sleap_roots.viewer.serializer import _get_frame_image_path

        series = MagicMock()
        # Mock an ImageVideo with only 2 frames
        type(series.video).filename = PropertyMock(
            return_value=["/path/0.jpg", "/path/1.jpg"]
        )

        result = _get_frame_image_path(series, 10, Path("/tmp/viewer.html"))

        assert result == ""

    def test_get_frame_image_path_exception_returns_empty(self):
        """Test _get_frame_image_path returns empty on exception."""
        from unittest.mock import MagicMock, PropertyMock
        from sleap_roots.viewer.serializer import _get_frame_image_path

        series = MagicMock()
        type(series.video).filename = PropertyMock(side_effect=Exception("Error"))

        result = _get_frame_image_path(series, 0, Path("/tmp/viewer.html"))

        assert result == ""

    def test_get_skeleton_edges(self):
        """Test get_skeleton_edges extracts edges correctly."""
        from unittest.mock import MagicMock
        from sleap_roots.viewer.serializer import get_skeleton_edges

        skeleton = MagicMock()
        skeleton.edge_inds = [(0, 1), (1, 2), (2, 3)]

        result = get_skeleton_edges(skeleton)

        assert result == [[0, 1], [1, 2], [2, 3]]


class TestH5FrameExtraction:
    """Tests for H5 frame extraction in client-render mode."""

    def test_frame_to_base64_returns_data_uri(self, canola_h5, canola_primary_slp):
        """Test that frame_to_base64 returns a valid data URI."""
        from sleap_roots.viewer.serializer import frame_to_base64

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        result = frame_to_base64(series, frame_idx=0)

        assert isinstance(result, str)
        assert result.startswith("data:image/jpeg;base64,")
        # Should have actual base64 content
        assert len(result) > 100

    def test_frame_to_base64_respects_format_png(self, canola_h5, canola_primary_slp):
        """Test that frame_to_base64 respects PNG format."""
        from sleap_roots.viewer.serializer import frame_to_base64

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        result = frame_to_base64(series, frame_idx=0, image_format="png")

        assert result.startswith("data:image/png;base64,")

    def test_frame_to_base64_respects_quality(self, canola_h5, canola_primary_slp):
        """Test that frame_to_base64 produces different sizes for different quality."""
        from sleap_roots.viewer.serializer import frame_to_base64

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        low_quality = frame_to_base64(series, frame_idx=0, quality=10)
        high_quality = frame_to_base64(series, frame_idx=0, quality=95)

        # Higher quality should produce larger file
        assert len(high_quality) > len(low_quality)

    def test_frame_to_base64_returns_empty_on_no_video(self, canola_primary_slp):
        """Test that frame_to_base64 returns empty string when no video."""
        from sleap_roots.viewer.serializer import frame_to_base64

        series = Series.load(
            series_name="test",
            h5_path=None,
            primary_path=canola_primary_slp,
        )

        result = frame_to_base64(series, frame_idx=0)

        assert result == ""

    def test_is_h5_video_returns_true_for_h5(self, canola_h5, canola_primary_slp):
        """Test is_h5_video returns True for H5-backed video."""
        from sleap_roots.viewer.serializer import is_h5_video

        series = Series.load(
            series_name="test",
            h5_path=canola_h5,
            primary_path=canola_primary_slp,
        )

        assert is_h5_video(series) is True

    def test_is_h5_video_returns_false_for_no_video(self, canola_primary_slp):
        """Test is_h5_video returns False when no video."""
        from sleap_roots.viewer.serializer import is_h5_video

        series = Series.load(
            series_name="test",
            h5_path=None,
            primary_path=canola_primary_slp,
        )

        assert is_h5_video(series) is False

    def test_client_render_with_h5_embeds_base64_images(self, canola_folder, tmp_path):
        """Test that client-render mode with H5 source embeds base64 images."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2)

        content = output_path.read_text(encoding="utf-8")

        # For H5 sources, client-render should embed base64 images
        assert "data:image/jpeg;base64," in content

    def test_client_render_with_h5_still_has_predictions(self, canola_folder, tmp_path):
        """Test that client-render with H5 still includes predictions for canvas."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2)

        content = output_path.read_text(encoding="utf-8")

        # Should have prediction data for canvas rendering
        assert '"predictions":' in content
        assert '"points":' in content
        assert '"edges":' in content


class TestPreRenderedMode:
    """Tests for pre-rendered mode (--render flag)."""

    def test_generate_render_creates_images_directory(self, canola_folder, tmp_path):
        """Test that render=True creates viewer_images directory."""
        output_path = tmp_path / "viewer.html"
        images_dir = tmp_path / "viewer_images"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, render=True)

        # Images directory should be created
        assert images_dir.exists()
        assert images_dir.is_dir()

    def test_generate_render_saves_images_to_disk(self, canola_folder, tmp_path):
        """Test that render=True saves images as files."""
        output_path = tmp_path / "viewer.html"
        images_dir = tmp_path / "viewer_images"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, render=True)

        # Should have image files in the directory
        image_files = list(images_dir.glob("**/*.jpeg")) + list(
            images_dir.glob("**/*.jpg")
        )
        assert len(image_files) > 0

    def test_generate_render_html_references_images(self, canola_folder, tmp_path):
        """Test that render=True HTML references image files."""
        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, render=True)

        content = output_path.read_text(encoding="utf-8")

        # HTML should reference external images (not base64)
        assert "viewer_images/" in content or ".jpeg" in content or ".jpg" in content

    def test_generate_render_uses_jpeg_by_default(self, canola_folder, tmp_path):
        """Test that render=True uses JPEG format by default."""
        output_path = tmp_path / "viewer.html"
        images_dir = tmp_path / "viewer_images"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, render=True)

        # Should have JPEG files
        jpeg_files = list(images_dir.glob("**/*.jpeg")) + list(
            images_dir.glob("**/*.jpg")
        )
        png_files = list(images_dir.glob("**/*.png"))
        assert len(jpeg_files) > 0
        assert len(png_files) == 0

    def test_generate_render_png_format(self, canola_folder, tmp_path):
        """Test that render=True with image_format='png' saves PNG images."""
        output_path = tmp_path / "viewer.html"
        images_dir = tmp_path / "viewer_images"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, render=True, image_format="png")

        # Should have PNG files
        png_files = list(images_dir.glob("**/*.png"))
        assert len(png_files) > 0

    def test_generate_render_calls_matplotlib_functions(self, canola_folder, tmp_path):
        """Test that render=True calls matplotlib render functions."""
        from unittest.mock import patch

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))

        # Mock matplotlib rendering functions to track calls
        with patch(
            "sleap_roots.viewer.generator.render_frame_root_type"
        ) as mock_root_type, patch(
            "sleap_roots.viewer.generator.render_frame_confidence"
        ) as mock_confidence:
            generator.generate(output_path, max_frames=2, render=True)

            # In render mode, matplotlib functions should be called
            assert mock_root_type.called or mock_confidence.called

    def test_figure_to_file_saves_with_correct_format(self, canola_folder, tmp_path):
        """Test that _save_figure_to_file saves figure to disk correctly."""
        generator = ViewerGenerator(Path(canola_folder))

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])

        # Test JPEG format
        jpeg_path = tmp_path / "test.jpeg"
        generator._save_figure_to_file(fig, jpeg_path, "jpeg", 85)
        assert jpeg_path.exists()
        with open(jpeg_path, "rb") as f:
            header = f.read(2)
            assert header == b"\xff\xd8"  # JPEG magic number

        # Test PNG format
        png_path = tmp_path / "test.png"
        generator._save_figure_to_file(fig, png_path, "png", 85)
        assert png_path.exists()
        with open(png_path, "rb") as f:
            header = f.read(8)
            assert header == b"\x89PNG\r\n\x1a\n"  # PNG magic number

        plt.close(fig)


class TestCLI:
    """Tests for the viewer CLI command."""

    def test_cli_basic_invocation(self, rice_10do_pipeline_output_folder, tmp_path):
        """Test basic CLI invocation creates viewer."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "cli_viewer.html"

        result = runner.invoke(
            viewer,
            [
                str(rice_10do_pipeline_output_folder),
                "--output",
                str(output_path),
                "--max-frames",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_path.exists()

    def test_cli_with_images_dir(self, canola_folder, tmp_path):
        """Test CLI with explicit images directory."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "viewer.html"

        result = runner.invoke(
            viewer,
            [
                str(canola_folder),
                "--images",
                str(canola_folder),
                "--output",
                str(output_path),
                "--max-frames",
                "2",
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_path.exists()

    def test_cli_embed_mode(self, canola_folder, tmp_path):
        """Test CLI with --embed flag."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "embedded.html"

        result = runner.invoke(
            viewer,
            [
                str(canola_folder),
                "--output",
                str(output_path),
                "--max-frames",
                "2",
                "--embed",
            ],
        )

        assert result.exit_code == 0, result.output
        content = output_path.read_text(encoding="utf-8")
        assert "data:image/png;base64," in content

    def test_cli_render_mode(self, canola_folder, tmp_path):
        """Test CLI with --render flag."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "rendered.html"

        result = runner.invoke(
            viewer,
            [
                str(canola_folder),
                "--output",
                str(output_path),
                "--max-frames",
                "2",
                "--render",
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_path.exists()
        images_dir = tmp_path / "rendered_images"
        assert images_dir.exists()

    def test_cli_render_and_embed_mutually_exclusive(self, canola_folder, tmp_path):
        """Test that --render and --embed together raises error."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "viewer.html"

        result = runner.invoke(
            viewer,
            [
                str(canola_folder),
                "--output",
                str(output_path),
                "--render",
                "--embed",
            ],
        )

        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_cli_zip_flag(self, canola_folder, tmp_path):
        """Test CLI with --zip flag."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "viewer.html"

        result = runner.invoke(
            viewer,
            [
                str(canola_folder),
                "--output",
                str(output_path),
                "--max-frames",
                "2",
                "--embed",
                "--zip",
            ],
        )

        assert result.exit_code == 0, result.output
        zip_path = tmp_path / "viewer.zip"
        assert zip_path.exists()

    def test_cli_format_and_quality_options(self, canola_folder, tmp_path):
        """Test CLI with --format and --quality options."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "viewer.html"

        result = runner.invoke(
            viewer,
            [
                str(canola_folder),
                "--output",
                str(output_path),
                "--max-frames",
                "2",
                "--render",
                "--format",
                "png",
                "--quality",
                "90",
            ],
        )

        assert result.exit_code == 0, result.output
        images_dir = tmp_path / "viewer_images"
        png_files = list(images_dir.glob("**/*.png"))
        assert len(png_files) > 0

    def test_cli_no_limit_flag(self, canola_folder, tmp_path):
        """Test CLI with --no-limit flag."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        output_path = tmp_path / "viewer.html"

        result = runner.invoke(
            viewer,
            [
                str(canola_folder),
                "--output",
                str(output_path),
                "--max-frames",
                "0",
                "--no-limit",
            ],
        )

        assert result.exit_code == 0, result.output
        assert output_path.exists()

    def test_cli_empty_directory_creates_empty_viewer(self, tmp_path):
        """Test CLI with empty directory creates viewer with no scans."""
        from click.testing import CliRunner
        from sleap_roots.viewer.cli import viewer

        runner = CliRunner()
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_path = tmp_path / "viewer.html"

        result = runner.invoke(
            viewer,
            [str(empty_dir), "--output", str(output_path)],
        )

        # CLI succeeds but creates an HTML with no scan data
        assert result.exit_code == 0
        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "scansData = []" in content


class TestZIPArchive:
    """Tests for ZIP archive creation."""

    def test_generate_create_zip_creates_zip_file(
        self, rice_10do_pipeline_output_folder, tmp_path
    ):
        """Test that generate(create_zip=True) creates a zip file."""
        import zipfile

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        generator.generate(output_path, max_frames=2, create_zip=True)

        zip_path = tmp_path / "viewer.zip"
        assert zip_path.exists()
        assert zipfile.is_zipfile(zip_path)

    def test_client_render_zip_copies_source_images(
        self, rice_10do_pipeline_output_folder, tmp_path
    ):
        """Test that client-render zip copies source images into archive."""
        import zipfile

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        generator.generate(output_path, max_frames=2, create_zip=True)

        zip_path = tmp_path / "viewer.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # Should have image files in the archive
            image_files = [n for n in names if n.endswith((".jpg", ".jpeg", ".png"))]
            assert len(image_files) > 0

    def test_client_render_zip_rewrites_paths_in_html(
        self, rice_10do_pipeline_output_folder, tmp_path
    ):
        """Test that client-render zip rewrites image paths in HTML."""
        import zipfile

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(rice_10do_pipeline_output_folder))
        generator.generate(output_path, max_frames=2, create_zip=True)

        zip_path = tmp_path / "viewer.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Read the HTML from the archive
            html_content = zf.read("viewer.html").decode("utf-8")
            # Paths should be relative within the archive (images/ prefix)
            assert "images/" in html_content
            # Should NOT have absolute paths or paths outside the archive
            assert "tests/data/" not in html_content

    def test_pre_rendered_zip_contains_html_and_images(self, canola_folder, tmp_path):
        """Test that pre-rendered zip contains HTML + images directory."""
        import zipfile

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, render=True, create_zip=True)

        zip_path = tmp_path / "viewer.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # Should have HTML file
            assert "viewer.html" in names
            # Should have images in viewer_images directory
            image_files = [n for n in names if "viewer_images/" in n]
            assert len(image_files) > 0

    def test_embedded_zip_contains_only_html(self, canola_folder, tmp_path):
        """Test that embedded zip contains only HTML file."""
        import zipfile

        output_path = tmp_path / "viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, embed=True, create_zip=True)

        zip_path = tmp_path / "viewer.zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            # Should only have the HTML file (all images embedded)
            assert names == ["viewer.html"]

    def test_zip_file_named_correctly(self, canola_folder, tmp_path):
        """Test that zip file is named {output_stem}.zip."""
        output_path = tmp_path / "my_custom_viewer.html"
        generator = ViewerGenerator(Path(canola_folder))
        generator.generate(output_path, max_frames=2, embed=True, create_zip=True)

        # Should create my_custom_viewer.zip
        zip_path = tmp_path / "my_custom_viewer.zip"
        assert zip_path.exists()
