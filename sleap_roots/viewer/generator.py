"""HTML viewer generator for SLEAP prediction visualization."""

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import attrs
import jinja2
import matplotlib.pyplot as plt

from sleap_roots.series import Series, find_all_slp_paths
from sleap_roots.viewer.renderer import (
    figure_to_base64,
    render_frame_confidence,
    render_frame_root_type,
)

# Frame limit constants
DEFAULT_MAX_FRAMES = 10
WARNING_FRAME_THRESHOLD = 100
ERROR_FRAME_THRESHOLD = 1000


class FrameLimitExceededError(Exception):
    """Raised when total frames exceed the limit without --no-limit."""

    pass


def select_frame_indices(total_frames: int, max_frames: int) -> List[int]:
    """Select frame indices for sampling.

    Selects evenly-distributed frames including first and last.

    Args:
        total_frames: Total number of frames in the scan.
        max_frames: Maximum frames to select. If 0 or negative, returns all frames.

    Returns:
        List of frame indices to render.
    """
    if max_frames <= 0 or total_frames <= max_frames:
        return list(range(total_frames))

    if total_frames == 1:
        return [0]

    if max_frames == 1:
        return [0]

    # Always include first and last, distribute rest evenly
    indices = [0]
    if max_frames > 2:
        step = (total_frames - 1) / (max_frames - 1)
        for i in range(1, max_frames - 1):
            indices.append(round(i * step))
    indices.append(total_frames - 1)

    return indices


def confidence_to_hex(normalized_score: float, colormap: str = "viridis") -> str:
    """Convert a normalized confidence score (0-1) to a hex color using a colormap.

    Args:
        normalized_score: Score in 0-1 range.
        colormap: Matplotlib colormap name. Default is viridis.

    Returns:
        Hex color string (e.g., "#440154").
    """
    cmap = plt.get_cmap(colormap)
    rgba = cmap(float(normalized_score))
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


@attrs.define
class ScanInfo:
    """Information about a single scan for the viewer.

    Attributes:
        name: Unique identifier for the scan (usually the series name).
        primary_path: Path to primary root predictions file.
        lateral_path: Path to lateral root predictions file.
        crown_path: Path to crown root predictions file.
        h5_path: Path to HDF5 image file.
        frame_count: Number of frames in the scan.
    """

    name: str
    primary_path: Optional[Path] = None
    lateral_path: Optional[Path] = None
    crown_path: Optional[Path] = None
    h5_path: Optional[Path] = None
    frame_count: int = 0


class ViewerGenerator:
    """Generates static HTML viewers for SLEAP prediction visualization.

    Attributes:
        predictions_dir: Directory containing .slp prediction files.
        images_dir: Optional directory containing source images.
    """

    def __init__(
        self,
        predictions_dir: Path,
        images_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the viewer generator.

        Args:
            predictions_dir: Directory containing .slp prediction files.
            images_dir: Optional directory containing source images.
        """
        self.predictions_dir = Path(predictions_dir)
        self.images_dir = Path(images_dir) if images_dir else None

    def discover_scans(self, recursive: bool = True) -> List[ScanInfo]:
        """Discover all scans in the predictions directory.

        Args:
            recursive: If True, search subdirectories recursively (default).
                This is useful for pipeline output where scans are organized
                into subdirectories.

        Returns:
            List of ScanInfo objects for each discovered scan.

        Raises:
            FileNotFoundError: If predictions_dir does not exist.
        """
        if not self.predictions_dir.exists():
            raise FileNotFoundError(
                f"Predictions directory not found: {self.predictions_dir}"
            )

        # Find all .slp files in the predictions directory
        if recursive:
            # Search recursively for nested pipeline output directories
            slp_paths = [str(p) for p in self.predictions_dir.glob("**/*.slp")]
        else:
            slp_paths = find_all_slp_paths(str(self.predictions_dir))

        if not slp_paths:
            return []

        # Group by series name (extract from filename)
        # Expected patterns:
        # - series_name.primary.predictions.slp
        # - series_name.lateral.predictions.slp
        # - series_name.crown.predictions.slp
        # - series_name.model{id}.rootprimary.slp (pipeline output)
        series_dict = {}

        for slp_path in slp_paths:
            path = Path(slp_path)
            filename = path.name

            # Extract series name (first part before any dot)
            series_name = filename.split(".")[0]

            if series_name not in series_dict:
                series_dict[series_name] = {
                    "primary_path": None,
                    "lateral_path": None,
                    "crown_path": None,
                    "h5_path": None,
                }

            # Determine root type from filename
            lower_filename = filename.lower()
            if "primary" in lower_filename:
                series_dict[series_name]["primary_path"] = path
            elif "lateral" in lower_filename:
                series_dict[series_name]["lateral_path"] = path
            elif "crown" in lower_filename:
                series_dict[series_name]["crown_path"] = path

        # Look for corresponding h5 files
        images_dir = self.images_dir or self.predictions_dir
        for series_name in series_dict:
            h5_path = images_dir / f"{series_name}.h5"
            if h5_path.exists():
                series_dict[series_name]["h5_path"] = h5_path

        # Create ScanInfo objects with frame counts
        scans = []
        for series_name, paths in series_dict.items():
            # Load series to get frame count
            try:
                series = Series.load(
                    series_name=series_name,
                    h5_path=str(paths["h5_path"]) if paths["h5_path"] else None,
                    primary_path=(
                        str(paths["primary_path"]) if paths["primary_path"] else None
                    ),
                    lateral_path=(
                        str(paths["lateral_path"]) if paths["lateral_path"] else None
                    ),
                    crown_path=(
                        str(paths["crown_path"]) if paths["crown_path"] else None
                    ),
                )
                frame_count = len(series)
            except Exception:
                frame_count = 0

            scan_info = ScanInfo(
                name=series_name,
                primary_path=paths["primary_path"],
                lateral_path=paths["lateral_path"],
                crown_path=paths["crown_path"],
                h5_path=paths["h5_path"],
                frame_count=frame_count,
            )
            scans.append(scan_info)

        return scans

    def _load_series(self, scan: ScanInfo) -> Optional[Series]:
        """Load a Series object from ScanInfo.

        Args:
            scan: ScanInfo object with paths.

        Returns:
            Loaded Series or None if loading fails.
        """
        try:
            return Series.load(
                series_name=scan.name,
                h5_path=str(scan.h5_path) if scan.h5_path else None,
                primary_path=str(scan.primary_path) if scan.primary_path else None,
                lateral_path=str(scan.lateral_path) if scan.lateral_path else None,
                crown_path=str(scan.crown_path) if scan.crown_path else None,
            )
        except Exception:
            return None

    def _find_and_remap_video(self, series: Series) -> bool:
        """Find local image directory and remap video paths.

        Pipeline output .slp files use ImageVideo backend with embedded absolute
        paths that don't exist on the local machine. This method extracts the
        image directory name from the embedded paths and searches for a matching
        local directory, then remaps the video to use local paths.

        Args:
            series: Series object with potentially invalid video paths.

        Returns:
            True if video was successfully remapped, False otherwise.
        """
        # Get video from one of the labels files
        video = None
        for labels in [
            series.crown_labels,
            series.primary_labels,
            series.lateral_labels,
        ]:
            if labels is not None and labels.videos:
                video = labels.videos[0]
                break

        if video is None:
            return False

        # Check if video already works
        try:
            if video.exists():
                series.video = video
                return True
        except Exception as exc:
            warnings.warn(f"Failed to check video existence for {video!r}: {exc}")

        # Get embedded paths - for ImageVideo, filename is a list
        try:
            filenames = video.filename
            if not filenames:
                return False

            # Handle both list and single path cases
            if isinstance(filenames, (list, tuple)):
                first_path = Path(filenames[0])
            else:
                first_path = Path(filenames)

            # Extract image directory name from embedded path
            img_dir_name = first_path.parent.name
        except Exception:
            return False

        # Search for local image directory
        search_dirs = [self.predictions_dir]
        if self.images_dir:
            search_dirs.append(self.images_dir)

        local_img_dir = None
        for search_dir in search_dirs:
            # Direct match
            candidate = search_dir / img_dir_name
            if candidate.is_dir():
                local_img_dir = candidate
                break

            # Recursive search
            for subdir in search_dir.glob(f"**/{img_dir_name}"):
                if subdir.is_dir():
                    local_img_dir = subdir
                    break

            if local_img_dir:
                break

        if local_img_dir is None:
            return False

        # Get sorted list of local images
        # Support common image extensions (case-insensitive)
        image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        local_images = [
            p for p in local_img_dir.iterdir() if p.suffix.lower() in image_exts
        ]

        if not local_images:
            return False

        # Sort images by numeric filename (e.g., 1.jpg, 2.jpg, ...)
        # Non-numeric filenames sort alphabetically after numeric ones
        def sort_key(p: Path):
            """Sort key that prefers numeric ordering with alphabetic fallback."""
            try:
                return (0, int(p.stem))
            except ValueError:
                return (1, p.stem)

        local_images = sorted(local_images, key=sort_key)

        # Remap video paths
        try:
            video.replace_filename([str(p) for p in local_images])
            series.video = video
            return True
        except Exception:
            return False

    def _get_instance_counts(self, series: Series, frame_idx: int) -> Dict[str, int]:
        """Get instance counts per root type for a frame.

        Args:
            series: Series object.
            frame_idx: Frame index.

        Returns:
            Dictionary of root type to instance count.
        """
        counts = {}
        frames = series.get_frame(frame_idx)

        for root_type in ["primary", "lateral", "crown"]:
            if root_type in frames and frames[root_type] is not None:
                counts[root_type] = len(frames[root_type].instances)

        return counts

    def _get_mean_confidence(self, series: Series, frame_idx: int) -> Optional[float]:
        """Get mean confidence score for a frame.

        Args:
            series: Series object.
            frame_idx: Frame index.

        Returns:
            Mean confidence score or None if not available.
        """
        frames = series.get_frame(frame_idx)
        scores = []

        for root_type in ["primary", "lateral", "crown"]:
            if root_type in frames and frames[root_type] is not None:
                for instance in frames[root_type].instances:
                    if hasattr(instance, "score") and instance.score is not None:
                        # Convert numpy float to Python float for JSON serialization
                        scores.append(float(instance.score))

        return float(sum(scores) / len(scores)) if scores else None

    def generate(
        self,
        output_path: Path,
        max_frames: int = DEFAULT_MAX_FRAMES,
        no_limit: bool = False,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """Generate the HTML viewer.

        Args:
            output_path: Path to write the HTML file.
            max_frames: Maximum frames to render per scan. Default is 10.
                If 0, all frames are rendered.
            no_limit: If True, disable the 1000 total frame limit.
            progress_callback: Optional callback for progress updates.
                Called with (scan_name, frames_done, total_frames) after each frame.

        Raises:
            FrameLimitExceededError: If total frames exceed 1000 without no_limit.
        """
        output_path = Path(output_path)

        # Discover all scans
        scans = self.discover_scans()

        # Calculate total frames and check limits
        scan_frame_counts = []
        for scan in scans:
            if scan.frame_count > 0:
                sampled_count = len(select_frame_indices(scan.frame_count, max_frames))
                scan_frame_counts.append(sampled_count)

        total_frames = sum(scan_frame_counts)

        if total_frames > WARNING_FRAME_THRESHOLD:
            warnings.warn(
                f"Generating viewer with {total_frames} total frames. "
                f"This may produce a large HTML file.",
                UserWarning,
            )

        if total_frames > ERROR_FRAME_THRESHOLD and not no_limit:
            raise FrameLimitExceededError(
                f"Total frames ({total_frames}) exceeds limit of {ERROR_FRAME_THRESHOLD}. "
                f"Use --no-limit to override, or reduce --max-frames."
            )

        # Prepare scan data for template
        scans_data: List[Dict[str, Any]] = []
        scans_template: List[Dict[str, Any]] = []

        for scan in scans:
            series = self._load_series(scan)
            if series is None or scan.frame_count == 0:
                continue

            # Try to remap video paths for pipeline output with image directories
            if series.video is None:
                self._find_and_remap_video(series)

            # Check if video is available for rendering
            if series.video is None:
                continue

            # Select frames to render
            frame_indices = select_frame_indices(scan.frame_count, max_frames)

            scan_data: Dict[str, Any] = {
                "name": scan.name,
                "frame_count": len(frame_indices),
                "frames_root_type": [],
                "frames_confidence": [],
            }

            total_frames_in_scan = len(frame_indices)

            # Render selected frames
            for i, frame_idx in enumerate(frame_indices):
                instance_counts = self._get_instance_counts(series, frame_idx)
                mean_confidence = self._get_mean_confidence(series, frame_idx)

                # Render root type view
                try:
                    fig_root = render_frame_root_type(series, frame_idx)
                    img_root = figure_to_base64(fig_root, close=True)
                except Exception:
                    img_root = ""

                # Render confidence view
                try:
                    fig_conf = render_frame_confidence(series, frame_idx)
                    img_conf = figure_to_base64(fig_conf, close=True)
                except Exception:
                    img_conf = img_root  # Fallback to root type

                scan_data["frames_root_type"].append(
                    {
                        "image": img_root,
                        "instance_counts": instance_counts,
                        "mean_confidence": mean_confidence,
                    }
                )
                scan_data["frames_confidence"].append(
                    {
                        "image": img_conf,
                        "instance_counts": instance_counts,
                        "mean_confidence": mean_confidence,
                    }
                )

                # Call progress callback after each frame
                if progress_callback is not None:
                    progress_callback(scan.name, i + 1, total_frames_in_scan)

            # Use first frame as thumbnail
            thumbnail = (
                scan_data["frames_root_type"][0]["image"]
                if scan_data["frames_root_type"]
                else ""
            )

            # Calculate overall raw mean confidence for the scan
            all_confidences = [
                f["mean_confidence"]
                for f in scan_data["frames_root_type"]
                if f["mean_confidence"] is not None
            ]
            overall_confidence = (
                float(sum(all_confidences) / len(all_confidences))
                if all_confidences
                else None
            )

            scans_data.append(scan_data)
            scans_template.append(
                {
                    "name": scan.name,
                    "frame_count": scan.frame_count,
                    "thumbnail": thumbnail,
                    "mean_confidence": overall_confidence,
                }
            )

        # Close any remaining matplotlib figures
        plt.close("all")

        # Normalize confidence scores across all scans using global min/max
        from sleap_roots.viewer.renderer import normalize_confidence

        all_raw_scores = []
        for scan_data in scans_data:
            for frame in scan_data["frames_root_type"]:
                if frame["mean_confidence"] is not None:
                    all_raw_scores.append(frame["mean_confidence"])

        if all_raw_scores:
            global_min = min(all_raw_scores)
            global_max = max(all_raw_scores)

            # Normalize frame-level confidences
            for scan_data in scans_data:
                for view in ["frames_root_type", "frames_confidence"]:
                    for frame in scan_data[view]:
                        if frame["mean_confidence"] is not None:
                            frame["mean_confidence"] = normalize_confidence(
                                frame["mean_confidence"], global_min, global_max
                            )

            # Normalize scan-level confidences and compute viridis colors
            for scan_t in scans_template:
                if scan_t["mean_confidence"] is not None:
                    scan_t["mean_confidence"] = normalize_confidence(
                        scan_t["mean_confidence"], global_min, global_max
                    )
                    scan_t["confidence_color"] = confidence_to_hex(
                        scan_t["mean_confidence"]
                    )
                else:
                    scan_t["confidence_color"] = None

        # Load and render template
        template_dir = Path(__file__).parent / "templates"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=True,
        )
        template = env.get_template("viewer.html")

        html_content = template.render(
            scans=scans_template,
            scans_json=json.dumps(scans_data),
        )

        # Write output file
        output_path.write_text(html_content, encoding="utf-8")
