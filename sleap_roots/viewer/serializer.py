"""Prediction data serialization for client-side rendering.

This module provides functions to serialize SLEAP prediction data (skeletons,
instances, scores) into JSON-friendly dictionaries for JavaScript Canvas rendering.
"""

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sleap_roots.series import Series


def frame_to_base64(
    series: Series,
    frame_idx: int,
    image_format: str = "jpeg",
    quality: int = 85,
) -> str:
    """Extract a frame from video and encode as base64 data URI.

    Args:
        series: The Series object containing the video.
        frame_idx: The frame index to extract.
        image_format: Image format ("jpeg" or "png").
        quality: JPEG quality (1-100), ignored for PNG.

    Returns:
        Base64 data URI string (e.g., "data:image/jpeg;base64,...").
        Empty string if extraction fails.
    """
    if series.video is None:
        return ""

    try:
        # Get frame as numpy array
        img = series.video[frame_idx]

        # Import PIL for image encoding
        from PIL import Image

        # Handle different image formats from video
        if img.ndim == 2:
            # Grayscale
            pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
        elif img.ndim == 3 and img.shape[2] == 1:
            # Grayscale with channel dimension
            pil_img = Image.fromarray(img[:, :, 0].astype(np.uint8), mode="L")
        elif img.ndim == 3 and img.shape[2] == 3:
            # RGB
            pil_img = Image.fromarray(img.astype(np.uint8), mode="RGB")
        elif img.ndim == 3 and img.shape[2] == 4:
            # RGBA
            pil_img = Image.fromarray(img.astype(np.uint8), mode="RGBA")
        else:
            return ""

        # Convert grayscale to RGB for JPEG
        if image_format.lower() in ("jpeg", "jpg") and pil_img.mode in ("L", "LA"):
            pil_img = pil_img.convert("RGB")

        # Encode to bytes
        buf = io.BytesIO()
        if image_format.lower() in ("jpeg", "jpg"):
            pil_img.save(buf, format="JPEG", quality=quality)
            mime_type = "image/jpeg"
        else:
            pil_img.save(buf, format="PNG")
            mime_type = "image/png"

        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()

        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return ""


def is_h5_video(series: Series) -> bool:
    """Check if the series video is an H5 file (not an image directory).

    Args:
        series: The Series object.

    Returns:
        True if video is H5-backed, False if it's an image directory.
    """
    if series.video is None:
        return False

    try:
        filenames = series.video.filename
        if isinstance(filenames, (list, tuple)):
            # ImageVideo backend - list of image paths
            return False
        else:
            # Single filename - check if it's an H5 file
            return str(filenames).lower().endswith((".h5", ".hdf5"))
    except Exception:
        return False


def serialize_instance(
    instance: Any,
    skeleton: Any,
    root_type: str,
) -> Dict[str, Any]:
    """Serialize a single instance for JavaScript rendering.

    Args:
        instance: A sleap_io Instance or PredictedInstance object.
        skeleton: The skeleton associated with the instance.
        root_type: The root type ("primary", "lateral", or "crown").

    Returns:
        Dictionary with keys:
            - points: List of [x, y] coordinates (None for invisible nodes)
            - edges: List of [i, j] node index pairs
            - score: Instance confidence score (float or None)
            - root_type: The root type string
    """
    # Extract points as numpy array
    points_array = instance.numpy(invisible_as_nan=True)

    # Convert to list, replacing NaN with None for JSON serialization
    points = []
    for point in points_array:
        x, y = point[0], point[1]
        if math.isnan(x) or math.isnan(y):
            points.append([None, None])
        else:
            points.append([float(x), float(y)])

    # Extract edges as list of [i, j] pairs
    edges = [[int(i), int(j)] for i, j in skeleton.edge_inds]

    # Extract score (may be None for ground truth instances)
    score = None
    if hasattr(instance, "score") and instance.score is not None:
        score = float(instance.score)

    return {
        "points": points,
        "edges": edges,
        "score": score,
        "root_type": root_type,
    }


def serialize_frame_predictions(
    series: Series,
    frame_idx: int,
    html_path: Path,
) -> Dict[str, Any]:
    """Serialize all predictions for a single frame.

    Args:
        series: The Series object containing predictions.
        frame_idx: The frame index to serialize.
        html_path: Path to the output HTML file (for computing relative image paths).

    Returns:
        Dictionary with keys:
            - instances: List of serialized instance dictionaries
            - image_path: Path to the source image (relative to HTML or absolute)
    """
    instances = []

    # Get labeled frames for this index
    frames = series.get_frame(frame_idx)

    # Serialize instances from each root type
    for root_type in ["primary", "lateral", "crown"]:
        if root_type not in frames or frames[root_type] is None:
            continue

        labeled_frame = frames[root_type]

        for instance in labeled_frame.instances:
            # Get skeleton from instance
            skeleton = instance.skeleton
            if skeleton is None:
                continue
            serialized = serialize_instance(instance, skeleton, root_type)
            instances.append(serialized)

    # Determine image path
    image_path = _get_frame_image_path(series, frame_idx, html_path)

    return {
        "instances": instances,
        "image_path": image_path,
    }


def serialize_scan_predictions(
    series: Series,
    frame_indices: List[int],
    html_path: Path,
) -> List[Dict[str, Any]]:
    """Serialize predictions for multiple frames in a scan.

    Args:
        series: The Series object containing predictions.
        frame_indices: List of frame indices to serialize.
        html_path: Path to the output HTML file (for computing relative image paths).

    Returns:
        List of frame prediction dictionaries, one per frame index.
    """
    return [
        serialize_frame_predictions(series, frame_idx, html_path)
        for frame_idx in frame_indices
    ]


def _get_frame_image_path(
    series: Series,
    frame_idx: int,
    html_path: Path,
) -> str:
    """Get the image path for a frame, relative to the HTML file if possible.

    Args:
        series: The Series object.
        frame_idx: The frame index.
        html_path: Path to the output HTML file.

    Returns:
        Image path as a string (relative to HTML directory if possible).
    """
    if series.video is None:
        return ""

    # Get the video filename(s)
    try:
        filenames = series.video.filename
        if isinstance(filenames, (list, tuple)):
            # ImageVideo backend - list of image paths
            if frame_idx < len(filenames):
                image_path = Path(filenames[frame_idx])
            else:
                return ""
        else:
            # HDF5Video or other - single filename
            # For h5 files, we'll need to extract frames separately
            return str(filenames)
    except Exception:
        return ""

    # Try to make path relative to HTML directory
    try:
        html_dir = html_path.parent.resolve()
        image_path_resolved = image_path.resolve()
        relative_path = image_path_resolved.relative_to(html_dir)
        return str(relative_path).replace("\\", "/")
    except (ValueError, OSError):
        # Can't make relative, return absolute
        return str(image_path).replace("\\", "/")


def get_skeleton_edges(skeleton: Any) -> List[List[int]]:
    """Extract edge indices from a skeleton.

    Args:
        skeleton: A sleap_io Skeleton object.

    Returns:
        List of [source_idx, dest_idx] pairs for each edge.
    """
    return [[int(i), int(j)] for i, j in skeleton.edge_inds]
