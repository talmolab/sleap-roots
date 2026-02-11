"""Rendering functions for SLEAP prediction visualization.

This module provides functions to render prediction overlays on images
using either root type coloring or confidence-based colormaps.
"""

import base64
import io
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt

from sleap_roots.series import Series, plot_img, plot_instance


def normalize_confidence(
    score: float,
    min_score: float,
    max_score: float,
) -> float:
    """Normalize a confidence score to 0-1 range.

    Args:
        score: The confidence score to normalize.
        min_score: Minimum score in the dataset.
        max_score: Maximum score in the dataset.

    Returns:
        Normalized score in 0-1 range.
    """
    if max_score == min_score:
        return 0.5  # Default to middle when all scores are the same
    return (score - min_score) / (max_score - min_score)


def get_confidence_range(
    series: Series,
    frame_idx: int,
) -> Tuple[float, float]:
    """Get the min and max confidence scores for a frame.

    Args:
        series: Series object containing predictions.
        frame_idx: Frame index.

    Returns:
        Tuple of (min_score, max_score).
    """
    frames = series.get_frame(frame_idx)
    scores = []

    for root_type in ["primary", "lateral", "crown"]:
        if root_type not in frames:
            continue
        labeled_frame = frames[root_type]
        if labeled_frame is None:
            continue

        for instance in labeled_frame.instances:
            if hasattr(instance, "score") and instance.score is not None:
                scores.append(float(instance.score))

    if not scores:
        return (0.0, 1.0)  # Default range if no scores

    return (min(scores), max(scores))


def render_frame_root_type(
    series: Series,
    frame_idx: int,
    scale: float = 1.0,
) -> matplotlib.figure.Figure:
    """Render a frame with root type colored overlay.

    Uses distinct colors for each root type (primary, lateral, crown).

    Args:
        series: Series object containing predictions and video.
        frame_idx: Frame index to render.
        scale: Relative size of the visualized image.

    Returns:
        matplotlib.figure.Figure with the rendered frame.

    Raises:
        IndexError: If frame_idx is out of range.
    """
    # Validate frame index
    if frame_idx < 0 or frame_idx >= len(series):
        raise IndexError(
            f"Frame index {frame_idx} out of range for series with {len(series)} frames"
        )

    # Use the existing Series.plot() method which handles root type coloring
    return series.plot(frame_idx, scale=scale)


def render_frame_confidence(
    series: Series,
    frame_idx: int,
    scale: float = 1.0,
    colormap: str = "viridis",
) -> matplotlib.figure.Figure:
    """Render a frame with confidence-based colormap overlay.

    Colors predictions based on their confidence scores using a continuous
    colormap.

    Args:
        series: Series object containing predictions and video.
        frame_idx: Frame index to render.
        scale: Relative size of the visualized image.
        colormap: Matplotlib colormap name for confidence visualization.

    Returns:
        matplotlib.figure.Figure with the rendered frame.

    Raises:
        IndexError: If frame_idx is out of range.
    """
    # Validate frame index
    if frame_idx < 0 or frame_idx >= len(series):
        raise IndexError(
            f"Frame index {frame_idx} out of range for series with {len(series)} frames"
        )

    # Check if video is available
    if series.video is None:
        raise ValueError("Video is not available. Specify the h5_path to load it.")

    # Get the image for this frame
    img = series.video[frame_idx]

    # Plot the base image
    fig = plot_img(img, scale=scale)

    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Get confidence range for normalization
    min_conf, max_conf = get_confidence_range(series, frame_idx)

    # Retrieve all available frames/predictions
    frames = series.get_frame(frame_idx)

    # Define the order of preference for the predictions
    prediction_order = ["primary", "lateral", "crown"]

    # Collect all instances with their confidence scores
    for prediction_type in prediction_order:
        if prediction_type not in frames:
            continue

        labeled_frame = frames[prediction_type]
        if labeled_frame is None:
            continue

        # Get skeleton for this prediction type
        skeleton = None
        if prediction_type == "primary" and series.primary_labels is not None:
            skeleton = series.primary_labels.skeletons[0]
        elif prediction_type == "lateral" and series.lateral_labels is not None:
            skeleton = series.lateral_labels.skeletons[0]
        elif prediction_type == "crown" and series.crown_labels is not None:
            skeleton = series.crown_labels.skeletons[0]

        # Plot each instance with confidence-based color
        for instance in labeled_frame.instances:
            # Get instance confidence score
            if hasattr(instance, "score") and instance.score is not None:
                confidence = float(instance.score)
            else:
                # Default if no confidence available
                confidence = (min_conf + max_conf) / 2

            # Normalize confidence to 0-1 for colormap
            normalized_conf = normalize_confidence(confidence, min_conf, max_conf)

            # Map normalized confidence to color
            color = cmap(normalized_conf)

            # Plot the instance with confidence color
            plot_instance(
                instance,
                skeleton=skeleton,
                cmap=[color] * 20,  # Use same color for all edges
                lw=2,
                ms=6,
                scale=scale,
            )

    return fig


def figure_to_base64(
    fig: matplotlib.figure.Figure,
    close: bool = False,
    format: str = "png",
    dpi: int = 72,
) -> str:
    """Convert a matplotlib figure to a base64-encoded string.

    Args:
        fig: Matplotlib figure to encode.
        close: Whether to close the figure after encoding.
        format: Image format (default: "png").
        dpi: Dots per inch for the output image.

    Returns:
        Base64-encoded string of the image.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()

    if close:
        plt.close(fig)

    return encoded
