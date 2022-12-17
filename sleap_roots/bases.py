"""Trait calculations that rely on bases (i.e., dicot-only)."""

import numpy as np


def get_bases(pts: np.ndarray) -> np.ndarray:
    """Return bases (r1) from each lateral root.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)

    Returns:
        Array of bases (instances, (x, y)).
    """
    # Exceptions
    if len(pts) == 0 or np.isnan(pts[:, 0].all()):
        # (instances, 2)
        base_pts = np.empty((0, 2))

    else:
        # (instances, 2)
        base_pts = pts[:, 0]
        base_pts = base_pts[~np.isnan(base_pts[:, 0])]
    return base_pts


def get_root_lengths(pts: np.ndarray) -> np.ndarray:
    """Return root lengths for all roots in a frame.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2).

    Returns:
        Array of root lengths of shape (instances,).
    """
    segment_diffs = np.diff(pts, axis=1)
    segment_lengths = np.linalg.norm(segment_diffs, axis=-1)
    total_lengths = np.nansum(segment_lengths, axis=-1)
    return total_lengths
