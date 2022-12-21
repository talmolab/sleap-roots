"""Trait calculations that rely on bases (i.e., dicot-only)."""

import numpy as np


def get_bases(pts: np.ndarray) -> np.ndarray:
    """Return bases (r1) from each lateral root.

    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)

    Returns:
        Array of bases (instances, (x, y)).
        If there is no root, or the roots don't have bases, an empty array of shape
        (0,2) is returned.
    """
    # Check for edge cases where the root is missing or doesn't have bases.
    if len(pts) == 0 or np.isnan(pts[:, 0].all()):
        # Shape is (0, 2)
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
        If there is no root, or the roots is one point only (all of the rest of the
        points are NaNs), an array of NaNs with shape (len(pts),) is returned.
        This is also the case for non-contiguous points at the moment.
    """
    segment_diffs = np.diff(pts, axis=1)
    segment_lengths = np.linalg.norm(segment_diffs, axis=-1)

    if np.isnan(segment_lengths).all():
        total_lengths = np.empty((len(pts),)) ** np.nan
    else:
        total_lengths = np.nansum(segment_lengths, axis=-1)
    return total_lengths
