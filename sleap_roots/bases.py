"""Trait calculations that rely on bases (i.e., dicot-only)."""


def get_bases(pts):
    """Return bases (r1) from each lateral root.
    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)

    Returns:
        Array of bases (instances, (x, y)).
    """
    # (instances, 2)
    base_pts = pts[:, 0]
    return base_pts

