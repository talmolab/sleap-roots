"""Trait calculations that rely on tips."""


def get_tips(pts):
    """Return tips (last node) from each lateral root.
    Args:
        pts: Root landmarks as array of shape (instances, nodes, 2)

    Returns:
        Array of tips (instances, (x, y)).
        If there is no root, or the roots don't have tips an array of shape
        (instances, 2) of NaNs will be returned.
    """
    # Get the last point of each instance. Shape is (instances, 2)
    tip_pts = pts[:, -1]
    return tip_pts
