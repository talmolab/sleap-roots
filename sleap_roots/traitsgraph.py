"""Create traits graph and get destinations."""

import networkx as nx


def get_traits_graph():
    """Get traits graph using networkx.

    Args:
        None

    Return:
        Destination nodes.
    """
    G = nx.DiGraph()
    edge_list = [
        ("pts", "primary_pts"),
        ("pts", "lateral_pts"),
        ("primary_pts", "primary_base_pt"),
        ("primary_pts", "primary_angle_proximal"),
        ("primary_pts", "primary_angle_distal"),
        ("primary_pts", "primary_length"),
        ("primary_base_pt", "primary_base_pt_y"),
        ("primary_pts", "primary_tip_pt"),
        ("primary_base_pt_y", "primary_base_tip_dist"),
        ("primary_tip_pt_y", "primary_base_tip_dist"),
        # ("primary_tip_pt_y", "primary_depth"),
        ("lateral_pts", "lateral_count"),
        ("primary_tip_pt", "primary_tip_pt_y"),
        ("primary_length_max", "grav_index"),
        ("primary_base_tip_dist", "grav_index"),
        ("lateral_base_ys", "base_length"),
        # ("lateral_base_ys", "lateral_base_y_max"),
        # ("lateral_base_y_min", "base_length"),
        # ("lateral_base_y_max", "base_length"),
        # ("lateral_base_pts", "base_ct_density"),
        # ("primary_length", "base_ct_density"),
        # ("convex_hull", "chull_perimeter"),
        # ("convex_hull", "chull_area"),
        # ("convex_hull", "chull_max_width"),
        # ("convex_hull", "chull_max_height"),
        ("primary_pts", "ellipse"),
        ("chull_area", "network_solidity"),
        ("lateral_lengths", "network_solidity"),
        ("primary_length", "network_solidity"),
        ("lateral_pts", "bounding_box"),
        ("primary_pts", "bounding_box"),
        ("lateral_lengths", "network_distribution_ratio"),
        ("primary_length", "network_distribution_ratio"),
        ("primary_length", "network_length_lower"),
        ("lateral_lengths", "network_length_lower"),
        ("bounding_box", "network_distribution_ratio"),
        ("bounding_box", "network_length_lower"),
        # ("scanline_intersection_counts", "scanline_last_ind"),
        # ("scanline_intersection_counts", "scanline_first_ind"),
        ("lateral_pts", "lateral_angles_proximal"),
        ("lateral_pts", "lateral_angles_distal"),
        ("lateral_pts", "lateral_lengths"),
        ("primary_pts", "stem_widths"),
        ("lateral_pts", "lateral_base_pts"),
        ("lateral_base_pts", "stem_widths"),
        ("lateral_base_pts", "lateral_base_xs"),
        ("lateral_base_pts", "lateral_base_ys"),
        ("lateral_pts", "lateral_tip_pts"),
        ("lateral_tip_pts", "lateral_tip_xs"),
        ("lateral_tip_pts", "lateral_tip_ys"),
        ("primary_pts", "convex_hull"),
        ("lateral_pts", "convex_hull"),
        ("convex_hull", "chull_line_lengths"),
        ("primary_pts", "scanline_intersections"),
        ("lateral_pts", "scanline_intersections"),
        ("scanline_intersections", "scanline_intersection_counts"),
    ]

    G.add_edges_from(edge_list)
    dts = [dst for (src, dst) in list(nx.bfs_tree(G, "pts").edges())[2:]]
    return dts
