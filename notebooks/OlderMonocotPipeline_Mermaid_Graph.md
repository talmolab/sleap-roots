```mermaid
graph LR
    crown_pts --> pts_all_array
    crown_pts --> crown_count
    crown_pts --> crown_proximal_node_inds
    crown_pts --> crown_distal_node_inds
    crown_pts --> crown_lengths
    crown_pts --> crown_base_pts
    crown_pts --> crown_tip_pts
    crown_pts --> scanline_intersection_counts
    crown_pts --> crown_angles_distal
    crown_distal_node_inds --> crown_angles_distal
    crown_pts --> crown_angles_proximal
    crown_proximal_node_inds --> crown_angles_proximal
    crown_pts --> network_length_lower
    bounding_box --> network_length_lower
    crown_pts --> ellipse
    crown_pts --> bounding_box
    crown_pts --> convex_hull
    crown_tip_pts --> crown_tip_xs
    crown_tip_pts --> crown_tip_ys
    network_length --> network_distribution_ratio
    network_length_lower --> network_distribution_ratio
    crown_lengths --> network_length
    crown_base_pts --> crown_base_tip_dists
    crown_tip_pts --> crown_base_tip_dists
    crown_lengths --> crown_curve_indices
    crown_base_tip_dists --> crown_curve_indices
    network_length --> network_solidity
    chull_area --> network_solidity
    ellipse --> ellipse_a
    ellipse --> ellipse_b
    bounding_box --> network_width_depth_ratio
    convex_hull --> chull_perimeter
    convex_hull --> chull_area
    convex_hull --> chull_max_width
    convex_hull --> chull_max_height
    convex_hull --> chull_line_lengths
    ellipse --> ellipse_ratio
    scanline_intersection_counts --> scanline_last_ind
    scanline_intersection_counts --> scanline_first_ind
    crown_pts --> crown_r1_pts
    crown_base_pts --> chull_r1_intersection_vectors
    crown_r1_pts --> chull_r1_intersection_vectors
    crown_pts --> chull_r1_intersection_vectors
    convex_hull --> chull_r1_intersection_vectors
    chull_r1_intersection_vectors --> chull_r1_left_intersection_vector
    chull_r1_intersection_vectors --> chull_r1_right_intersection_vector
    chull_r1_left_intersection_vector --> angle_chull_r1_left_intersection_vector
    chull_r1_right_intersection_vector --> angle_chull_r1_right_intersection_vector
    crown_r1_pts --> chull_areas_r1_intersection
    crown_pts --> chull_areas_r1_intersection
    convex_hull --> chull_areas_r1_intersection
    chull_areas_r1_intersection --> chull_area_above_r1_intersection
    chull_areas_r1_intersection --> chull_area_below_r1_intersection
```

Distinct Traits for OlderMonocotPipeline: 42
