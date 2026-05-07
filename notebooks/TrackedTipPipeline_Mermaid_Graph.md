```mermaid
graph LR
    track_xy --> track_first_xy
    track_xy --> track_last_xy
    track_first_xy --> tip_displacement_net
    track_last_xy --> tip_displacement_net
    track_xy --> tip_trajectory_length
    n_frames_tracked --> tracking_coverage
    n_frames_total --> tracking_coverage
```

Distinct Traits for TrackedTipPipeline: 8
