"""TrackedTipPipeline: per-track tip-trajectory substrate from tracked .slp predictions.

Issue #129 (Workstream 2 of the 2026-04-23 timelapse design). Consumes
SLEAP-tracked predictions and emits a minimum-viable substrate of per-track
geometric scalars plus the raw trajectory rows. **Scope is frozen** —
velocity / curvature / circumnutation traits NEVER belong here; they live in
separate downstream pipeline classes that REUSE this pipeline's
``Series.get_tracked_tips`` accessor and trajectory output as their input
substrate.

Lives in its own file (NOT appended to the 3763-line ``trait_pipelines.py``
megafile) — starts the per-pipeline-module pattern; the existing megafile
split is tracked in #189.

See:
- ``docs/superpowers/specs/2026-04-23-timelapse-diff-and-tip-kinematics-design.md``
  § Workstream 2 — design rationale and brainstorm decisions.
- ``openspec/changes/add-tracked-tip-pipeline/`` — formal contract.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import attrs
import numpy as np
import pandas as pd

from sleap_roots.bases import get_base_tip_dist
from sleap_roots.lengths import get_root_lengths
from sleap_roots.series import Series, validate_series_for_tracked_tip
from sleap_roots.trait_pipelines import (
    NumpyArrayEncoder,
    Pipeline,
    TraitDef,
    _json_sanitize,
)


logger = logging.getLogger(__name__)


_TRACKED_TIP_UNITS: Dict[str, str] = {
    "lengths": "pixels",
    "ratios": "dimensionless",
    "counts": "dimensionless",
    "time": "unspecified",
}

_SCHEMA_VERSION = 1

_SUMMARY_COLUMNS: List[str] = [
    "series",
    "sample_uid",
    "timepoint",
    "track_id",
    "n_frames_tracked",
    "n_frames_total",
    "tracking_coverage",
    "tip_trajectory_length",
    "tip_displacement_net",
]
_TRAJECTORY_COLUMNS: List[str] = [
    "series",
    "sample_uid",
    "timepoint",
    "track_id",
    "frame",
    "tip_x",
    "tip_y",
]


def _tracking_coverage_fn(n_tracked: int, n_total: int) -> float:
    """Fraction of frames in which a track has an instance.

    ``np.nan`` when ``n_total == 0`` (defensive — pipeline iteration handles
    empty input upstream so this is rarely reached).
    """
    if n_total == 0:
        return float("nan")
    return float(n_tracked) / float(n_total)


@attrs.define
class TrackedTipPipeline(Pipeline):
    """Pipeline emitting per-track tip-trajectory substrate from tracked .slp.

    Reuses existing trait functions DIRECTLY via TraitDef DAG composition —
    no wrapper module. ``tip_displacement_net`` delegates to
    ``sleap_roots.bases.get_base_tip_dist``; ``tip_trajectory_length``
    delegates to ``sleap_roots.lengths.get_root_lengths``. The DAG provides
    per-track input slicing (``track_first_xy = xy[0]``,
    ``track_last_xy = xy[-1]``) so existing functions plug in unchanged.

    Iteration unit is **per-track** (not per-frame as in DicotPipeline et
    al.). The pipeline calls ``series.get_tracked_tips()`` to obtain a
    long-format DataFrame sorted by ``(track_id, frame)``, groups by
    ``track_id``, and runs the DAG once per group with the per-track inputs.
    """

    def define_traits(self) -> List[TraitDef]:
        """Return the per-track TraitDef DAG.

        Inputs (pre-populated per track when ``compute_frame_traits`` is
        called): ``track_xy`` (Nx2 frame-sorted), ``n_frames_tracked``,
        ``n_frames_total``.
        """
        return [
            # Per-track slicing — DAG provides endpoint extraction for the
            # existing distance function below.
            TraitDef(
                name="track_first_xy",
                fn=lambda xy: xy[0],
                input_traits=["track_xy"],
                scalar=False,
                include_in_csv=False,
                description="First frame's tip (x, y) for this track.",
            ),
            TraitDef(
                name="track_last_xy",
                fn=lambda xy: xy[-1],
                input_traits=["track_xy"],
                scalar=False,
                include_in_csv=False,
                description="Last frame's tip (x, y) for this track.",
            ),
            # Trait scalars — existing trait functions used directly via DAG
            # composition.
            TraitDef(
                name="tip_displacement_net",
                fn=get_base_tip_dist,  # ← from bases.py, no wrapper
                input_traits=["track_first_xy", "track_last_xy"],
                scalar=True,
                include_in_csv=True,
                description=(
                    "Euclidean distance from first-tracked-frame tip to "
                    "last-tracked-frame tip (pixels)."
                ),
            ),
            TraitDef(
                name="tip_trajectory_length",
                fn=get_root_lengths,  # ← from lengths.py, no wrapper
                input_traits=["track_xy"],
                scalar=True,
                include_in_csv=True,
                description=(
                    "Cumulative arclength of the tip trajectory across all "
                    "tracked frames (pixels). NaN for single-frame tracks "
                    "(codebase NaN-on-empty-segments convention)."
                ),
            ),
            TraitDef(
                name="tracking_coverage",
                fn=_tracking_coverage_fn,
                input_traits=["n_frames_tracked", "n_frames_total"],
                scalar=True,
                include_in_csv=True,
                description=(
                    "n_frames_tracked / n_frames_total (dimensionless ratio "
                    "in [0, 1])."
                ),
            ),
        ]

    def compute_tracked_tip_traits(
        self,
        series: Series,
        *,
        write_csv: bool = False,
        write_json: bool = False,
        output_dir: str = ".",
        emit_trajectories: bool = True,
        csv_summary_suffix: str = ".tracked_tip_traits.csv",
        csv_trajectory_suffix: str = ".tracked_tip_trajectories.csv",
        json_suffix: str = ".tracked_tip_traits.json",
    ) -> Dict[str, Any]:
        """Compute per-track tip-trajectory traits for one Series.

        Args:
            series: ``Series`` whose tracked .slp will be processed.
            write_csv: When True, write the summary CSV and (unless
                ``emit_trajectories=False``) the trajectory CSV to
                ``output_dir``.
            write_json: When True, write the per-series JSON file.
            output_dir: Directory to write outputs to. Created if absent.
            emit_trajectories: When False, suppress writing the trajectory
                CSV and emit ``trajectories=[]`` in the in-memory and JSON
                outputs.
            csv_summary_suffix: Filename suffix for the summary CSV.
            csv_trajectory_suffix: Filename suffix for the trajectory CSV.
            json_suffix: Filename suffix for the JSON.

        Returns:
            A dict with keys ``schema_version`` (1), ``pipeline``
            (``"TrackedTipPipeline"``), ``units`` (structured dict),
            ``series`` (str), ``sample_uid`` (str), ``timepoint`` (float or
            NaN), ``tracks`` (list of per-track dicts), ``trajectories``
            (list of per-frame dicts; empty when ``emit_trajectories=False``).
        """
        # Validate input early — raises ValueError on untracked instances or
        # zero/multiple paths populated without root_type.
        validate_series_for_tracked_tip(series)

        # Long-format trajectory rows (sorted by (track_id, frame)).
        df = series.get_tracked_tips()
        n_frames_total = len(series)

        result: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "pipeline": "TrackedTipPipeline",
            "units": dict(_TRACKED_TIP_UNITS),
            "series": series.series_name,
            "sample_uid": series.sample_uid,
            "timepoint": series.timepoint,
            "tracks": [],
            "trajectories": [],
        }

        # Per-track DAG iteration. Empty df → no iterations, empty arrays.
        for track_id, group in df.groupby("track_id"):
            track_xy = group[["tip_x", "tip_y"]].to_numpy(dtype=float)
            initial_traits = {
                "track_xy": track_xy,
                "n_frames_tracked": len(group),
                "n_frames_total": n_frames_total,
            }
            computed = self.compute_frame_traits(initial_traits)
            result["tracks"].append(
                {
                    "track_id": str(track_id),
                    "n_frames_tracked": int(initial_traits["n_frames_tracked"]),
                    "n_frames_total": int(n_frames_total),
                    "tracking_coverage": float(computed["tracking_coverage"]),
                    "tip_trajectory_length": float(computed["tip_trajectory_length"]),
                    "tip_displacement_net": float(computed["tip_displacement_net"]),
                }
            )

        if emit_trajectories:
            for _, row in df.iterrows():
                result["trajectories"].append(
                    {
                        "track_id": str(row["track_id"]),
                        "frame": int(row["frame"]),
                        "tip_x": float(row["tip_x"]),
                        "tip_y": float(row["tip_y"]),
                    }
                )

        if write_csv or write_json:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if write_csv:
            summary_df, trajectory_df = self._build_dataframes(
                result, emit_trajectories=emit_trajectories
            )
            summary_path = (
                Path(output_dir) / f"{series.series_name}{csv_summary_suffix}"
            )
            summary_df.to_csv(summary_path.as_posix(), index=False)
            if emit_trajectories and trajectory_df is not None:
                trajectory_path = (
                    Path(output_dir) / f"{series.series_name}{csv_trajectory_suffix}"
                )
                trajectory_df.to_csv(trajectory_path.as_posix(), index=False)

        if write_json:
            json_path = Path(output_dir) / f"{series.series_name}{json_suffix}"
            sanitized = _json_sanitize(result)
            with open(json_path.as_posix(), "w") as f:
                json.dump(
                    sanitized,
                    f,
                    cls=NumpyArrayEncoder,
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=4,
                )

        return result

    def _build_dataframes(
        self,
        result: Dict[str, Any],
        *,
        emit_trajectories: bool,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Build per-series summary + trajectory DataFrames for CSV emission.

        Returns ``(summary_df, trajectory_df)``. ``trajectory_df`` is ``None``
        when ``emit_trajectories`` is False.
        """
        # Repeat top-level scalars on every row of both DataFrames.
        series_name = result["series"]
        sample_uid = result["sample_uid"]
        timepoint = result["timepoint"]

        summary_rows = [
            {
                "series": series_name,
                "sample_uid": sample_uid,
                "timepoint": timepoint,
                **row,
            }
            for row in result["tracks"]
        ]
        summary_df = pd.DataFrame(summary_rows, columns=_SUMMARY_COLUMNS)

        if not emit_trajectories:
            return summary_df, None

        trajectory_rows = [
            {
                "series": series_name,
                "sample_uid": sample_uid,
                "timepoint": timepoint,
                **row,
            }
            for row in result["trajectories"]
        ]
        trajectory_df = pd.DataFrame(trajectory_rows, columns=_TRAJECTORY_COLUMNS)
        return summary_df, trajectory_df

    def compute_batch_tracked_tip_traits(
        self,
        all_series: List[Series],
        *,
        write_csv: bool = False,
        write_json: bool = False,
        output_dir: str = ".",
        csv_summary_name: str = "tracked_tip_batch_traits.csv",
        csv_trajectory_name: str = "tracked_tip_batch_trajectories.csv",
        json_name: str = "tracked_tip_batch_traits.json",
        emit_trajectories: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[Dict[str, Any]]]:
        """Run ``compute_tracked_tip_traits`` across multiple Series; concatenate.

        Mirrors the existing ``compute_batch_plate_traits`` pattern in
        ``trait_pipelines.py``: walks ``all_series``, calls the per-series
        method, concatenates per-series DataFrames into one, collects
        per-series result dicts into a list.

        Args:
            all_series: Sequence of ``Series`` objects to process.
            write_csv: When True, write batch summary CSV and (unless
                ``emit_trajectories=False``) batch trajectory CSV.
            write_json: When True, write a single JSON file containing a
                list of per-series result dicts.
            output_dir: Directory to write outputs to. Created if absent.
            csv_summary_name: Filename for the batch summary CSV.
            csv_trajectory_name: Filename for the batch trajectory CSV.
            json_name: Filename for the batch JSON.
            emit_trajectories: When False, suppress trajectory output across
                all series.

        Returns:
            ``(batch_summary_df, batch_trajectory_df_or_None, per_series_results)``.
            ``batch_trajectory_df_or_None`` is ``None`` when
            ``emit_trajectories`` is False; otherwise a concatenated
            DataFrame across all series.
        """
        per_series_results: List[Dict[str, Any]] = []
        per_series_summary_dfs: List[pd.DataFrame] = []
        per_series_trajectory_dfs: List[pd.DataFrame] = []
        for series in all_series:
            result = self.compute_tracked_tip_traits(
                series, emit_trajectories=emit_trajectories
            )
            per_series_results.append(result)
            summary_df, trajectory_df = self._build_dataframes(
                result, emit_trajectories=emit_trajectories
            )
            per_series_summary_dfs.append(summary_df)
            if emit_trajectories and trajectory_df is not None:
                per_series_trajectory_dfs.append(trajectory_df)

        if per_series_summary_dfs:
            batch_summary_df = pd.concat(
                per_series_summary_dfs, axis=0, ignore_index=True
            )
        else:
            batch_summary_df = pd.DataFrame(columns=_SUMMARY_COLUMNS)

        if emit_trajectories:
            if per_series_trajectory_dfs:
                batch_trajectory_df = pd.concat(
                    per_series_trajectory_dfs, axis=0, ignore_index=True
                )
            else:
                batch_trajectory_df = pd.DataFrame(columns=_TRAJECTORY_COLUMNS)
        else:
            batch_trajectory_df = None

        if write_csv or write_json:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        if write_csv:
            batch_summary_df.to_csv(
                (Path(output_dir) / csv_summary_name).as_posix(), index=False
            )
            if emit_trajectories and batch_trajectory_df is not None:
                batch_trajectory_df.to_csv(
                    (Path(output_dir) / csv_trajectory_name).as_posix(), index=False
                )

        if write_json:
            json_path = Path(output_dir) / json_name
            sanitized = _json_sanitize(per_series_results)
            with open(json_path.as_posix(), "w") as f:
                json.dump(
                    sanitized,
                    f,
                    cls=NumpyArrayEncoder,
                    allow_nan=False,
                    ensure_ascii=False,
                    indent=4,
                )

        return batch_summary_df, batch_trajectory_df, per_series_results
