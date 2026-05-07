"""TrackedTipPipeline for per-track tip-trajectory analysis (issue #129)."""

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

# Column orderings are tuples (not lists) so they cannot be mutated by
# downstream callers. Pandas does not copy the iterable passed to the
# `columns=` argument; mutation would silently leak across pipeline calls.
# Matches the convention `_VALID_ROOT_TYPES = ("primary", "lateral", "crown")`
# in sleap_roots/series.py.
_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "series",
    "sample_uid",
    "timepoint",
    "track_id",
    "n_frames_tracked",
    "n_frames_total",
    "tracking_coverage",
    "tip_trajectory_length",
    "tip_displacement_net",
)
_TRAJECTORY_COLUMNS: Tuple[str, ...] = (
    "series",
    "sample_uid",
    "timepoint",
    "track_id",
    "frame",
    "tip_x",
    "tip_y",
)


def _get_first_xy(xy: np.ndarray) -> np.ndarray:
    """Return the first row of an `(N, 2)` xy array.

    Used as a TraitDef function in `TrackedTipPipeline.define_traits`.
    Defined at module level (not as an inline lambda) so that the pipeline's
    trait list is picklable, which is a precondition for any future
    `multiprocessing`-based parallelization of the trait DAG.

    Args:
        xy: An `(N, 2)` array of frame-sorted `(tip_x, tip_y)` rows.

    Returns:
        The first row, shape `(2,)`.
    """
    return xy[0]


def _get_last_xy(xy: np.ndarray) -> np.ndarray:
    """Return the last row of an `(N, 2)` xy array.

    Module-level (not lambda) for picklability. See `_get_first_xy` for
    rationale.

    Args:
        xy: An `(N, 2)` array of frame-sorted `(tip_x, tip_y)` rows.

    Returns:
        The last row, shape `(2,)`.
    """
    return xy[-1]


def _tracking_coverage_fn(n_tracked: int, n_total: int) -> float:
    """Compute the fraction of frames in which a track has an instance.

    Args:
        n_tracked: Number of frames in which the track has an instance.
        n_total: Total number of frames in the series.

    Returns:
        `n_tracked / n_total` as a float in `[0, 1]`. Returns `np.nan` when
        `n_total == 0` (defensive — pipeline iteration handles empty input
        upstream so this branch is rarely reached).
    """
    if n_total == 0:
        return float("nan")
    return float(n_tracked) / float(n_total)


@attrs.define
class TrackedTipPipeline(Pipeline):
    """Pipeline emitting per-track tip-trajectory substrate from tracked .slp.

    Reuses existing trait functions directly via TraitDef DAG composition —
    no wrapper module. `tip_displacement_net` delegates to
    `sleap_roots.bases.get_base_tip_dist`; `tip_trajectory_length` delegates
    to `sleap_roots.lengths.get_root_lengths`. The DAG provides per-track
    input slicing (`track_first_xy = xy[0]`, `track_last_xy = xy[-1]`) so
    existing functions plug in unchanged.

    Iteration unit is per-track (not per-frame as in `DicotPipeline` et al.).
    `compute_tracked_tip_traits` calls `series.get_tracked_tips()` to obtain
    a long-format DataFrame sorted by `(track_id, frame)`, groups by
    `track_id`, and runs the DAG once per group with the per-track inputs.

    Attributes:
        traits: List of `TraitDef` objects (inherited from `Pipeline`,
            populated from `define_traits()`).
        trait_map: Dictionary mapping trait names to their definitions
            (inherited from `Pipeline`).
        trait_computation_order: List of trait names in topological order
            (inherited from `Pipeline`).

    Methods:
        define_traits: Return the per-track TraitDef DAG (5 trait nodes —
            `track_first_xy`, `track_last_xy`, `tip_displacement_net`,
            `tip_trajectory_length`, `tracking_coverage`).
        compute_tracked_tip_traits: Compute per-track traits for one Series
            and optionally write CSV/JSON outputs.
        compute_batch_tracked_tip_traits: Run `compute_tracked_tip_traits`
            across multiple Series and concatenate per-series outputs.
    """

    def define_traits(self) -> List[TraitDef]:
        """Return the per-track TraitDef DAG.

        Inputs (pre-populated per track when `compute_frame_traits` is
        called): `track_xy` (Nx2 frame-sorted), `n_frames_tracked`,
        `n_frames_total`.
        """
        return [
            # Per-track slicing — DAG provides endpoint extraction for the
            # existing distance function below.
            TraitDef(
                name="track_first_xy",
                fn=_get_first_xy,
                input_traits=["track_xy"],
                scalar=False,
                include_in_csv=False,
                description="First frame's tip (x, y) for this track.",
            ),
            TraitDef(
                name="track_last_xy",
                fn=_get_last_xy,
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
            series: `Series` whose tracked .slp will be processed.
            write_csv: When True, write the summary CSV and (unless
                `emit_trajectories=False`) the trajectory CSV to
                `output_dir`.
            write_json: When True, write the per-series JSON file.
            output_dir: Directory to write outputs to. Created if absent.
            emit_trajectories: When False, suppress writing the trajectory
                CSV and emit `trajectories=[]` in the in-memory and JSON
                outputs.
            csv_summary_suffix: Filename suffix for the summary CSV.
            csv_trajectory_suffix: Filename suffix for the trajectory CSV.
            json_suffix: Filename suffix for the JSON.

        Returns:
            A dict with keys `schema_version` (1), `pipeline`
            (`"TrackedTipPipeline"`), `units` (structured dict),
            `series` (str), `sample_uid` (str), `timepoint` (float or
            NaN), `tracks` (list of per-track dicts), `trajectories`
            (list of per-frame dicts; empty when `emit_trajectories=False`).
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
            # n_frames_tracked is the count of UNIQUE frames in which the
            # track has an instance — NOT len(group). Pathological tracker
            # output (over-eager merging) can produce duplicate
            # (track_id, frame) rows; counting instances would push
            # tracking_coverage above 1.0 and break the [0, 1] contract.
            # The trajectory CSV preserves duplicates for debugging; only
            # the per-track summary deduplicates.
            initial_traits = {
                "track_xy": track_xy,
                "n_frames_tracked": int(group["frame"].nunique()),
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

        Args:
            result: Per-series result dict from `compute_tracked_tip_traits`.
                Must contain `series`, `sample_uid`, `timepoint`, `tracks`,
                and `trajectories` keys.
            emit_trajectories: When True, build the trajectory DataFrame.
                When False, return `None` for the trajectory DataFrame.

        Returns:
            A tuple `(summary_df, trajectory_df)`. `summary_df` carries the
            per-track scalar columns with `series` / `sample_uid` /
            `timepoint` repeated on every row. `trajectory_df` carries the
            per-frame tip rows (or `None` when `emit_trajectories=False`).
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
        """Run `compute_tracked_tip_traits` across multiple Series; concatenate.

        Mirrors the existing `compute_batch_plate_traits` pattern in
        `trait_pipelines.py`: walks `all_series`, calls the per-series
        method, concatenates per-series DataFrames into one, collects
        per-series result dicts into a list.

        Args:
            all_series: Sequence of `Series` objects to process.
            write_csv: When True, write batch summary CSV and (unless
                `emit_trajectories=False`) batch trajectory CSV.
            write_json: When True, write a single JSON file containing a
                list of per-series result dicts.
            output_dir: Directory to write outputs to. Created if absent.
            csv_summary_name: Filename for the batch summary CSV.
            csv_trajectory_name: Filename for the batch trajectory CSV.
            json_name: Filename for the batch JSON.
            emit_trajectories: When False, suppress trajectory output across
                all series.

        Returns:
            `(batch_summary_df, batch_trajectory_df_or_None, per_series_results)`.
            `batch_trajectory_df_or_None` is `None` when
            `emit_trajectories` is False; otherwise a concatenated
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
