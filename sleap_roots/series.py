"""Series-level data loader."""

import attrs
import logging
import math
import numpy as np
import sleap_io as sio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Any, Dict, Optional, Tuple, List, Union
from pathlib import Path


logger = logging.getLogger(__name__)


@attrs.define
class Series:
    """Data and predictions for a single image series.

    Attributes:
        series_name: Unique identifier for the series.
        h5_path: Optional path to the HDF5-formatted image series.
        primary_path: Optional path to the primary root predictions file. At least one
            of the primary, lateral, or crown paths must be provided.
        lateral_path: Optional path to the lateral root predictions file. At least one
            of the primary, lateral, or crown paths must be provided.
        crown_path: Optional path to the crown predictions file. At least one of the
            primary, lateral, or crown paths must be provided.
        primary_labels: Optional `sio.Labels` corresponding to the primary root predictions.
        lateral_labels: Optional `sio.Labels` corresponding to the lateral root predictions.
        crown_labels: Optional `sio.Labels` corresponding to the crown predictions.
        video: Optional `sio.Video` corresponding to the image series.
        csv_path: Optional path to a metadata CSV. The CSV is keyed on a
            `plant_qr_code` column (matched against `sample_uid`). Any column
            in the CSV is readable via `Series.get_metadata(column)`. The
            `expected_count`, `group`, `qc_fail`, and `timepoint` properties
            are thin wrappers that read specific well-known columns; users
            can add arbitrary additional columns and read them via
            `get_metadata`. See `sleap_roots.metadata.build_metadata_csv` for
            a helper that emits a CSV with the canonical column ordering.
        sample_uid: Optional cross-scan stable identity for this series. Defaults to
            `series_name` when unset or empty. Coerced to `str` so CSV `plant_qr_code`
            lookups have predictable equality semantics.

    Methods:
        load: Load a set of predictions for this series.
        get_metadata: Generic CSV-column accessor keyed on `sample_uid` (and optional
            `plant_id`).
        __len__: Length of the series (number of images).
        __getitem__: Return labeled frames for predictions.
        __iter__: Iterator for looping through predictions.
        get_frame: Return labeled frames for predictions.
        plot: Plot predictions on top of the image.
        get_primary_points: Get primary root points.
        get_lateral_points: Get lateral root points.
        get_crown_points: Get crown root points.

    Properties:
        expected_count: Fetch the expected plant count for this series from the CSV.
        group: Group name for the series from the CSV.
        qc_fail: Flag to indicate if the series failed QC from the CSV.
        timepoint: Numeric time-axis value for this series, coerced to float.
    """

    series_name: str
    h5_path: Optional[str] = None
    primary_path: Optional[str] = None
    lateral_path: Optional[str] = None
    crown_path: Optional[str] = None
    primary_labels: Optional[sio.Labels] = None
    lateral_labels: Optional[sio.Labels] = None
    crown_labels: Optional[sio.Labels] = None
    video: Optional[sio.Video] = None
    csv_path: Optional[str] = None
    sample_uid: Optional[str] = None
    _warned_missing_plant_id_column: bool = attrs.field(
        default=False, init=False, repr=False
    )

    def __attrs_post_init__(self) -> None:
        """Default `sample_uid` to `series_name` and coerce to `str`."""
        if self.sample_uid is None or self.sample_uid == "":
            self.sample_uid = self.series_name
        self.sample_uid = str(self.sample_uid)

    @classmethod
    def load(
        cls,
        series_name: str,
        h5_path: Optional[str] = None,
        primary_path: Optional[str] = None,
        lateral_path: Optional[str] = None,
        crown_path: Optional[str] = None,
        csv_path: Optional[str] = None,
        sample_uid: Optional[str] = None,
    ) -> "Series":
        """Load a set of predictions for this series.

        Args:
            series_name: Unique identifier for the series.
            h5_path: Optional path to the HDF5-formatted image series, which will be
                used to load the video.
            primary_path: Optional path to the primary root '.slp' predictions file.
            lateral_path: Optional path to the lateral root '.slp' predictions file.
            crown_path: Optional path to the crown '.slp' predictions file.
            csv_path: Optional path to a metadata CSV (keyed on `plant_qr_code`,
                matched against `sample_uid`). Read via `get_metadata(column)`
                or the wrapper properties (`expected_count`, `group`, `qc_fail`,
                `timepoint`).
            sample_uid: Optional cross-scan stable identity. Defaults to `series_name`
                when unset or empty. Used as the CSV `plant_qr_code` lookup key by
                `get_metadata` and the `expected_count`/`group`/`qc_fail` properties.

        Returns:
            An instance of Series loaded with the specified predictions.
        """
        # Initialize the labels as None
        primary_labels, lateral_labels, crown_labels = None, None, None

        # Attempt to load the predictions, with error handling
        try:
            if primary_path:
                # Make path object
                primary_path = Path(primary_path)
                # Check if the file exists
                if primary_path.exists():
                    # Make the primary_path POSIX-compliant
                    primary_path = primary_path.as_posix()
                    # Load the primary predictions
                    primary_labels = sio.load_slp(primary_path)
                else:
                    print(f"Primary prediction file not found: {primary_path}")
            if lateral_path:
                # Make path object
                lateral_path = Path(lateral_path)
                # Check if the file exists
                if lateral_path.exists():
                    # Make the lateral_path POSIX-compliant
                    lateral_path = lateral_path.as_posix()
                    # Load the lateral predictions
                    lateral_labels = sio.load_slp(lateral_path)
                else:
                    print(f"Lateral prediction file not found: {lateral_path}")
            if crown_path:
                # Make path object
                crown_path = Path(crown_path)
                # Check if the file exists
                if crown_path.exists():
                    # Make the crown_path POSIX-compliant
                    crown_path = crown_path.as_posix()
                    # Load the crown predictions
                    crown_labels = sio.load_slp(crown_path)
                else:
                    print(f"Crown prediction file not found: {crown_path}")
        except Exception as e:
            print(f"Error loading prediction files: {e}")

        # Attempt to load the video, with error handling
        video = None
        try:
            if h5_path:
                # Make path object
                h5_path = Path(h5_path)
                # Check if the file exists
                if h5_path.exists():
                    # Make the h5_path POSIX-compliant
                    h5_path = h5_path.as_posix()
                    # Load the video
                    video = sio.Video.from_filename(h5_path)
                    # Replace the filename in the labels with the h5_path
                    for labels in [primary_labels, lateral_labels, crown_labels]:
                        if labels is not None:
                            labels.video.replace_filename(h5_path)
                else:
                    print(f"Video file not found: {h5_path}")
        except Exception as e:
            print(f"Error loading video file {h5_path}: {e}")

        # Make the csv path POSIX-compliant
        if csv_path:
            csv_path = Path(csv_path).as_posix()

        return cls(
            series_name=series_name,
            h5_path=h5_path,
            primary_path=primary_path,
            lateral_path=lateral_path,
            crown_path=crown_path,
            primary_labels=primary_labels,
            lateral_labels=lateral_labels,
            crown_labels=crown_labels,
            video=video,
            csv_path=csv_path,
            sample_uid=sample_uid,
        )

    def get_metadata(self, column: str, plant_id: Optional[int] = None) -> Any:
        """Fetch a CSV column value keyed on `sample_uid` (and optional `plant_id`).

        Looks up `df[df["plant_qr_code"] == self.sample_uid]`. If `plant_id` is given
        AND the CSV has a `plant_id` column, the lookup is the composite
        (sample_uid, plant_id) match. If `plant_id` is given but the CSV has no
        `plant_id` column, the argument is silently ignored, the sample-uid-only
        lookup is used, and a one-shot WARNING is emitted (per Series instance).

        Args:
            column: The CSV column name to fetch.
            plant_id: Optional per-plant disambiguator. Used only when the CSV has a
                `plant_id` column.

        Returns:
            The value in the first matching row's `column` field. Returns `np.nan`
            when the CSV is missing, the requested `column` is missing, the
            `plant_qr_code` lookup-key column is missing (with WARNING logged),
            or no row matches.
        """
        # Reject bool-typed plant_id — Python booleans are ints (False == 0,
        # True == 1) and pandas equality matches them against integer plant_id
        # columns. A caller passing plant_id=False (e.g. from a CLI arg parser
        # or a `flag or False` ternary) would silently fetch the row for
        # plant_id=0. Fail loudly instead of producing wrong rows.
        if isinstance(plant_id, bool):
            raise TypeError(
                f"Series.get_metadata: plant_id must be int or None, not "
                f"bool ({plant_id!r}). Booleans match int 0/1 in pandas "
                f"equality and would silently fetch the wrong row."
            )
        if not self.csv_path or not Path(self.csv_path).exists():
            return np.nan
        df = pd.read_csv(self.csv_path)
        if column not in df.columns:
            return np.nan
        if "plant_qr_code" not in df.columns:
            # Lookup key absent — preserve fail-soft contract instead of
            # raising KeyError. Surface the misconfiguration via WARNING so
            # users notice they wrote a CSV without the required column.
            logger.warning(
                "Series '%s': metadata CSV %s has no 'plant_qr_code' column; "
                "cannot perform sample_uid lookup. Returning NaN. The CSV "
                "MUST contain a 'plant_qr_code' column to be useful (see "
                "build_metadata_csv).",
                self.series_name,
                self.csv_path,
            )
            return np.nan
        sample_match = df[df["plant_qr_code"] == self.sample_uid]
        if plant_id is not None and "plant_id" in df.columns:
            sample_match = sample_match[sample_match["plant_id"] == plant_id]
        elif plant_id is not None and not self._warned_missing_plant_id_column:
            logger.warning(
                "Series '%s': plant_id=%r ignored because CSV %s has no "
                "'plant_id' column; falling back to sample_uid-only lookup. "
                "Add a 'plant_id' column to disambiguate per-plant rows.",
                self.series_name,
                plant_id,
                self.csv_path,
            )
            self._warned_missing_plant_id_column = True
        if len(sample_match) == 0:
            return np.nan
        return sample_match[column].iloc[0]

    @property
    def expected_count(self) -> Union[float, int]:
        """Fetch the expected plant count for this series from the CSV."""
        if not self.csv_path or not Path(self.csv_path).exists():
            print("CSV path is not set or the file does not exist.")
            return np.nan
        value = self.get_metadata("number_of_plants_cylinder")
        if pd.isna(value):
            print(f"No expected count found for series {self.series_name} in CSV.")
            return np.nan
        return value

    @property
    def group(self) -> str:
        """Group name for the series from the CSV."""
        if not self.csv_path or not Path(self.csv_path).exists():
            print("CSV path is not set or the file does not exist.")
            return np.nan
        value = self.get_metadata("genotype")
        if pd.isna(value):
            print(f"No group found for series {self.series_name} in CSV.")
            return np.nan
        return value

    @property
    def qc_fail(self) -> Union[int, float]:
        """Flag to indicate if the series failed QC from the CSV."""
        if not self.csv_path or not Path(self.csv_path).exists():
            print("CSV path is not set or the file does not exist.")
            return np.nan
        value = self.get_metadata("qc_cylinder")
        if pd.isna(value):
            print(f"No QC flag found for series {self.series_name} in CSV.")
            return np.nan
        return value

    @property
    def timepoint(self) -> float:
        """Numeric time-axis value for this series, coerced to float.

        Returns `np.nan` when the CSV is absent, the `timepoint` column is missing,
        or no row matches `sample_uid`. Raises `ValueError` when a matching row
        contains a non-numeric string value (e.g. a date) OR a non-finite value
        (`inf`, `-inf`) — failing loudly at the metadata layer beats silently
        producing nonsensical results in downstream timepoint arithmetic.
        """
        value = self.get_metadata("timepoint")
        if pd.isna(value):
            return np.nan
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Series '{self.series_name}': could not coerce 'timepoint' "
                f"value {value!r} to float ({exc})."
            ) from exc
        if not math.isfinite(result):
            raise ValueError(
                f"Series '{self.series_name}': non-finite 'timepoint' value "
                f"{value!r}. Timepoints must be finite floats; downstream "
                f"arithmetic on inf would silently produce nonsensical deltas."
            )
        return result

    def __len__(self) -> int:
        """Length of the series (number of images)."""
        if self.video is not None:
            return len(self.video)
        else:
            # Check all labels if video is None
            for labels in [self.primary_labels, self.lateral_labels, self.crown_labels]:
                if labels is not None:
                    return len(labels)
            # If all labels are None, return 0
            return 0

    def __getitem__(self, idx: int) -> Dict[str, Optional[sio.LabeledFrame]]:
        """Return labeled frames for primary and/or lateral and/or crown predictions."""
        return self.get_frame(idx)

    def __iter__(self):
        """Iterator for looping through predictions."""
        for i in range(len(self)):
            yield self[i]

    def get_frame(self, frame_idx: int) -> dict:
        """Return labeled frames for primary, lateral, and crown predictions.

        Args:
            frame_idx: Integer frame number.

        Returns:
            Dictionary with keys 'primary', 'lateral', and 'crown', each corresponding
            to the `sio.LabeledFrame` from each set of predictions on the same frame. If
            any set of predictions is not available, its value will be None.
        """
        frames = {}

        # For primary predictions
        if self.primary_labels is not None:
            frames["primary"] = self.primary_labels.find(
                self.primary_labels.video, frame_idx, return_new=True
            )[0]
        else:
            frames["primary"] = None

        # For lateral predictions
        if self.lateral_labels is not None:
            frames["lateral"] = self.lateral_labels.find(
                self.lateral_labels.video, frame_idx, return_new=True
            )[0]
        else:
            frames["lateral"] = None

        # For crown predictions
        if self.crown_labels is not None:
            frames["crown"] = self.crown_labels.find(
                self.crown_labels.video, frame_idx, return_new=True
            )[0]
        else:
            frames["crown"] = None

        return frames

    def plot(
        self, frame_idx: int, scale: float = 1.0, **kwargs
    ) -> matplotlib.figure.Figure:
        """Plot predictions on top of the image.

        Args:
            frame_idx: Frame index to visualize.
            scale: Relative size of the visualized image. Useful for plotting smaller
                images within notebooks.

        Returns:
            matplotlib.figure.Figure object that shows predictions on top of images.
        """
        # Check if the video is available
        if self.video is None:
            raise ValueError("Video is not available. Specify the h5_path to load it.")

        # Retrieve all available frames
        frames = self.get_frame(frame_idx)

        # Generate the color palette from seaborn
        cmap = sns.color_palette("tab10")

        # Define the order of preference for the predictions for plotting the image
        prediction_order = ["primary", "lateral", "crown"]

        # Variable to keep track if the image has been plotted
        image_plotted = False

        # First, find the first available prediction to plot the image
        for prediction in prediction_order:
            labeled_frame = frames.get(prediction)
            if labeled_frame is not None and not image_plotted:
                # Plot the image
                plot_img(labeled_frame.image, scale=scale)
                # Set the flag to True to avoid plotting the image again
                image_plotted = True

        # Then, iterate through all predictions to plot instances
        for i, prediction in enumerate(prediction_order):
            labeled_frame = frames.get(prediction)
            if labeled_frame is not None:
                # Use the color map index for each prediction type
                # Modulo the length of the color map to avoid index out of range
                color = cmap[i % len(cmap)]

                # Plot the instances
                plot_instances(labeled_frame.instances, cmap=[color], **kwargs)

        # Capture the current plot after calling plot_img and plot_instances.
        fig = plt.gcf()

        # Close the captured fig to avoid duplicate rendering from Jupyter.
        plt.close(fig)

        # Return the figure. In a cell, Jupyter will automatically render the plot.
        return fig

    def get_primary_points(self, frame_idx: int) -> np.ndarray:
        """Get primary root points.

        Args:
            frame_idx: Frame index.

        Returns:
            Primary root points as array of shape `(n_instances, n_nodes, 2)`.
        """
        # Check that self.primary_labels is not None
        if self.primary_labels is None:
            raise ValueError("Primary labels are not available.")
        # Retrieve all available frames
        frames = self.get_frame(frame_idx)
        # Get the primary labeled frame
        primary_lf = frames.get("primary")
        # Get the ground truth instances and unused predictions
        gt_instances_pr = primary_lf.user_instances + primary_lf.unused_predictions
        # If there are no instances, return an empty array
        if len(gt_instances_pr) == 0:
            primary_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        # Otherwise, stack the instances into an array
        else:
            primary_pts = np.stack([inst.numpy() for inst in gt_instances_pr], axis=0)
        return primary_pts

    def get_lateral_points(self, frame_idx: int) -> np.ndarray:
        """Get lateral root points.

        Args:
            frame_idx: Frame index.

        Returns:
            Lateral root points as array of shape `(n_instances, n_nodes, 2)`.
        """
        # Check that self.lateral_labels is not None
        if self.lateral_labels is None:
            raise ValueError("Lateral labels are not available.")
        # Retrieve all available frames
        frames = self.get_frame(frame_idx)
        # Get the lateral labeled frame
        lateral_lf = frames.get("lateral")
        # Get the ground truth instances and unused predictions
        gt_instances_lr = lateral_lf.user_instances + lateral_lf.unused_predictions
        # If there are no instances, return an empty array
        if len(gt_instances_lr) == 0:
            lateral_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        # Otherwise, stack the instances into an array
        else:
            lateral_pts = np.stack([inst.numpy() for inst in gt_instances_lr], axis=0)
        return lateral_pts

    def get_crown_points(self, frame_idx: int) -> np.ndarray:
        """Get crown root points.

        Args:
            frame_idx: Frame index.

        Returns:
            Crown root points as array of shape `(n_instances, n_nodes, 2)`.
        """
        # Check that self.crown_labels is not None
        if self.crown_labels is None:
            raise ValueError("Crown labels are not available.")
        # Retrieve all available frames
        frames = self.get_frame(frame_idx)
        # Get the crown labeled frame
        crown_lf = frames.get("crown")
        # Get the ground truth instances and unused predictions
        gt_instances_cr = crown_lf.user_instances + crown_lf.unused_predictions
        # If there are no instances, return an empty array
        if len(gt_instances_cr) == 0:
            crown_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        # Otherwise, stack the instances into an array
        else:
            crown_pts = np.stack([inst.numpy() for inst in gt_instances_cr], axis=0)
        return crown_pts

    def get_tracked_tips(
        self,
        root_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return per-frame tracked-tip rows in long format.

        Args:
            root_type: One of `"primary"`, `"lateral"`, `"crown"`. When
                `None` (default), auto-detects from whichever single
                `<root_type>_path` is populated. Raises `ValueError` when
                zero or more than one path is populated and `root_type` is
                `None`.

        Returns:
            A `pandas.DataFrame` with columns `["track_id", "frame",
            "tip_x", "tip_y"]` in that order, sorted by `(track_id, frame)`
            with a clean range index. One row per `(track_id, frame)` where
            the track has an instance.

            The tip coordinate is the LAST node of the skeleton
            (`inst.numpy()[-1]`), matching the convention used by
            `sleap_roots.tips.get_tips`. This works for both single-node
            skeletons (e.g. `["r0"]`) and multi-node skeletons (e.g.
            `["base", "mid", "tip"]`).

        Raises:
            ValueError: When `root_type` is `None` and zero or more than
                one of `primary_path` / `lateral_path` / `crown_path`
                is populated. Also raised when any instance has
                `inst.track is None` or empty `inst.track.name` — the
                pipeline requires SLEAP-tracked predictions.
        """
        # Resolve root_type via the shared helper (validates explicit
        # values + auto-detects when None).
        path_attrs = {
            "primary": self.primary_path,
            "lateral": self.lateral_path,
            "crown": self.crown_path,
        }
        root_type = _resolve_root_type(path_attrs, root_type)
        # Get the labels object for the resolved root type.
        labels_attr = f"{root_type}_labels"
        labels = getattr(self, labels_attr, None)
        if labels is None:
            raise ValueError(
                f"{labels_attr} is not available — was {root_type}_path set "
                f"on Series.load?"
            )

        # Iterate per-instance — tracker output does NOT preserve positional
        # ordering across frames (verified during the #129 brainstorm: frame 0
        # may be [track_0, 1, 2, 3, 4, 5] but frame 1 may be
        # [track_0, 3, 4, 2, 1, 5]). We must read inst.track.name per instance.
        rows = []
        offending_frames = []
        for lf in labels.labeled_frames:
            for inst in lf.instances:
                track = inst.track
                if track is None or not getattr(track, "name", None):
                    offending_frames.append(lf.frame_idx)
                    continue
                pts = inst.numpy()
                tip_xy = pts[-1]  # last node = tip by skeleton convention
                rows.append(
                    {
                        "track_id": track.name,
                        "frame": lf.frame_idx,
                        "tip_x": float(tip_xy[0]),
                        "tip_y": float(tip_xy[1]),
                    }
                )

        if offending_frames:
            unique_frames = sorted(set(offending_frames))
            raise ValueError(
                f"TrackedTipPipeline requires tracked .slp predictions; "
                f"untracked instances found at frame indices {unique_frames}. "
                f"See https://sleap.ai/tutorials/tracking.html for tracking "
                f"setup."
            )

        df = pd.DataFrame(rows, columns=["track_id", "frame", "tip_x", "tip_y"])
        df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
        return df


def find_all_h5_paths(data_folders: Union[str, List[str]]) -> List[str]:
    """Find all .h5 paths from a list of folders.

    Args:
        data_folders: Path or list of paths to folders containing .h5 paths.

    Returns:
        A list of filenames to .h5 paths.
    """
    if type(data_folders) != list:
        data_folders = [data_folders]

    h5_paths = []
    for data_folder in data_folders:
        h5_paths.extend([Path(p).as_posix() for p in Path(data_folder).glob("*.h5")])
    return h5_paths


def find_all_slp_paths(data_folders: Union[str, List[str]]) -> List[str]:
    """Find all .slp paths from a list of folders.

    Args:
        data_folders: Path or list of paths to folders containing .slp paths.

    Returns:
        A list of filenames to .slp paths.
    """
    if type(data_folders) != list:
        data_folders = [data_folders]

    slp_paths = []
    for data_folder in data_folders:
        slp_paths.extend([Path(p).as_posix() for p in Path(data_folder).glob("*.slp")])
    return slp_paths


def load_series_from_h5s(
    h5_paths: List[str], model_id: str, csv_path: Optional[str] = None
) -> List[Series]:
    """Load a list of Series from a list of .h5 paths.

    To load the `Series`, the files must be named with the following convention:
    h5_path: '/path/to/scan/series_name.h5'
    primary_path: '/path/to/scan/series_name.model{model_id}.rootprimary.slp'
    lateral_path: '/path/to/scan/series_name.model{model_id}.rootlateral.slp'
    crown_path: '/path/to/scan/series_name.model{model_id}.rootcrown.slp'

    Our pipeline outputs prediction files with this format:
    /<output_folder>/scan{scan_id}.model{model_id}.root{model_type}.slp

    Args:
        h5_paths: List of paths to .h5 files.
        csv_path: Optional path to the CSV file containing the expected plant count.

    Returns:
        A list of Series loaded with the specified .h5 files.
    """
    series_list = []
    for h5_path in h5_paths:
        # Extract the series name from the h5 path
        series_name = Path(h5_path).name.split(".")[0]
        # Generate the paths for the primary, lateral, and crown predictions
        primary_path = h5_path.replace(".h5", f".model{model_id}.rootprimary.slp")
        lateral_path = h5_path.replace(".h5", f".model{model_id}.rootlateral.slp")
        crown_path = h5_path.replace(".h5", f".model{model_id}.rootcrown.slp")
        # Load the Series
        series = Series.load(
            series_name,
            h5_path=h5_path,
            primary_path=primary_path,
            lateral_path=lateral_path,
            crown_path=crown_path,
            csv_path=csv_path,
        )
        series_list.append(series)
    return series_list


def load_series_from_slps(
    slp_paths: List[str], h5s: bool = False, csv_path: Optional[str] = None
) -> List[Series]:
    """Load a list of Series from a list of .slp paths.

    To load the `Series`, the files must be named with the following convention.
    The `slp_paths` are expeted to have the `series_name` in the filename and "primary",
    "lateral", or "crown" in the filename to differentiate the predictions.
    h5_path: '/path/to/scan/series_name.h5'
    Note that everything is expected to be in the same folder.

    Our pipeline outputs prediction files with this format:
    /<output_folder>/scan{scan_id}.model{model_id}.root{model_type}.slp


    Args:
        slp_paths: List of paths to .slp files.
        h5s: Boolean flag to indicate if the .h5 files are available. Default is False.
        csv_path: Optional path to the CSV file containing the expected plant count.
    """
    series_list = []
    series_names = list(set([Path(p).name.split(".")[0] for p in slp_paths]))
    for series_name in series_names:
        # Generate the paths for the primary, lateral, and crown predictions
        primary_path = [p for p in slp_paths if series_name in p and "primary" in p]
        lateral_path = [p for p in slp_paths if series_name in p and "lateral" in p]
        crown_path = [p for p in slp_paths if series_name in p and "crown" in p]
        # Check if the .h5 files are available
        if h5s:
            # Get directory of the h5s
            h5_dir = Path(slp_paths[0]).parent
            # Create path to the .h5 file
            h5_path = h5_dir / f"{series_name}.h5"
        else:
            h5_path = None
        # Load the Series
        series = Series.load(
            series_name,
            primary_path=primary_path[0] if primary_path else None,
            lateral_path=lateral_path[0] if lateral_path else None,
            crown_path=crown_path[0] if crown_path else None,
            h5_path=h5_path,
            csv_path=csv_path,
        )
        series_list.append(series)
    return series_list


def imgfig(
    size: Union[float, Tuple] = 6, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Create a tight figure for image plotting.

    Args:
        size: Scalar or 2-tuple specifying the (width, height) of the figure in inches.
            If scalar, will assume equal width and height.
        dpi: Dots per inch, controlling the resolution of the image.
        scale: Factor to scale the size of the figure by. This is a convenience for
            increasing the size of the plot at the same DPI.

    Returns:
        A matplotlib.figure.Figure to use for plotting.
    """
    if not isinstance(size, (tuple, list)):
        size = (size, size)
    fig = plt.figure(figsize=(scale * size[0], scale * size[1]), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    return fig


def plot_img(
    img: np.ndarray, dpi: int = 72, scale: float = 1.0
) -> matplotlib.figure.Figure:
    """Plot an image in a tight figure.

    Args:
        img: Image to plot. Can be a numpy array or a `tf.Tensor`.
        dpi: Dots per inch, controlling the resolution of the image.
        scale: Factor to scale the size of the figure by. This is a convenience for
            increasing the size of the plot at the same DPI.

    Returns:
        A matplotlib.figure.Figure containing the image.
    """
    if hasattr(img, "numpy"):
        img = img.numpy()

    if img.shape[0] == 1:
        # Squeeze out batch singleton dimension.
        img = img.squeeze(axis=0)

    # Check if image is grayscale (single channel).
    grayscale = img.shape[-1] == 1
    if grayscale:
        # Squeeze out singleton channel.
        img = img.squeeze(axis=-1)

    # Normalize the range of pixel values.
    img_min = img.min()
    img_max = img.max()
    if img_min < 0.0 or img_max > 1.0:
        img = (img - img_min) / (img_max - img_min)

    fig = imgfig(
        size=(float(img.shape[1]) / dpi, float(img.shape[0]) / dpi),
        dpi=dpi,
        scale=scale,
    )

    ax = fig.gca()
    ax.imshow(
        img,
        cmap="gray" if grayscale else None,
        origin="upper",
        extent=[-0.5, img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5],
    )
    return fig


def plot_instance(
    instance,
    skeleton=None,
    cmap=None,
    color_by_node=False,
    lw=2,
    ms=10,
    bbox=None,
    scale=1.0,
    **kwargs,
):
    """Plot a single instance with edge coloring."""
    if cmap is None:
        cmap = sns.color_palette("tab20")

    if skeleton is None and hasattr(instance, "skeleton"):
        skeleton = instance.skeleton

    if skeleton is None:
        color_by_node = True
    else:
        if len(skeleton.edges) == 0:
            color_by_node = True

    if hasattr(instance, "numpy"):
        inst_pts = instance.numpy()
    else:
        inst_pts = instance

    h_lines = []
    if color_by_node:
        for k, (x, y) in enumerate(inst_pts):
            if bbox is not None:
                x -= bbox[1]
                y -= bbox[0]

            x *= scale
            y *= scale

            h_lines_k = plt.plot(x, y, ".", ms=ms, c=cmap[k % len(cmap)], **kwargs)
            h_lines.append(h_lines_k)

    else:
        for k, (src_ind, dst_ind) in enumerate(skeleton.edge_inds):
            src_pt = inst_pts[src_ind]
            dst_pt = inst_pts[dst_ind]

            x = np.array([src_pt[0], dst_pt[0]])
            y = np.array([src_pt[1], dst_pt[1]])

            if bbox is not None:
                x -= bbox[1]
                y -= bbox[0]

            x *= scale
            y *= scale

            h_lines_k = plt.plot(
                x, y, ".-", ms=ms, lw=lw, c=cmap[k % len(cmap)], **kwargs
            )

            h_lines.append(h_lines_k)

    return h_lines


def plot_instances(
    instances: List,
    skeleton: Optional["sio.Skeleton"] = None,
    cmap: Optional[List] = None,
    color_by_track: bool = False,
    tracks: Optional[List] = None,
    **kwargs: Any,
) -> List:
    """Plot a list of instances with identity coloring.

    Args:
        instances: List of instances to plot.
        skeleton: Skeleton to use for edge coloring. If not provided, will use node
            coloring.
        cmap: Color map to use for coloring. If not provided, will use the default
            seaborn tab10 color palette.
        color_by_track: If True, will color instances by their track. If False, will
            color instances by their identity in the list.
        tracks: List of tracks to use for coloring. If not provided, will infer tracks
            from the instances.
        **kwargs: Additional keyword arguments to pass to `plot_instance`.

    Returns:
        A list of handles to the plotted lines.
    """
    if cmap is None:
        cmap = sns.color_palette("tab10")

    if color_by_track and tracks is None:
        # Infer tracks for ordering if not provided.
        tracks = set()
        for instance in instances:
            tracks.add(instance.track)

        # Sort by spawned frame.
        tracks = sorted(list(tracks), key=lambda track: track.name)

    h_lines = []
    for i, instance in enumerate(instances):
        if color_by_track:
            if instance.track is None:
                raise ValueError(
                    "Instances must have a set track when coloring by track."
                )

            if instance.track not in tracks:
                raise ValueError("Instance has a track not found in specified tracks.")

            color = cmap[tracks.index(instance.track) % len(cmap)]

        else:
            # Color by identity (order in list).
            color = cmap[i % len(cmap)]

        h_lines_i = plot_instance(instance, skeleton=skeleton, cmap=[color], **kwargs)
        h_lines.append(h_lines_i)

    return h_lines


_VALID_ROOT_TYPES = ("primary", "lateral", "crown")


def _resolve_root_type(
    path_attrs: Dict[str, Optional[str]],
    root_type: Optional[str],
) -> str:
    """Resolve `root_type` from a populated-path mapping.

    Shared helper used by `Series.get_tracked_tips` and
    `validate_series_for_tracked_tip` to keep root-type validation +
    auto-detection consistent across both call sites.

    Args:
        path_attrs: Dict mapping root-type strings (`'primary'` / `'lateral'`
            / `'crown'`) to either a path or `None`.
        root_type: Either `None` (auto-detect from `path_attrs`) or one of
            the valid root-type strings.

    Returns:
        The resolved root-type string (one of `'primary'`, `'lateral'`,
        `'crown'`).

    Raises:
        ValueError: When `root_type` is not `None` but is not in
            `{'primary', 'lateral', 'crown'}`; when `root_type` is `None`
            and zero or more than one path is populated.
    """
    if root_type is not None:
        if root_type not in _VALID_ROOT_TYPES:
            raise ValueError(
                f"Invalid root_type={root_type!r}; must be one of "
                f"{list(_VALID_ROOT_TYPES)}."
            )
        return root_type

    populated = [k for k, v in path_attrs.items() if v is not None]
    if len(populated) == 0:
        raise ValueError(
            "Cannot auto-detect root_type: none of primary_path, "
            "lateral_path, or crown_path is populated. Pass an explicit "
            f"root_type kwarg (one of {list(_VALID_ROOT_TYPES)})."
        )
    if len(populated) > 1:
        raise ValueError(
            f"Cannot auto-detect root_type: multiple paths populated "
            f"({populated}). Pass an explicit root_type kwarg "
            f"(one of {list(_VALID_ROOT_TYPES)})."
        )
    return populated[0]


def validate_tracked_slp(slp_path: Union[str, Path]) -> None:
    """Validate that every instance in a .slp file has a non-empty track.

    Used as an input precondition for `TrackedTipPipeline`.

    Args:
        slp_path: Path to the .slp file to validate.

    Returns:
        `None` when every instance has a non-empty `inst.track.name`.

    Raises:
        ValueError: When any instance has `inst.track is None` or empty
            `inst.track.name`. The error message lists ALL offending frame
            indices and points to the SLEAP tracking documentation.
    """
    labels = sio.load_slp(str(slp_path))
    offending = []
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            track = inst.track
            if track is None or not getattr(track, "name", None):
                offending.append(lf.frame_idx)
                break  # one offender per frame is enough to flag the frame
    if offending:
        unique_frames = sorted(set(offending))
        raise ValueError(
            f"TrackedTipPipeline requires tracked .slp predictions; "
            f"untracked instances found at frame indices {unique_frames} in "
            f"{slp_path}. See https://sleap.ai/tutorials/tracking.html for "
            f"tracking setup."
        )


def validate_series_for_tracked_tip(
    series: Series,
    root_type: Optional[str] = None,
) -> None:
    """Validate a `Series` is loadable by `TrackedTipPipeline`.

    Composite check: resolves `root_type` (auto-detect when `None`),
    asserts the corresponding `<root_type>_path` is set and the loaded
    skeleton has at least one node, then calls `validate_tracked_slp` on
    the resolved path.

    Args:
        series: The `Series` to validate.
        root_type: One of `"primary"`, `"lateral"`, `"crown"`. When
            `None`, auto-detects from whichever single `<root_type>_path`
            is populated.

    Returns:
        `None` when the series is valid for tracked-tip processing.

    Raises:
        ValueError: When `root_type` is `None` and zero or more than one
            of `primary_path` / `lateral_path` / `crown_path` is
            populated; when the resolved path's skeleton has zero nodes;
            or when `validate_tracked_slp` raises (propagated).
    """
    path_attrs = {
        "primary": series.primary_path,
        "lateral": series.lateral_path,
        "crown": series.crown_path,
    }
    root_type = _resolve_root_type(path_attrs, root_type)

    resolved_path = path_attrs[root_type]
    if resolved_path is None:
        raise ValueError(
            f"root_type={root_type!r} requested but {root_type}_path is None "
            f"on the series."
        )

    labels_attr = f"{root_type}_labels"
    labels = getattr(series, labels_attr, None)
    if labels is not None:
        # Skeleton sanity check (every Labels has a `skeletons` list).
        sks = getattr(labels, "skeletons", None) or []
        if sks and len(sks[0].nodes) == 0:
            raise ValueError(
                f"{root_type} skeleton has zero nodes — cannot extract tips."
            )

    validate_tracked_slp(resolved_path)
