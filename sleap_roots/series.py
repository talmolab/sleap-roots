"""Series-level data loader."""

import attrs
import numpy as np
import sleap_io as sio
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from typing import Dict, Optional, Tuple, List, Union
from pathlib import Path


@attrs.define
class Series:
    """Data and predictions for a single image series.

    Attributes:
        series_name: Unique identifier for the series.
        h5_path: Optional path to the HDF5-formatted image series.
        primary_labels: Optional `sio.Labels` corresponding to the primary root predictions.
        lateral_labels: Optional `sio.Labels` corresponding to the lateral root predictions.
        crown_labels: Optional `sio.Labels` corresponding to the crown predictions.
        video: Optional `sio.Video` corresponding to the image series.
        csv_path: Optional path to the CSV file containing the expected plant count.

    Methods:
        load: Load a set of predictions for this series.
        __len__: Length of the series (number of images).
        __getitem__: Return labeled frames for predictions.
        __iter__: Iterator for looping through predictions.
        get_frame: Return labeled frames for predictions.
        plot: Plot predictions on top of the image.
        get_primary_points: Get primary root points.
        get_lateral_points: Get lateral root points.
        get_crown_points: Get crown root points.

    Properties:
        expected_count: Expected plant count for this series from the CSV.
        group: Group name for the series from the CSV.
        qc_fail: Flag to indicate if the series failed QC from the CSV.
    """

    series_name: str
    h5_path: Optional[str] = None
    primary_labels: Optional[sio.Labels] = None
    lateral_labels: Optional[sio.Labels] = None
    crown_labels: Optional[sio.Labels] = None
    video: Optional[sio.Video] = None
    csv_path: Optional[str] = None

    @classmethod
    def load(
        cls,
        series_name: Optional[str] = None,
        h5_path: Optional[str] = None,
        primary_name: Optional[str] = None,
        lateral_name: Optional[str] = None,
        crown_name: Optional[str] = None,
        csv_path: Optional[str] = None,
    ) -> "Series":
        """Load a set of predictions for this series.

        Args:
            series_name: Unique identifier for the scan. The prediction files are
                expected to be named "{series_name}.{prediction_name}.slp".
            h5_path: Path to the HDF5-formatted image series. If the video in any of the
                labels is not found, the path will be used to replace the filename. This
                is necessary for plotting.
            primary_name: Optional path to primary prediction file ending with .slp. If
                a string is provided that does not end with .slp, the string is assumed
                to be the ending of predictions filewith the format
                "{series_name}.{primary_name}.slp".
            lateral_name: Optional path to lateral prediction file ending with .slp. If
                a string is provided that does not end with .slp, the string is assumed
                to be the ending of predictions file with the format
                "{series_name}.{lateral_name}.slp".
            crown_name: Optional path to crown prediction file ending with .slp. If a
                string is provided that does not end with .slp, the string is assumed to
                be the ending of predictions file with the format
                "{series_name}.{crown_name}.slp".
            csv_path: Optional path to the CSV file containing the expected plant count.

        Returns:
            An instance of Series loaded with the specified predictions.
        """
        if series_name is None:
            if h5_path is not None:
                series_name = Path(h5_path).name.split(".")[0]
            else:
                raise ValueError("series_name or h5_path must be provided.")

        # Initialize the labels as None
        primary_labels, lateral_labels, crown_labels = None, None, None

        # Attempt to load the predictions, with error handling
        try:
            if primary_name:
                if Path(primary_name).as_posix().endswith(".slp"):
                    primary_path = primary_name
                else:
                    primary_path = (
                        Path(series_name).with_suffix(f".{primary_name}.slp").as_posix()
                    )
                if Path(primary_path).exists():
                    primary_labels = sio.load_slp(primary_path)
                else:
                    print(f"Primary prediction file not found: {primary_path}")
            if lateral_name:
                if lateral_name.endswith(".slp"):
                    lateral_path = lateral_name
                else:
                    lateral_path = (
                        Path(series_name).with_suffix(f".{lateral_name}.slp").as_posix()
                    )
                if Path(lateral_path).exists():
                    lateral_labels = sio.load_slp(lateral_path)
                else:
                    print(f"Lateral prediction file not found: {lateral_path}")
            if crown_name:
                if crown_name.endswith(".slp"):
                    crown_path = crown_name
                else:
                    crown_path = (
                        Path(series_name).with_suffix(f".{crown_name}.slp").as_posix()
                    )
                if Path(crown_path).exists():
                    crown_labels = sio.load_slp(crown_path)
                else:
                    print(f"Crown prediction file not found: {crown_path}")
        except Exception as e:
            print(f"Error loading prediction files: {e}")

        # Replace the filename in the labels with the h5_path if it is provided.
        if h5_path is not None:
            for labels in [primary_labels, lateral_labels, crown_labels]:
                if labels is not None:
                    if not labels.video.exists():
                        labels.video.replace_filename(h5_path)

        # Attempt to load the video, with error handling
        video = None
        for labels in [primary_labels, lateral_labels, crown_labels]:
            if labels is not None:
                if labels.video.exists():
                    video = labels.video
                    break

        return cls(
            series_name=series_name,
            h5_path=h5_path,
            primary_labels=primary_labels,
            lateral_labels=lateral_labels,
            crown_labels=crown_labels,
            video=video,
            csv_path=csv_path,
        )

    @property
    def expected_count(self) -> Union[float, int]:
        """Fetch the expected plant count for this series from the CSV."""
        if not self.csv_path or not Path(self.csv_path).exists():
            print("CSV path is not set or the file does not exist.")
            return np.nan
        df = pd.read_csv(self.csv_path)
        try:
            # Match the series_name (or plant_qr_code in the CSV) to fetch the expected count
            return df[df["plant_qr_code"] == self.series_name][
                "number_of_plants_cylinder"
            ].iloc[0]
        except IndexError:
            print(f"No expected count found for series {self.series_name} in CSV.")
            return np.nan

    @property
    def group(self) -> str:
        """Group name for the series from the CSV."""
        if not self.csv_path or not Path(self.csv_path).exists():
            print("CSV path is not set or the file does not exist.")
            return np.nan
        df = pd.read_csv(self.csv_path)
        try:
            # Match the series_name (or plant_qr_code in the CSV) to fetch the group
            return df[df["plant_qr_code"] == self.series_name]["genotype"].iloc[0]
        except IndexError:
            print(f"No group found for series {self.series_name} in CSV.")
            return np.nan

    @property
    def qc_fail(self) -> Union[int, float]:
        """Flag to indicate if the series failed QC from the CSV."""
        if not self.csv_path or not Path(self.csv_path).exists():
            print("CSV path is not set or the file does not exist.")
            return np.nan
        df = pd.read_csv(self.csv_path)
        try:
            # Match the series_name (or plant_qr_code in the CSV) to fetch the QC flag
            return df[df["plant_qr_code"] == self.series_name]["qc_cylinder"].iloc[0]
        except IndexError:
            print(f"No QC flag found for series {self.series_name} in CSV.")
            return np.nan

    def __len__(self) -> int:
        """Length of the series (number of images)."""
        return len(self.video)

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

    def plot(self, frame_idx: int, scale: float = 1.0, **kwargs):
        """Plot predictions on top of the image.

        Args:
            frame_idx: Frame index to visualize.
            scale: Relative size of the visualized image. Useful for plotting smaller
                images within notebooks.
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


def find_all_h5_series(data_folders: Union[str, List[str]]) -> List[str]:
    """Find all .h5 series from a list of folders.

    Args:
        data_folders: Path or list of paths to folders containing .h5 series.

    Returns:
        A list of filenames to .h5 series.
    """
    if type(data_folders) != list:
        data_folders = [data_folders]

    h5_series = []
    for data_folder in data_folders:
        h5_series.extend([Path(p).as_posix() for p in Path(data_folder).glob("*.h5")])
    return h5_series


def find_all_slp_series(data_folders: Union[str, List[str]]) -> List[str]:
    """Find all .slp series from a list of folders.

    Args:
        data_folders: Path or list of paths to folders containing .slp series.

    Returns:
        A list of unique series names derived from the filenames.
    """
    if type(data_folders) != list:
        data_folders = [data_folders]

    slp_series = []
    for data_folder in data_folders:
        slp_series.extend([Path(p).as_posix() for p in Path(data_folder).glob("*.slp")])
    return list(set([Path(p).name.split(".")[0] for p in slp_series]))


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
    instances, skeleton=None, cmap=None, color_by_track=False, tracks=None, **kwargs
):
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
