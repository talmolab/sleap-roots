"""Series-level data loader."""

import attrs
import numpy as np
from pathlib import Path
import sleap_io as sio
from typing import Optional, Tuple, List, Union

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


@attrs.define
class Series:
    """Data and predictions for a single image series.

    Attributes:
        h5_path: Path to the HDF5-formatted image series.
        primary_labels: A `sleap.Labels` corresponding to the primary root predictions.
        lateral_labels: A `sleap.Labels` corresponding to the lateral root predictions.
    """

    h5_path: Optional[str] = None
    primary_labels: Optional[sio.Labels] = None
    lateral_labels: Optional[sio.Labels] = None

    @classmethod
    def load(
        cls,
        h5_path: str,
        primary_name: str = "primary_multi_day",
        lateral_name: str = "lateral__nodes",
    ):
        """Load a set of predictions for this series.

        Args:
            h5_path: Path to the HDF5-formatted image series.
            primary_name: Name of the primary root predictions. The predictions file is
                expected to be named `"{h5_path}.{primary_name}.predictions.slp"`.
            lateral_name: Name of the lateral root predictions. The predictions file is
                expected to be named `"{h5_path}.{lateral_name}.predictions.slp"`.
        """
        primary_path = (
            Path(h5_path).with_suffix(f".{primary_name}.predictions.slp").as_posix()
        )
        lateral_path = (
            Path(h5_path).with_suffix(f".{lateral_name}.predictions.slp").as_posix()
        )

        return cls(
            h5_path,
            primary_labels=sio.load_slp(primary_path),
            lateral_labels=sio.load_slp(lateral_path),
        )

    @property
    def series_name(self) -> str:
        """Name of the series derived from the HDF5 filename."""
        return Path(self.h5_path).name.split(".")[0]

    @property
    def video(self) -> sio.Video:
        """The `sleap.Video` corresponding to the image series."""
        return self.primary_labels.video

    def __len__(self) -> int:
        """Length of the series (number of images)."""
        return len(self.video)

    def __getitem__(self, idx: int) -> Tuple[sio.LabeledFrame, sio.LabeledFrame]:
        """Return labeled frames for primary and lateral predictions."""
        return self.get_frame(idx)

    def __iter__(self):
        """Iterator for looping through predictions."""
        for i in range(len(self)):
            yield self[i]

    def get_frame(self, frame_idx: int) -> Tuple[sio.LabeledFrame, sio.LabeledFrame]:
        """Return labeled frames for primary and lateral predictions.

        Args:
            frame_idx: Integer frame number.

        Returns:
            Tuple of (primary_lf, lateral_lf) corresponding to the
            `sleap.LabeledFrame` from each set of predictions on the same frame.
        """
        lf_primary = self.primary_labels.find(
            self.primary_labels.video, frame_idx, return_new=True
        )[0]
        lf_lateral = self.lateral_labels.find(
            self.lateral_labels.video, frame_idx, return_new=True
        )[0]
        return lf_primary, lf_lateral

    def plot(self, frame_idx: int, scale: float = 1.0, **kwargs):
        """Plot predictions on top of the image.

        Args:
            frame_idx: Frame index to visualize.
            scale: Relative size of the visualized image. Useful for plotting smaller
                images within notebooks.
        """
        primary_lf, lateral_lf = self.get_frame(frame_idx)
        plot_img(primary_lf.image, scale=scale)
        plot_instances(primary_lf.instances, cmap=["r"], **kwargs)
        plot_instances(lateral_lf.instances, cmap=["g"], **kwargs)

    def get_primary_points(self, frame_idx: int) -> np.ndarray:
        """Get primary root points.

        Args:
            frame_idx: frame index to get primary root points in shape (# instance,
            # node, 2)
        """
        primary_lf, lateral_lf = self.get_frame(frame_idx)
        gt_instances_pr = primary_lf.user_instances + primary_lf.unused_predictions
        if len(gt_instances_pr) == 0:
            return []
        else:
            primary_pts = np.stack([inst.numpy() for inst in gt_instances_pr], axis=0)
        return primary_pts

    def get_lateral_points(self, frame_idx: int) -> np.ndarray:
        """Get lateral root points.

        Args:
            frame_idx: frame index to get lateral root points in shape (# instance,
            # node, 2)
        """
        primary_lf, lateral_lf = self.get_frame(frame_idx)
        gt_instances_lr = lateral_lf.user_instances + lateral_lf.unused_predictions
        if len(gt_instances_lr) == 0:
            return []
        else:
            lateral_pts = np.stack([inst.numpy() for inst in gt_instances_lr], axis=0)
        return lateral_pts


def find_all_series(data_folders: Union[str, List[str]]) -> List[str]:
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
