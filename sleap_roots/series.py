"""Series-level data loader."""

import attrs
import numpy as np
from pathlib import Path
import sleap
from typing import Optional, Tuple, List, Union


@attrs.define
class Series:
    """Data and predictions for a single image series.

    Attributes:
        h5_path: Path to the HDF5-formatted image series.
        primary_labels: A `sleap.Labels` corresponding to the primary root predictions.
        lateral_labels: A `sleap.Labels` corresponding to the lateral root predictions.
    """

    h5_path: Optional[str] = None
    primary_labels: Optional[sleap.Labels] = None
    lateral_labels: Optional[sleap.Labels] = None

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
            primary_labels=sleap.load_file(primary_path),
            lateral_labels=sleap.load_file(lateral_path),
        )

    @property
    def series_name(self) -> str:
        """Name of the series derived from the HDF5 filename."""
        return Path(self.h5_path).stem

    @property
    def video(self) -> sleap.Video:
        """The `sleap.Video` corresponding to the image series."""
        return self.primary_labels.video

    def __len__(self) -> int:
        """Length of the series (number of images)."""
        return len(self.video)

    def __getitem__(self, idx: int) -> Tuple[sleap.LabeledFrame, sleap.LabeledFrame]:
        """Return labeled frames for primary and lateral predictions."""
        return self.get_frame(idx)

    def __iter__(self):
        """Iterator for looping through predictions."""
        for i in range(len(self)):
            yield self[i]

    def get_frame(
        self, frame_idx: int
    ) -> Tuple[sleap.LabeledFrame, sleap.LabeledFrame]:
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
        sleap.nn.viz.plot_img(primary_lf.image, scale=scale)
        sleap.nn.viz.plot_instances(primary_lf.instances, cmap=["r"], **kwargs)
        sleap.nn.viz.plot_instances(lateral_lf.instances, cmap=["g"], **kwargs)

    def get_primary_points(self, frame_idx: int) -> np.ndarray:
        """Get primary root points.

        Args:
            frame_idx: frame index to get primary root points in shape (# instance,
            # node, 2)
        """
        primary_lf, lateral_lf = self.get_frame(frame_idx)
        return primary_lf.numpy()

    def get_lateral_points(self, frame_idx: int) -> np.ndarray:
        """Get lateral root points.

        Args:
            frame_idx: frame index to get lateral root points in shape (# instance,
            # node, 2)
        """
        primary_lf, lateral_lf = self.get_frame(frame_idx)
        return lateral_lf.numpy()


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
