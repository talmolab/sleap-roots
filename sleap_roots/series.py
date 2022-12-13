"""Series-level data loader."""

import attrs
from typing import Optional, Tuple
from pathlib import Path

try:
    import sleap
except:
    import sleap_io as sleap


@attrs.define
class Series:
    h5_path: Optional[str] = None
    primary_name: str = "primary_multi_day"
    lateral_name: str = "lateral__nodes"
    primary_labels: Optional[sleap.Labels] = None
    lateral_labels: Optional[sleap.Labels] = None
    
    @classmethod
    def load(cls, h5_path: str, primary_name: str = "primary_multi_day", lateral_name: str = "lateral__nodes"):
        primary_path = Path(h5_path).with_suffix(f".{primary_name}.predictions.slp").as_posix()
        lateral_path = Path(h5_path).with_suffix(f".{lateral_name}.predictions.slp").as_posix()
        
        return cls(
            h5_path,
            primary_name=primary_name,
            lateral_name=lateral_name,
            primary_labels=sleap.load_file(primary_path),
            lateral_labels=sleap.load_file(lateral_path),
        )
    
    @property
    def series_name(self) -> str:
        return Path(self.h5_path).stem
    
    @property
    def video(self) -> sleap.Video:
        return self.primary_labels.video
    
    def __len__(self) -> int:
        return len(self.video)
    
    def __getitem__(self, idx: int) ->  Tuple[sleap.LabeledFrame, sleap.LabeledFrame]:
        return self.get_frame(idx)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def get_frame(self, frame_idx: int) -> Tuple[sleap.LabeledFrame, sleap.LabeledFrame]:
        """Return labeled frames for primary and lateral predictions.
        
        Args:
            frame_idx: Integer frame number.
        
        Returns:
            Tuple of (primary_lf, lateral_lf) corresponding to the
            sleap.LabeledFrames from each set of predictions on the same frame.
        """
        lf_primary = self.primary_labels.find(self.primary_labels.video, frame_idx, return_new=True)[0]
        lf_lateral = self.lateral_labels.find(self.lateral_labels.video, frame_idx, return_new=True)[0]
        return lf_primary, lf_lateral

    def plot(self, frame_idx: int, scale: float = 1.0, **kwargs):
        primary_lf, lateral_lf = self.get_frame(frame_idx)
        sleap.nn.viz.plot_img(primary_lf.image, scale=scale)
        sleap.nn.viz.plot_instances(primary_lf.instances, cmap=["r"], **kwargs)
        sleap.nn.viz.plot_instances(lateral_lf.instances, cmap=["g"], **kwargs)


def find_all_series(data_folders: list[str]) -> list[str]:
    """Find all .h5 series from a list of folders.
    
    Args:
        data_folders: List of paths to folders containing .h5 series.
    
    Returns:
        A list of filenames to .h5 series.
    """
    if type(data_folders) != list:
        data_folders = [data_folders]

    h5_series = []
    for data_folder in data_folders:
        h5_series.extend([Path(p).as_posix() for p in Path(data_folder).glob("*.h5")])
    return h5_series