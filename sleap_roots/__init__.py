"""High-level imports."""

import sleap_roots.angle
import sleap_roots.bases
import sleap_roots.tips
import sleap_roots.convhull
import sleap_roots.ellipse
import sleap_roots.networklength
import sleap_roots.lengths
import sleap_roots.points
import sleap_roots.scanline
import sleap_roots.series
import sleap_roots.summary
import sleap_roots.trait_pipelines
from sleap_roots.trait_pipelines import (
    DicotPipeline,
    TraitDef,
    YoungerMonocotPipeline,
    OlderMonocotPipeline,
    MultipleDicotPipeline,
)
from sleap_roots.series import (
    Series,
    find_all_h5_paths,
    find_all_slp_paths,
    load_series_from_h5s,
    load_series_from_slps,
)

# Define package version.
# This is read dynamically by setuptools in pyproject.toml to determine the release version.
__version__ = "0.1.2"
