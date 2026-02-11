"""HTML viewer for SLEAP prediction visualization.

This module provides tools for generating static HTML reports that visualize
SLEAP prediction overlays on root images, enabling quick validation of model
predictions without requiring the SLEAP GUI.
"""

from sleap_roots.viewer.generator import ViewerGenerator

__all__ = ["ViewerGenerator"]
