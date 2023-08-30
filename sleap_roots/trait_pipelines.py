"""Extract traits in a pipeline based on a trait graph."""

import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import attrs
import networkx as nx
import numpy as np
import pandas as pd

from sleap_roots.angle import get_node_ind, get_root_angle
from sleap_roots.bases import (
    get_base_ct_density,
    get_base_length,
    get_base_length_ratio,
    get_base_median_ratio,
    get_base_tip_dist,
    get_base_xs,
    get_base_ys,
    get_bases,
    get_lateral_count,
    get_root_widths_inds,
    get_root_widths_package,
    get_root_widths_left_bases,
    get_root_widths_right_bases,
    get_root_widths,
)
from sleap_roots.convhull import (
    get_chull_area,
    get_chull_line_lengths,
    get_chull_max_height,
    get_chull_max_width,
    get_chull_perimeter,
    get_convhull,
)
from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)
from sleap_roots.lengths import get_grav_index, get_max_length_pts, get_root_lengths
from sleap_roots.networklength import (
    get_bbox,
    get_network_distribution,
    get_network_distribution_ratio,
    get_network_length,
    get_network_solidity,
    get_network_width_depth_ratio,
)
from sleap_roots.points import get_all_pts_array
from sleap_roots.scanline import (
    count_scanline_intersections,
    get_scanline_first_ind,
    get_scanline_last_ind,
)
from sleap_roots.series import Series
from sleap_roots.summary import SUMMARY_SUFFIXES, get_summary
from sleap_roots.tips import get_tip_xs, get_tip_ys, get_tips

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in intersection",
    category=RuntimeWarning,
    module="shapely",
)
warnings.filterwarnings(
    "ignore", message="All-NaN slice encountered", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore", message="All-NaN axis encountered", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="Degrees of freedom <= 0 for slice.",
    category=RuntimeWarning,
    module="numpy",
)
warnings.filterwarnings(
    "ignore", message="Mean of empty slice", category=RuntimeWarning
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in sqrt",
    category=RuntimeWarning,
    module="skimage",
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in double_scalars",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar divide",
    category=RuntimeWarning,
    module="ellipse",
)


@attrs.define
class TraitDef:
    """Definition of how to compute a trait.

    Attributes:
        name: Unique identifier for the trait.
        fn: Function used to compute the trait's value.
        input_traits: List of trait names that should be computed before the current
            trait and are expected as input positional arguments to `fn`.
        scalar: Indicates if the trait is scalar (has a dimension of 0 per frame). If
            `True`, the trait is also listed in `SCALAR_TRAITS`.
        include_in_csv: `True `indicates the trait should be included in downstream CSV
            files.
        kwargs: Additional keyword arguments to be passed to the `fn` function. These
            arguments are not reused from previously computed traits.
        description: String describing the trait for documentation purposes.

    Notes:
        The `fn` specified will be called with a pattern like:

        ```
        trait_def = TraitDef(
            name="my_trait",
            fn=compute_my_trait,
            input_traits=["input_trait_1", "input_trait_2"],
            scalar=True,
            include_in_csv=True,
            kwargs={"kwarg1": True}
        )
        traits[trait_def.name] = trait_def.fn(
            *[traits[input_trait] for input_trait in trait_def.input_traits],
            **trait_def.kwargs
        )
        ```

        For this example, the last line is equivalent to:

        ```
        traits["my_trait"] = trait_def.fn(
            traits["input_trait_1"], traits["input_trait_2"],
            kwarg1=True
        )
        ```
    """

    name: str
    fn: Callable
    input_traits: List[str]
    scalar: bool
    include_in_csv: bool
    kwargs: Dict[str, Any] = attrs.field(factory=dict)
    description: Optional[str] = None


@attrs.define
class Pipeline:
    """Pipeline for computing traits.

    Attributes:
        traits: List of `TraitDef` objects.
        trait_map: Dictionary mapping trait names to their definitions.
        trait_computation_order: List of trait names in the order they should be
            computed.
    """

    traits: List[TraitDef] = attrs.field(init=False)
    trait_map: Dict[str, TraitDef] = attrs.field(init=False)
    trait_computation_order: List[str] = attrs.field(init=False)

    def __attrs_post_init__(self):
        """Build pipeline objects from traits list."""
        # Build list of trait definitions.
        self.traits = self.define_traits()

        # Check that trait names are unique.
        trait_names = [trait.name for trait in self.traits]
        if len(trait_names) != len(set(trait_names)):
            raise ValueError("Trait names must be unique.")

        # Map trait names to their definitions.
        self.trait_map = {trait_def.name: trait_def for trait_def in self.traits}

        # Determine computation order by topologically sorting the nodes.
        self.trait_computation_order = self.get_computation_order()

    def define_traits(self) -> List[TraitDef]:
        """Return list of `TraitDef` objects."""
        raise NotImplementedError

    def get_computation_order(self) -> List[str]:
        """Determine computation order by topologically sorting the nodes.

        Returns:
            A list of trait names in the order they should be computed.
        """
        # Infer edges from trait map.
        edges = []
        for trait_def in self.traits:
            for input_trait in trait_def.input_traits:
                edges.append((input_trait, trait_def.name))

        # Build networkx graph from inferred edges.
        G = nx.DiGraph()
        G.add_edges_from(edges)

        # Determine computation order by topologically sorting the nodes.
        trait_computation_order = list(nx.topological_sort(G))

        return trait_computation_order

    @property
    def summary_traits(self) -> List[str]:
        """List of traits to include in the summary CSV."""
        return [
            trait.name
            for trait in self.traits
            if trait.include_in_csv and not trait.scalar
        ]

    @property
    def csv_traits(self) -> List[str]:
        """List of frame-level traits to include in the CSV."""
        csv_traits = []
        for trait in self.traits:
            if trait.include_in_csv:
                if trait.scalar:
                    csv_traits.append(trait.name)
                else:
                    csv_traits.extend(
                        [f"{trait.name}_{suffix}" for suffix in SUMMARY_SUFFIXES]
                    )
        return csv_traits

    def compute_frame_traits(self, traits: Dict[str, Any]) -> Dict[str, Any]:
        """Compute traits based on the pipeline.

        Args:
            traits: Dictionary of traits where keys are trait names and values are
                the trait values.

        Returns:
            A dictionary of computed traits.
        """
        # Initialize traits container with initial data.
        traits = traits.copy()

        # Compute traits!
        for trait_name in self.trait_computation_order:
            if trait_name in traits:
                # Skip traits already computed.
                continue

            # Get trait definition.
            trait_def = self.trait_map[trait_name]

            # Compute trait based on trait definition.
            traits[trait_name] = trait_def.fn(
                *[traits[input_trait] for input_trait in trait_def.input_traits],
                **trait_def.kwargs,
            )

        return traits

    def get_initial_frame_traits(self, plant: Series, frame_idx: int) -> Dict[str, Any]:
        """Return initial traits for a plant frame.

        Args:
            plant: The plant `Series` object.
            frame_idx: The index of the current frame.

        Returns:
            A dictionary of initial traits.

            This is defined on a per-pipeline basis as different plant species will have
            different initial points to be used as starting traits.

            Most commonly, this will be the primary and lateral root points for the
            current frame.
        """
        raise NotImplementedError

    def compute_plant_traits(
        self,
        plant: Series,
        write_csv: bool = False,
        csv_suffix: str = ".traits.csv",
        return_non_scalar: bool = False,
    ) -> pd.DataFrame:
        """Compute traits for a plant.

        Args:
            plant: The plant image series as a `Series` object.
            write_csv: A boolean value. If True, it writes per plant detailed
                CSVs with traits for every instance on every frame.
            csv_suffix: If `write_csv` is `True`, a CSV file will be saved with the same
                name as the plant's `{plant.series_name}{csv_suffix}`.
            return_non_scalar: If `True`, return all non-scalar traits as well as the
                summarized traits.

        Returns:
            The computed traits as a pandas DataFrame.
        """
        traits = []
        for frame in range(len(plant)):
            # Get initial traits for the frame.
            initial_traits = self.get_initial_frame_traits(plant, frame)

            # Compute traits via the frame-level pipeline.
            frame_traits = self.compute_frame_traits(initial_traits)

            # Compute trait summaries.
            for trait_name in self.summary_traits:
                trait_summary = get_summary(
                    frame_traits[trait_name], prefix=f"{trait_name}_"
                )
                frame_traits.update(trait_summary)

            # Add metadata.
            frame_traits["plant_name"] = plant.series_name
            frame_traits["frame_idx"] = frame
            traits.append(frame_traits)
        traits = pd.DataFrame(traits)

        # Move metadata columns to the front.
        plant_name = traits.pop("plant_name")
        frame_idx = traits.pop("frame_idx")
        traits = pd.concat([plant_name, frame_idx, traits], axis=1)

        if write_csv:
            csv_name = Path(plant.h5_path).with_suffix(csv_suffix)
            traits[["plant_name", "frame_idx"] + self.csv_traits].to_csv(
                csv_name, index=False
            )

        if return_non_scalar:
            return traits
        else:
            return traits[["plant_name", "frame_idx"] + self.csv_traits]

    def compute_batch_traits(
        self,
        plants: List[Series],
        write_csv: bool = False,
        csv_path: str = "traits.csv",
    ) -> pd.DataFrame:
        """Compute traits for a batch of plants.

        Args:
            plants: List of `Series` objects.
            write_csv: If `True`, write the computed traits to a CSV file.
            csv_path: Path to write the CSV file to.

        Returns:
            A pandas DataFrame of computed traits summarized over all frames of each
            plant. The resulting dataframe will have a row for each plant and a column
            for each plant-level summarized trait.

            Summarized traits are prefixed with the trait name and an underscore,
            followed by the summary statistic.
        """
        all_traits = []
        for plant in plants:
            # Compute frame level traits for the plant.
            plant_traits = self.compute_plant_traits(plant)

            # Summarize frame level traits.
            plant_summary = {"plant_name": plant.series_name}
            for trait_name in self.csv_traits:
                trait_summary = get_summary(
                    plant_traits[trait_name], prefix=f"{trait_name}_"
                )
                plant_summary.update(trait_summary)
            all_traits.append(plant_summary)

        # Build dataframe from list of frame-level summaries.
        all_traits = pd.DataFrame(all_traits)

        if write_csv:
            all_traits.to_csv(csv_path, index=False)
        return all_traits


@attrs.define
class DicotPipeline(Pipeline):
    """Pipeline for computing traits for dicot plants.

    Attributes:
        img_height: Image height.
        img_width: Image width.
        root_width_tolerance: Difference in projection norm between right and left side.
        n_scanlines: Number of scan lines, np.nan for no interaction.
        network_fraction: Length found in the lower fraction value of the network.
    """

    img_height: int = 1080
    img_width: int = 2048
    root_width_tolerance: float = 0.02
    n_scanlines: int = 50
    network_fraction: float = 2 / 3

    def define_traits(self) -> List[TraitDef]:
        """Define the trait computation pipeline for dicot plants."""
        trait_definitions = [
            TraitDef(
                name="primary_max_length_pts",
                fn=get_max_length_pts,
                input_traits=["primary_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Points of the primary root with maximum length.",
            ),
            TraitDef(
                name="pts_all_array",
                fn=get_all_pts_array,
                input_traits=["primary_max_length_pts", "lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"monocots": False},
                description="Landmark points within a given frame as a flat array"
                "of coordinates.",
            ),
            TraitDef(
                name="root_widths_package",
                fn=get_root_widths_package,
                input_traits=["primary_max_length_pts", "lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"tolerance": self.root_width_tolerance, "monocots": False},
                description="Returns estimation of root width using bases of lateral "
                "roots with indices and coordinates of matched base pairs.",
            ),
            TraitDef(
                name="root_widths",
                fn=get_root_widths,
                input_traits=["root_widths_package"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Returns estimation of root width using bases of lateral "
                "roots.",
            ),
            TraitDef(
                name="root_widths_inds",
                fn=get_root_widths_inds,
                input_traits=["root_widths_package"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Returns indices of matched roots used to calculate the"
                "root widths.",
            ),
            TraitDef(
                name="root_widths_left_bases",
                fn=get_root_widths_left_bases,
                input_traits=["root_widths_package"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Returns the left bases of matched roots used to calculate"
                "the root widths.",
            ),
            TraitDef(
                name="root_widths_right_bases",
                fn=get_root_widths_right_bases,
                input_traits=["root_widths_package"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Returns the right bases of matched roots used to calculate"
                "the root widths.",
            ),
            TraitDef(
                name="lateral_count",
                fn=get_lateral_count,
                input_traits=["lateral_pts"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Get the number of lateral roots.",
            ),
            TraitDef(
                name="lateral_proximal_node_inds",
                fn=get_node_ind,
                input_traits=["lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"proximal": True},
                description="Get the indices of the proximal nodes of lateral roots.",
            ),
            TraitDef(
                name="lateral_distal_node_inds",
                fn=get_node_ind,
                input_traits=["lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"proximal": False},
                description="Get the indices of the distal nodes of lateral roots.",
            ),
            TraitDef(
                name="lateral_lengths",
                fn=get_root_lengths,
                input_traits=["lateral_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of lateral root lengths of shape `(instances,)`.",
            ),
            TraitDef(
                name="lateral_base_pts",
                fn=get_bases,
                input_traits=["lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"monocots": False},
                description="Array of lateral bases `(instances, (x, y))`.",
            ),
            TraitDef(
                name="lateral_tip_pts",
                fn=get_tips,
                input_traits=["lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Array of lateral tips `(instances, (x, y))`.",
            ),
            TraitDef(
                name="scanline_intersection_counts",
                fn=count_scanline_intersections,
                input_traits=["primary_max_length_pts", "lateral_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={
                    "height": self.img_height,
                    "width": self.img_width,
                    "n_line": self.n_scanlines,
                    "monocots": False,
                },
                description="Array of intersections of each scanline `(n_scanlines,)`.",
            ),
            TraitDef(
                name="lateral_angles_distal",
                fn=get_root_angle,
                input_traits=["lateral_pts", "lateral_distal_node_inds"],
                scalar=False,
                include_in_csv=True,
                kwargs={"proximal": False, "base_ind": 0},
                description="Array of lateral distal angles in degrees `(instances,)`.",
            ),
            TraitDef(
                name="lateral_angles_proximal",
                fn=get_root_angle,
                input_traits=["lateral_pts", "lateral_proximal_node_inds"],
                scalar=False,
                include_in_csv=True,
                kwargs={"proximal": True, "base_ind": 0},
                description="Array of lateral proximal angles in degrees "
                "`(instances,)`.",
            ),
            TraitDef(
                name="network_solidity",
                fn=get_network_solidity,
                input_traits=["network_length", "chull_area"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of the total network length divided by the network"
                "convex area.",
            ),
            TraitDef(
                name="ellipse",
                fn=fit_ellipse,
                input_traits=["pts_all_array"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Tuple of (a, b, ratio) containing the semi-major axis "
                "length, semi-minor axis length, and the ratio of the major to minor "
                "lengths.",
            ),
            TraitDef(
                name="bounding_box",
                fn=get_bbox,
                input_traits=["pts_all_array"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Tuple of four parameters in bounding box.",
            ),
            TraitDef(
                name="convex_hull",
                fn=get_convhull,
                input_traits=["pts_all_array"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Convex hull of the points.",
            ),
            TraitDef(
                name="primary_proximal_node_ind",
                fn=get_node_ind,
                input_traits=["primary_max_length_pts"],
                scalar=True,
                include_in_csv=False,
                kwargs={"proximal": True},
                description="Get the indices of the proximal nodes of primary roots.",
            ),
            TraitDef(
                name="primary_angle_proximal",
                fn=get_root_angle,
                input_traits=["primary_max_length_pts", "primary_proximal_node_ind"],
                scalar=True,
                include_in_csv=True,
                kwargs={"proximal": True, "base_ind": 0},
                description="Array of primary proximal angles in degrees "
                "`(instances,)`.",
            ),
            TraitDef(
                name="primary_distal_node_ind",
                fn=get_node_ind,
                input_traits=["primary_max_length_pts"],
                scalar=True,
                include_in_csv=False,
                kwargs={"proximal": False},
                description="Get the indices of the distal nodes of primary roots.",
            ),
            TraitDef(
                name="primary_angle_distal",
                fn=get_root_angle,
                input_traits=["primary_max_length_pts", "primary_distal_node_ind"],
                scalar=True,
                include_in_csv=True,
                kwargs={"proximal": False, "base_ind": 0},
                description="Array of primary distal angles in degrees `(instances,)`.",
            ),
            TraitDef(
                name="primary_length",
                fn=get_root_lengths,
                input_traits=["primary_max_length_pts"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of primary root length.",
            ),
            TraitDef(
                name="primary_base_pt",
                fn=get_bases,
                input_traits=["primary_max_length_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"monocots": False},
                description="Primary root base point.",
            ),
            TraitDef(
                name="primary_tip_pt",
                fn=get_tips,
                input_traits=["primary_max_length_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Primary root tip point.",
            ),
            TraitDef(
                name="network_length_lower",
                fn=get_network_distribution,
                input_traits=["primary_max_length_pts", "lateral_pts", "bounding_box"],
                scalar=True,
                include_in_csv=True,
                kwargs={"fraction": self.network_fraction, "monocots": False},
                description="Scalar of the root network length in the lower fraction "
                "of the plant.",
            ),
            TraitDef(
                name="lateral_base_xs",
                fn=get_base_xs,
                input_traits=["lateral_base_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={"monocots": False},
                description="Array of the x-coordinates of lateral bases "
                "`(instances,)`.",
            ),
            TraitDef(
                name="lateral_base_ys",
                fn=get_base_ys,
                input_traits=["lateral_base_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={"monocots": False},
                description="Array of the y-coordinates of lateral bases "
                "`(instances,)`.",
            ),
            TraitDef(
                name="base_ct_density",
                fn=get_base_ct_density,
                input_traits=["primary_length", "lateral_base_pts"],
                scalar=True,
                include_in_csv=True,
                kwargs={"monocots": False},
                description="Scalar of base count density.",
            ),
            TraitDef(
                name="lateral_tip_xs",
                fn=get_tip_xs,
                input_traits=["lateral_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of the x-coordinates of lateral tips `(instance,)`.",
            ),
            TraitDef(
                name="lateral_tip_ys",
                fn=get_tip_ys,
                input_traits=["lateral_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of the y-coordinates of lateral tips `(instance,)`.",
            ),
            TraitDef(
                name="network_distribution_ratio",
                fn=get_network_distribution_ratio,
                input_traits=[
                    "primary_length",
                    "lateral_lengths",
                    "network_length_lower",
                ],
                scalar=True,
                include_in_csv=True,
                kwargs={"fraction": self.network_fraction, "monocots": False},
                description="Scalar of ratio of the root network length in the lower "
                "fraction of the plant over all root length.",
            ),
            TraitDef(
                name="network_length",
                fn=get_network_length,
                input_traits=["primary_length", "lateral_lengths"],
                scalar=True,
                include_in_csv=False,
                kwargs={"monocots": False},
                description="Scalar of all roots network length.",
            ),
            TraitDef(
                name="primary_base_pt_y",
                fn=get_base_ys,
                input_traits=["primary_base_pt"],
                scalar=True,
                include_in_csv=False,
                kwargs={"monocots": False},
                description="Y-coordinate of the primary root base node.",
            ),
            TraitDef(
                name="primary_tip_pt_y",
                fn=get_tip_ys,
                input_traits=["primary_tip_pt"],
                scalar=True,
                include_in_csv=False,
                kwargs={},
                description="Y-coordinate of the primary root tip node.",
            ),
            TraitDef(
                name="ellipse_a",
                fn=get_ellipse_a,
                input_traits=["ellipse"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of semi-major axis length.",
            ),
            TraitDef(
                name="ellipse_b",
                fn=get_ellipse_b,
                input_traits=["ellipse"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of semi-minor axis length.",
            ),
            TraitDef(
                name="network_width_depth_ratio",
                fn=get_network_width_depth_ratio,
                input_traits=["bounding_box"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of bounding box width to depth ratio of root "
                "network.",
            ),
            TraitDef(
                name="chull_perimeter",
                fn=get_chull_perimeter,
                input_traits=["convex_hull"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of convex hull perimeter.",
            ),
            TraitDef(
                name="chull_area",
                fn=get_chull_area,
                input_traits=["convex_hull"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of convex hull area.",
            ),
            TraitDef(
                name="chull_max_width",
                fn=get_chull_max_width,
                input_traits=["convex_hull"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of convex hull maximum width.",
            ),
            TraitDef(
                name="chull_max_height",
                fn=get_chull_max_height,
                input_traits=["convex_hull"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of convex hull maximum height.",
            ),
            TraitDef(
                name="chull_line_lengths",
                fn=get_chull_line_lengths,
                input_traits=["convex_hull"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of line lengths connecting any two vertices on the"
                "convex hull.",
            ),
            TraitDef(
                name="base_length",
                fn=get_base_length,
                input_traits=["lateral_base_ys"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of the distance between the top and deepest base"
                "y-coordinates.",
            ),
            TraitDef(
                name="base_median_ratio",
                fn=get_base_median_ratio,
                input_traits=["lateral_base_ys", "primary_tip_pt_y"],
                scalar=True,
                include_in_csv=True,
                kwargs={"monocots": False},
                description="Scalar of base median ratio.",
            ),
            TraitDef(
                name="grav_index",
                fn=get_grav_index,
                input_traits=["primary_length", "primary_base_tip_dist"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of primary root gravity index.",
            ),
            TraitDef(
                name="base_length_ratio",
                fn=get_base_length_ratio,
                input_traits=["primary_length", "base_length"],
                scalar=True,
                include_in_csv=True,
                kwargs={"monocots": False},
                description="Scalar of base length ratio.",
            ),
            TraitDef(
                name="primary_base_tip_dist",
                fn=get_base_tip_dist,
                input_traits=["primary_base_pt", "primary_tip_pt"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of distance from primary root base to tip.",
            ),
            TraitDef(
                name="ellipse_ratio",
                fn=get_ellipse_ratio,
                input_traits=["ellipse"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of ratio of the minor to major lengths.",
            ),
            TraitDef(
                name="scanline_last_ind",
                fn=get_scanline_last_ind,
                input_traits=["scanline_intersection_counts"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of count_scanline_interaction index for the last"
                "interaction.",
            ),
            TraitDef(
                name="scanline_first_ind",
                fn=get_scanline_first_ind,
                input_traits=["scanline_intersection_counts"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of count_scanline_interaction index for the first"
                "interaction.",
            ),
        ]

        return trait_definitions

    def get_initial_frame_traits(self, plant: Series, frame_idx: int) -> Dict[str, Any]:
        """Return initial traits for a plant frame.

        Args:
            plant: The plant `Series` object.
            frame_idx: The index of the current frame.

        Returns:
            A dictionary of initial traits with keys:
                - "primary_pts": Array of primary root points.
                - "lateral_pts": Array of lateral root points.
        """
        # Get the root instances.
        primary, lateral = plant[frame_idx]
        gt_instances_pr = primary.user_instances + primary.unused_predictions
        gt_instances_lr = lateral.user_instances + lateral.unused_predictions

        # Convert the instances to numpy arrays.
        if len(gt_instances_lr) == 0:
            lateral_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        else:
            lateral_pts = np.stack([inst.numpy() for inst in gt_instances_lr], axis=0)

        if len(gt_instances_pr) == 0:
            primary_pts = np.array([[(np.nan, np.nan), (np.nan, np.nan)]])
        else:
            primary_pts = np.stack([inst.numpy() for inst in gt_instances_pr], axis=0)

        return {"primary_pts": primary_pts, "lateral_pts": lateral_pts}
