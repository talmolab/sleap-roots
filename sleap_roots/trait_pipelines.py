"""Extract traits in a pipeline based on a trait graph."""

import json
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import attrs
import networkx as nx
import numpy as np
import pandas as pd

from sleap_roots.angle import (
    get_node_ind,
    get_root_angle,
    get_vector_angles_from_gravity,
)
from sleap_roots.bases import (
    get_base_ct_density,
    get_base_length,
    get_base_length_ratio,
    get_base_median_ratio,
    get_base_tip_dist,
    get_base_xs,
    get_base_ys,
    get_bases,
    get_root_widths,
)
from sleap_roots.convhull import (
    get_chull_area,
    get_chull_intersection_vectors,
    get_chull_intersection_vectors_left,
    get_chull_intersection_vectors_right,
    get_chull_line_lengths,
    get_chull_max_height,
    get_chull_max_width,
    get_chull_perimeter,
    get_convhull,
    get_chull_areas_via_intersection,
    get_chull_area_via_intersection_below,
    get_chull_area_via_intersection_above,
)
from sleap_roots.ellipse import (
    fit_ellipse,
    get_ellipse_a,
    get_ellipse_b,
    get_ellipse_ratio,
)
from sleap_roots.lengths import get_curve_index, get_max_length_pts, get_root_lengths
from sleap_roots.networklength import (
    get_bbox,
    get_network_distribution,
    get_network_distribution_ratio,
    get_network_length,
    get_network_solidity,
    get_network_width_depth_ratio,
)
from sleap_roots.points import (
    associate_lateral_to_primary,
    filter_plants_with_unexpected_ct,
    filter_roots_with_nans,
    get_all_pts_array,
    get_count,
    get_filtered_lateral_pts,
    get_filtered_primary_pts,
    get_nodes,
    join_pts,
)
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


class NumpyArrayEncoder(json.JSONEncoder):
    """Custom encoder for NumPy array types."""

    def default(self, obj):
        """Serialize NumPy arrays to lists.

        Args:
            obj: The object to serialize.

        Returns:
            A list representation of the NumPy array.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64):
            return int(obj)
        # Let the base class default method raise the TypeError
        return json.JSONEncoder.default(self, obj)


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

    @property
    def csv_traits_multiple_plants(self) -> List[str]:
        """List of frame-level traits to include in the CSV for multiple plants."""
        csv_traits = []
        for trait in self.traits:
            if trait.include_in_csv:
                csv_traits.append(trait.name)
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
        output_dir: str = ".",
        csv_suffix: str = ".traits.csv",
        return_non_scalar: bool = False,
    ) -> pd.DataFrame:
        """Compute traits for a plant.

        Args:
            plant: The plant image series as a `Series` object.
            write_csv: A boolean value. If True, it writes per plant detailed
                CSVs with traits for every instance on every frame.
            output_dir: The directory to write the CSV files to.
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
            csv_name = Path(output_dir) / f"{plant.series_name}{csv_suffix}"
            traits[["plant_name", "frame_idx"] + self.csv_traits].to_csv(
                csv_name, index=False
            )

        if return_non_scalar:
            return traits
        else:
            return traits[["plant_name", "frame_idx"] + self.csv_traits]

    def compute_multiple_dicots_traits(
        self,
        series: Series,
        write_json: bool = False,
        json_suffix: str = ".all_frames_traits.json",
        write_csv: bool = False,
        csv_suffix: str = ".all_frames_summary.csv",
    ):
        """Computes plant traits for pipelines with multiple plants over all frames in a series.

        Args:
            series: The Series object containing the primary and lateral root points.
            write_json: Whether to write the aggregated traits to a JSON file. Default is False.
            json_suffix: The suffix to append to the JSON file name. Default is ".all_frames_traits.json".
            write_csv: Whether to write the summary statistics to a CSV file. Default is False.
            csv_suffix: The suffix to append to the CSV file name. Default is ".all_frames_summary.csv".

        Returns:
            A dictionary containing the series name, group, qc_fail, aggregated traits, and summary statistics.
        """
        # Initialize the return structure with the series name and group
        result = {
            "series": str(series.series_name),
            "group": str(series.group),
            "qc_fail": series.qc_fail,
            "traits": {},
            "summary_stats": {},
        }

        # Check if the series has frames to process
        if len(series) == 0:
            print(f"Series '{series.series_name}' contains no frames to process.")
            # Return early with the initialized structure
            return result

        # Initialize a separate dictionary to hold the aggregated traits across all frames
        aggregated_traits = {}

        # Iterate over frames in series
        for frame in range(len(series)):
            # Get initial points and number of plants per frame
            initial_frame_traits = self.get_initial_frame_traits(series, frame)
            # Compute initial associations and perform filter operations
            frame_traits = self.compute_frame_traits(initial_frame_traits)

            # Instantiate DicotPipeline
            dicot_pipeline = DicotPipeline()

            # Extract the plant associations for this frame
            associations = frame_traits["plant_associations_dict"]

            for primary_idx, assoc in associations.items():
                primary_pts = assoc["primary_points"]
                lateral_pts = assoc["lateral_points"]
                # Get the initial frame traits for this plant using the primary and lateral points
                initial_frame_traits = {
                    "primary_pts": primary_pts,
                    "lateral_pts": lateral_pts,
                }
                # Use the dicot pipeline to compute the plant traits on this frame
                plant_traits = dicot_pipeline.compute_frame_traits(initial_frame_traits)

                # For each plant's traits in the frame
                for trait_name, trait_value in plant_traits.items():
                    # Not all traits are added to the aggregated traits dictionary
                    if trait_name in dicot_pipeline.csv_traits_multiple_plants:
                        if trait_name not in aggregated_traits:
                            # Initialize the trait array if it's the first frame
                            aggregated_traits[trait_name] = [np.atleast_1d(trait_value)]
                        else:
                            # Append new trait values for subsequent frames
                            aggregated_traits[trait_name].append(
                                np.atleast_1d(trait_value)
                            )

        # After processing, update the result dictionary with computed traits
        for trait, arrays in aggregated_traits.items():
            aggregated_traits[trait] = np.concatenate(arrays, axis=0)
        result["traits"] = aggregated_traits

        # Write to JSON if requested
        if write_json:
            json_name = f"{series.series_name}{json_suffix}"
            try:
                with open(json_name, "w") as f:
                    json.dump(
                        result, f, cls=NumpyArrayEncoder, ensure_ascii=False, indent=4
                    )
                print(f"Aggregated traits saved to {json_name}")
            except IOError as e:
                print(f"Error writing JSON file '{json_name}': {e}")

        # Compute summary statistics and update result
        summary_stats = {}
        for trait_name, trait_values in aggregated_traits.items():
            trait_stats = get_summary(trait_values, prefix=f"{trait_name}_")
            summary_stats.update(trait_stats)
        result["summary_stats"] = summary_stats

        # Optionally write summary stats to CSV
        if write_csv:
            csv_name = f"{series.series_name}{csv_suffix}"
            try:
                summary_df = pd.DataFrame([summary_stats])
                summary_df.insert(0, "series", series.series_name)
                summary_df.to_csv(csv_name, index=False)
                print(f"Summary statistics saved to {csv_name}")
            except IOError as e:
                print(f"Failed to write CSV file '{csv_name}': {e}")

        # Return the final result structure
        return result

    def compute_multiple_dicots_traits_for_groups(
        self,
        series_list: List[Series],
        output_dir: str = "grouped_traits",
        write_json: bool = False,
        json_suffix: str = ".grouped_traits.json",
        write_csv: bool = False,
        csv_suffix: str = ".grouped_summary.csv",
    ) -> List[
        Dict[str, Union[str, List[str], Dict[str, Union[List[float], np.ndarray]]]]
    ]:
        """Aggregates plant traits over groups of samples.

        Args:
            series_list: A list of Series objects containing the primary and lateral root points for each sample.
            output_dir: The directory to write the JSON and CSV files to. Default is "grouped_traits".
            write_json: Whether to write the aggregated traits to a JSON file. Default is False.
            json_suffix: The suffix to append to the JSON file name. Default is ".grouped_traits.json".
            write_csv: Whether to write the summary statistics to a CSV file. Default is False.
            csv_suffix: The suffix to append to the CSV file name. Default is ".grouped_summary.csv".

        Returns:
            A list of dictionaries containing the aggregated traits and summary statistics for each group.
        """
        # Input Validation
        if not isinstance(series_list, list) or not all(
            isinstance(series, Series) for series in series_list
        ):
            raise ValueError("series_list must be a list of Series objects.")

        # Group series by their group property
        series_groups = {}
        for series in series_list:
            # Exclude series with qc_fail flag set to 1
            if int(series.qc_fail) == 1:
                print(f"Skipping series '{series.series_name}' due to qc_fail flag.")
                continue
            # Get the group name from the series object
            group_name = str(series.group)
            if group_name not in series_groups:
                series_groups[group_name] = {"names": [], "series": []}
            # Store series names and objects in the dictionary
            series_groups[group_name]["names"].append(str(series.series_name))
            series_groups[group_name]["series"].append(series)  # Store Series objects

        # Initialize the list to hold the results for each group
        grouped_results = []
        # Iterate over each group of series
        for group_name, group_data in series_groups.items():
            # Initialize the return structure with the group name
            group_result = {
                "group": group_name,
                "series": group_data["names"],  # Use series names
                "traits": {},
            }

            # Aggregate traits over all samples in the group
            aggregated_traits = {}
            # Iterate over each series in the group
            for series in group_data["series"]:
                print(f"Processing series '{series.series_name}'")
                # Get the trait results for each series in the group
                result = self.compute_multiple_dicots_traits(
                    series=series, write_json=False, write_csv=False
                )
                # Aggregate the series traits into the group traits
                for trait, values in result["traits"].items():
                    # Ensure values are at least 1D
                    values = np.atleast_1d(values)
                    if trait not in aggregated_traits:
                        aggregated_traits[trait] = values
                    else:
                        # Concatenate the current values with the existing array
                        aggregated_traits[trait] = np.concatenate(
                            (aggregated_traits[trait], values)
                        )

            group_result["traits"] = aggregated_traits
            print(f"Finished processing group '{group_name}'")

            # Write to JSON if requested
            if write_json:
                # Make the output directory if it doesn't exist
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                # Construct the JSON file name
                json_name = f"{group_name}{json_suffix}"
                # Join the output directory with the JSON file name
                json_path = Path(output_dir) / json_name
                try:
                    with open(json_path, "w") as f:
                        json.dump(
                            group_result,
                            f,
                            cls=NumpyArrayEncoder,
                            ensure_ascii=False,
                            indent=4,
                        )
                    print(
                        f"Aggregated traits for group {group_name} saved to {str(json_path)}"
                    )
                except IOError as e:
                    print(f"Error writing JSON file '{str(json_path)}': {e}")

            # Compute summary statistics
            summary_stats = {}
            for trait, trait_values in aggregated_traits.items():
                trait_stats = get_summary(trait_values, prefix=f"{trait}_")
                summary_stats.update(trait_stats)

            group_result["summary_stats"] = summary_stats

            # Write summary stats to CSV if requested
            if write_csv:
                # Make the output directory if it doesn't exist
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                # Construct the CSV file name
                csv_name = f"{group_name}{csv_suffix}"
                # Join the output directory with the CSV file name
                csv_path = Path(output_dir) / csv_name
                try:
                    summary_df = pd.DataFrame([summary_stats])
                    summary_df.insert(0, "genotype", group_name)
                    summary_df.to_csv(csv_path, index=False)
                    print(
                        f"Summary statistics for group {group_name} saved to {str(csv_path)}"
                    )
                except IOError as e:
                    print(f"Failed to write CSV file '{str(csv_path)}': {e}")

            # Append the group result to the list of results
            grouped_results.append(group_result)

        return grouped_results

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
            print(f"Processing series: {plant.series_name}")
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
            print(f"Batch traits saved to {csv_path}")
        return all_traits

    def compute_batch_multiple_dicots_traits(
        self,
        all_series: List[Series],
        write_csv: bool = False,
        csv_path: str = "traits.csv",
    ) -> pd.DataFrame:
        """Compute traits for a batch of series with multiple dicots.

        Args:
            all_series: List of `Series` objects.
            write_csv: If `True`, write the computed traits to a CSV file.
            csv_path: Path to write the CSV file to.

        Returns:
            A pandas DataFrame of computed traits summarized over all frames of each
            series. The resulting dataframe will have a row for each series and a column
            for each series-level summarized trait.

            Summarized traits are prefixed with the trait name and an underscore,
            followed by the summary statistic.
        """
        all_series_summaries = []

        for series in all_series:
            print(f"Processing series '{series.series_name}'")
            # Use the updated function and access its return value
            series_result = self.compute_multiple_dicots_traits(
                series, write_json=False, write_csv=False
            )
            # Prepare the series-level summary.
            series_summary = {
                "series_name": series_result["series"],
                **series_result["summary_stats"],  # Unpack summary_stats
            }
            all_series_summaries.append(series_summary)

        # Convert list of dictionaries to a DataFrame
        all_series_summaries_df = pd.DataFrame(all_series_summaries)

        # Write to CSV if requested
        if write_csv:
            all_series_summaries_df.to_csv(csv_path, index=False)
            print(f"Computed traits for all series saved to {csv_path}")

        return all_series_summaries_df

    def compute_batch_multiple_dicots_traits_for_groups(
        self,
        all_series: List[Series],
        output_dir: str = "grouped_traits",
        write_json: bool = False,
        write_csv: bool = False,
        csv_path: str = "group_summarized_traits.csv",
    ) -> pd.DataFrame:
        """Compute traits for a batch of grouped series with multiple dicots.

        Args:
            all_series: List of `Series` objects.
            output_dir: The directory to write the JSON and CSV files to. Default is "grouped_traits".
            write_json: If `True`, write each set of group traits to a JSON file.
            write_csv: If `True`, write the computed traits to a CSV file.
            csv_path: Path to write the CSV file to.

        Returns:
            A pandas DataFrame of computed traits summarized over all frames of each
            group. The resulting dataframe will have a row for each series and a column
            for each series-level summarized trait.

            Summarized traits are prefixed with the trait name and an underscore,
            followed by the summary statistic.
        """
        # Check if the input list is empty
        if not all_series:
            raise ValueError("The input list 'all_series' is empty.")

        try:
            # Compute traits for each group of series
            grouped_results = self.compute_multiple_dicots_traits_for_groups(
                all_series,
                output_dir=output_dir,
                write_json=write_json,
                write_csv=False,
            )
        except Exception as e:
            raise RuntimeError(f"Error computing traits for groups: {e}")

        # Prepare the list of dictionaries for the DataFrame
        all_group_summaries = []
        for group_result in grouped_results:
            # Validate the expected key exists in the result
            if "summary_stats" not in group_result:
                raise KeyError(
                    "Expected key 'summary_stats' not found in group result."
                )

            # Assuming 'group' key exists in group_result and it indicates the genotype
            genotype = group_result.get(
                "group", "Unknown Genotype"
            )  # Default to "Unknown Genotype" if not found

            # Start with a dictionary containing the genotype
            group_summary = {"genotype": genotype}

            # Add each trait statistic from the summary_stats dictionary to the group_summary
            # This assumes summary_stats is a dictionary where keys are trait names and values are the statistics
            for trait, statistic in group_result["summary_stats"].items():
                group_summary[trait] = statistic

            all_group_summaries.append(group_summary)

        # Create a DataFrame from the list of dictionaries
        all_group_summaries_df = pd.DataFrame(all_group_summaries)

        # Write to CSV if requested
        if write_csv:
            try:
                all_group_summaries_df.to_csv(csv_path, index=False)
                print(f"Computed traits for all groups saved to {csv_path}")
            except Exception as e:
                raise IOError(f"Failed to write computed traits to CSV: {e}")

        return all_group_summaries_df


@attrs.define
class DicotPipeline(Pipeline):
    """Pipeline for computing traits for dicot plants (primary + lateral roots).

    Attributes:
        img_height: Image height.
        root_width_tolerance: Difference in projection norm between right and left side.
        n_scanlines: Number of scan lines, np.nan for no interaction.
        network_fraction: Length found in the lower fraction value of the network.
    """

    img_height: int = 1080
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
                kwargs={},
                description="Landmark points within a given frame as a flat array"
                "of coordinates.",
            ),
            TraitDef(
                name="pts_list",
                fn=join_pts,
                input_traits=["primary_max_length_pts", "lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="A list of instance arrays, each having shape `(nodes, 2)`.",
            ),
            TraitDef(
                name="root_widths",
                fn=get_root_widths,
                input_traits=["primary_max_length_pts", "lateral_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={
                    "tolerance": self.root_width_tolerance,
                    "return_inds": False,
                },
                description="Estimate root width using bases of lateral roots.",
            ),
            TraitDef(
                name="lateral_count",
                fn=get_count,
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
                kwargs={},
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
                input_traits=["pts_list"],
                scalar=False,
                include_in_csv=True,
                kwargs={
                    "height": self.img_height,
                    "n_line": self.n_scanlines,
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
                kwargs={},
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
                input_traits=["pts_list", "bounding_box"],
                scalar=True,
                include_in_csv=True,
                kwargs={
                    "fraction": self.network_fraction,
                },
                description="Scalar of the root network length in the lower fraction "
                "of the plant.",
            ),
            TraitDef(
                name="lateral_base_xs",
                fn=get_base_xs,
                input_traits=["lateral_base_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Get x coordinates of the base of each lateral root.",
            ),
            TraitDef(
                name="lateral_base_ys",
                fn=get_base_ys,
                input_traits=["lateral_base_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of the y-coordinates of lateral bases "
                "`(instances,)`.",
            ),
            TraitDef(
                name="base_ct_density",
                fn=get_base_ct_density,
                input_traits=["primary_length", "lateral_base_pts"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
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
                input_traits=["network_length", "network_length_lower"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of ratio of the root network length in the lower "
                "fraction of the plant over all root length.",
            ),
            TraitDef(
                name="network_length",
                fn=get_network_length,
                input_traits=["primary_length", "lateral_lengths"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of all roots network length.",
            ),
            TraitDef(
                name="primary_base_pt_y",
                fn=get_base_ys,
                input_traits=["primary_base_pt"],
                scalar=True,
                include_in_csv=False,
                kwargs={},
                description="Y-coordinate of the primary root base node.",
            ),
            TraitDef(
                name="primary_tip_pt_y",
                fn=get_tip_ys,
                input_traits=["primary_tip_pt"],
                scalar=True,
                include_in_csv=True,
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
                kwargs={},
                description="Scalar of base median ratio.",
            ),
            TraitDef(
                name="curve_index",
                fn=get_curve_index,
                input_traits=["primary_length", "primary_base_tip_dist"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of primary root curvature index.",
            ),
            TraitDef(
                name="base_length_ratio",
                fn=get_base_length_ratio,
                input_traits=["primary_length", "base_length"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
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
        primary_pts = plant.get_primary_points(frame_idx)
        lateral_pts = plant.get_lateral_points(frame_idx)
        return {"primary_pts": primary_pts, "lateral_pts": lateral_pts}


@attrs.define
class YoungerMonocotPipeline(Pipeline):
    """Pipeline for computing traits for young monocot plants (primary + crown roots).

    Attributes:
        img_height: Image height.
        n_scanlines: Number of scan lines, np.nan for no interaction.
        network_fraction: Lower fraction value. Defaults to 2/3.
    """

    img_height: int = 1080
    n_scanlines: int = 50
    network_fraction: float = 2 / 3

    def define_traits(self) -> List[TraitDef]:
        """Define the trait computation pipeline for younger monocot plants."""
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
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Crown root points within a given frame as a flat array"
                "of coordinates.",
            ),
            TraitDef(
                name="crown_count",
                fn=get_count,
                input_traits=["crown_pts"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Get the number of crown roots.",
            ),
            TraitDef(
                name="crown_proximal_node_inds",
                fn=get_node_ind,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"proximal": True},
                description="Get the indices of the proximal nodes of crown roots.",
            ),
            TraitDef(
                name="crown_distal_node_inds",
                fn=get_node_ind,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"proximal": False},
                description="Get the indices of the distal nodes of crown roots.",
            ),
            TraitDef(
                name="crown_lengths",
                fn=get_root_lengths,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of crown root lengths of shape `(instances,)`.",
            ),
            TraitDef(
                name="crown_base_pts",
                fn=get_bases,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Array of crown bases `(instances, (x, y))`.",
            ),
            TraitDef(
                name="crown_tip_pts",
                fn=get_tips,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Array of crown tips `(instances, (x, y))`.",
            ),
            TraitDef(
                name="scanline_intersection_counts",
                fn=count_scanline_intersections,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={
                    "height": self.img_height,
                    "n_line": self.n_scanlines,
                },
                description="Array of intersections of each scanline"
                "`(n_scanlines,)`.",
            ),
            TraitDef(
                name="crown_angles_distal",
                fn=get_root_angle,
                input_traits=["crown_pts", "crown_distal_node_inds"],
                scalar=False,
                include_in_csv=True,
                kwargs={"proximal": False, "base_ind": 0},
                description="Array of crown distal angles in degrees `(instances,)`.",
            ),
            TraitDef(
                name="crown_angles_proximal",
                fn=get_root_angle,
                input_traits=["crown_pts", "crown_proximal_node_inds"],
                scalar=False,
                include_in_csv=True,
                kwargs={"proximal": True, "base_ind": 0},
                description="Array of crown proximal angles in degrees "
                "`(instances,)`.",
            ),
            TraitDef(
                name="network_length_lower",
                fn=get_network_distribution,
                input_traits=[
                    "crown_pts",
                    "bounding_box",
                ],
                scalar=True,
                include_in_csv=True,
                kwargs={
                    "fraction": self.network_fraction,
                },
                description="Scalar of the root network length in the lower fraction "
                "of the plant.",
            ),
            TraitDef(
                name="ellipse",
                fn=fit_ellipse,
                input_traits=["crown_pts"],
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
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Tuple of four parameters in bounding box.",
            ),
            TraitDef(
                name="convex_hull",
                fn=get_convhull,
                input_traits=["crown_pts"],
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
                input_traits=[
                    "primary_max_length_pts",
                    "primary_proximal_node_ind",
                ],
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
                kwargs={},
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
                name="crown_tip_xs",
                fn=get_tip_xs,
                input_traits=["crown_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of the x-coordinates of crown tips `(instance,)`.",
            ),
            TraitDef(
                name="crown_tip_ys",
                fn=get_tip_ys,
                input_traits=["crown_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of the y-coordinates of crown tips `(instance,)`.",
            ),
            TraitDef(
                name="network_distribution_ratio",
                fn=get_network_distribution_ratio,
                input_traits=[
                    "network_length",
                    "network_length_lower",
                ],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of ratio of the root network length in the lower"
                "fraction of the plant over all root length.",
            ),
            TraitDef(
                name="network_length",
                fn=get_network_length,
                input_traits=["crown_lengths"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of all roots network length.",
            ),
            TraitDef(
                name="crown_base_tip_dists",
                fn=get_base_tip_dist,
                input_traits=["crown_base_pts", "crown_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Straight-line distance(s) from the base(s) to the"
                "tip(s) of the crown root(s).",
            ),
            TraitDef(
                name="crown_curve_indices",
                fn=get_curve_index,
                input_traits=["crown_lengths", "crown_base_tip_dists"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Curvature index for each crown root.",
            ),
            TraitDef(
                name="network_solidity",
                fn=get_network_solidity,
                input_traits=["network_length", "chull_area"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of the total network length divided by the"
                "network convex hull area.",
            ),
            TraitDef(
                name="primary_tip_pt_y",
                fn=get_tip_ys,
                input_traits=["primary_tip_pt"],
                scalar=True,
                include_in_csv=True,
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
                name="curve_index",
                fn=get_curve_index,
                input_traits=["primary_length", "primary_base_tip_dist"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of primary root curvature index.",
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
                - "crown_pts": Array of crown root points.
        """
        primary_pts = plant.get_primary_points(frame_idx)
        crown_pts = plant.get_crown_points(frame_idx)
        return {"primary_pts": primary_pts, "crown_pts": crown_pts}


@attrs.define
class OlderMonocotPipeline(Pipeline):
    """Pipeline for computing traits for older monocot plants (crown roots only).

    Attributes:
        img_height: Image height.
        n_scanlines: Number of scan lines, np.nan for no interaction.
        network_fraction: Lower fraction value. Defaults to 2/3.
    """

    img_height: int = 1080
    n_scanlines: int = 50
    network_fraction: float = 2 / 3

    def define_traits(self) -> List[TraitDef]:
        """Define the trait computation pipeline for older monocot plants (crown roots)."""
        trait_definitions = [
            TraitDef(
                name="pts_all_array",
                fn=get_all_pts_array,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Landmark points within a given frame as a flat array"
                "of coordinates.",
            ),
            TraitDef(
                name="crown_count",
                fn=get_count,
                input_traits=["crown_pts"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Get the number of crown roots.",
            ),
            TraitDef(
                name="crown_proximal_node_inds",
                fn=get_node_ind,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"proximal": True},
                description="Get the indices of the proximal nodes of crown roots.",
            ),
            TraitDef(
                name="crown_distal_node_inds",
                fn=get_node_ind,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"proximal": False},
                description="Get the indices of the distal nodes of crown roots.",
            ),
            TraitDef(
                name="crown_lengths",
                fn=get_root_lengths,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of crown root lengths of shape `(instances,)`.",
            ),
            TraitDef(
                name="crown_base_pts",
                fn=get_bases,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Array of crown bases `(instances, (x, y))`.",
            ),
            TraitDef(
                name="crown_tip_pts",
                fn=get_tips,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Array of crown tips `(instances, (x, y))`.",
            ),
            TraitDef(
                name="scanline_intersection_counts",
                fn=count_scanline_intersections,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={
                    "height": self.img_height,
                    "n_line": self.n_scanlines,
                },
                description="Array of intersections of each scanline"
                "`(n_scanlines,)`.",
            ),
            TraitDef(
                name="crown_angles_distal",
                fn=get_root_angle,
                input_traits=["crown_pts", "crown_distal_node_inds"],
                scalar=False,
                include_in_csv=True,
                kwargs={"proximal": False, "base_ind": 0},
                description="Array of crown distal angles in degrees `(instances,)`.",
            ),
            TraitDef(
                name="crown_angles_proximal",
                fn=get_root_angle,
                input_traits=["crown_pts", "crown_proximal_node_inds"],
                scalar=False,
                include_in_csv=True,
                kwargs={"proximal": True, "base_ind": 0},
                description="Array of crown proximal angles in degrees "
                "`(instances,)`.",
            ),
            TraitDef(
                name="network_length_lower",
                fn=get_network_distribution,
                input_traits=[
                    "crown_pts",
                    "bounding_box",
                ],
                scalar=True,
                include_in_csv=True,
                kwargs={
                    "fraction": self.network_fraction,
                },
                description="Scalar of the root network length in the lower fraction "
                "of the plant.",
            ),
            TraitDef(
                name="ellipse",
                fn=fit_ellipse,
                input_traits=["crown_pts"],
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
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Tuple of four parameters representing bounding box.",
            ),
            TraitDef(
                name="convex_hull",
                fn=get_convhull,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Convex hull of the crown points.",
            ),
            TraitDef(
                name="crown_tip_xs",
                fn=get_tip_xs,
                input_traits=["crown_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of the x-coordinates of crown tips `(instance,)`.",
            ),
            TraitDef(
                name="crown_tip_ys",
                fn=get_tip_ys,
                input_traits=["crown_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Array of the y-coordinates of crown tips `(instance,)`.",
            ),
            TraitDef(
                name="network_distribution_ratio",
                fn=get_network_distribution_ratio,
                input_traits=[
                    "network_length",
                    "network_length_lower",
                ],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of ratio of the root network length in the lower"
                "fraction of the plant over all root length.",
            ),
            TraitDef(
                name="network_length",
                fn=get_network_length,
                input_traits=["crown_lengths"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of all roots network length.",
            ),
            TraitDef(
                name="crown_base_tip_dists",
                fn=get_base_tip_dist,
                input_traits=["crown_base_pts", "crown_tip_pts"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Straight-line distance(s) from the base(s) to the"
                "tip(s) of the crown root(s).",
            ),
            TraitDef(
                name="crown_curve_indices",
                fn=get_curve_index,
                input_traits=["crown_lengths", "crown_base_tip_dists"],
                scalar=False,
                include_in_csv=True,
                kwargs={},
                description="Curvature index for each crown root.",
            ),
            TraitDef(
                name="network_solidity",
                fn=get_network_solidity,
                input_traits=["network_length", "chull_area"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of the total network length divided by the"
                "network convex hull area.",
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
            TraitDef(
                name="crown_r1_pts",
                fn=get_nodes,
                input_traits=["crown_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={"node_index": 1},
                description="Array of crown bases `(instances, (x, y))`.",
            ),
            TraitDef(
                name="chull_r1_intersection_vectors",
                fn=get_chull_intersection_vectors,
                input_traits=[
                    "crown_base_pts",
                    "crown_r1_pts",
                    "crown_pts",
                    "convex_hull",
                ],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="A tuple containing vectors from the top left point to the"
                "left intersection point, and from the top right point to the right"
                "intersection point with the convex hull.",
            ),
            TraitDef(
                name="chull_r1_left_intersection_vector",
                fn=get_chull_intersection_vectors_left,
                input_traits=["chull_r1_intersection_vectors"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Vector from the base point to the left"
                "intersection point with the convex hull.",
            ),
            TraitDef(
                name="chull_r1_right_intersection_vector",
                fn=get_chull_intersection_vectors_right,
                input_traits=["chull_r1_intersection_vectors"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Vector from the base point to the right"
                "intersection point with the convex hull.",
            ),
            TraitDef(
                name="angle_chull_r1_left_intersection_vector",
                fn=get_vector_angles_from_gravity,
                input_traits=["chull_r1_left_intersection_vector"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Angle of the left intersection vector from gravity.",
            ),
            TraitDef(
                name="angle_chull_r1_right_intersection_vector",
                fn=get_vector_angles_from_gravity,
                input_traits=["chull_r1_right_intersection_vector"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Angle of the right intersection vector from gravity.",
            ),
            TraitDef(
                name="chull_areas_r1_intersection",
                fn=get_chull_areas_via_intersection,
                input_traits=["crown_r1_pts", "crown_pts", "convex_hull"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Tuple of the convex hull areas above and below the r1"
                "intersection.",
            ),
            TraitDef(
                name="chull_area_above_r1_intersection",
                fn=get_chull_area_via_intersection_above,
                input_traits=["chull_areas_r1_intersection"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of the convex hull area above the r1 intersection.",
            ),
            TraitDef(
                name="chull_area_below_r1_intersection",
                fn=get_chull_area_via_intersection_below,
                input_traits=["chull_areas_r1_intersection"],
                scalar=True,
                include_in_csv=True,
                kwargs={},
                description="Scalar of the convex hull area below the r1 intersection.",
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
                - "crown_pts": Array of crown root points.
        """
        crown_pts = plant.get_crown_points(frame_idx)
        return {"crown_pts": crown_pts}


@attrs.define
class MultipleDicotPipeline(Pipeline):
    """Pipeline for computing traits for multiple dicot plants."""

    def define_traits(self) -> List[TraitDef]:
        """Define the trait computation pipeline for primary roots."""
        trait_definitions = [
            TraitDef(
                name="primary_pts_no_nans",
                fn=filter_roots_with_nans,
                input_traits=["primary_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Primary roots without any NaNs.",
            ),
            TraitDef(
                name="lateral_pts_no_nans",
                fn=filter_roots_with_nans,
                input_traits=["lateral_pts"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Lateral roots without any NaNs.",
            ),
            TraitDef(
                name="filtered_pts_expected_plant_ct",
                fn=filter_plants_with_unexpected_ct,
                input_traits=[
                    "primary_pts_no_nans",
                    "lateral_pts_no_nans",
                    "expected_plant_ct",
                ],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Tuple of filtered points with expected plant count.",
            ),
            TraitDef(
                name="primary_pts_expected_plant_ct",
                fn=get_filtered_primary_pts,
                input_traits=["filtered_pts_expected_plant_ct"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Filtered primary root points with expected plant count.",
            ),
            TraitDef(
                name="lateral_pts_expected_plant_ct",
                fn=get_filtered_lateral_pts,
                input_traits=["filtered_pts_expected_plant_ct"],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Filtered lateral root points with expected plant count.",
            ),
            TraitDef(
                name="plant_associations_dict",
                fn=associate_lateral_to_primary,
                input_traits=[
                    "primary_pts_expected_plant_ct",
                    "lateral_pts_expected_plant_ct",
                ],
                scalar=False,
                include_in_csv=False,
                kwargs={},
                description="Dictionary of plant associations.",
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
                - "expected_ct": Expected number of plants as a float.
        """
        primary_pts = plant.get_primary_points(frame_idx)
        lateral_pts = plant.get_lateral_points(frame_idx)
        expected_plant_ct = plant.expected_count
        return {
            "primary_pts": primary_pts,
            "lateral_pts": lateral_pts,
            "expected_plant_ct": expected_plant_ct,
        }
