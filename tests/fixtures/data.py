import pytest


@pytest.fixture
def canola_folder():
    """Path to a folder with the predictions for 7 day old canola."""
    return "tests/data/canola_7do"


@pytest.fixture
def canola_h5():
    """Path to primary root image stack for 7 day old canola."""
    return "tests/data/canola_7do/919QDUH.h5"


@pytest.fixture
def canola_primary_slp():
    """Path to primary root predictions for 7 day old canola."""
    return "tests/data/canola_7do/919QDUH.primary.predictions.slp"


@pytest.fixture
def canola_lateral_slp():
    """Path to lateral root predictions for 7 day old canola."""
    return "tests/data/canola_7do/919QDUH.lateral.predictions.slp"


@pytest.fixture
def canola_traits_csv():
    """Path to computed traits csv for 7 day old canola."""
    return "tests/data/canola_7do/919QDUH.traits.csv"


@pytest.fixture
def canola_batch_traits_csv():
    """Path to computed batch traits csv for 7 day old canola."""
    return "tests/data/canola_7do/919QDUH.batch_traits.csv"


@pytest.fixture
def rice_pipeline_output_folder():
    """Path to the folder with the output of the rice pipeline."""
    return "tests/data/rice_3do_pipeline_output"


@pytest.fixture
def rice_10do_pipeline_output_folder():
    """Path to the folder with the output of the 10 day old rice pipeline."""
    return "tests/data/rice_10do_pipeline_output"


@pytest.fixture
def rice_folder():
    """Path to a folder with the predictions for 3 day old rice."""
    return "tests/data/rice_3do"


@pytest.fixture
def rice_h5():
    """Path to root image stack for 3 day old rice."""
    return "tests/data/rice_3do/YR39SJX.h5"


@pytest.fixture
def rice_long_slp():
    """Path to longest root predictions for 3 day old rice."""
    return "tests/data/rice_3do/YR39SJX.primary.predictions.slp"


@pytest.fixture
def rice_main_slp():
    """Path to main root predictions for 3 day old rice."""
    return "tests/data/rice_3do/YR39SJX.crown.predictions.slp"


@pytest.fixture
def rice_3do_0K9E8B1_traits_csv():
    """Path to computed traits csv for one sample of 3 day old rice."""
    return "tests/data/rice_3do/0K9E8BI.traits.csv"


@pytest.fixture
def rice_3do_YR39SJX_traits_csv():
    """Path to computed traits csv for one sample of 3 day old rice."""
    return "tests/data/rice_3do/YR39SJX.traits.csv"


@pytest.fixture
def rice_3do_batch_traits_csv():
    """Path to computed batch traits csv for 3 day old rice."""
    return "tests/data/rice_3do/rice_3do.batch_traits.csv"


@pytest.fixture
def rice_10do_folder():
    """Path to a folder with the predictions for 10 day old rice."""
    return "tests/data/rice_10do"


@pytest.fixture
def rice_main_10do_h5():
    """Path to root image stack for 10 day old rice."""
    return "tests/data/rice_10do/0K9E8BI.h5"


@pytest.fixture
def rice_main_10do_slp():
    """Path to main root predictions for 10 day old rice."""
    return "tests/data/rice_10do/0K9E8BI.crown.predictions.slp"


@pytest.fixture
def rice_10do_traits_csv():
    """Path to computed traits csv for 10 day old rice."""
    return "tests/data/rice_10do/0K9E8BI.traits.csv"


@pytest.fixture
def rice_10do_batch_traits_csv():
    """Path to computed batch traits csv for 10 day old rice."""
    return "tests/data/rice_10do/rice_10do.batch_traits.csv"


@pytest.fixture
def rice_10do_stunted_slp():
    """Path to stunted root predictions for 10 day old rice."""
    return "tests/data/rice_10do_pipeline_output/scan_7859150.model_221208_113552.multi_instance.n=574.root_crown.slp"


@pytest.fixture
def soy_folder():
    """Path to a folder with the predictions for 6 day old soy."""
    return "tests/data/soy_6do"


@pytest.fixture
def soy_h5():
    """Path to image stack for 6 day old soy."""
    return "tests/data/soy_6do/6PR6AA22JK.h5"


@pytest.fixture
def soy_primary_slp():
    """Path to primary root predictions for 6 day old soy."""
    return "tests/data/soy_6do/6PR6AA22JK.primary.predictions.slp"


@pytest.fixture
def soy_lateral_slp():
    """Path to lateral root predictions for 6 day old soy."""
    return "tests/data/soy_6do/6PR6AA22JK.lateral.predictions.slp"


@pytest.fixture
def soy_traits_csv():
    """Path to computed traits csv for 6 day old soy."""
    return "tests/data/soy_6do/6PR6AA22JK.traits.csv"


@pytest.fixture
def soy_batch_traits_csv():
    """Path to computed batch traits csv for 6 day old soy."""
    return "tests/data/soy_6do/6PR6AA22JK.batch_traits.csv"


@pytest.fixture
def multiple_arabidopsis_11do_folder():
    """Path to a folder with the predictions for 3, 11 day old arabidopsis."""
    return "tests/data/multiple_arabidopsis_11do"


@pytest.fixture
def multiple_arabidopsis_11do_h5():
    """Path to image stack for 11 day old arabidopsis."""
    return "tests/data/multiple_arabidopsis_11do/997_1.h5"


@pytest.fixture
def multiple_arabidopsis_11do_primary_slp():
    """Path to primary root predictions for 11 day old arabidopsis."""
    return "tests/data/multiple_arabidopsis_11do/997_1.primary.predictions.slp"


@pytest.fixture
def multiple_arabidopsis_11do_lateral_slp():
    """Path to lateral root predictions for 11 day old arabidopsis."""
    return "tests/data/multiple_arabidopsis_11do/997_1.lateral.predictions.slp"


@pytest.fixture
def multiple_arabidopsis_11do_csv():
    """Path to the CSV file with expected count and group information."""
    return "tests/data/multiple_arabidopsis_11do/merged_proofread_samples_03122024.csv"


@pytest.fixture
def multiple_arabidopsis_11do_group_batch_traits_csv_MultipleDicotPipeline():
    """Path to the CSV file with aggregated traits per group."""
    return "tests/data/multiple_arabidopsis_11do/multiple_dicot_pipeline/MultipleDicotPipeline.group_batch_summarized_traits.csv"


@pytest.fixture
def multiple_arabidopsis_11do_batch_traits_csv_MultipleDicotPipeline():
    """Path to the CSV file with batch traits for all arabidopsis samples."""
    return "tests/data/multiple_arabidopsis_11do/multiple_dicot_pipeline/MultipleDicotPipeline.batch_traits.csv"


@pytest.fixture
def multiple_arabidopsis_11do_batch_traits_MultiplePrimaryRootPipeline():
    """Path to the CSV file with expected count and group information."""
    return "tests/data/multiple_arabidopsis_11do/multiple_primary_root_pipeline/MultiplePrimaryRootPipeline.batch_traits.csv"


@pytest.fixture
def multiple_arabidopsis_11do_group_batch_traits_MultiplePrimaryRootPipeline():
    """Path to the CSV file with expected count and group information."""
    return "tests/data/multiple_arabidopsis_11do/multiple_primary_root_pipeline/MultiplePrimaryRootPipeline.group_batch_traits.csv"


@pytest.fixture
def rice_graviscan_folder_path():
    """Path to the folder containing rice graviscan (plate) data."""
    return "tests/data/rice_graviscan"


@pytest.fixture
def rice_graviscan_flattened_traits_csv_MultiplePrimaryRootPipeline():
    """Path to the CSV file with computed traits per instance for the rice graviscan (plate) dataset."""
    return "tests/data/rice_graviscan/plates_24hr.flattened_traits.csv"


@pytest.fixture
def sleap_roots_pipeline_output_folder():
    """Path to the folder with the output of the sleap_roots pipeline."""
    return "tests/data/sleap-roots-pipeline-outputs"
