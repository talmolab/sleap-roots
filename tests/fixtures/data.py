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
    return "tests/data/canola_7do/919QDUH.primary_multi_day.predictions.slp"


@pytest.fixture
def canola_lateral_slp():
    """Path to lateral root predictions for 7 day old canola."""
    return "tests/data/canola_7do/919QDUH.lateral_3_nodes.predictions.slp"


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
    return "tests/data/rice_3do/YR39SJX.longest_3do_6nodes.predictions.slp"


@pytest.fixture
def rice_main_slp():
    """Path to main root predictions for 3 day old rice."""
    return "tests/data/rice_3do/YR39SJX.main_3do_6nodes.predictions.slp"


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
    return "tests/data/soy_6do/6PR6AA22JK.primary_multi_day.predictions.slp"


@pytest.fixture
def soy_lateral_slp():
    """Path to lateral root predictions for 6 day old soy."""
    return "tests/data/soy_6do/6PR6AA22JK.lateral__nodes.predictions.slp"
