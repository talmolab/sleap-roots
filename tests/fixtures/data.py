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
