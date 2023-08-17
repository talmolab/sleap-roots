import numpy as np
from sleap_roots.summary import get_summary


def test_get_summary():
    summary = get_summary(np.array([-1, 0, 1]))
    assert summary["min"] == -1
    assert summary["max"] == 1
    assert summary["mean"] == 0
    assert summary["median"] == 0
    assert summary["std"] == np.std([-1, 0, 1])
    assert summary["p5"] == np.percentile([-1, 0, 1], 5)
    assert summary["p25"] == np.percentile([-1, 0, 1], 25)
    assert summary["p75"] == np.percentile([-1, 0, 1], 75)
    assert summary["p95"] == np.percentile([-1, 0, 1], 95)


def test_get_summary_empty():
    summary = get_summary([])
    np.testing.assert_array_equal(summary["min"], np.nan)
    np.testing.assert_array_equal(summary["max"], np.nan)
    np.testing.assert_array_equal(summary["mean"], np.nan)
    np.testing.assert_array_equal(summary["median"], np.nan)
    np.testing.assert_array_equal(summary["std"], np.nan)
    np.testing.assert_array_equal(summary["p5"], np.nan)
    np.testing.assert_array_equal(summary["p25"], np.nan)
    np.testing.assert_array_equal(summary["p75"], np.nan)
    np.testing.assert_array_equal(summary["p95"], np.nan)


def test_get_summary_prefix():
    summary = get_summary([], prefix="test_")
    assert "test_min" in summary
    assert "test_max" in summary
    assert "test_mean" in summary
    assert "test_median" in summary
    assert "test_std" in summary
    assert "test_p5" in summary
    assert "test_p25" in summary
    assert "test_p75" in summary
    assert "test_p95" in summary
    