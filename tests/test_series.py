from sleap_roots.series import Series, find_all_series


def test_series_load(canola_h5):
    series = Series.load(
        canola_h5, primary_name="primary_multi_day", lateral_name="lateral_3_nodes"
    )
    assert len(series) == 72


def test_find_all_series(canola_folder):
    all_series_files = find_all_series(canola_folder)
    assert len(all_series_files) == 1
