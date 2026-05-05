"""Tests for sleap_roots.metadata helpers."""

import re
from pathlib import Path

import pandas as pd
import pytest


def test_build_metadata_csv_canonical_column_order(tmp_path):
    from sleap_roots.metadata import build_metadata_csv

    rows = [
        {"plant_qr_code": "a", "genotype": "X", "timepoint": 0},
        {"plant_qr_code": "b", "genotype": "Y", "timepoint": 1},
    ]
    out = build_metadata_csv(rows, tmp_path / "out.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == ["plant_qr_code", "genotype", "timepoint"]
    assert df["plant_qr_code"].tolist() == ["a", "b"]


def test_build_metadata_csv_omits_unused_columns(tmp_path):
    from sleap_roots.metadata import build_metadata_csv

    rows = [{"plant_qr_code": "a", "timepoint": 0}]  # no genotype
    out = build_metadata_csv(rows, tmp_path / "out.csv")
    df = pd.read_csv(out)
    assert "genotype" not in df.columns
    assert list(df.columns) == ["plant_qr_code", "timepoint"]


def test_build_metadata_csv_raises_on_missing_plant_qr_code(tmp_path):
    from sleap_roots.metadata import build_metadata_csv

    rows = [{"genotype": "X"}]
    with pytest.raises(ValueError) as excinfo:
        build_metadata_csv(rows, tmp_path / "out.csv")
    assert "plant_qr_code" in str(excinfo.value)


def test_build_metadata_csv_returns_path(tmp_path):
    from sleap_roots.metadata import build_metadata_csv

    rows = [{"plant_qr_code": "a"}]
    target = tmp_path / "x.csv"
    out = build_metadata_csv(rows, target)
    assert Path(out) == target
    assert target.exists()


def test_build_metadata_csv_accepts_str_path(tmp_path):
    from sleap_roots.metadata import build_metadata_csv

    rows = [{"plant_qr_code": "a"}]
    target = str(tmp_path / "x.csv")
    out = build_metadata_csv(rows, target)
    assert Path(out).exists()


def test_build_metadata_csv_extras_sorted(tmp_path):
    from sleap_roots.metadata import build_metadata_csv

    rows = [
        {
            "plant_qr_code": "a",
            "genotype": "X",
            "zzz_extra": 1,
            "aaa_extra": 2,
        }
    ]
    out = build_metadata_csv(rows, tmp_path / "out.csv")
    df = pd.read_csv(out)
    assert list(df.columns) == [
        "plant_qr_code",
        "genotype",
        "aaa_extra",
        "zzz_extra",
    ]


def test_build_metadata_csv_overwrites_existing(tmp_path):
    from sleap_roots.metadata import build_metadata_csv

    target = tmp_path / "x.csv"
    target.write_text("garbage,prior,content\n1,2,3\n")
    rows = [{"plant_qr_code": "a", "genotype": "X"}]
    build_metadata_csv(rows, target)
    df = pd.read_csv(target)
    assert list(df.columns) == ["plant_qr_code", "genotype"]
    assert df["plant_qr_code"].tolist() == ["a"]


def test_infer_timepoints_from_filenames_named_groups():
    from sleap_roots.metadata import infer_timepoints_from_filenames

    paths = [Path("plant1_0.slp"), Path("plant1_5.slp"), Path("plant1_10.slp")]
    pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"
    result = infer_timepoints_from_filenames(paths, pattern)
    assert result == {"plant1_0": 0.0, "plant1_5": 5.0, "plant1_10": 10.0}


def test_infer_timepoints_from_filenames_missing_named_groups():
    from sleap_roots.metadata import infer_timepoints_from_filenames

    with pytest.raises(ValueError) as excinfo:
        infer_timepoints_from_filenames([Path("plant1_0.slp")], r".+_\d+")
    msg = str(excinfo.value)
    assert "series_name" in msg
    assert "timepoint" in msg


def test_infer_timepoints_from_filenames_skips_non_matches_with_warning(caplog):
    from sleap_roots.metadata import infer_timepoints_from_filenames

    paths = [Path("plant1_0.slp"), Path("garbage.slp")]
    pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"
    with caplog.at_level("WARNING", logger="sleap_roots.metadata"):
        result = infer_timepoints_from_filenames(paths, pattern)
    assert result == {"plant1_0": 0.0}
    warnings = [
        r
        for r in caplog.records
        if r.name == "sleap_roots.metadata" and r.levelname == "WARNING"
    ]
    assert any("garbage" in r.message for r in warnings)
    assert any(
        "pattern" in r.message.lower() or "match" in r.message.lower() for r in warnings
    )


def test_infer_timepoints_from_filenames_skips_non_numeric_with_warning(caplog):
    from sleap_roots.metadata import infer_timepoints_from_filenames

    paths = [Path("plant1_5.slp"), Path("plant1_abc.slp")]
    pattern = r"(?P<series_name>.+?)_(?P<timepoint>[^.]+)"
    with caplog.at_level("WARNING", logger="sleap_roots.metadata"):
        result = infer_timepoints_from_filenames(paths, pattern)
    assert result == {"plant1_5": 5.0}
    warnings = [
        r
        for r in caplog.records
        if r.name == "sleap_roots.metadata" and r.levelname == "WARNING"
    ]
    assert any("plant1_abc" in r.message for r in warnings)
    # The float-cast warning has a different reason text than pattern-mismatch.
    assert any(
        re.search(r"convert|numeric|float", r.message, re.IGNORECASE) for r in warnings
    )


def test_infer_timepoints_from_filenames_casts_to_float():
    from sleap_roots.metadata import infer_timepoints_from_filenames

    paths = [Path("plant1_42.slp")]
    pattern = r"(?P<series_name>.+?)_(?P<timepoint>\d+)"
    result = infer_timepoints_from_filenames(paths, pattern)
    assert result == {"plant1_42": 42.0}
    assert isinstance(list(result.values())[0], float)
