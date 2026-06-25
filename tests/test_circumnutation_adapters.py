"""Tests for `sleap_roots.circumnutation.adapters.series_to_inputs` (PR #17).

The adapter is the `Series` → `CircumnutationInputs` bridge. These tests use the
flexible `_build_tracked_slp` builder (custom track names, optional metadata CSV)
from `tests.test_series`, plus the synthetic builders from
`tests.test_circumnutation_cli`, so every case runs without the Git-LFS fixtures.
"""

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleap_roots.circumnutation._types import ROW_IDENTITY_COLUMNS
from tests.test_series import _build_tracked_slp


def _positions(n_frames, x0=10.0, y0=20.0):
    """A simple monotonic tip path of length ``n_frames``."""
    return [(x0 + i, y0 + 2 * i) for i in range(n_frames)]


def _series_with_tracks(
    tmp_path,
    track_names,
    *,
    n_frames=4,
    csv_content=None,
    sample_uid=None,
    series_name="plate",
):
    """Build a `Series` whose tracks are named exactly ``track_names``."""
    track_positions = {name: _positions(n_frames) for name in track_names}
    return _build_tracked_slp(
        tmp_path,
        series_name,
        n_frames,
        track_positions,
        csv_content=csv_content,
        sample_uid=sample_uid,
    )


_CSV_HEADER = "plant_qr_code,genotype,treatment,number_of_plants_cylinder,timepoint\n"


# ---------------------------------------------------------------------------
# 2.1 — adapter behavior, one test per spec scenario
# ---------------------------------------------------------------------------


def test_returns_inputs_and_provenance_with_identity(tmp_path):
    """Returns (inputs, identity_provenance); 8 identity cols; int track/plant ids."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["track_0", "track_1"], n_frames=4)
    inputs, prov = series_to_inputs(
        series, cadence_s=300.0, sample_uid="plate_001", genotype="Nipponbare"
    )
    df = inputs.trajectory_df
    for col in ROW_IDENTITY_COLUMNS:
        assert col in df.columns
    assert df["track_id"].dtype == np.int64 and df["plant_id"].dtype == np.int64
    assert set(df["track_id"]) == {0, 1}
    assert (df["plant_id"] == df["track_id"]).all()
    assert (df["sample_uid"] == "plate_001").all()
    assert (df["series"] == series.series_name).all()
    assert (df["genotype"] == "Nipponbare").all()
    assert isinstance(prov, dict)
    assert inputs.cadence_s == 300.0


def test_bare_numeric_track_name(tmp_path):
    """A track literally named "5" (no prefix) coerces to int 5."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["5"], n_frames=3)
    inputs, _ = series_to_inputs(series, cadence_s=300.0, sample_uid="s")
    assert set(inputs.trajectory_df["track_id"]) == {5}


def test_non_integer_track_name_raises(tmp_path):
    """`track_2a` (non-integer remainder) raises ValueError naming the offender."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["track_2a"], n_frames=3)
    with pytest.raises(ValueError, match="track_2a"):
        series_to_inputs(series, cadence_s=300.0, sample_uid="s")


def test_interior_track_prefix_raises_not_corrupts(tmp_path):
    """`track_track_1` raises (anchored strip leaves `track_1`); NOT silently 1."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["track_track_1"], n_frames=3)
    with pytest.raises(ValueError, match="track_track_1"):
        series_to_inputs(series, cadence_s=300.0, sample_uid="s")


def test_flag_overrides_csv_value_and_logs(tmp_path, caplog):
    """A flag overrides a real CSV value and logs the override at INFO."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    csv = _CSV_HEADER + "plate_001,Nipponbare,MOCK,6,0\n"
    series = _series_with_tracks(
        tmp_path, ["track_0"], n_frames=3, csv_content=csv, sample_uid="plate_001"
    )
    with caplog.at_level(logging.INFO, logger="sleap_roots.circumnutation.adapters"):
        inputs, prov = series_to_inputs(
            series, cadence_s=300.0, sample_uid="plate_001", genotype="KitaakeX"
        )
    assert (inputs.trajectory_df["genotype"] == "KitaakeX").all()
    assert prov["identity_source"]["genotype"] == "flag"
    assert any(
        "genotype" in r.message and "Nipponbare" in r.message for r in caplog.records
    )


def test_csv_used_when_no_flag_blank_cell_no_spurious_log(tmp_path, caplog):
    """CSV value used when no flag; a blank cell yields NaN with no override log."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    # genotype populated, treatment blank.
    csv = _CSV_HEADER + "plate_001,Nipponbare,,6,0\n"
    series = _series_with_tracks(
        tmp_path, ["track_0"], n_frames=3, csv_content=csv, sample_uid="plate_001"
    )
    with caplog.at_level(logging.INFO, logger="sleap_roots.circumnutation.adapters"):
        inputs, prov = series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001")
    df = inputs.trajectory_df
    assert (df["genotype"] == "Nipponbare").all()
    assert df["treatment"].isna().all()
    assert prov["identity_source"]["genotype"] == "metadata_csv"
    assert prov["identity_source"]["treatment"] == "absent"
    assert not any("override" in r.message.lower() for r in caplog.records)


def test_neither_csv_nor_flag_yields_nan(tmp_path):
    """With no CSV and no flags, identity fields are NaN and labeled absent."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["track_0"], n_frames=3)
    inputs, prov = series_to_inputs(series, cadence_s=300.0, sample_uid="s")
    df = inputs.trajectory_df
    for col in ("genotype", "treatment", "timepoint", "plate_id"):
        assert df[col].isna().all()
        assert prov["identity_source"][col] == "absent"


def test_timepoint_numeric_csv_cell_stringified(tmp_path):
    """A numeric CSV timepoint cell (0) becomes the string "0" (object dtype)."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    csv = _CSV_HEADER + "plate_001,Nipponbare,MOCK,6,0\n"
    series = _series_with_tracks(
        tmp_path, ["track_0"], n_frames=3, csv_content=csv, sample_uid="plate_001"
    )
    inputs, prov = series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001")
    df = inputs.trajectory_df
    assert (df["timepoint"] == "0").all()
    assert df["timepoint"].map(type).eq(str).all()
    assert prov["identity_source"]["timepoint"] == "metadata_csv"


def test_malformed_metadata_csv_raises_valueerror(tmp_path):
    """A non-parseable metadata CSV surfaces a clear ValueError, not a pandas error."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    # Ragged/garbage content that pandas cannot parse as a table.
    bad = 'a,b,c\n"unterminated,1,2\n3,4\n'
    series = _series_with_tracks(
        tmp_path, ["track_0"], n_frames=3, csv_content=bad, sample_uid="plate_001"
    )
    with pytest.raises(ValueError, match="metadata"):
        series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001", genotype=None)


def test_identity_provenance_per_field(tmp_path):
    """identity_provenance carries metadata_csv_path, sha256, and a total source map."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    csv = _CSV_HEADER + "plate_001,Nipponbare,MOCK,6,0\n"
    series = _series_with_tracks(
        tmp_path, ["track_0"], n_frames=3, csv_content=csv, sample_uid="plate_001"
    )
    csv_path = Path(series.csv_path)
    inputs, prov = series_to_inputs(
        series, cadence_s=300.0, sample_uid="plate_001", treatment="DROUGHT"
    )
    assert prov["metadata_csv_path"] == str(csv_path.resolve())
    assert (
        prov["metadata_csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    )
    src = prov["identity_source"]
    assert set(src) == {
        "series",
        "sample_uid",
        "timepoint",
        "plate_id",
        "genotype",
        "treatment",
    }
    assert src["series"] == "default"
    assert src["sample_uid"] == "flag"
    assert src["genotype"] == "metadata_csv"
    assert src["treatment"] == "flag"
    assert src["timepoint"] == "metadata_csv"
    assert src["plate_id"] == "absent"


def test_no_csv_provenance_is_null(tmp_path):
    """With no metadata CSV, the path/hash are None and no source is metadata_csv."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["track_0"], n_frames=3)
    _, prov = series_to_inputs(series, cadence_s=300.0, sample_uid="s", genotype="WT")
    assert prov["metadata_csv_path"] is None
    assert prov["metadata_csv_sha256"] is None
    assert "metadata_csv" not in prov["identity_source"].values()
    assert prov["identity_source"]["genotype"] == "flag"


@pytest.mark.parametrize("bad", ["track_-1", "track_1_2", "track_ 1", "track_+1"])
def test_int_like_but_invalid_track_names_raise(tmp_path, bad):
    """Names Python's int() would silently mis-coerce raise instead (strict \\d+)."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, [bad], n_frames=3)
    with pytest.raises(ValueError, match="track_id"):
        series_to_inputs(series, cadence_s=300.0, sample_uid="s")


def test_blank_genotype_flag_is_absent(tmp_path):
    """An empty/whitespace --genotype is treated as not supplied (absent/NaN)."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["track_0"], n_frames=3)
    inputs, prov = series_to_inputs(
        series, cadence_s=300.0, sample_uid="s", genotype="   "
    )
    assert inputs.trajectory_df["genotype"].isna().all()
    assert prov["identity_source"]["genotype"] == "absent"


def test_blank_csv_genotype_cell_is_absent(tmp_path):
    """A whitespace-only CSV genotype cell resolves to absent (not a real value)."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    csv = _CSV_HEADER + "plate_001,   ,MOCK,6,0\n"
    series = _series_with_tracks(
        tmp_path, ["track_0"], n_frames=3, csv_content=csv, sample_uid="plate_001"
    )
    inputs, prov = series_to_inputs(series, cadence_s=300.0, sample_uid="plate_001")
    assert inputs.trajectory_df["genotype"].isna().all()
    assert prov["identity_source"]["genotype"] == "absent"


def test_identity_columns_are_object_dtype(tmp_path):
    """The string identity columns are object dtype even when all-NaN."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    series = _series_with_tracks(tmp_path, ["track_0"], n_frames=3)
    inputs, _ = series_to_inputs(series, cadence_s=300.0, sample_uid="s")
    df = inputs.trajectory_df
    for col in (
        "series",
        "sample_uid",
        "timepoint",
        "plate_id",
        "genotype",
        "treatment",
    ):
        assert df[col].dtype == object


def test_unmatched_sample_uid_warns(tmp_path, caplog):
    """A --metadata-csv whose plant_qr_code never matches sample_uid logs a WARNING."""
    from sleap_roots.circumnutation.adapters import series_to_inputs

    csv = _CSV_HEADER + "plate_001,Nipponbare,MOCK,6,0\n"
    series = _series_with_tracks(
        tmp_path, ["track_0"], n_frames=3, csv_content=csv, sample_uid="WRONG_QR"
    )
    with caplog.at_level(logging.WARNING, logger="sleap_roots.circumnutation.adapters"):
        series_to_inputs(series, cadence_s=300.0, sample_uid="WRONG_QR")
    assert any("matches no plant_qr_code" in r.message for r in caplog.records)
