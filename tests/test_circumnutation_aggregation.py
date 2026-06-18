"""Tests for ``sleap_roots.circumnutation.aggregation`` (PR #15).

Covers the per-genotype aggregation API + the per-genotype CSV/sidecar I/O.
Unit tests drive a minimal-but-complete synthetic per-plant frame; the real
plate-001 integration test (gated on the Git-LFS fixture) is the validation.

Tests that import ``pipeline`` to build a real frame do so function-locally
(the documented pattern that avoids the cross-module reload pollution that bit
PR #14: ``test_no_root_handlers_added_at_import`` swaps module generations).
"""

import numpy as np
import pandas as pd
import pytest

from sleap_roots.circumnutation.aggregation import (
    _float_trait_columns,
    _stat_plan,
    aggregate_by_genotype,
)
from sleap_roots.circumnutation._io import (
    _validate_units_coverage,
    gather_run_metadata,
    read_per_genotype_csv,
    write_per_genotype_csv,
)

# A minimal-but-complete per-plant schema: 8 identity + 2 representative float
# traits + the special columns the function requires/handles.
_BASE_UNITS = {
    "series": "string",
    "sample_uid": "string",
    "timepoint": "string",
    "plate_id": "string",
    "plant_id": "int",
    "track_id": "int",
    "genotype": "string",
    "treatment": "string",
    "v_total_median_px_per_frame": "px/frame",
    "T_nutation_median": "s",
    "is_nutating": "bool",
    "handedness": "int",
    "track_is_clean": "bool",
    "qc_failure_reason": "string",
    "growth_axis_unreliable": "bool",
    "principal_axis_angle": "rad",
    "helix_signed_area_px2": "px²",
}


def _make_per_plant(rows):
    """Build a (per_plant_df, units) pair from per-plant override dicts.

    Each entry in ``rows`` overrides the defaults for one plant; ``plant_id`` and
    ``track_id`` default to the row index unless overridden.
    """
    defaults = dict(
        series="s1",
        sample_uid="u1",
        timepoint="T0",
        plate_id="plate_001",
        plant_id=0,
        track_id=0,
        genotype="Nipponbare",
        treatment="none",
        v_total_median_px_per_frame=1.0,
        T_nutation_median=10.0,
        is_nutating=True,
        handedness=1,
        track_is_clean=True,
        qc_failure_reason="",
        growth_axis_unreliable=False,
        principal_axis_angle=0.1,
        helix_signed_area_px2=100.0,
    )
    recs = []
    for i, r in enumerate(rows):
        rec = dict(defaults)
        rec["plant_id"] = i
        rec["track_id"] = i
        rec.update(r)
        recs.append(rec)
    df = pd.DataFrame(recs, columns=list(_BASE_UNITS.keys()))
    if not df.empty:
        for c in ("plant_id", "track_id", "handedness"):
            df[c] = df[c].astype("int64")
        for c in ("is_nutating", "track_is_clean", "growth_axis_unreliable"):
            df[c] = df[c].astype(bool)
    return df, dict(_BASE_UNITS)


# ---------------------------------------------------------------------------
# 1. Callability
# ---------------------------------------------------------------------------


def test_aggregate_by_genotype_is_callable_returns_tuple():
    df, units = _make_per_plant([{}, {}])
    out = aggregate_by_genotype(df, units)
    assert isinstance(out, tuple) and len(out) == 2
    per_genotype_df, per_genotype_units = out
    assert isinstance(per_genotype_df, pd.DataFrame)
    assert isinstance(per_genotype_units, dict)


# ---------------------------------------------------------------------------
# 2. Input validation
# ---------------------------------------------------------------------------


def test_units_not_covering_columns_raises_naming_column():
    df, units = _make_per_plant([{}])
    units.pop("T_nutation_median")
    with pytest.raises(ValueError, match="T_nutation_median"):
        aggregate_by_genotype(df, units)


def test_one_to_one_plant_guard_raises_on_duplicate_plant():
    df, units = _make_per_plant([{}, {}])
    # Force both rows to the same plant_id within one (plate,genotype,treatment).
    df["plant_id"] = 7
    with pytest.raises(ValueError, match="7"):
        aggregate_by_genotype(df, units)


def test_inputs_are_not_mutated():
    df, units = _make_per_plant([{}, {"handedness": -1}])
    df_before = df.copy(deep=True)
    units_before = dict(units)
    aggregate_by_genotype(df, units)
    pd.testing.assert_frame_equal(df, df_before)
    assert units == units_before


# ---------------------------------------------------------------------------
# 3. Float-trait classification + median/IQR
# ---------------------------------------------------------------------------


def test_float_trait_detection_excludes_special_and_circular():
    df, units = _make_per_plant([{}])
    floats = _float_trait_columns(list(df.columns), units)
    assert set(floats) == {"v_total_median_px_per_frame", "T_nutation_median"}
    # principal_axis_angle (rad) and helix_signed_area_px2 are NOT plain floats
    assert "principal_axis_angle" not in floats
    assert "helix_signed_area_px2" not in floats
    # helix is in the plan as a magnitude
    mags = [src for src, _p, is_mag in _stat_plan(list(df.columns), units) if is_mag]
    assert mags == ["helix_signed_area_px2"]


def test_median_and_iqr_over_passing_plants():
    df, units = _make_per_plant(
        [
            {"v_total_median_px_per_frame": 1.0},
            {"v_total_median_px_per_frame": 2.0},
            {"v_total_median_px_per_frame": 3.0},
            {"v_total_median_px_per_frame": 4.0},
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert len(g) == 1
    assert g["v_total_median_px_per_frame_median"].iloc[0] == 2.5
    # IQR of [1,2,3,4] (linear) = Q75(3.25) - Q25(1.75) = 1.5
    assert g["v_total_median_px_per_frame_iqr"].iloc[0] == pytest.approx(1.5)


def test_iqr_nan_for_single_passing_plant():
    df, units = _make_per_plant([{"v_total_median_px_per_frame": 5.0}])
    g, _ = aggregate_by_genotype(df, units)
    assert g["v_total_median_px_per_frame_median"].iloc[0] == 5.0
    assert np.isnan(g["v_total_median_px_per_frame_iqr"].iloc[0])


def test_iqr_nan_when_trait_has_one_finite_value_in_multi_plant_group():
    df, units = _make_per_plant(
        [
            {"T_nutation_median": 12.0},
            {"T_nutation_median": np.nan},
            {"T_nutation_median": np.nan},
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert g["T_nutation_median_median"].iloc[0] == 12.0
    assert np.isnan(g["T_nutation_median_iqr"].iloc[0])


def test_trait_all_nan_among_passing_plants_median_and_iqr_nan():
    """A trait that is NaN for every passing plant -> median & IQR NaN, no crash.

    The plants still pass QC (so frac_nutating reflects them); only the trait's
    own per-trait sample is empty. Locks the module-docstring's headline caveat
    that per-trait finite n can be 0 while n_plants_passing_qc > 0.
    """
    df, units = _make_per_plant(
        [
            {"T_nutation_median": np.nan, "is_nutating": False},
            {"T_nutation_median": np.nan, "is_nutating": False},
            {"T_nutation_median": np.nan, "is_nutating": True},
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert g["n_plants_passing_qc"].iloc[0] == 3
    assert np.isnan(g["T_nutation_median_median"].iloc[0])
    assert np.isnan(g["T_nutation_median_iqr"].iloc[0])
    # frac_nutating still computed over all 3 passing plants (1 of 3 nutating)
    assert g["frac_nutating"].iloc[0] == pytest.approx(1 / 3)


def test_scipy_iqr_behavior_guard():
    """Pin the behavior _iqr_or_nan depends on across scipy versions."""
    import scipy.stats

    assert scipy.stats.iqr([5.0, float("nan")], nan_policy="omit") == 0.0
    assert scipy.stats.iqr(
        [1.0, 2.0, 3.0, 4.0], nan_policy="omit", interpolation="linear"
    ) == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# 4. Special-column aggregation
# ---------------------------------------------------------------------------


def test_frac_nutating_is_mean_of_is_nutating():
    df, units = _make_per_plant(
        [{"is_nutating": True}, {"is_nutating": True}, {"is_nutating": False}]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert g["frac_nutating"].iloc[0] == pytest.approx(2 / 3)
    assert g["frac_nutating"].dtype == np.float64


def test_handedness_mode_clear_majority():
    df, units = _make_per_plant(
        [{"handedness": 1}, {"handedness": 1}, {"handedness": 1}, {"handedness": -1}]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert g["handedness_mode"].iloc[0] == 1
    assert g["handedness_consensus_frac"].iloc[0] == pytest.approx(0.75)


def test_handedness_mode_tiebreak():
    df, units = _make_per_plant([{"handedness": 1}, {"handedness": -1}])
    g, _ = aggregate_by_genotype(df, units)
    assert g["handedness_mode"].iloc[0] == -1  # smallest signed among tied abs
    assert g["handedness_consensus_frac"].iloc[0] == pytest.approx(0.5)

    df2, units2 = _make_per_plant(
        [{"handedness": 0}, {"handedness": 0}, {"handedness": 1}, {"handedness": 1}]
    )
    g2, _ = aggregate_by_genotype(df2, units2)
    assert g2["handedness_mode"].iloc[0] == 0  # smallest abs wins the tie


def test_growth_axis_and_principal_axis_not_emitted_identity_dropped():
    df, units = _make_per_plant([{}, {}])
    g, _ = aggregate_by_genotype(df, units)
    for col in (
        "growth_axis_unreliable",
        "growth_axis_unreliable_median",
        "growth_axis_unreliable_iqr",
        "principal_axis_angle",
        "principal_axis_angle_median",
        "principal_axis_angle_iqr",
        "series",
        "sample_uid",
        "timepoint",
        "plant_id",
        "track_id",
    ):
        assert col not in g.columns


def test_helix_aggregated_as_magnitude():
    df, units = _make_per_plant(
        [{"helix_signed_area_px2": 1000.0}, {"helix_signed_area_px2": -1000.0}]
    )
    g, ug = aggregate_by_genotype(df, units)
    assert "helix_signed_area_abs_px2_median" in g.columns
    assert "helix_signed_area_px2_median" not in g.columns
    assert "helix_signed_area_px2_iqr" not in g.columns
    # median of |[1000, -1000]| = 1000, NOT ~0
    assert g["helix_signed_area_abs_px2_median"].iloc[0] == pytest.approx(1000.0)
    assert ug["helix_signed_area_abs_px2_median"] == "px²"
    assert ug["helix_signed_area_abs_px2_iqr"] == "px²"


# ---------------------------------------------------------------------------
# 5. Grouping + QC exclusion + counts + reasons
# ---------------------------------------------------------------------------


def test_group_by_plate_genotype_treatment_sorted_not_pooled():
    df, units = _make_per_plant(
        [
            {"plate_id": "plate_002"},
            {"plate_id": "plate_001"},
            {"plate_id": "plate_001"},
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert list(g["plate_id"]) == ["plate_001", "plate_002"]  # sorted, not pooled
    assert list(g["n_plants_passing_qc"]) == [2, 1]


def test_nan_group_key_is_grouped_not_dropped():
    df, units = _make_per_plant(
        [{"treatment": "none"}, {"treatment": "none"}, {"treatment": np.nan}]
    )
    g, _ = aggregate_by_genotype(df, units)
    # dropna=False: the NaN-treatment plant forms its OWN group (2 rows), not
    # silently merged into the "none" group.
    assert len(g) == 2
    assert (g["n_plants_passing_qc"] + g["n_plants_excluded"]).sum() == 3


def test_qc_exclusion_counts():
    df, units = _make_per_plant(
        [
            {"track_is_clean": True},
            {"track_is_clean": True},
            {"track_is_clean": True},
            {"track_is_clean": False, "qc_failure_reason": "worst_step_ratio_high"},
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert g["n_plants_passing_qc"].iloc[0] == 3
    assert g["n_plants_excluded"].iloc[0] == 1
    # the excluded plant must not contribute to the trait medians
    assert g["v_total_median_px_per_frame_median"].iloc[0] == 1.0


def test_exclusion_reasons_clause_count_ordering():
    df, units = _make_per_plant(
        [
            {"track_is_clean": True},
            {
                "track_is_clean": False,
                "qc_failure_reason": "frac_outlier_steps_high, worst_step_ratio_high",
            },
            {"track_is_clean": False, "qc_failure_reason": "frac_outlier_steps_high"},
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    # ordered by _FAILURE_CLAUSE_ORDER (frac_outlier_steps_high before worst_step_ratio_high)
    assert (
        g["exclusion_reasons"].iloc[0]
        == "frac_outlier_steps_high:2; worst_step_ratio_high:1"
    )


def test_exclusion_reasons_empty_when_none_excluded():
    df, units = _make_per_plant([{}, {}])
    g, _ = aggregate_by_genotype(df, units)
    assert g["exclusion_reasons"].iloc[0] == ""


def test_failure_clause_tokens_are_separator_safe():
    """_FAILURE_CLAUSE_ORDER tokens must be parse-safe for the ':'/'; ' encoding."""
    import re

    from sleap_roots.circumnutation import qc

    for clause in qc._FAILURE_CLAUSE_ORDER:
        assert re.fullmatch(r"[a-z0-9_]+", clause), clause


# ---------------------------------------------------------------------------
# 6. Degenerate groups
# ---------------------------------------------------------------------------


def test_all_excluded_group_emits_nan_row():
    df, units = _make_per_plant(
        [
            {"track_is_clean": False, "qc_failure_reason": "d2_msd_agreement_high"},
            {"track_is_clean": False, "qc_failure_reason": "d2_msd_agreement_high"},
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert g["n_plants_passing_qc"].iloc[0] == 0
    assert g["n_plants_excluded"].iloc[0] == 2
    assert np.isnan(g["v_total_median_px_per_frame_median"].iloc[0])
    assert np.isnan(g["v_total_median_px_per_frame_iqr"].iloc[0])
    assert np.isnan(g["frac_nutating"].iloc[0])
    assert np.isnan(g["handedness_consensus_frac"].iloc[0])
    assert g["handedness_mode"].iloc[0] == 0


def test_int_dtype_survives_mixed_passing_and_all_excluded_frame():
    df, units = _make_per_plant(
        [
            {"plate_id": "plate_001", "track_is_clean": True},
            {
                "plate_id": "plate_002",
                "track_is_clean": False,
                "qc_failure_reason": "d2_msd_agreement_high",
            },
        ]
    )
    g, _ = aggregate_by_genotype(df, units)
    assert g["handedness_mode"].dtype == np.int64
    assert g["n_plants_passing_qc"].dtype == np.int64
    assert g["n_plants_excluded"].dtype == np.int64


def test_empty_per_plant_frame_yields_empty_per_genotype_frame():
    df, units = _make_per_plant([])
    g, ug = aggregate_by_genotype(df, units)
    assert len(g) == 0
    # full column set still present + units 1:1
    assert "v_total_median_px_per_frame_median" in g.columns
    assert "helix_signed_area_abs_px2_median" in g.columns
    assert set(ug.keys()) == set(g.columns)
    # count columns keep int dtype even when empty
    assert g["n_plants_passing_qc"].dtype == np.int64
    assert g["handedness_mode"].dtype == np.int64


# ---------------------------------------------------------------------------
# 7. Output schema order + units (full composed schema)
# ---------------------------------------------------------------------------


def _full_composed_frame(n=2):
    """Build a (df, units) over the REAL composed schema via the tier maps."""
    from sleap_roots.circumnutation.pipeline import (
        _COMPOSED_COLUMN_ORDER,
        _assemble_units,
    )

    units = _assemble_units()
    cols = list(_COMPOSED_COLUMN_ORDER)
    recs = []
    for i in range(n):
        rec = {}
        for c in cols:
            u = units[c]
            if c in ("plant_id", "track_id"):
                rec[c] = i
            elif u == "int":
                rec[c] = 1
            elif u == "bool":
                rec[c] = c != "growth_axis_unreliable"  # clean, growth-axis ok
            elif u == "string":
                rec[c] = "" if c == "qc_failure_reason" else "x"
            else:
                rec[c] = float(i + 1)
        rec["track_is_clean"] = True
        rec["genotype"] = "Nipponbare"
        rec["treatment"] = "none"
        rec["plate_id"] = "plate_001"
        recs.append(rec)
    df = pd.DataFrame(recs, columns=cols)
    df["plant_id"] = df["plant_id"].astype("int64")
    df["track_id"] = df["track_id"].astype("int64")
    return df, units


def test_full_composed_schema_output_columns_and_order():
    df, units = _full_composed_frame(n=2)
    g, ug = aggregate_by_genotype(df, units)

    cols = list(g.columns)
    # leading identity + counts + reasons
    assert cols[:6] == [
        "plate_id",
        "genotype",
        "treatment",
        "n_plants_passing_qc",
        "n_plants_excluded",
        "exclusion_reasons",
    ]
    # verbose names present; circular angle absent; helix magnitude present
    assert "T_nutation_median_median" in cols
    assert "T_nutation_median_iqr" in cols
    assert "angular_amplitude_median" in cols
    assert "helix_signed_area_abs_px2_median" in cols
    assert "principal_axis_angle_median" not in cols
    assert "helix_signed_area_px2_median" not in cols
    # trailing special trio
    assert cols[-3:] == [
        "frac_nutating",
        "handedness_mode",
        "handedness_consensus_frac",
    ]
    # 73-column schema
    assert len(cols) == 73
    # units 1:1 cover, all in vocabulary
    from sleap_roots.circumnutation._constants import PIPELINE_UNIT_VOCABULARY

    assert set(ug.keys()) == set(cols)
    assert all(v in PIPELINE_UNIT_VOCABULARY for v in ug.values())
    assert ug["helix_signed_area_abs_px2_median"] == "px²"
    assert ug["frac_nutating"] == "—"
    assert ug["handedness_mode"] == "int"


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------


def test_aggregation_is_deterministic():
    df, units = _make_per_plant(
        [{"handedness": 1}, {"handedness": -1}, {"plate_id": "plate_002"}]
    )
    a, _ = aggregate_by_genotype(df, units)
    b, _ = aggregate_by_genotype(df, units)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# 9. Per-genotype CSV + sidecar I/O
# ---------------------------------------------------------------------------


def test_validate_units_coverage_is_shared_and_per_plant_writer_unchanged(tmp_path):
    """The extracted helper exists and write_per_plant_csv still raises on coverage."""
    from sleap_roots.circumnutation._io import write_per_plant_csv

    df = pd.DataFrame({"a": [1.0]})
    # coverage helper names the offending column
    with pytest.raises(ValueError, match="a"):
        _validate_units_coverage(df, {}, fn_name="probe")
    # write_per_plant_csv still raises (before any write) on coverage mismatch
    with pytest.raises(ValueError):
        write_per_plant_csv(tmp_path / "x.csv", df, {}, {})
    assert not (tmp_path / "x.csv").exists()


def test_write_read_per_genotype_round_trip(tmp_path):
    df, units = _make_per_plant([{}, {"handedness": -1}])
    g, ug = aggregate_by_genotype(df, units)
    md = gather_run_metadata("dummy.slp", cadence_s=300.0)
    out = tmp_path / "per_genotype.csv"
    write_per_genotype_csv(out, g, ug, md)
    assert out.exists()
    assert (tmp_path / "per_genotype.units.json").exists()
    assert (tmp_path / "run_metadata.json").exists()
    g2, ug2, md2 = read_per_genotype_csv(out)
    assert list(g2.columns) == list(g.columns)
    assert len(g2) == len(g)
    assert ug2 == ug
    assert md2["cadence_s"] == 300.0
    assert md2["_constants_version"] == 6


def test_write_per_genotype_rejects_out_of_vocab_unit(tmp_path):
    df, units = _make_per_plant([{}])
    g, ug = aggregate_by_genotype(df, units)
    ug["frac_nutating"] = "mm"  # out of vocabulary
    out = tmp_path / "bad.csv"
    with pytest.raises(ValueError, match="PIPELINE_UNIT_VOCABULARY"):
        write_per_genotype_csv(out, g, ug, {})
    assert not out.exists()


def test_write_per_genotype_rejects_non_1to1_units(tmp_path):
    df, units = _make_per_plant([{}])
    g, ug = aggregate_by_genotype(df, units)
    ug.pop("frac_nutating")  # missing a column
    out = tmp_path / "bad2.csv"
    with pytest.raises(ValueError, match="frac_nutating"):
        write_per_genotype_csv(out, g, ug, {})
    assert not out.exists()


def test_read_per_genotype_missing_sidecars_returns_empty(tmp_path):
    df, units = _make_per_plant([{}])
    g, ug = aggregate_by_genotype(df, units)
    md = gather_run_metadata("dummy.slp", cadence_s=300.0)
    out = tmp_path / "pg.csv"
    write_per_genotype_csv(out, g, ug, md)
    (tmp_path / "pg.units.json").unlink()
    (tmp_path / "run_metadata.json").unlink()
    g2, ug2, md2 = read_per_genotype_csv(out)
    assert len(g2) == len(g)
    assert ug2 == {}
    assert md2 == {}


# ---------------------------------------------------------------------------
# 10. Real plate-001 integration test (the validation)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

_PROOFREAD_FIXTURE = (
    Path(__file__).parent
    / "data"
    / "circumnutation_nipponbare_plate_001"
    / "plate_001_greyscale.tracked_proofread.slp"
)


def _load_plate001_inputs():
    from sleap_roots.circumnutation._types import CircumnutationInputs
    from sleap_roots.series import Series

    series = Series.load(series_name="plate_001", primary_path=str(_PROOFREAD_FIXTURE))
    df = series.get_tracked_tips()
    df["track_id"] = df["track_id"].str.replace("track_", "", regex=False).astype(int)
    df = df.copy()
    df["series"] = "plate_001"
    df["sample_uid"] = "plate_001"
    df["timepoint"] = "T0"
    df["plate_id"] = "plate_001"
    df["plant_id"] = df["track_id"]
    df["genotype"] = "Nipponbare"
    df["treatment"] = "none"
    return CircumnutationInputs(trajectory_df=df, cadence_s=300.0)


@pytest.mark.skipif(
    not _PROOFREAD_FIXTURE.exists(),
    reason=f"Git-LFS proofread fixture not present: {_PROOFREAD_FIXTURE}",
)
def test_real_plate001_aggregation_round_trip(tmp_path):
    """Round-4 empirical anchor: plate-001 is 1 passing / 5 excluded."""
    from sleap_roots.circumnutation import pipeline

    inputs = _load_plate001_inputs()
    per_plant_df, _, units = pipeline.compute_traits(inputs)

    g, ug = aggregate_by_genotype(per_plant_df, units)

    # Exactly one (plate_001, Nipponbare, none) row.
    assert len(g) == 1
    assert g["plate_id"].iloc[0] == "plate_001"
    assert g["genotype"].iloc[0] == "Nipponbare"
    assert g["treatment"].iloc[0] == "none"

    # The empirically-verified profile (update if the fixture/QC changes).
    assert g["n_plants_passing_qc"].iloc[0] == 1
    assert g["n_plants_excluded"].iloc[0] == 5
    assert g["n_plants_passing_qc"].iloc[0] + g["n_plants_excluded"].iloc[0] == 6
    assert g["exclusion_reasons"].iloc[0] == "d2_msd_agreement_high:5"
    assert g["frac_nutating"].iloc[0] == pytest.approx(1.0)
    assert g["handedness_mode"].iloc[0] == 1
    assert g["handedness_consensus_frac"].iloc[0] == pytest.approx(1.0)

    # All medians finite; all IQRs NaN (only 1 passing plant → n < 2).
    median_cols = [c for c in g.columns if c.endswith("_median")]
    iqr_cols = [c for c in g.columns if c.endswith("_iqr")]
    assert median_cols and iqr_cols
    for c in median_cols:
        assert np.isfinite(g[c].iloc[0]), c
    for c in iqr_cols:
        assert np.isnan(g[c].iloc[0]), c

    # Round-trip to a DISTINCT subdir (avoid the fixed run_metadata.json clobber).
    out_dir = tmp_path / "per_genotype"
    out_dir.mkdir()
    md = gather_run_metadata(str(_PROOFREAD_FIXTURE), cadence_s=300.0)
    out = out_dir / "per_genotype.csv"
    write_per_genotype_csv(out, g, ug, md)
    g2, ug2, md2 = read_per_genotype_csv(out)
    assert len(g2) == 1
    assert ug2 == ug
    assert md2["cadence_s"] == 300.0
