"""Foundation contract tests for `sleap_roots.circumnutation`.

Exercises the package-layout, calibration-independence, row-identity
schema, units-sidecar, run-metadata sidecar, module-constants, and
logging contracts established by the foundation OpenSpec change
``add-circumnutation-foundation``. No spectral analysis or trait
computation is exercised here; tier PRs add those tests in later
files.
"""

import importlib
import json
import logging
import re
import sys

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Stub-module canonical-callable map
# ---------------------------------------------------------------------------

STUB_MODULES = [
    ("kinematics", "compute", 2),
    ("qc", "compute", 3),
    ("synthetic", "generate_trajectory", 4),
    ("temporal_cwt", "compute_scaleogram", 5),
    ("psi_g", "compute_psi_g", 7),
    ("midline", "reconstruct", 8),
    ("spatial_cwt", "compute_scaleogram", 9),
    ("parametric", "compute", 11),
    ("plotting", "scaleogram", 16),
    ("pipeline", "compute_traits", 14),
]


# ---------------------------------------------------------------------------
# Constants the foundation must expose with documented defaults
# ---------------------------------------------------------------------------

EXPECTED_CONSTANTS = {
    "NOISE_MASK_K": 2,
    "LGZ_STEADY_STATE_RESIDUAL_MAX": 0.2,
    "NYQUIST_RATIO_MAX": 0.25,
    "SG_D2_AGREEMENT_MAX": 1.5,
    "LGZ_NMIN_RESOLVABLE": 5,
    "COI_FRACTION_MAX": 0.5,
    "BAND_POWER_NOISE_RATIO": 3,
    "WAVELET_DEFAULT_TEMPORAL": "cmor1.5-1.0",
    "WAVELET_DEFAULT_SPATIAL": "cgau2",
    "SG_WINDOW_SHORT": 5,
    "SG_DEGREE": 3,
    "SG_WINDOW_DETREND": 23,
    "OUTLIER_STEP_RATIO": 2,
    "GROWTH_AXIS_RELIABILITY_K": 10,
}


# ---------------------------------------------------------------------------
# Row-identity schema as declared by the foundation spec
# ---------------------------------------------------------------------------

EXPECTED_ROW_IDENTITY_COLUMNS = (
    "series",
    "sample_uid",
    "timepoint",
    "plate_id",
    "plant_id",
    "track_id",
    "genotype",
    "treatment",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_trajectory_df():
    """Build a minimal trajectory DataFrame with all 8 row-identity columns + tip coords."""
    rows = []
    for track in range(6):
        for frame in range(10):
            rows.append(
                {
                    "series": "plate_001",
                    "sample_uid": "test_sample",
                    "timepoint": "T0",
                    "plate_id": "plate_001",
                    "plant_id": track,
                    "track_id": track,
                    "genotype": np.nan,
                    "treatment": np.nan,
                    "frame": frame,
                    "tip_x": float(track + frame * 0.1),
                    "tip_y": float(track * 10 + frame),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stub module tests (spec: Package layout)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", [name for name, _, _ in STUB_MODULES])
def test_stub_module_imports_cleanly(module_name):
    """Each stub module imports without raising."""
    mod = importlib.import_module(f"sleap_roots.circumnutation.{module_name}")
    assert mod is not None


@pytest.mark.parametrize("module_name,callable_name,expected_pr", STUB_MODULES)
def test_stub_callable_raises_with_correct_pr(module_name, callable_name, expected_pr):
    """Stub callables raise NotImplementedError with regex-matching PR# message."""
    mod = importlib.import_module(f"sleap_roots.circumnutation.{module_name}")
    fn = getattr(mod, callable_name)
    with pytest.raises(NotImplementedError) as exc_info:
        fn()
    msg = str(exc_info.value)
    match = re.match(r"^PR #(\d+) — see docs/circumnutation/roadmap\.md$", msg)
    assert match is not None, f"Message {msg!r} does not match expected regex"
    assert int(match.group(1)) == expected_pr


def test_import_sleap_roots_succeeds():
    """Top-level sleap_roots imports cleanly with foundation re-exports available."""
    import sleap_roots

    assert hasattr(sleap_roots, "CircumnutationInputs")
    assert hasattr(sleap_roots, "convert_to_mm")


# ---------------------------------------------------------------------------
# Constants tests (spec: Module-level constants)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", list(EXPECTED_CONSTANTS.items()))
def test_constant_has_documented_default(name, expected):
    """Each module-level constant exists in `_constants` with the documented default."""
    from sleap_roots.circumnutation import _constants

    assert hasattr(_constants, name), f"_constants is missing {name}"
    assert getattr(_constants, name) == expected


def test_pipeline_unit_vocabulary_is_px_only():
    """PIPELINE_UNIT_VOCABULARY contains only px-based + calibration-independent units.

    Regression test for Copilot PR #200 finding: the original
    `VALID_UNIT_VOCABULARY` contained mm-based units, weakening the
    pure-pixel sidecar contract. The split establishes one vocabulary
    for what the pipeline emits (this one — px only) and a separate
    one for `convert_to_mm` outputs.
    """
    from sleap_roots.circumnutation._constants import PIPELINE_UNIT_VOCABULARY

    forbidden_in_pipeline = {"mm", "mm²", "mm/hr", "mm/frame", "mm·hr⁻¹"}
    assert forbidden_in_pipeline.isdisjoint(PIPELINE_UNIT_VOCABULARY)
    assert "px" in PIPELINE_UNIT_VOCABULARY
    assert "px²" in PIPELINE_UNIT_VOCABULARY
    assert "px/hr" in PIPELINE_UNIT_VOCABULARY
    # Calibration-independent units must remain valid
    assert "hr" in PIPELINE_UNIT_VOCABULARY
    assert "rad" in PIPELINE_UNIT_VOCABULARY
    assert "bool" in PIPELINE_UNIT_VOCABULARY


def test_converted_unit_vocabulary_is_mm_only():
    """CONVERTED_UNIT_VOCABULARY contains only mm-based units (convert_to_mm output range)."""
    from sleap_roots.circumnutation._constants import CONVERTED_UNIT_VOCABULARY

    forbidden_in_converted = {"px", "px²", "px/hr", "px/frame", "px·hr⁻¹"}
    assert forbidden_in_converted.isdisjoint(CONVERTED_UNIT_VOCABULARY)
    assert "mm" in CONVERTED_UNIT_VOCABULARY
    assert "mm²" in CONVERTED_UNIT_VOCABULARY
    assert "mm/hr" in CONVERTED_UNIT_VOCABULARY


def test_valid_unit_vocabulary_is_union_of_pipeline_and_converted():
    """VALID_UNIT_VOCABULARY = PIPELINE_UNIT_VOCABULARY | CONVERTED_UNIT_VOCABULARY."""
    from sleap_roots.circumnutation._constants import (
        CONVERTED_UNIT_VOCABULARY,
        PIPELINE_UNIT_VOCABULARY,
        VALID_UNIT_VOCABULARY,
    )

    assert VALID_UNIT_VOCABULARY == PIPELINE_UNIT_VOCABULARY | CONVERTED_UNIT_VOCABULARY


def test_schema_and_constants_versions_are_integers_equal_to_one():
    """`_SCHEMA_VERSION` and `_CONSTANTS_VERSION` exist as integers equal to 1."""
    from sleap_roots.circumnutation import _constants

    assert isinstance(_constants._SCHEMA_VERSION, int)
    assert _constants._SCHEMA_VERSION == 1
    assert isinstance(_constants._CONSTANTS_VERSION, int)
    assert _constants._CONSTANTS_VERSION == 1


# ---------------------------------------------------------------------------
# CircumnutationInputs validation tests (spec: CircumnutationInputs data class)
# ---------------------------------------------------------------------------


def test_valid_construction(valid_trajectory_df):
    """CircumnutationInputs builds without exception when all required fields are valid."""
    from sleap_roots.circumnutation import CircumnutationInputs

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df,
        cadence_s=300.0,
        R_px=2.4,
        run_id="plate_001",
    )
    assert inputs.trajectory_df is valid_trajectory_df
    assert inputs.cadence_s == 300.0
    assert inputs.R_px == 2.4
    assert inputs.run_id == "plate_001"


@pytest.mark.parametrize("missing_col", list(EXPECTED_ROW_IDENTITY_COLUMNS))
def test_missing_row_identity_column_raises(valid_trajectory_df, missing_col):
    """Dropping any required row-identity column raises a ValueError that names the column."""
    from sleap_roots.circumnutation import CircumnutationInputs

    df = valid_trajectory_df.drop(columns=[missing_col])
    with pytest.raises(ValueError, match=missing_col):
        CircumnutationInputs(trajectory_df=df, cadence_s=300.0)


def test_empty_trajectory_df_raises(valid_trajectory_df):
    """An empty trajectory DataFrame raises a ValueError mentioning empty/trajectory."""
    from sleap_roots.circumnutation import CircumnutationInputs

    df = valid_trajectory_df.iloc[0:0]
    with pytest.raises(ValueError, match=r"empty|trajectory_df"):
        CircumnutationInputs(trajectory_df=df, cadence_s=300.0)


@pytest.mark.parametrize("bad_cadence", [0.0, -1.0, float("nan")])
def test_invalid_cadence_s_raises(valid_trajectory_df, bad_cadence):
    """Non-positive or NaN cadence_s raises a ValueError naming `cadence_s`."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="cadence_s"):
        CircumnutationInputs(trajectory_df=valid_trajectory_df, cadence_s=bad_cadence)


@pytest.mark.parametrize("bad_R", [0.0, -2.4, float("nan")])
def test_invalid_R_px_raises(valid_trajectory_df, bad_R):
    """Non-positive or NaN R_px raises a ValueError naming `R_px`."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="R_px"):
        CircumnutationInputs(
            trajectory_df=valid_trajectory_df,
            cadence_s=300.0,
            R_px=bad_R,
        )


def test_R_px_none_succeeds(valid_trajectory_df):
    """R_px=None is accepted (it is optional)."""
    from sleap_roots.circumnutation import CircumnutationInputs

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df,
        cadence_s=300.0,
        R_px=None,
    )
    assert inputs.R_px is None


def test_non_dataframe_trajectory_raises(valid_trajectory_df):
    """A non-DataFrame `trajectory_df` raises `ValueError` mentioning DataFrame."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="DataFrame"):
        CircumnutationInputs(trajectory_df={"not": "a df"}, cadence_s=300.0)


def test_unconvertible_R_px_raises(valid_trajectory_df):
    """Non-numeric R_px (e.g. a string that can't be float()'d) raises a ValueError naming R_px."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="R_px"):
        CircumnutationInputs(
            trajectory_df=valid_trajectory_df,
            cadence_s=300.0,
            R_px="abc",
        )


def test_unconvertible_cadence_s_raises(valid_trajectory_df):
    """Non-numeric cadence_s (e.g. a string) raises a ValueError naming cadence_s.

    Regression test for Copilot PR #200 review finding: the original
    `_validate_cadence_s` raised TypeError from `float(value)` without
    naming the field, contradicting the docstring contract that "the
    message names the offending field". The fix mirrors the try/except
    pattern already used by `_validate_R_px`.
    """
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="cadence_s"):
        CircumnutationInputs(
            trajectory_df=valid_trajectory_df,
            cadence_s="abc",
        )


def test_cadence_s_string_coerced_to_float(valid_trajectory_df):
    """Numeric-string `cadence_s="300"` is coerced to float, not stored as a string.

    Regression test for Copilot PR #200 second-round finding: validation
    accepted convertible strings but didn't actually convert, so
    `inputs.cadence_s` would still be the string and downstream numeric
    code would fail. Fix uses an attrs `converter=` so the stored value
    is always a `float`.
    """
    from sleap_roots.circumnutation import CircumnutationInputs

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df,
        cadence_s="300",
    )
    assert isinstance(inputs.cadence_s, float)
    assert inputs.cadence_s == 300.0


def test_R_px_string_coerced_to_float(valid_trajectory_df):
    """Numeric-string `R_px="2.4"` is coerced to float; `R_px=None` stays None."""
    from sleap_roots.circumnutation import CircumnutationInputs

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df,
        cadence_s=300.0,
        R_px="2.4",
    )
    assert isinstance(inputs.R_px, float)
    assert inputs.R_px == 2.4

    inputs_none = CircumnutationInputs(
        trajectory_df=valid_trajectory_df,
        cadence_s=300.0,
        R_px=None,
    )
    assert inputs_none.R_px is None


def test_importable_from_top_level():
    """`from sleap_roots import CircumnutationInputs, convert_to_mm` succeeds."""
    from sleap_roots import CircumnutationInputs, convert_to_mm

    assert CircumnutationInputs is not None
    assert callable(convert_to_mm)


# ---------------------------------------------------------------------------
# Schema tests (spec: Trait CSV row-identity schema)
# ---------------------------------------------------------------------------


def test_per_plant_template_columns_and_dtypes(valid_trajectory_df):
    """build_per_plant_template emits 8 row-identity columns in order; track_id int; plant_id == track_id."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import build_per_plant_template

    inputs = CircumnutationInputs(trajectory_df=valid_trajectory_df, cadence_s=300.0)
    df = build_per_plant_template(inputs)

    # 6 distinct tracks → 6 rows
    assert len(df) == 6
    # First 8 columns in declared order
    assert list(df.columns[:8]) == list(EXPECTED_ROW_IDENTITY_COLUMNS)
    # track_id is integer dtype
    assert pd.api.types.is_integer_dtype(df["track_id"])
    # plant_id column-wise equals track_id
    assert (df["plant_id"] == df["track_id"]).all()


def test_sort_order_numeric_for_track_id():
    """track_id=2 precedes track_id=10 after sort (numeric, not lexicographic)."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import build_per_plant_template

    rows = []
    for track_id in [10, 2]:  # input intentionally not sorted
        for frame in range(3):
            rows.append(
                {
                    "series": "S",
                    "sample_uid": "U",
                    "timepoint": "T0",
                    "plate_id": "P",
                    "plant_id": track_id,
                    "track_id": track_id,
                    "genotype": np.nan,
                    "treatment": np.nan,
                    "frame": frame,
                    "tip_x": 1.0,
                    "tip_y": 2.0,
                }
            )
    df = pd.DataFrame(rows)
    inputs = CircumnutationInputs(trajectory_df=df, cadence_s=300.0)
    template = build_per_plant_template(inputs)
    assert list(template["track_id"]) == [2, 10]


def test_pipeline_output_is_calibration_independent(valid_trajectory_df):
    """The foundation does not emit any [mm]-bearing unit string."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._constants import VALID_UNIT_VOCABULARY
    from sleap_roots.circumnutation._io import (
        build_per_plant_template,
        default_units_for_template,
    )

    inputs = CircumnutationInputs(trajectory_df=valid_trajectory_df, cadence_s=300.0)
    df = build_per_plant_template(inputs)
    units = default_units_for_template(df)
    forbidden = {"mm", "mm²", "mm/hr", "mm/frame", "mm·hr⁻¹"}
    assert all(u not in forbidden for u in units.values())
    assert all(u in VALID_UNIT_VOCABULARY for u in units.values())


# ---------------------------------------------------------------------------
# convert_to_mm tests (spec: convert_to_mm utility)
# ---------------------------------------------------------------------------


def test_convert_to_mm_identity_at_one():
    """At px_per_mm=1.0 the values are unchanged but the columns are renamed."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"length_px": [47.24]})
    units = {"length_px": "px"}
    out_df, out_units = convert_to_mm(df, units, px_per_mm=1.0)

    assert "length_mm" in out_df.columns
    assert out_df["length_mm"].iloc[0] == pytest.approx(47.24)
    assert out_units == {"length_mm": "mm"}
    # Inputs not mutated
    assert "length_px" in df.columns
    assert units == {"length_px": "px"}


def test_convert_to_mm_1200_dpi():
    """At px_per_mm=47.24 (1200 DPI), 47.24 px → 1.0 mm."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"length_px": [47.24]})
    units = {"length_px": "px"}
    out_df, out_units = convert_to_mm(df, units, px_per_mm=47.24)

    assert out_df["length_mm"].iloc[0] == pytest.approx(1.0)
    assert out_units == {"length_mm": "mm"}


def test_convert_to_mm_velocity_units():
    """px/hr → mm/hr; px/frame → mm/frame."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame(
        {
            "v_long_px_per_hr": [47.24],
            "v_total_px_per_frame": [4.724],
        }
    )
    units = {
        "v_long_px_per_hr": "px/hr",
        "v_total_px_per_frame": "px/frame",
    }
    out_df, out_units = convert_to_mm(df, units, px_per_mm=47.24)

    assert "v_long_mm_per_hr" in out_df.columns
    assert "v_total_mm_per_frame" in out_df.columns
    assert out_df["v_long_mm_per_hr"].iloc[0] == pytest.approx(1.0)
    assert out_df["v_total_mm_per_frame"].iloc[0] == pytest.approx(0.1)
    assert out_units["v_long_mm_per_hr"] == "mm/hr"
    assert out_units["v_total_mm_per_frame"] == "mm/frame"


def test_convert_to_mm_passes_through_non_px_columns():
    """Columns with non-px units (hr, bool, dimensionless) pass through unchanged."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame(
        {
            "T_nutation_hr": [0.926],
            "is_nutating": [True],
            "B_balance_number": [3.0],
        }
    )
    units = {
        "T_nutation_hr": "hr",
        "is_nutating": "bool",
        "B_balance_number": "—",
    }
    out_df, out_units = convert_to_mm(df, units, px_per_mm=47.24)

    assert list(out_df.columns) == list(df.columns)
    assert out_units == units
    assert out_df["T_nutation_hr"].iloc[0] == pytest.approx(0.926)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_convert_to_mm_invalid_px_per_mm(bad):
    """ValueError mentioning px_per_mm when calibration is not a positive finite float."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"length_px": [47.24]})
    units = {"length_px": "px"}
    with pytest.raises(ValueError, match="px_per_mm"):
        convert_to_mm(df, units, px_per_mm=bad)


def test_convert_to_mm_unconvertible_px_per_mm():
    """A non-numeric px_per_mm (e.g. a list) raises ValueError naming px_per_mm."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"length_px": [47.24]})
    units = {"length_px": "px"}
    with pytest.raises(ValueError, match="px_per_mm"):
        convert_to_mm(df, units, px_per_mm=[1, 2, 3])


def test_convert_to_mm_px_squared_area():
    """px² area columns scale by (1/px_per_mm)² and rename to mm²."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"helix_signed_area_px²": [47.24**2]})
    units = {"helix_signed_area_px²": "px²"}
    out_df, out_units = convert_to_mm(df, units, px_per_mm=47.24)

    assert "helix_signed_area_mm²" in out_df.columns
    assert out_df["helix_signed_area_mm²"].iloc[0] == pytest.approx(1.0)
    assert out_units["helix_signed_area_mm²"] == "mm²"


def test_convert_to_mm_px_hr_inverse_alternate_notation():
    """The alternate `px·hr⁻¹` unit string scales identically to `px/hr`."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"v_other_px_per_hr": [94.48]})
    units = {"v_other_px_per_hr": "px·hr⁻¹"}
    out_df, out_units = convert_to_mm(df, units, px_per_mm=47.24)

    assert "v_other_mm_per_hr" in out_df.columns
    assert out_df["v_other_mm_per_hr"].iloc[0] == pytest.approx(2.0)
    assert out_units["v_other_mm_per_hr"] == "mm/hr"


# ---------------------------------------------------------------------------
# Sidecar I/O tests (spec: Units sidecar JSON, Run-metadata sidecar)
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_csv_setup(tmp_path, valid_trajectory_df):
    """Build a CircumnutationInputs and a per-plant template DataFrame."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import build_per_plant_template

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df,
        cadence_s=300.0,
        run_id="plate_001",
    )
    df = build_per_plant_template(inputs)
    return tmp_path, inputs, df


def test_units_sidecar_exists_and_parses(tmp_csv_setup):
    """Sibling units.json exists, parses, every column present, every value in vocabulary."""
    from sleap_roots.circumnutation._constants import VALID_UNIT_VOCABULARY
    from sleap_roots.circumnutation._io import (
        default_units_for_template,
        gather_run_metadata,
        write_per_plant_csv,
    )

    tmp_path, _inputs, df = tmp_csv_setup
    units = default_units_for_template(df)
    metadata = gather_run_metadata(input_path="test_input.slp", run_id="plate_001")
    out_path = tmp_path / "traits_per_plant.csv"
    write_per_plant_csv(out_path, df, units, metadata)

    units_path = tmp_path / "traits_per_plant.units.json"
    assert units_path.exists()
    parsed = json.loads(units_path.read_text(encoding="utf-8"))
    for col in df.columns:
        assert col in parsed, f"Column {col} missing from units sidecar"
        assert parsed[col] in VALID_UNIT_VOCABULARY


def test_units_sidecar_utf8_round_trip(tmp_path):
    """`px²` survives a round-trip through the sidecar writer/reader."""
    from sleap_roots.circumnutation._io import read_units_sidecar, write_units_sidecar

    units = {"helix_signed_area": "px²"}
    out_path = tmp_path / "t.units.json"
    write_units_sidecar(out_path, units)
    parsed = read_units_sidecar(out_path)
    assert parsed == units
    assert "²" in out_path.read_text(encoding="utf-8")


def test_sidecar_path_with_dots_in_csv_stem(tmp_path, valid_trajectory_df):
    """CSV filenames with intermediate dots (`traits.per.plant.csv`) place the units sidecar at the right name.

    Regression test for Copilot PR #200 second-round finding: the
    original implementation used
    `csv_path.with_suffix("").with_suffix(".units.json")`, which
    incorrectly stripped intermediate dotted segments (so
    `traits.per.plant.csv` produced `traits.per.units.json`). The fix
    uses `csv_path.parent / f"{csv_path.stem}.units.json"`, which only
    strips the final `.csv` extension.
    """
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import (
        build_per_plant_template,
        default_units_for_template,
        gather_run_metadata,
        read_per_plant_csv,
        write_per_plant_csv,
    )

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df, cadence_s=300.0, run_id="plate_001"
    )
    df = build_per_plant_template(inputs)
    units = default_units_for_template(df)
    metadata = gather_run_metadata(input_path="test_input.slp", run_id="plate_001")

    # CSV filename with intermediate dots — this is the failure case.
    csv_path = tmp_path / "traits.per.plant.csv"
    write_per_plant_csv(csv_path, df, units, metadata)

    # Sidecar MUST be at traits.per.plant.units.json — not traits.per.units.json.
    correct_sidecar = tmp_path / "traits.per.plant.units.json"
    wrong_sidecar = tmp_path / "traits.per.units.json"
    assert (
        correct_sidecar.exists()
    ), f"Expected sidecar at {correct_sidecar}; got files: {list(tmp_path.iterdir())}"
    assert not wrong_sidecar.exists(), "Old bug: sidecar landed at the wrong path"

    # Reader must use the same convention to find the sibling.
    df_back, units_back, meta_back = read_per_plant_csv(csv_path)
    assert units_back == units


def test_run_metadata_required_fields(tmp_csv_setup):
    """run_metadata.json contains every required field."""
    from sleap_roots.circumnutation._io import (
        default_units_for_template,
        gather_run_metadata,
        read_run_metadata,
        write_per_plant_csv,
    )

    tmp_path, _inputs, df = tmp_csv_setup
    units = default_units_for_template(df)
    metadata = gather_run_metadata(input_path="test_input.slp", run_id="plate_001")
    out_path = tmp_path / "traits_per_plant.csv"
    write_per_plant_csv(out_path, df, units, metadata)

    meta_path = tmp_path / "run_metadata.json"
    assert meta_path.exists()
    parsed = read_run_metadata(meta_path)

    required = {
        "input_path",
        "sleap_roots_git_sha",
        "sleap_roots_version",
        "sleap_io_version",
        "python_version",
        "timestamp",
        "run_id",
        "_schema_version",
        "_constants_version",
        "_constants_snapshot",
    }
    assert required.issubset(parsed.keys())
    assert isinstance(parsed["_schema_version"], int)
    assert isinstance(parsed["_constants_version"], int)
    snapshot = parsed["_constants_snapshot"]
    for name in EXPECTED_CONSTANTS:
        assert name in snapshot, f"Constant {name} missing from snapshot"


def test_constants_snapshot_reflects_override():
    """Custom `ConstantsT` overrides surface in the snapshot; defaults remain unmodified."""
    from sleap_roots.circumnutation._constants import ConstantsT
    from sleap_roots.circumnutation._io import gather_run_metadata

    custom = ConstantsT(BAND_POWER_NOISE_RATIO=4)
    metadata = gather_run_metadata(input_path="x", run_id="r1", constants=custom)
    snapshot = metadata["_constants_snapshot"]

    assert snapshot["BAND_POWER_NOISE_RATIO"] == 4
    # Other constants retain their defaults
    assert snapshot["NOISE_MASK_K"] == EXPECTED_CONSTANTS["NOISE_MASK_K"]
    assert (
        snapshot["WAVELET_DEFAULT_TEMPORAL"]
        == EXPECTED_CONSTANTS["WAVELET_DEFAULT_TEMPORAL"]
    )


# ---------------------------------------------------------------------------
# Logging tests (spec: Per-module logger convention)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_name",
    [name for name, _, _ in STUB_MODULES] + ["_constants", "_types", "_io", "units"],
)
def test_module_logger_is_namespaced(module_name):
    """If the module declares a `logger`, it MUST be the module-namespaced logger."""
    full_name = f"sleap_roots.circumnutation.{module_name}"
    mod = importlib.import_module(full_name)
    if hasattr(mod, "logger"):
        assert mod.logger.name == full_name


def test_no_root_handlers_added_at_import():
    """Importing the package does not configure root logger handlers."""
    root = logging.getLogger()
    pre_handlers = list(root.handlers)
    # Force a fresh import in case earlier tests mutated state
    for name in list(sys.modules):
        if name.startswith("sleap_roots.circumnutation"):
            del sys.modules[name]
    importlib.import_module("sleap_roots.circumnutation")
    post_handlers = list(root.handlers)
    assert pre_handlers == post_handlers


# ---------------------------------------------------------------------------
# Review-round-2 regression tests (B1-B5, I1, I3, I4, I8, I9, I10, I11)
# ---------------------------------------------------------------------------

# B1 — inf must be rejected by all three validators


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_cadence_s_inf_rejected(valid_trajectory_df, bad):
    """B1: cadence_s=±inf raises ValueError naming cadence_s (spec says positive *finite*)."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="cadence_s"):
        CircumnutationInputs(trajectory_df=valid_trajectory_df, cadence_s=bad)


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_R_px_inf_rejected(valid_trajectory_df, bad):
    """B1: R_px=±inf raises ValueError naming R_px."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="R_px"):
        CircumnutationInputs(
            trajectory_df=valid_trajectory_df, cadence_s=300.0, R_px=bad
        )


@pytest.mark.parametrize("bad", [float("inf"), float("-inf")])
def test_convert_to_mm_inf_rejected(bad):
    """B1: convert_to_mm px_per_mm=±inf raises ValueError naming px_per_mm."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"length_px": [47.24]})
    units = {"length_px": "px"}
    with pytest.raises(ValueError, match="px_per_mm"):
        convert_to_mm(df, units, px_per_mm=bad)


# B2 — every stub whose tier needs cross-cutting overrides must accept constants= kwarg


STUBS_WITH_CONSTANTS_KWARG = [
    ("kinematics", "compute"),
    ("qc", "compute"),
    ("temporal_cwt", "compute_scaleogram"),
    ("psi_g", "compute_psi_g"),
    ("midline", "reconstruct"),
    ("spatial_cwt", "compute_scaleogram"),
    ("pipeline", "compute_traits"),
]


@pytest.mark.parametrize("module_name,callable_name", STUBS_WITH_CONSTANTS_KWARG)
def test_stub_accepts_constants_kwarg(module_name, callable_name):
    """B2: stubs whose tier PR will use ConstantsT accept `constants=...` kwarg now.

    Calling with `constants=...` must raise NotImplementedError (not TypeError).
    """
    mod = importlib.import_module(f"sleap_roots.circumnutation.{module_name}")
    fn = getattr(mod, callable_name)
    with pytest.raises(NotImplementedError):
        fn(constants=object())  # any sentinel; should not be argument-validated


# B3 — conflicting per-frame metadata raises clear error


def test_conflicting_genotype_across_frames_raises():
    """B3: build_per_plant_template raises ValueError when same plant has different genotype across frames."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import build_per_plant_template

    rows = []
    for frame, genotype in enumerate(["WT", "MOCK"]):
        rows.append(
            {
                "series": "S",
                "sample_uid": "U",
                "timepoint": "T0",
                "plate_id": "P",
                "plant_id": 1,
                "track_id": 1,
                "genotype": genotype,
                "treatment": np.nan,
                "frame": frame,
                "tip_x": 1.0,
                "tip_y": 2.0,
            }
        )
    df = pd.DataFrame(rows)
    inputs = CircumnutationInputs(trajectory_df=df, cadence_s=300.0)
    with pytest.raises(ValueError, match=r"genotype|conflict"):
        build_per_plant_template(inputs)


def test_track_id_nan_raises_clear_error(valid_trajectory_df):
    """B3 (related): NaN in track_id surfaces as a ValueError naming the field, not pandas IntCastingNaNError."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import build_per_plant_template

    df = valid_trajectory_df.copy()
    df["track_id"] = df["track_id"].astype("float64")
    df.loc[df.index[0], "track_id"] = np.nan
    inputs = CircumnutationInputs(trajectory_df=df, cadence_s=300.0)
    with pytest.raises(ValueError, match="track_id"):
        build_per_plant_template(inputs)


# B4 — writer validates units against PIPELINE_UNIT_VOCABULARY


def test_write_per_plant_csv_rejects_invalid_unit(tmp_path, valid_trajectory_df):
    """B4: writer raises ValueError naming the column when a unit string is out-of-vocabulary."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import (
        build_per_plant_template,
        default_units_for_template,
        gather_run_metadata,
        write_per_plant_csv,
    )

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df, cadence_s=300.0, run_id="r1"
    )
    df = build_per_plant_template(inputs)
    units = default_units_for_template(df)
    # Inject an out-of-vocabulary unit (mm leaks pure-pixel contract).
    units["track_id"] = "mm"
    metadata = gather_run_metadata(input_path="x", run_id="r1")
    out_path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match=r"track_id|mm"):
        write_per_plant_csv(out_path, df, units, metadata)
    # No files should have been written.
    assert not out_path.exists()


def test_write_per_plant_csv_rejects_unknown_unit(tmp_path, valid_trajectory_df):
    """B4: writer raises on any unit not in PIPELINE_UNIT_VOCABULARY (e.g. typo `kg`)."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import (
        build_per_plant_template,
        default_units_for_template,
        gather_run_metadata,
        write_per_plant_csv,
    )

    inputs = CircumnutationInputs(
        trajectory_df=valid_trajectory_df, cadence_s=300.0, run_id="r1"
    )
    df = build_per_plant_template(inputs)
    units = default_units_for_template(df)
    units["track_id"] = "kg"
    metadata = gather_run_metadata(input_path="x", run_id="r1")
    out_path = tmp_path / "out.csv"
    with pytest.raises(ValueError, match=r"track_id|kg"):
        write_per_plant_csv(out_path, df, units, metadata)


# B5 — synthetic stub must NOT have px_per_mm in signature


def test_synthetic_signature_has_no_px_per_mm():
    """B5: synthetic.generate_trajectory has no `px_per_mm` parameter (pure-pixel contract)."""
    import inspect

    from sleap_roots.circumnutation import synthetic

    sig = inspect.signature(synthetic.generate_trajectory)
    assert "px_per_mm" not in sig.parameters, (
        f"synthetic.generate_trajectory must not include px_per_mm in its signature "
        f"(pure-pixel contract); got params: {list(sig.parameters)}"
    )


# I1 — schema validates frame/tip_x/tip_y presence


@pytest.mark.parametrize("missing_col", ["frame", "tip_x", "tip_y"])
def test_missing_per_frame_column_raises(valid_trajectory_df, missing_col):
    """I1: trajectory_df missing frame/tip_x/tip_y raises ValueError naming the column."""
    from sleap_roots.circumnutation import CircumnutationInputs

    df = valid_trajectory_df.drop(columns=[missing_col])
    with pytest.raises(ValueError, match=missing_col):
        CircumnutationInputs(trajectory_df=df, cadence_s=300.0)


# I3 — run-metadata includes dependency versions and platform


def test_run_metadata_includes_dependency_versions_and_platform():
    """I3: gather_run_metadata captures numpy, scipy, pandas versions and platform."""
    from sleap_roots.circumnutation._io import gather_run_metadata

    metadata = gather_run_metadata(input_path="x", run_id="r1")
    for key in ("numpy_version", "scipy_version", "pandas_version", "platform"):
        assert key in metadata, f"Missing required key {key}"
        assert metadata[key], f"{key} must not be empty"


# I4 — convert_to_mm detects rename collisions


def test_convert_to_mm_rename_collision_raises():
    """I4: convert_to_mm raises when `_px` → `_mm` rename would collide with an existing `_mm` column."""
    from sleap_roots.circumnutation import convert_to_mm

    df = pd.DataFrame({"length_px": [47.24], "length_mm": [99.0]})
    units = {"length_px": "px", "length_mm": "mm"}
    with pytest.raises(ValueError, match=r"length_px|length_mm|collision|conflict"):
        convert_to_mm(df, units, px_per_mm=47.24)


# I8 — per-plant template asserts object dtype for string row-identity columns


def test_per_plant_template_string_columns_have_object_dtype(valid_trajectory_df):
    """I8: series/sample_uid/timepoint/plate_id/genotype/treatment have object dtype per spec."""
    from sleap_roots.circumnutation import CircumnutationInputs
    from sleap_roots.circumnutation._io import build_per_plant_template

    inputs = CircumnutationInputs(trajectory_df=valid_trajectory_df, cadence_s=300.0)
    df = build_per_plant_template(inputs)
    for col in (
        "series",
        "sample_uid",
        "timepoint",
        "plate_id",
        "genotype",
        "treatment",
    ):
        assert (
            df[col].dtype == object
        ), f"{col} dtype should be object, got {df[col].dtype}"


# I9 — cadence_s and R_px accept string forms of nan/inf and raise


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf"])
def test_cadence_s_string_nan_inf_raises(valid_trajectory_df, bad):
    """I9: cadence_s='nan' / 'inf' (string) is coerced via float() then rejected by validator."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="cadence_s"):
        CircumnutationInputs(trajectory_df=valid_trajectory_df, cadence_s=bad)


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_R_px_string_nan_inf_raises(valid_trajectory_df, bad):
    """I9: R_px='nan' / 'inf' (string) raises ValueError naming R_px."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="R_px"):
        CircumnutationInputs(
            trajectory_df=valid_trajectory_df, cadence_s=300.0, R_px=bad
        )


# I10 — _get_git_sha and version helpers return 'unknown' on failure


def test_get_git_sha_returns_unknown_when_subprocess_fails(monkeypatch):
    """I10: _get_git_sha gracefully degrades to 'unknown' when subprocess fails."""
    from sleap_roots.circumnutation import _io

    def _raise_filenotfound(*args, **kwargs):
        raise FileNotFoundError("git not on PATH (test mock)")

    monkeypatch.setattr(_io.subprocess, "run", _raise_filenotfound)
    assert _io._get_git_sha() == "unknown"


def test_get_sleap_roots_version_returns_unknown_when_import_fails(monkeypatch):
    """I10: _get_sleap_roots_version degrades to 'unknown' on any exception."""
    import builtins

    from sleap_roots.circumnutation import _io

    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "sleap_roots":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    assert _io._get_sleap_roots_version() == "unknown"


def test_get_sleap_io_version_returns_unknown_when_import_fails(monkeypatch):
    """I10: _get_sleap_io_version degrades to 'unknown' on any exception."""
    import builtins

    from sleap_roots.circumnutation import _io

    original_import = builtins.__import__

    def _mock_import(name, *args, **kwargs):
        if name == "sleap_io":
            raise ImportError("mocked")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _mock_import)
    assert _io._get_sleap_io_version() == "unknown"


# I11 — cadence_s and R_px reject Python booleans (int subclass coercion footgun)


def test_cadence_s_bool_rejected(valid_trajectory_df):
    """I11: cadence_s=True is rejected (bools are int subclass; would otherwise pass as 1.0)."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="cadence_s"):
        CircumnutationInputs(trajectory_df=valid_trajectory_df, cadence_s=True)


def test_R_px_bool_rejected(valid_trajectory_df):
    """I11: R_px=True is rejected (bool footgun)."""
    from sleap_roots.circumnutation import CircumnutationInputs

    with pytest.raises(ValueError, match="R_px"):
        CircumnutationInputs(
            trajectory_df=valid_trajectory_df, cadence_s=300.0, R_px=True
        )


def test_no_records_emitted_at_import(caplog):
    """No log records emitted from the package during import."""
    for name in list(sys.modules):
        if name.startswith("sleap_roots.circumnutation"):
            del sys.modules[name]

    with caplog.at_level(logging.DEBUG, logger="sleap_roots.circumnutation"):
        importlib.import_module("sleap_roots.circumnutation")

    package_records = [
        r for r in caplog.records if r.name.startswith("sleap_roots.circumnutation")
    ]
    assert package_records == []
