"""Tests for MultipleDicotPlatePipeline (issue #126 PR 1).

Dedicated test file (separate from the 2900-line test_trait_pipelines.py)
so the synthetic `.slp` helper and 18 plate-pipeline tests live together.

Section 3 of `openspec/changes/add-multiple-dicot-plate-pipeline/tasks.md`.
All integration tests round-trip through synthetic `.slp` files written via
`sio.save_slp` and loaded via `Series.load`, mirroring the idiom in
`tests/test_pixel_units.py`.
"""

import json
import logging
import math

import numpy as np
import pandas as pd
import pytest
import sleap_io as sio
from PIL import Image

from sleap_roots import Series
from sleap_roots.trait_pipelines import (
    DicotPipeline,
    MultipleDicotPlatePipeline,
    Pipeline,
)


# ---------------------------------------------------------------------------
# Synthetic `.slp` builder (shared helper)
# ---------------------------------------------------------------------------

N_NODES = 6  # ≥3 nodes required to avoid the #150 get_node_ind crash on <3-node roots.


def _default_primary(x_base: float, y_base: float = 50.0) -> np.ndarray:
    """Return a 6-node straight-vertical primary root at x=x_base, y in [50, 150].

    The root spans 100px vertically (base at y=50, tip at y=150). 6 nodes
    evenly spaced at 20px intervals.
    """
    return np.array(
        [
            [x_base, y_base + i * 20.0]
            for i in range(N_NODES)
        ],
        dtype=float,
    )


def _default_lateral(x_base: float, y_base: float = 80.0) -> np.ndarray:
    """Return a 6-node lateral root originating near the primary at x_base.

    The lateral extends diagonally 20px right and 100px down from its base.
    """
    return np.array(
        [
            [x_base + i * 4.0, y_base + i * 20.0]
            for i in range(N_NODES)
        ],
        dtype=float,
    )


def _nan_instance() -> np.ndarray:
    """Return an (N_NODES, 2) all-NaN instance (explicit, not the
    get_primary_points NaN placeholder of shape (1, 2, 2))."""
    return np.full((N_NODES, 2), np.nan, dtype=float)


def _build_synthetic_slp(
    tmp_path,
    series_name: str,
    primary_pts_per_frame,
    lateral_pts_per_frame,
    csv_content: str = None,
) -> Series:
    """Build a synthetic plate Series by round-tripping through `.slp` files.

    Args:
        tmp_path: pytest tmp_path fixture.
        series_name: Series identifier used as plant_qr_code in the CSV lookup.
        primary_pts_per_frame: list of ndarrays, one per frame, each of shape
            (n_primaries, N_NODES, 2). Pass `None` for an empty frame (no
            primary instances) — but prefer an explicit all-NaN array to avoid
            the `(1, 2, 2)` NaN placeholder from `Series.get_primary_points`
            when the node count matters.
        lateral_pts_per_frame: list of ndarrays, one per frame, each of shape
            (n_laterals, N_NODES, 2). Pass a `(0, N_NODES, 2)` array for zero
            laterals.
        csv_content: Optional CSV text content (not a path). When provided,
            written to a sidecar CSV and attached via csv_path.

    Returns:
        A `Series` loaded via `Series.load` with `primary_path`, `lateral_path`,
        and optionally `csv_path`.
    """
    n_frames = len(primary_pts_per_frame)
    # Shared image backing for all frames.
    image_h, image_w = 400, 300
    img_array = np.zeros((image_h, image_w), dtype=np.uint8)
    tif_path = tmp_path / f"{series_name}.tif"
    # Write one single-frame TIFF; sio.Video will repeat it across frames.
    Image.fromarray(img_array).save(tif_path.as_posix(), dpi=(72, 72))
    video = sio.Video.from_filename(tif_path.as_posix())

    skeleton_primary = sio.Skeleton(
        nodes=[sio.Node(f"pnode_{i}") for i in range(N_NODES)]
    )
    skeleton_lateral = sio.Skeleton(
        nodes=[sio.Node(f"lnode_{i}") for i in range(N_NODES)]
    )

    primary_frames = []
    lateral_frames = []
    for frame_idx in range(n_frames):
        p_insts = [
            sio.Instance.from_numpy(pts, skeleton=skeleton_primary)
            for pts in primary_pts_per_frame[frame_idx]
        ]
        l_insts = [
            sio.Instance.from_numpy(pts, skeleton=skeleton_lateral)
            for pts in lateral_pts_per_frame[frame_idx]
        ]
        primary_frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=p_insts)
        )
        lateral_frames.append(
            sio.LabeledFrame(video=video, frame_idx=frame_idx, instances=l_insts)
        )

    primary_labels = sio.Labels(
        labeled_frames=primary_frames, skeletons=[skeleton_primary], videos=[video]
    )
    lateral_labels = sio.Labels(
        labeled_frames=lateral_frames, skeletons=[skeleton_lateral], videos=[video]
    )

    primary_path = tmp_path / f"{series_name}.primary.predictions.slp"
    lateral_path = tmp_path / f"{series_name}.lateral.predictions.slp"
    sio.save_slp(primary_labels, primary_path.as_posix())
    sio.save_slp(lateral_labels, lateral_path.as_posix())

    csv_path = None
    if csv_content is not None:
        csv_path = tmp_path / f"{series_name}.csv"
        csv_path.write_text(csv_content)

    return Series.load(
        series_name=series_name,
        primary_path=primary_path.as_posix(),
        lateral_path=lateral_path.as_posix(),
        csv_path=csv_path.as_posix() if csv_path else None,
    )


# ---------------------------------------------------------------------------
# 3a. Unit tests
# ---------------------------------------------------------------------------


def test_multiple_dicot_plate_pipeline_define_traits():
    """Req 1 scenarios: class is a Pipeline subclass; DAG has expected shape."""
    pipeline = MultipleDicotPlatePipeline()
    assert isinstance(pipeline, Pipeline)

    traits = pipeline.define_traits()
    names = [t.name for t in traits]
    expected_in_order = [
        "primary_pts_no_nans",
        "lateral_pts_no_nans",
        "detected_count",
        "plant_associations_dict",
        "plant_id_order",
    ]
    # Assert each expected name appears, in order.
    for name in expected_in_order:
        assert name in names, f"missing TraitDef: {name}"
    indices = [names.index(n) for n in expected_in_order]
    assert indices == sorted(indices), (
        f"TraitDefs not in expected order: got indices {indices} for {expected_in_order}"
    )

    by_name = {t.name: t for t in traits}
    assert by_name["detected_count"].scalar is True
    assert by_name["detected_count"].include_in_csv is True

    from sleap_roots.points import (
        filter_roots_with_nans,
        filter_plants_with_unexpected_ct,
        associate_lateral_to_primary,
        argsort_primaries_by_base_x,
    )
    from sleap_roots.points import get_count

    # Plates skip count-filter entirely (design D2).
    for t in traits:
        assert t.fn is not filter_plants_with_unexpected_ct, (
            "plate pipeline must NOT use filter_plants_with_unexpected_ct"
        )

    assert by_name["primary_pts_no_nans"].fn is filter_roots_with_nans
    assert by_name["primary_pts_no_nans"].input_traits == ["primary_pts"]
    assert by_name["lateral_pts_no_nans"].fn is filter_roots_with_nans
    assert by_name["lateral_pts_no_nans"].input_traits == ["lateral_pts"]
    assert by_name["detected_count"].fn is get_count
    assert by_name["detected_count"].input_traits == ["primary_pts_no_nans"]
    assert by_name["plant_associations_dict"].fn is associate_lateral_to_primary
    assert by_name["plant_associations_dict"].input_traits == [
        "primary_pts_no_nans",
        "lateral_pts_no_nans",
    ]
    assert by_name["plant_id_order"].fn is argsort_primaries_by_base_x
    assert by_name["plant_id_order"].input_traits == ["plant_associations_dict"]


def test_multiple_dicot_plate_pipeline_top_level_import():
    """Req 1 scenario: importable from top-level `sleap_roots` namespace."""
    import sleap_roots
    from sleap_roots import MultipleDicotPlatePipeline as PlatePipelineTop
    from sleap_roots.trait_pipelines import (
        MultipleDicotPlatePipeline as PlatePipelineNested,
    )
    assert PlatePipelineTop is PlatePipelineNested
    assert hasattr(sleap_roots, "MultipleDicotPlatePipeline")


def test_multiple_dicot_plate_pipeline_get_initial_frame_traits_keys(tmp_path):
    """Req 1 scenario: returned dict keys match the 5-key contract."""
    primaries = [np.stack([_default_primary(100.0)])]  # (1, N_NODES, 2)
    laterals = [np.stack([_default_lateral(100.0)])]  # (1, N_NODES, 2)
    series = _build_synthetic_slp(
        tmp_path, "test_keys", primaries, laterals, csv_content=None
    )
    pipeline = MultipleDicotPlatePipeline()
    initial = pipeline.get_initial_frame_traits(series, 0)
    assert set(initial.keys()) == {
        "primary_pts",
        "lateral_pts",
        "primary_sleap_idxs",
        "lateral_sleap_idxs",
        "expected_count",
    }


def test_multiple_dicot_plate_pipeline_get_initial_frame_traits_sleap_idxs(tmp_path):
    """Req 1 scenario: primary_sleap_idxs preserves original indices through NaN filter.

    4 primary instances of which SLEAP index 1 is all-NaN → primary_sleap_idxs
    is [0, 2, 3] (original indices of survivors), NOT [0, 1, 2] (post-filter).
    """
    primaries = [
        np.stack(
            [
                _default_primary(100.0),
                _nan_instance(),
                _default_primary(200.0),
                _default_primary(300.0),
            ]
        )
    ]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path, "test_sleap_idxs", primaries, laterals, csv_content=None
    )
    pipeline = MultipleDicotPlatePipeline()
    initial = pipeline.get_initial_frame_traits(series, 0)
    assert initial["primary_sleap_idxs"] == [0, 2, 3]
    # Raw pre-filter shape preserved — Req 1 scenario c explicitly requires this.
    assert initial["primary_pts"].shape == (4, N_NODES, 2)


# ---------------------------------------------------------------------------
# 3b. Integration tests
# ---------------------------------------------------------------------------


def test_multiple_dicot_plate_pipeline_basic(tmp_path):
    """Req 3 scenario: 3 primaries sorted left-to-right; plant_id + SLEAP idx mapped."""
    # SLEAP instance order: 0=x200, 1=x50, 2=x100
    primaries = [
        np.stack(
            [
                _default_primary(200.0),
                _default_primary(50.0),
                _default_primary(100.0),
            ]
        )
    ]
    # One lateral per primary.
    laterals = [
        np.stack(
            [
                _default_lateral(200.0),
                _default_lateral(50.0),
                _default_lateral(100.0),
            ]
        )
    ]
    series = _build_synthetic_slp(
        tmp_path, "test_basic", primaries, laterals, csv_content=None
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    plants = result["plants"]
    assert len(plants) == 3

    # Left-to-right: x=50 → plant_id=0 (SLEAP 1), x=100 → plant_id=1 (SLEAP 2),
    # x=200 → plant_id=2 (SLEAP 0).
    assert plants[0]["plant_id"] == 0
    assert plants[0]["primary_sleap_idx"] == 1
    assert plants[1]["plant_id"] == 1
    assert plants[1]["primary_sleap_idx"] == 2
    assert plants[2]["plant_id"] == 2
    assert plants[2]["primary_sleap_idx"] == 0

    for plant in plants:
        assert plant["detected_count"] == 3
        # Unchanged DicotPipeline trait names — no `_root_` infix or plate renames.
        for trait_name in (
            "primary_length",
            "lateral_count",
            "network_length",
            "primary_base_tip_dist",
        ):
            assert trait_name in plant["traits"]
        # Explicit negative assertion: no renamed variants.
        for forbidden in (
            "primary_root_length",
            "lateral_root_count",
            "avg_lateral_root_length",
        ):
            assert forbidden not in plant["traits"]


def test_multiple_dicot_plate_pipeline_sleap_idx_traceability(tmp_path):
    """Req 3 scenario: NaN primary at SLEAP index 1 drops, survivors keep {0, 2}."""
    primaries = [
        np.stack(
            [
                _default_primary(100.0),
                _nan_instance(),
                _default_primary(200.0),
            ]
        )
    ]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path, "test_trace", primaries, laterals, csv_content=None
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    plants = result["plants"]
    assert len(plants) == 2
    sleap_idx_set = {p["primary_sleap_idx"] for p in plants}
    assert sleap_idx_set == {0, 2}


def _plate_csv(series_name, expected_count=None, group=None, qc_cylinder=0):
    """Build a plate metadata CSV compatible with Series.expected_count/group/qc_fail."""
    rows = ["plant_qr_code,genotype,number_of_plants_cylinder,qc_cylinder"]
    rows.append(
        f"{series_name},{group or ''},"
        f"{expected_count if expected_count is not None else ''},{qc_cylinder}"
    )
    return "\n".join(rows) + "\n"


def test_multiple_dicot_plate_pipeline_expected_count_none(tmp_path, caplog):
    """Req 3 scenario: missing expected_count tolerated, both flags False, no log."""
    primaries = [
        np.stack(
            [
                _default_primary(100.0),
                _default_primary(200.0),
                _default_primary(300.0),
            ]
        )
    ]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path, "test_none", primaries, laterals, csv_content=None
    )
    with caplog.at_level(logging.WARNING, logger="sleap_roots.trait_pipelines"):
        result = MultipleDicotPlatePipeline().compute_plate_traits(series)

    assert pd.isna(result["qc_fail"])
    for plant in result["plants"]:
        assert pd.isna(plant["expected_count"])
        assert plant["count_validated"] is False
        assert plant["count_mismatch"] is False
        assert isinstance(plant["detected_count"], int) or (
            isinstance(plant["detected_count"], (np.integer,))
        )
        assert plant["detected_count"] >= 0
    plate_warnings = [
        r for r in caplog.records
        if r.name == "sleap_roots.trait_pipelines" and r.levelno >= logging.WARNING
    ]
    assert plate_warnings == [], (
        f"expected no WARNING records, got: {[r.message for r in plate_warnings]}"
    )


def test_multiple_dicot_plate_pipeline_expected_count_mismatch(tmp_path, caplog):
    """Req 3 scenario: mismatch sets count_mismatch=True and logs per frame."""
    primaries = [
        np.stack(
            [
                _default_primary(100.0),
                _default_primary(200.0),
                _default_primary(300.0),
            ]
        )
    ]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path,
        "test_mismatch",
        primaries,
        laterals,
        csv_content=_plate_csv("test_mismatch", expected_count=2),
    )
    with caplog.at_level(logging.WARNING, logger="sleap_roots.trait_pipelines"):
        result = MultipleDicotPlatePipeline().compute_plate_traits(series)

    plants = result["plants"]
    assert len(plants) == 3
    for plant in plants:
        assert plant["expected_count"] == 2
        assert plant["detected_count"] == 3
        assert plant["count_mismatch"] is True
        assert plant["count_validated"] is False

    plate_warnings = [
        r for r in caplog.records
        if r.name == "sleap_roots.trait_pipelines" and r.levelno >= logging.WARNING
    ]
    assert len(plate_warnings) >= 1, "expected at least one WARNING record"
    combined = " ".join(r.message for r in plate_warnings)
    assert "3" in combined and "2" in combined and "frame" in combined.lower()


def test_multiple_dicot_plate_pipeline_expected_count_match(tmp_path, caplog):
    """Req 3 scenario: match sets count_validated=True and logs nothing."""
    primaries = [
        np.stack(
            [
                _default_primary(100.0),
                _default_primary(200.0),
                _default_primary(300.0),
            ]
        )
    ]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path,
        "test_match",
        primaries,
        laterals,
        csv_content=_plate_csv("test_match", expected_count=3),
    )
    with caplog.at_level(logging.WARNING, logger="sleap_roots.trait_pipelines"):
        result = MultipleDicotPlatePipeline().compute_plate_traits(series)

    for plant in result["plants"]:
        assert plant["count_validated"] is True
        assert plant["count_mismatch"] is False

    plate_warnings = [
        r for r in caplog.records
        if r.name == "sleap_roots.trait_pipelines" and r.levelno >= logging.WARNING
    ]
    assert plate_warnings == []


def test_multiple_dicot_plate_pipeline_empty_frame(tmp_path):
    """Req 3 scenario: all-NaN primaries → empty plants list, no exception."""
    primaries = [np.stack([_nan_instance(), _nan_instance()])]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path, "test_empty", primaries, laterals, csv_content=None
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    assert result["plants"] == []


def test_multiple_dicot_plate_pipeline_zero_laterals(tmp_path):
    """Req 3 scenario: zero-laterals plant yields lateral_count==0 (not 1)."""
    primaries = [np.stack([_default_primary(100.0)])]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path, "test_zero_lat", primaries, laterals, csv_content=None
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    plant = result["plants"][0]
    assert plant["traits"]["lateral_count"] == 0
    lat_lengths = np.asarray(plant["traits"]["lateral_lengths"])
    assert lat_lengths.shape[0] == 0, f"expected empty array, got {lat_lengths!r}"
    # network_length == primary_length when no laterals.
    assert plant["traits"]["network_length"] == pytest.approx(
        plant["traits"]["primary_length"]
    )
    assert plant["lateral_sleap_idxs"] == []


def test_multiple_dicot_plate_pipeline_duplicate_lateral_coords(tmp_path):
    """Req 3 scenario: bit-identical duplicate laterals map to distinct SLEAP indices.

    Fails if the implementation uses np.array_equal first-match back-mapping;
    passes only if SLEAP indices are tracked alongside the distance association.
    """
    primaries = [np.stack([_default_primary(100.0)])]
    # Two laterals with bit-identical node coordinates.
    lateral_pts = _default_lateral(100.0)
    laterals = [np.stack([lateral_pts.copy(), lateral_pts.copy()])]
    series = _build_synthetic_slp(
        tmp_path, "test_dup", primaries, laterals, csv_content=None
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    plants = result["plants"]
    assert len(plants) == 1
    idxs = plants[0]["lateral_sleap_idxs"]
    assert len(idxs) == 2
    assert set(idxs) == {0, 1}


def test_multiple_dicot_plate_pipeline_timelapse_shape(tmp_path):
    """Req 3 scenario: 2 frames × 3 primaries → 6 rows, grouped by frame."""
    three_primaries = np.stack(
        [
            _default_primary(100.0),
            _default_primary(200.0),
            _default_primary(300.0),
        ]
    )
    three_laterals = np.stack(
        [
            _default_lateral(100.0),
            _default_lateral(200.0),
            _default_lateral(300.0),
        ]
    )
    primaries = [three_primaries.copy(), three_primaries.copy()]
    laterals = [three_laterals.copy(), three_laterals.copy()]
    series = _build_synthetic_slp(
        tmp_path, "test_timelapse", primaries, laterals, csv_content=None
    )
    result = MultipleDicotPlatePipeline().compute_plate_traits(series)
    plants = result["plants"]
    assert len(plants) == 6
    assert [p["frame"] for p in plants[:3]] == [0, 0, 0]
    assert [p["frame"] for p in plants[3:]] == [1, 1, 1]
    assert [p["plant_id"] for p in plants[:3]] == [0, 1, 2]
    assert [p["plant_id"] for p in plants[3:]] == [0, 1, 2]


def test_multiple_dicot_plate_pipeline_csv_output(tmp_path):
    """Req 5 scenario: CSV column order + DicotPipeline.csv_traits unchanged."""
    primaries = [
        np.stack(
            [
                _default_primary(100.0),
                _default_primary(200.0),
            ]
        )
    ]
    laterals = [
        np.stack(
            [
                _default_lateral(100.0),
                _default_lateral(200.0),
            ]
        )
    ]
    series = _build_synthetic_slp(
        tmp_path, "test_csv", primaries, laterals, csv_content=None
    )
    MultipleDicotPlatePipeline().compute_plate_traits(
        series, write_csv=True, output_dir=tmp_path.as_posix()
    )
    csv_path = tmp_path / "test_csv.plate_traits.csv"
    df = pd.read_csv(csv_path.as_posix())
    meta_cols = list(df.columns)[:6]
    assert meta_cols == [
        "series",
        "frame",
        "plant_id",
        "primary_sleap_idx",
        "expected_count",
        "detected_count",
    ]
    trait_cols = list(df.columns)[6:]
    assert trait_cols == DicotPipeline().csv_traits, (
        f"plate trait columns don't match DicotPipeline.csv_traits:\n"
        f"  plate: {trait_cols}\n"
        f"  dicot: {DicotPipeline().csv_traits}"
    )
    for col in df.columns:
        assert "_root_" not in col, f"forbidden _root_ infix in column: {col}"
    assert "lateral_sleap_idxs" not in df.columns
    assert "count_validated" not in df.columns
    assert "count_mismatch" not in df.columns


def test_multiple_dicot_plate_pipeline_csv_missing_expected_count(tmp_path):
    """Req 5 scenario: missing expected_count renders as empty cell in CSV."""
    primaries = [np.stack([_default_primary(100.0)])]
    laterals = [np.stack([_default_lateral(100.0)])]
    series = _build_synthetic_slp(
        tmp_path, "test_csv_none", primaries, laterals, csv_content=None
    )
    MultipleDicotPlatePipeline().compute_plate_traits(
        series, write_csv=True, output_dir=tmp_path.as_posix()
    )
    csv_path = tmp_path / "test_csv_none.plate_traits.csv"
    df = pd.read_csv(csv_path.as_posix())
    assert pd.isna(df.loc[0, "expected_count"])
    assert df.loc[0, "detected_count"] == 1
    assert "count_validated" not in df.columns
    assert "count_mismatch" not in df.columns


def test_multiple_dicot_plate_pipeline_json_output(tmp_path):
    """Req 4 scenario: written JSON is self-contained and round-trips cleanly."""
    primaries = [
        np.stack(
            [
                _default_primary(100.0),
                _default_primary(200.0),
            ]
        )
    ]
    laterals = [
        np.stack(
            [
                _default_lateral(100.0),
                _default_lateral(200.0),
            ]
        )
    ]
    series = _build_synthetic_slp(
        tmp_path, "test_json", primaries, laterals, csv_content=None
    )
    MultipleDicotPlatePipeline().compute_plate_traits(
        series, write_json=True, output_dir=tmp_path.as_posix()
    )
    json_path = tmp_path / "test_json.plate_traits.json"
    loaded = json.loads(json_path.read_text())

    assert set(loaded.keys()) == {
        "schema_version",
        "units",
        "series",
        "group",
        "qc_fail",
        "expected_count",
        "plants",
    }
    assert loaded["schema_version"] == 1
    assert loaded["units"] == {
        "lengths": "pixels",
        "angles": "degrees",
        "counts": "unitless",
        "ratios": "dimensionless",
    }

    for plant in loaded["plants"]:
        assert isinstance(plant["primary_points"], list)
        assert all(len(p) == 2 for p in plant["primary_points"])
        assert isinstance(plant["lateral_points"], list)
        for lat in plant["lateral_points"]:
            assert isinstance(lat, list)
            assert all(len(p) == 2 for p in lat)
        assert isinstance(plant["primary_sleap_idx"], int)
        assert isinstance(plant["lateral_sleap_idxs"], list)
        assert all(isinstance(i, int) for i in plant["lateral_sleap_idxs"])
        assert isinstance(plant["count_validated"], bool)
        assert isinstance(plant["count_mismatch"], bool)
        for trait_name in (
            "primary_length",
            "lateral_count",
            "network_length",
            "primary_base_tip_dist",
        ):
            assert trait_name in plant["traits"]


def test_multiple_dicot_plate_pipeline_json_rfc8259_valid_with_nested_nan(tmp_path):
    """Req 4 scenario: NaN deep in traits dict becomes JSON null; file is RFC-8259 valid.

    Zero-laterals plant produces NaN in traits (e.g. lateral_angles_distal).
    Series without CSV ensures top-level expected_count is also NaN.
    Both must survive as `null` in JSON, and no bare `NaN` literal must leak.
    """
    primaries = [np.stack([_default_primary(100.0)])]
    laterals = [np.empty((0, N_NODES, 2), dtype=float)]
    series = _build_synthetic_slp(
        tmp_path, "test_rfc8259", primaries, laterals, csv_content=None
    )
    MultipleDicotPlatePipeline().compute_plate_traits(
        series, write_json=True, output_dir=tmp_path.as_posix()
    )
    json_path = tmp_path / "test_rfc8259.plate_traits.json"
    text = json_path.read_text()
    assert "NaN" not in text, (
        f"bare NaN literal found in JSON — invalid per RFC 8259:\n{text[:500]}"
    )

    def _raise_on_constant(s):
        raise ValueError(f"bare constant {s!r}")

    # Strict RFC-8259 parse must NOT raise (no NaN, Infinity, -Infinity in text).
    json.loads(text, parse_constant=_raise_on_constant)

    loaded = json.loads(text)
    assert loaded["expected_count"] is None
    # Zero-laterals plant: lateral_sleap_idxs and lateral_points must be [].
    plant = loaded["plants"][0]
    assert plant["lateral_sleap_idxs"] == []
    assert plant["lateral_points"] == []
    # A trait that is deterministically NaN on zero-lateral plants:
    # get_root_angle on empty-node-ind input returns scalar NaN.
    assert plant["traits"]["lateral_angles_distal"] is None


def test_compute_batch_plate_traits(tmp_path):
    """Req 6 scenarios: batch concatenation of per-series rows + batch JSON list."""
    primaries_a = [
        np.stack(
            [
                _default_primary(100.0),
                _default_primary(200.0),
            ]
        )
    ]
    laterals_a = [
        np.stack(
            [
                _default_lateral(100.0),
                _default_lateral(200.0),
            ]
        )
    ]
    series_a = _build_synthetic_slp(
        tmp_path, "seriesA", primaries_a, laterals_a, csv_content=None
    )
    primaries_b = [
        np.stack(
            [
                _default_primary(50.0),
                _default_primary(150.0),
                _default_primary(250.0),
            ]
        )
    ]
    laterals_b = [
        np.stack(
            [
                _default_lateral(50.0),
                _default_lateral(150.0),
                _default_lateral(250.0),
            ]
        )
    ]
    series_b = _build_synthetic_slp(
        tmp_path, "seriesB", primaries_b, laterals_b, csv_content=None
    )

    pipeline = MultipleDicotPlatePipeline()
    df = pipeline.compute_batch_plate_traits([series_a, series_b])
    assert len(df) == 5
    assert list(df["series"][:2].unique()) == ["seriesA"]
    assert list(df["series"][2:].unique()) == ["seriesB"]

    pipeline.compute_batch_plate_traits(
        [series_a, series_b],
        write_json=True,
        output_dir=tmp_path.as_posix(),
        json_name="batch.json",
    )
    batch_text = (tmp_path / "batch.json").read_text()
    assert "NaN" not in batch_text

    def _raise_on_constant(s):
        raise ValueError(f"bare constant {s!r}")

    parsed = json.loads(batch_text, parse_constant=_raise_on_constant)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    for per_series in parsed:
        assert set(per_series.keys()) == {
            "schema_version",
            "units",
            "series",
            "group",
            "qc_fail",
            "expected_count",
            "plants",
        }
