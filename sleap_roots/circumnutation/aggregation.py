"""Per-genotype aggregation layer for the circumnutation pipeline (PR #15).

This module owns :func:`aggregate_by_genotype`, the first post-pipeline
aggregation layer. It consumes the composed per-plant trait frame produced by
:meth:`sleap_roots.circumnutation.pipeline.CircumnutationPipeline.compute_traits`
and produces a per-genotype summary: median ± IQR across plants grouped by
``(plate_id, genotype, treatment)`` (theory.md §7.7), with an explicit
``n_plants_passing_qc`` count and exclusion reasons.

Design notes (see ``openspec/changes/add-circumnutation-per-genotype-aggregation``):

- **Per-trait NaN-skip.** Medians/IQRs skip NaN per trait, so the number of
  finite values behind a given ``<trait>_median`` may be **smaller** than
  ``n_plants_passing_qc`` (a plant can pass QC yet carry NaN for a particular
  trait — e.g. a non-nutating plant's ``T_nutation_*``). ``n_plants_passing_qc``
  is therefore an upper bound on the per-trait sample size, and a NaN
  ``<trait>_iqr`` beside a finite ``<trait>_median`` means **fewer than 2 finite
  values** for that trait — not zero spread. Per-trait finite-count columns are
  deferred. A ``<trait>_iqr`` whose source trait is itself a dispersion measure
  (``T_nutation_iqr``, ``lambda_spatial_variation``, ``lambda_spatial_mad_px``,
  ``worst_step_ratio``) is a spread-of-spreads.
- **frac_nutating** is the mean of ``is_nutating`` over **all** passing plants;
  its denominator differs from a per-trait median's (finite-only).
- **handedness** is summarized by ``handedness_mode`` (majority sign, explicit
  value counts) + ``handedness_consensus_frac``. A ``handedness_consensus_frac``
  at or near ``0.5`` means the mode is a tie-break artifact, not a majority.
- **principal_axis_angle** (a wrapping circular angle) is dropped: a linear
  median/IQR is unsound near ±π and it is an absolute per-plant reference
  direction with arbitrary cross-plant orientation. A circular-statistics
  summary is deferred to a follow-up issue ("circular-statistics per-genotype
  summary for principal_axis_angle").
- **helix_signed_area_px2** is aggregated by its **magnitude**
  (``helix_signed_area_abs_px2_median``/``_iqr``); its sign is chirality, which
  is bimodal within a genotype, so a signed cross-plant median would cancel
  toward zero. Chirality direction is carried by ``handedness_mode`` /
  ``handedness_consensus_frac``. Other signed traits (``v_long_signed*``,
  ``v_lat_signed*``, ``period_residual_vs_derr_reference``) have
  per-plant-consistent sign semantics and are aggregated signed.
- **growth_axis_unreliable** is dropped: it is a ``track_is_clean`` failure
  clause, so it is always ``False`` among passing plants.
- **Excluded-plant audit.** The per-genotype row records counts + a clause→count
  summary, not the excluded plants' identities. The specific excluded plants of a
  group remain auditable by re-filtering the per-plant trait CSV for rows of that
  group with ``track_is_clean == False``.
"""

import logging
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
import scipy.stats

from sleap_roots.circumnutation import qc
from sleap_roots.circumnutation._io import _validate_units_coverage

logger = logging.getLogger(__name__)


# Group key: plates are never pooled (plate is a batch confound; theory.md §7.7
# "reports per-plate_id separately").
_GROUP_KEYS: tuple = ("plate_id", "genotype", "treatment")

# Identity columns dropped from the per-genotype output (vary within a group).
_AGG_IDENTITY_DROP: tuple = (
    "series",
    "sample_uid",
    "timepoint",
    "plant_id",
    "track_id",
)

# Identity columns retained as the group keys.
_AGG_IDENTITY_KEEP: tuple = _GROUP_KEYS

# Columns given bespoke handling — NOT a plain ``<trait>_median``/``_iqr``.
_AGG_SPECIAL_COLUMNS: tuple = (
    "is_nutating",  # -> frac_nutating
    "handedness",  # -> handedness_mode + handedness_consensus_frac
    "track_is_clean",  # the QC gate; drives counts (not a trait)
    "qc_failure_reason",  # feeds exclusion_reasons (not a trait)
    "growth_axis_unreliable",  # dropped (always False among passing plants)
    "principal_axis_angle",  # dropped (wrapping circular angle)
    "helix_signed_area_px2",  # aggregated as a magnitude (see below)
)

# Source column -> output prefix for traits aggregated by their magnitude |x|.
_MAGNITUDE_COLUMNS: dict = {"helix_signed_area_px2": "helix_signed_area_abs_px2"}

# Composed-schema columns the function requires to compute the special outputs.
_REQUIRED_COLUMNS: tuple = (
    "track_is_clean",
    "is_nutating",
    "handedness",
    "qc_failure_reason",
)

# Unit tags that mark a column as non-numeric (never a float trait).
_NON_FLOAT_UNITS = frozenset({"int", "bool", "string"})

# A spread is undefined below this many finite values (structural minimum, NOT a
# tunable constant — no _CONSTANTS_VERSION bump).
_MIN_FINITE_FOR_IQR = 2

# Dimensionless unit string (em-dash), member of PIPELINE_UNIT_VOCABULARY.
_DIMENSIONLESS = "—"


def _stat_plan(columns, units) -> List[Tuple[str, str, bool]]:
    """Build the ordered ``(source_col, out_prefix, is_magnitude)`` aggregation plan.

    A column gets a ``<prefix>_median``/``<prefix>_iqr`` pair iff it is a
    numeric-unit trait that is not a row-identity column and not in
    :data:`_AGG_SPECIAL_COLUMNS` — except ``helix_signed_area_px2``, which is
    aggregated by its magnitude (``is_magnitude=True``). Order follows the input
    column order.

    Args:
        columns: The per-plant DataFrame's columns, in order.
        units: Column-name → unit-string mapping (1:1 over ``columns``).

    Returns:
        Ordered list of ``(source_col, out_prefix, is_magnitude)``.
    """
    plan: List[Tuple[str, str, bool]] = []
    identity = set(_AGG_IDENTITY_DROP) | set(_AGG_IDENTITY_KEEP)
    for col in columns:
        if col in identity:
            continue
        if col in _MAGNITUDE_COLUMNS:
            plan.append((col, _MAGNITUDE_COLUMNS[col], True))
            continue
        if col in _AGG_SPECIAL_COLUMNS:
            continue
        if units[col] in _NON_FLOAT_UNITS:
            continue
        plan.append((col, col, False))
    return plan


def _float_trait_columns(columns, units) -> List[str]:
    """Return the directly-aggregated (non-magnitude) float trait columns, in order."""
    return [src for src, _prefix, is_mag in _stat_plan(columns, units) if not is_mag]


def _validate_one_row_per_plant(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if any plant maps to more than one row within its group.

    Enforces the current track↔plant 1:1 invariant: each
    ``(plate_id, genotype, treatment, plant_id)`` combination must map to exactly
    one row. A future track↔plant divergence (one plant carrying multiple
    tracks/rows) would make a per-row aggregation silently aggregate per-track;
    PR #15 fails loud instead. The two-level per-plant collapse (theory.md §7.7's
    literal shape) is deferred to a follow-up issue ("two-level plant_id collapse
    generalization of aggregate_by_genotype").

    Args:
        df: The per-plant DataFrame.

    Raises:
        ValueError: Naming the offending ``plant_id`` if any group/plant maps to
            more than one row.
    """
    if df.empty:
        return
    key = list(_GROUP_KEYS) + ["plant_id"]
    sizes = df.groupby(key, dropna=False, sort=False).size()
    dupes = sizes[sizes > 1]
    if len(dupes) > 0:
        offending = sorted({idx[-1] for idx in dupes.index})
        raise ValueError(
            f"aggregate_by_genotype: {len(dupes)} (plate_id, genotype, "
            f"treatment, plant_id) combination(s) map to more than one row "
            f"(offending plant_id(s): {offending}). The current track↔plant 1:1 "
            f"invariant is violated — one plant carries multiple tracks/rows. The "
            f"per-plant 'two-level plant_id collapse' is a deferred follow-up; "
            f"PR #15 raises rather than silently aggregate per-track."
        )


def _nanmedian(values: np.ndarray) -> float:
    """Median over finite values; NaN when none. No RuntimeWarning (pre-filtered)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _iqr_or_nan(values: np.ndarray) -> float:
    """IQR (Q75 − Q25, linear) over finite values; NaN when fewer than 2.

    Uses ``scipy.stats.iqr`` with ``interpolation="linear"`` pinned for
    cross-version determinism. Values are pre-filtered to finite so no
    ``SmallSampleWarning`` fires.
    """
    finite = values[np.isfinite(values)]
    if finite.size < _MIN_FINITE_FOR_IQR:
        return float("nan")
    return float(scipy.stats.iqr(finite, nan_policy="omit", interpolation="linear"))


def _handedness_mode_consensus(values: np.ndarray) -> Tuple[int, float]:
    """Return ``(mode, consensus_frac)`` for an array of integer handedness signs.

    The mode is the most frequent value by explicit value counts (NOT
    ``pandas.Series.mode``); ties are broken by smallest ``abs(value)`` first
    then smallest signed value, so a ``{+1, -1}`` tie resolves to ``-1`` and an
    ``{0, +1}`` tie resolves to ``0``. ``consensus_frac`` is the fraction of
    values equal to the mode.
    """
    counts = Counter(int(v) for v in values)
    max_count = max(counts.values())
    tied = [v for v, c in counts.items() if c == max_count]
    mode = sorted(tied, key=lambda v: (abs(v), v))[0]
    consensus = max_count / len(values)
    return int(mode), float(consensus)


def _build_exclusion_reasons(reasons) -> str:
    """Render the clause→count summary string over excluded plants' reasons.

    Clauses are split on ``", "`` (the ``qc`` join separator), counted by
    incidence (a plant failing multiple clauses counts toward each), ordered by
    :data:`qc._FAILURE_CLAUSE_ORDER` (the ``qc_inputs_insufficient`` sentinel
    sorts first), and rendered as ``"<clause>:<count>"`` joined by ``"; "``.
    Returns ``""`` when nothing is excluded.
    """
    counts: dict = {}
    for reason in reasons:
        if not isinstance(reason, str) or reason == "":
            continue
        for clause in reason.split(", "):
            clause = clause.strip()
            if clause:
                counts[clause] = counts.get(clause, 0) + 1
    if not counts:
        return ""
    ordered = [c for c in qc._FAILURE_CLAUSE_ORDER if c in counts]
    extra = sorted(c for c in counts if c not in qc._FAILURE_CLAUSE_ORDER)
    return "; ".join(f"{c}:{counts[c]}" for c in ordered + extra)


def _aggregate_one_group(key, group: pd.DataFrame, plan) -> dict:
    """Aggregate one ``(plate_id, genotype, treatment)`` group into one output row."""
    row = dict(zip(_GROUP_KEYS, key))

    # fillna(False) so a NaN clean-flag is conservatively EXCLUDED, never
    # silently counted as passing (a bare bool cast maps NaN -> True).
    clean = group["track_is_clean"].fillna(False).to_numpy(dtype=bool)
    passing = group.loc[clean]
    excluded = group.loc[~clean]
    n_pass = int(len(passing))
    row["n_plants_passing_qc"] = n_pass
    row["n_plants_excluded"] = int(len(excluded))
    row["exclusion_reasons"] = _build_exclusion_reasons(excluded["qc_failure_reason"])

    for source_col, prefix, is_mag in plan:
        vals = passing[source_col].to_numpy(dtype=float)
        if is_mag:
            vals = np.abs(vals)
        row[f"{prefix}_median"] = _nanmedian(vals)
        row[f"{prefix}_iqr"] = _iqr_or_nan(vals)

    if n_pass == 0:
        row["frac_nutating"] = float("nan")
        row["handedness_mode"] = (
            0  # int-preserving neutral fill; n_pass==0 disambiguates
        )
        row["handedness_consensus_frac"] = float("nan")
    else:
        row["frac_nutating"] = float(
            passing["is_nutating"].to_numpy(dtype=float).mean()
        )
        mode, consensus = _handedness_mode_consensus(passing["handedness"].to_numpy())
        row["handedness_mode"] = mode
        row["handedness_consensus_frac"] = consensus
    return row


def aggregate_by_genotype(
    per_plant_df: pd.DataFrame, units: dict
) -> "tuple[pd.DataFrame, dict]":
    """Aggregate a composed per-plant trait frame into a per-genotype summary.

    Groups by ``(plate_id, genotype, treatment)`` (plates never pooled) and emits
    one row per group with median ± IQR across the **passing** plants
    (``track_is_clean == True``) of every numeric trait, plus QC counts and an
    exclusion-reason summary. See the module docstring for the per-column rules.

    Args:
        per_plant_df: The composed per-plant DataFrame (the first element of
            :meth:`CircumnutationPipeline.compute_traits`'s return). Must carry
            the row-identity columns and the QC/special columns
            (``track_is_clean``, ``is_nutating``, ``handedness``,
            ``qc_failure_reason``). The composed frame guarantees the dtypes this
            function relies on: ``track_is_clean`` / ``is_nutating`` are boolean
            (the QC and Tier-1 emitters coerce via ``astype(bool)`` /
            ``fillna(False)``) and ``handedness`` is integer (psi_g
            ``fillna(0).astype(int64)``). A NaN ``track_is_clean`` is treated as
            *not clean* (conservatively excluded); a non-integer ``handedness``
            among passing plants is outside this contract.
        units: The per-plant column → unit-string mapping (the third element of
            ``compute_traits``), 1:1 covering ``per_plant_df``'s columns.

    Returns:
        ``(per_genotype_df, per_genotype_units)``: one row per
        ``(plate_id, genotype, treatment)`` group sorted by that key, and a 1:1
        column → unit mapping covering ``per_genotype_df``. Pure — neither input
        is mutated; no filesystem I/O.

    Raises:
        ValueError: If ``units`` does not 1:1 cover ``per_plant_df``'s columns;
            if a required composed column is missing; or if the track↔plant 1:1
            invariant is violated (see :func:`_validate_one_row_per_plant`).
    """
    _validate_units_coverage(per_plant_df, units, fn_name="aggregate_by_genotype")
    missing = [c for c in _REQUIRED_COLUMNS if c not in per_plant_df.columns]
    if missing:
        raise ValueError(
            f"aggregate_by_genotype: per_plant_df is missing required composed "
            f"columns {sorted(missing)}; it must be the output of "
            f"CircumnutationPipeline.compute_traits."
        )
    _validate_one_row_per_plant(per_plant_df)

    df = per_plant_df.copy()
    plan = _stat_plan(list(df.columns), units)

    # Column order: group keys, counts, reasons, then median/iqr per planned
    # trait (input order), then the special trio.
    ordered_cols: List[str] = list(_GROUP_KEYS) + [
        "n_plants_passing_qc",
        "n_plants_excluded",
        "exclusion_reasons",
    ]
    for _src, prefix, _is_mag in plan:
        ordered_cols += [f"{prefix}_median", f"{prefix}_iqr"]
    ordered_cols += ["frac_nutating", "handedness_mode", "handedness_consensus_frac"]

    # No warning suppression needed: _nanmedian / _iqr_or_nan / the n_pass==0
    # branch pre-filter to finite and short-circuit before any np.median /
    # scipy.stats.iqr / mean on an empty or <2 slice, so degenerate groups emit
    # NaN without a RuntimeWarning. (A blanket filter here would also hide
    # unrelated warnings and mutate process-global state.)
    rows: List[dict] = []
    if not df.empty:
        for key, group in df.groupby(list(_GROUP_KEYS), dropna=False, sort=True):
            rows.append(_aggregate_one_group(key, group, plan))

    # Build the frame column-wise with explicit dtypes (preserves int64 for
    # counts/handedness_mode even on a mixed or empty frame — a list-of-dicts
    # construction, not groupby().apply(Series) which would upcast).
    int_cols = {"n_plants_passing_qc", "n_plants_excluded", "handedness_mode"}
    string_cols = {"plate_id", "genotype", "treatment", "exclusion_reasons"}
    data: dict = {}
    for col in ordered_cols:
        series = pd.Series([row.get(col) for row in rows], dtype="object")
        if col in int_cols:
            data[col] = series.astype("int64")
        elif col in string_cols:
            data[col] = series.astype("object")
        else:
            data[col] = series.astype("float64")
    per_genotype_df = pd.DataFrame(data, columns=ordered_cols)

    per_genotype_units = _build_units(plan, units)
    logger.info(
        "aggregate_by_genotype: %d per-plant rows -> %d per-genotype rows",
        len(df),
        len(per_genotype_df),
    )
    return per_genotype_df, per_genotype_units


def _build_units(plan, units) -> dict:
    """Build the per-genotype units mapping (1:1 over the output columns)."""
    out: dict = {}
    for key in _GROUP_KEYS:
        out[key] = units[key]
    out["n_plants_passing_qc"] = "int"
    out["n_plants_excluded"] = "int"
    out["exclusion_reasons"] = "string"
    for source_col, prefix, _is_mag in plan:
        out[f"{prefix}_median"] = units[source_col]
        out[f"{prefix}_iqr"] = units[source_col]
    out["frac_nutating"] = _DIMENSIONLESS
    out["handedness_mode"] = "int"
    out["handedness_consensus_frac"] = _DIMENSIONLESS
    return out
