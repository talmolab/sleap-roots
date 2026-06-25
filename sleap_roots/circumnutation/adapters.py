"""``Series`` → :class:`CircumnutationInputs` adapter (PR #17).

This module owns :func:`series_to_inputs`, the single bridge from
:class:`sleap_roots.series.Series` into the pure circumnutation core. It
formalizes the mechanical transform of the ``_load_plate001_inputs()`` test
blueprint — ``Series.get_tracked_tips()`` → prefix-anchored ``track_id`` integer
coercion → row-identity population → :class:`CircumnutationInputs` — while
replacing that blueprint's hardcoded test literals with CSV / flag / NaN identity
sourcing.

The adapter carries NO ``click`` dependency (it is the testable seam between the
CLI and the pipeline) and never imports ``matplotlib``. Alongside the
:class:`CircumnutationInputs` it returns an **identity-provenance dict** recording
the resolved metadata-CSV path, its SHA-256, and a per-field
``identity_source`` map (``"flag"`` / ``"metadata_csv"`` / ``"default"`` /
``"absent"``) so the CLI can persist where each row-identity field came from
(see the ``Run-metadata sidecar`` spec requirement).
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sleap_roots.circumnutation._types import CircumnutationInputs


logger = logging.getLogger(__name__)


_TRACK_PREFIX = "track_"

# The six row-identity fields whose source is recorded in identity_source.
# plant_id / track_id are EXCLUDED — they are mechanically derived from the
# .slp track names, not flag/CSV-sourced (their provenance is input_path).
_PROVENANCE_FIELDS: tuple = (
    "series",
    "sample_uid",
    "timepoint",
    "plate_id",
    "genotype",
    "treatment",
)


def _coerce_track_id(track_id: pd.Series) -> pd.Series:
    r"""Coerce a ``"track_<int>"`` (or bare-integer) track-name Series to int64.

    The leading ``"track_"`` prefix is stripped **once** (anchored, not a global
    ``str.replace``) and the remainder must be an unsigned run of digits
    (``re.fullmatch(r"\d+", ...)``). A name that is not ``"track_<int>"`` (e.g.
    ``"track_2a"`` or the interior-prefix ``"track_track_1"``, which leaves
    ``"track_1"``) raises ``ValueError`` naming the offender rather than being
    silently mangled — the pipeline's "raise rather than silently corrupt" contract.
    The strict digit match also rejects names that Python's ``int()`` would silently
    accept: signed (``"track_-1"``), whitespace-padded (``"track_ 1"``), and
    underscore-separated (``"track_1_2"`` → ``int`` would yield ``12``).

    Args:
        track_id: The ``track_id`` column from ``Series.get_tracked_tips()``.

    Returns:
        An int64 ``pandas.Series`` aligned to ``track_id``'s index.

    Raises:
        ValueError: If any track name does not yield a pure unsigned-integer string
            after the anchored prefix strip; the message names the offending name(s).
    """
    coerced = []
    offenders = []
    for raw in track_id:
        name = str(raw)
        remainder = (
            name[len(_TRACK_PREFIX) :] if name.startswith(_TRACK_PREFIX) else name
        )
        if re.fullmatch(r"\d+", remainder):
            coerced.append(int(remainder))
        else:
            offenders.append(name)
    if offenders:
        raise ValueError(
            f"Cannot derive an integer track_id from track name(s) "
            f"{sorted(set(offenders))}; expected 'track_<int>' (the leading "
            f"'track_' is stripped once and the remainder must be an integer)."
        )
    return pd.Series(coerced, index=track_id.index, dtype="int64")


def _sha256_file(path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes (newline-agnostic)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _warn_if_sample_uid_unmatched(csv_path, sample_uid) -> None:
    """Log a WARNING if ``sample_uid`` matches no ``plant_qr_code`` row in the CSV.

    ``Series.get_metadata`` joins ``plant_qr_code == sample_uid`` and returns ``NaN``
    on a miss, so a mistyped ``--sample-uid`` would silently yield all-NaN metadata.
    A malformed/unreadable CSV is ignored here — the field-resolution path raises a
    clear ``ValueError`` for that case.
    """
    try:
        meta = pd.read_csv(csv_path)
    except Exception:  # noqa: BLE001 — resolution path raises the clear error
        return
    if "plant_qr_code" not in meta.columns:
        return
    if str(sample_uid) not in meta["plant_qr_code"].astype(str).values:
        logger.warning(
            "--sample-uid %r matches no plant_qr_code row in metadata CSV %s; "
            "metadata-sourced identity fields (genotype/treatment/timepoint) will "
            "be NaN.",
            sample_uid,
            str(csv_path),
        )


def _resolve_field(
    series,
    field: str,
    flag_value,
    *,
    has_csv: bool,
    csv_path,
    stringify: bool = False,
) -> Tuple[object, str]:
    """Resolve one identity field to ``(value, source_label)``.

    Precedence: an explicit, **non-blank** ``flag_value`` always wins; otherwise a
    non-null, **non-blank** metadata-CSV cell (via ``Series.get_metadata`` — the raw
    cell, never a coercing property); otherwise ``NaN``. A blank value (``None``,
    empty string, or whitespace-only) is treated as **not supplied** for both the
    flag and the CSV cell — so ``--genotype ""`` or a whitespace CSV cell resolves to
    ``"absent"`` (``NaN``), not a degenerate empty-string identity group. The
    override-notice is logged at INFO only when a non-blank flag actually shadows a
    real (non-blank, different) CSV value.

    Args:
        series: The :class:`~sleap_roots.series.Series`.
        field: The metadata column / identity field name.
        flag_value: The explicit CLI / adapter argument, or ``None``.
        has_csv: Whether ``series`` carries a ``csv_path``.
        csv_path: The metadata-CSV path (for error messages); may be ``None``.
        stringify: When ``True`` (``timepoint``), ``str()``-normalize the value.

    Returns:
        ``(resolved_value, source_label)`` where the label is one of ``"flag"``,
        ``"metadata_csv"``, or ``"absent"``.

    Raises:
        ValueError: If reading the metadata CSV fails (e.g. a non-parseable file).
    """

    def _csv_cell():
        if not has_csv:
            return np.nan
        try:
            return series.get_metadata(field)
        except Exception as exc:  # noqa: BLE001 — normalize pandas/IO errors
            raise ValueError(
                f"Failed to read metadata CSV {str(csv_path)!r}: {exc}"
            ) from exc

    def _nonblank(value) -> bool:
        """True iff ``value`` is a real, non-blank value (not None/NaN/whitespace)."""
        return pd.notna(value) and str(value).strip() != ""

    if _nonblank(flag_value):
        csv_value = _csv_cell()
        if _nonblank(csv_value) and str(flag_value) != str(csv_value):
            logger.info(
                "--%s overrides metadata-csv value %r -> %r",
                field,
                csv_value,
                flag_value,
            )
        return (str(flag_value) if stringify else flag_value), "flag"

    csv_value = _csv_cell()
    if _nonblank(csv_value):
        return (str(csv_value) if stringify else csv_value), "metadata_csv"
    return np.nan, "absent"


def series_to_inputs(
    series,
    *,
    cadence_s: float,
    sample_uid: str,
    series_name: Optional[str] = None,
    timepoint: Optional[str] = None,
    plate_id: Optional[str] = None,
    genotype: Optional[str] = None,
    treatment: Optional[str] = None,
    r_px: Optional[float] = None,
    run_id: Optional[str] = None,
) -> "Tuple[CircumnutationInputs, dict]":
    """Build a :class:`CircumnutationInputs` from a tracked :class:`Series`.

    Reads ``series.get_tracked_tips()``, derives an integer ``track_id`` (anchored
    ``"track_"`` strip), sets ``plant_id = track_id``, populates the eight
    row-identity columns (``sample_uid`` from the required argument, ``series`` from
    ``series_name`` defaulting to ``series.series_name``, and
    ``timepoint``/``plate_id``/``genotype``/``treatment`` via CSV-as-source /
    flag-as-override precedence), and constructs the inputs.

    Args:
        series: A loaded :class:`sleap_roots.series.Series` (its ``csv_path`` /
            ``sample_uid`` drive metadata resolution).
        cadence_s: Frame cadence in seconds (validated by
            :class:`CircumnutationInputs`).
        sample_uid: Stable sample identifier (the metadata-CSV join key); required.
        series_name: Optional human recording label; defaults to
            ``series.series_name``.
        timepoint: Optional timepoint label (object/string).
        plate_id: Optional plate identifier.
        genotype: Optional genotype label.
        treatment: Optional treatment label.
        r_px: Optional root cross-section radius in pixels.
        run_id: Optional human-readable run identifier.

    Returns:
        A 2-tuple ``(inputs, identity_provenance)`` where ``identity_provenance`` is
        ``{"metadata_csv_path": <abs str | None>, "metadata_csv_sha256":
        <hex str | None>, "identity_source": {<field>: "flag" | "metadata_csv" |
        "default" | "absent"}}`` (the ``identity_source`` map is total over the six
        fields ``series``, ``sample_uid``, ``timepoint``, ``plate_id``,
        ``genotype``, ``treatment``).

    Raises:
        ValueError: If a track name is not ``"track_<int>"``, if the metadata CSV
            cannot be read, or if :class:`CircumnutationInputs` validation fails
            (empty trajectory, non-positive/non-finite ``cadence_s``, etc.).
    """
    df = series.get_tracked_tips().copy()
    df["track_id"] = _coerce_track_id(df["track_id"])
    df["plant_id"] = df["track_id"].to_numpy()

    csv_path = getattr(series, "csv_path", None)
    has_csv = bool(csv_path) and Path(csv_path).exists()
    metadata_csv_path = str(Path(csv_path).resolve()) if has_csv else None
    metadata_csv_sha256 = _sha256_file(csv_path) if has_csv else None

    # Warn loudly if a metadata CSV is supplied but `sample_uid` matches no row:
    # `get_metadata` would then silently return NaN for every field, so a mistyped
    # `--sample-uid` ships an all-NaN-metadata trait CSV with no other signal.
    if has_csv:
        _warn_if_sample_uid_unmatched(csv_path, sample_uid)

    identity_source = {}

    resolved_series = series_name if series_name is not None else series.series_name
    identity_source["series"] = "flag" if series_name is not None else "default"
    identity_source["sample_uid"] = "flag"

    timepoint_val, identity_source["timepoint"] = _resolve_field(
        series,
        "timepoint",
        timepoint,
        has_csv=has_csv,
        csv_path=csv_path,
        stringify=True,
    )
    plate_val, identity_source["plate_id"] = _resolve_field(
        series,
        "plate_id",
        plate_id,
        has_csv=has_csv,
        csv_path=csv_path,
    )
    genotype_val, identity_source["genotype"] = _resolve_field(
        series,
        "genotype",
        genotype,
        has_csv=has_csv,
        csv_path=csv_path,
    )
    treatment_val, identity_source["treatment"] = _resolve_field(
        series,
        "treatment",
        treatment,
        has_csv=has_csv,
        csv_path=csv_path,
    )

    df["series"] = resolved_series
    df["sample_uid"] = sample_uid
    df["timepoint"] = timepoint_val
    df["plate_id"] = plate_val
    df["genotype"] = genotype_val
    df["treatment"] = treatment_val

    # Force object dtype on the string identity columns so an all-NaN field is
    # object (string + NaN), not float64 — stable regardless of data content, and
    # matching the per-plant template builder's own coercion (_io.py).
    for col in _PROVENANCE_FIELDS:
        df[col] = df[col].astype(object)

    inputs = CircumnutationInputs(
        trajectory_df=df, cadence_s=cadence_s, R_px=r_px, run_id=run_id
    )
    identity_provenance = {
        "metadata_csv_path": metadata_csv_path,
        "metadata_csv_sha256": metadata_csv_sha256,
        "identity_source": identity_source,
    }
    logger.debug(
        "series_to_inputs: %d tracks, identity_source=%s",
        df["track_id"].nunique(),
        identity_source,
    )
    return inputs, identity_provenance
