"""CSV-builder helpers for `Series` metadata.

Two pure functions for users to assemble the metadata CSV consumed by
`Series.get_metadata` (and the wrapper properties `expected_count`, `group`,
`qc_fail`, `timepoint`):

- `build_metadata_csv(rows, path)` — write a CSV from a list of row dicts,
  validating that every row carries `plant_qr_code` (the lookup key).
- `infer_timepoints_from_filenames(paths, pattern)` — parse `(series_name,
  timepoint)` from filename stems via a regex with named groups. Convenience
  for common conventions; users who name files differently can build the CSV
  any other way.

Both functions are re-exported at the package top level (`sleap_roots`).
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd


logger = logging.getLogger(__name__)


_CANONICAL_COLUMNS = (
    "plant_qr_code",
    "genotype",
    "number_of_plants_cylinder",
    "qc_cylinder",
    "qc_code",
    "timepoint",
)


def build_metadata_csv(
    rows: List[Dict[str, Any]], path: Union[str, Path]
) -> Path:
    """Write a metadata CSV from row dicts.

    Validates that every row carries `plant_qr_code` (the lookup key used by
    `Series.get_metadata`). Column ordering is canonical:
    `plant_qr_code, genotype, number_of_plants_cylinder, qc_cylinder, qc_code,
    timepoint, <extras in sorted order>`. Canonical columns absent from every
    row are omitted (so a user emitting only `plant_qr_code` + `timepoint` does
    not get empty `genotype` / `qc_cylinder` columns). Any non-canonical keys
    are appended in sorted order. An existing file at `path` is overwritten
    silently (pandas `to_csv` default).

    Args:
        rows: List of row dicts. Each row MUST contain `plant_qr_code`.
        path: Destination path (str or Path).

    Returns:
        The Path the CSV was written to.

    Raises:
        ValueError: If any row lacks a `plant_qr_code` key.
    """
    path = Path(path)
    for i, row in enumerate(rows):
        if "plant_qr_code" not in row:
            raise ValueError(
                f"build_metadata_csv: row {i} is missing required key "
                "'plant_qr_code'."
            )

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())

    columns = [c for c in _CANONICAL_COLUMNS if c in all_keys]
    extras = sorted(all_keys - set(_CANONICAL_COLUMNS))
    columns.extend(extras)

    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)
    return path


def infer_timepoints_from_filenames(
    slp_paths: List[Path], pattern: str
) -> Dict[str, float]:
    """Parse `(series_name, timepoint)` from filename stems via named groups.

    The pattern MUST contain named groups `series_name` AND `timepoint`. The
    returned dict keys on the full matched portion of the stem (regex group 0),
    NOT on the `series_name` group alone — this keeps keys uniquely identifying
    the source path even when `series_name` is a lazy match.

    Paths whose stems don't match the pattern, OR whose `timepoint` group can't
    be cast to `float`, are skipped silently from the return dict but logged at
    WARNING level (logger `sleap_roots.metadata`) with a reason string that
    distinguishes pattern-mismatch from float-cast failure.

    Args:
        slp_paths: Iterable of `.slp` paths (or any paths whose stems carry the
            series identity + timepoint).
        pattern: A regex containing named groups `series_name` and `timepoint`.

    Returns:
        Dict mapping the matched stem segment (group 0) to the timepoint as
        `float`.

    Raises:
        ValueError: If `pattern` lacks `series_name` or `timepoint` named
            groups.
    """
    compiled = re.compile(pattern)
    if (
        "series_name" not in compiled.groupindex
        or "timepoint" not in compiled.groupindex
    ):
        raise ValueError(
            "infer_timepoints_from_filenames: pattern must contain named "
            "groups 'series_name' and 'timepoint'. Got pattern: "
            f"{pattern!r}."
        )

    out: Dict[str, float] = {}
    for path in slp_paths:
        stem = Path(path).stem
        m = compiled.search(stem)
        if m is None:
            logger.warning(
                "infer_timepoints_from_filenames: skipping %s — pattern did "
                "not match stem %r.",
                path,
                stem,
            )
            continue
        try:
            value = float(m.group("timepoint"))
        except (TypeError, ValueError):
            logger.warning(
                "infer_timepoints_from_filenames: skipping %s — could not "
                "convert timepoint group %r to float.",
                path,
                m.group("timepoint"),
            )
            continue
        out[m.group(0)] = value
    return out
