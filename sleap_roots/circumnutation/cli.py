"""``sleap-roots circumnutation`` CLI (PR #17).

Defines the ``circumnutation`` ``click`` group and its ``analyze`` command, the
user-facing entry point that composes the whole pipeline on one ``.slp``:
``Series.load`` → :func:`~sleap_roots.circumnutation.adapters.series_to_inputs`
→ :func:`~sleap_roots.circumnutation.pipeline.compute_traits` → assemble
provenance once → ``write_per_plant_csv`` → (optionally) ``aggregate_by_genotype``
+ ``write_per_genotype_csv`` → (optionally) ``save_plots``.

The group is registered on the root CLI via ``main.add_command(circumnutation)``
(mirroring the viewer). Heavy imports (``Series``, the pipeline, ``matplotlib``)
are deferred into the command body so importing this module — and therefore
``sleap_roots.cli`` — stays cheap.

Outputs are pure-pixel (CC-3): there is no ``--px-per-mm`` option. Compose
:func:`sleap_roots.circumnutation.units.convert_to_mm` on the trait CSV for mm.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import click


logger = logging.getLogger(__name__)

_LOG_HANDLER_TAG = "_circumnutation_cli_handler"


def _configure_logging(verbose: int):
    """Set the ``sleap_roots`` log level + a stderr handler from the ``-v`` count.

    ``0 → WARNING`` (quiet default), ``1 → INFO`` (per-plate progress), ``≥2 →
    DEBUG`` (per-plant detail). Logs go to stderr so stdout stays clean for the
    result summary. Idempotent across repeated invocations (the prior tagged
    handler is removed before a fresh one binds the current stderr). Also quiets
    matplotlib's ``font_manager`` ``findfont`` fallback WARNINGs — the scaleogram's
    ``LogNorm`` colorbar renders mathtext, whose Computer-Modern font-family lookup
    falls back to DejaVu Sans (cosmetic, not actionable); ``ERROR`` keeps genuine
    font errors visible.

    Returns:
        A zero-arg restore callable that undoes every change (level on both loggers,
        and the tagged handler) — call it in a ``finally`` so the CLI's global
        logging config does not leak into a long-lived host process / the test
        session.
    """
    level = (
        logging.WARNING
        if verbose <= 0
        else logging.INFO if verbose == 1 else logging.DEBUG
    )
    pkg = logging.getLogger("sleap_roots")
    fontlog = logging.getLogger("matplotlib.font_manager")
    prior_pkg_level = pkg.level
    prior_font_level = fontlog.level

    pkg.setLevel(level)
    pkg.handlers = [
        h for h in pkg.handlers if getattr(h, "_tag", None) != _LOG_HANDLER_TAG
    ]
    handler = logging.StreamHandler()  # binds the current sys.stderr
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    handler._tag = _LOG_HANDLER_TAG  # type: ignore[attr-defined]
    pkg.addHandler(handler)
    fontlog.setLevel(logging.ERROR)

    def _restore() -> None:
        pkg.setLevel(prior_pkg_level)
        pkg.handlers = [
            h for h in pkg.handlers if getattr(h, "_tag", None) != _LOG_HANDLER_TAG
        ]
        fontlog.setLevel(prior_font_level)

    return _restore


@click.group()
def circumnutation() -> None:
    """Circumnutation trait analysis from SLEAP-tracked tip trajectories."""
    pass


@circumnutation.command()
@click.argument(
    "slp_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--cadence-s",
    type=float,
    required=True,
    help="Frame cadence in seconds (required; sets every period trait).",
)
@click.option(
    "--sample-uid",
    type=str,
    required=True,
    help="Stable sample id (e.g. plate QR code); the metadata-CSV join key.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: ./<series-name>_circumnutation/).",
)
@click.option(
    "--series-name",
    type=str,
    default=None,
    help="Recording label (default: the .slp filename stem).",
)
@click.option(
    "--metadata-csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional metadata CSV (genotype/treatment/timepoint).",
)
@click.option("--timepoint", type=str, default=None, help="Timepoint label (string).")
@click.option("--plate-id", type=str, default=None, help="Plate identifier.")
@click.option(
    "--genotype",
    type=str,
    default=None,
    help="Genotype label (required for aggregation unless --no-aggregate).",
)
@click.option("--treatment", type=str, default=None, help="Treatment label.")
@click.option(
    "--r-px",
    type=float,
    default=None,
    help="Optional root cross-section radius in pixels.",
)
@click.option("--run-id", type=str, default=None, help="Human-readable run identifier.")
@click.option(
    "--no-plots",
    is_flag=True,
    default=False,
    help="Skip diagnostic plots (and the matplotlib import).",
)
@click.option(
    "--no-aggregate",
    is_flag=True,
    default=False,
    help="Skip per-genotype aggregation (no genotype required).",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase log verbosity (-v INFO, -vv DEBUG; logs to stderr).",
)
def analyze(
    slp_path: Path,
    cadence_s: float,
    sample_uid: str,
    output_dir: Optional[Path],
    series_name: Optional[str],
    metadata_csv: Optional[Path],
    timepoint: Optional[str],
    plate_id: Optional[str],
    genotype: Optional[str],
    treatment: Optional[str],
    r_px: Optional[float],
    run_id: Optional[str],
    no_plots: bool,
    no_aggregate: bool,
    verbose: int,
) -> None:
    r"""Run the circumnutation pipeline on a tracked ``.slp``.

    SLP_PATH is a SLEAP-tracked ``.slp`` of tip trajectories (one track per plant).

    \b
    Outputs (under --output-dir) are PURE-PIXEL (CC-3): lengths in px, areas in
    px2, rates in px/frame, angles in rad, times in s/hr, ratios dimensionless.
    There is no pixel-to-mm calibration option; for mm output, compose
    sleap_roots.circumnutation.units.convert_to_mm on the trait CSV.

    \b
    Coordinates are image pixels: origin top-left, y increases DOWNWARD (the SLEAP
    convention). The sign of direction-bearing traits depends on this.
    The `timepoint` column is a free-form string label — cast it before joining a
    numeric metadata table.

    \b
    Output tree:
      <output-dir>/run_metadata.json          canonical run provenance
                  /per_plant/                  per-plant trait CSV + sidecars
                  /per_genotype/               per-genotype CSV (unless --no-aggregate)
                  /plots/                      diagnostic PNGs (unless --no-plots)

    Re-running on the same .slp overwrites prior outputs; use a distinct
    --output-dir for runs with different --cadence-s / --r-px.
    """
    _restore_logging = _configure_logging(verbose)

    # Lazy imports (keep module import — and `import sleap_roots.cli` — cheap).
    from sleap_roots.circumnutation._io import (
        gather_run_metadata,
        write_per_genotype_csv,
        write_per_plant_csv,
        write_run_metadata,
    )
    from sleap_roots.circumnutation.adapters import series_to_inputs
    from sleap_roots.circumnutation.aggregation import aggregate_by_genotype
    from sleap_roots.circumnutation.pipeline import compute_traits
    from sleap_roots.series import Series

    resolved_slp = slp_path.resolve()
    resolved_name = series_name if series_name is not None else slp_path.stem
    out_dir = (
        output_dir
        if output_dir is not None
        else Path(f"./{resolved_name}_circumnutation")
    )
    aggregate = not no_aggregate

    try:
        series = Series.load(
            series_name=resolved_name,
            primary_path=str(slp_path),
            csv_path=str(metadata_csv) if metadata_csv is not None else None,
            sample_uid=sample_uid,
        )
        inputs, identity_provenance = series_to_inputs(
            series,
            cadence_s=cadence_s,
            sample_uid=sample_uid,
            series_name=series_name,
            timepoint=timepoint,
            plate_id=plate_id,
            genotype=genotype,
            treatment=treatment,
            r_px=r_px,
            run_id=run_id,
        )

        # Genotype gate (before any output is written): aggregation is meaningless
        # without genotype labels. genotype is filled uniformly per run, so an
        # all-NaN genotype means no source supplied one.
        if aggregate and inputs.trajectory_df["genotype"].isna().any():
            raise click.ClickException(
                "genotype is required for per-genotype aggregation; supply "
                "--genotype or a --metadata-csv with a genotype column, or pass "
                "--no-aggregate to skip aggregation."
            )

        # Create the output tree only after the inputs + genotype gate pass.
        out_dir.mkdir(parents=True, exist_ok=True)
        per_plant_dir = out_dir / "per_plant"
        per_plant_dir.mkdir(parents=True, exist_ok=True)
        if aggregate:
            (out_dir / "per_genotype").mkdir(parents=True, exist_ok=True)

        logger.info("Computing circumnutation traits from %s", resolved_slp)
        per_plant_df, _trajectory_df, units = compute_traits(inputs)

        # Assemble provenance ONCE; write it to all sidecars byte-identically.
        run_metadata = gather_run_metadata(
            input_path=str(resolved_slp),
            run_id=inputs.run_id,
            constants=None,
            cadence_s=inputs.cadence_s,
            R_px=inputs.R_px,
            metadata_csv_path=identity_provenance["metadata_csv_path"],
            metadata_csv_sha256=identity_provenance["metadata_csv_sha256"],
            identity_source=identity_provenance["identity_source"],
        )
        write_run_metadata(out_dir / "run_metadata.json", run_metadata)
        write_per_plant_csv(
            per_plant_dir / "traits_per_plant.csv", per_plant_df, units, run_metadata
        )

        if aggregate:
            per_genotype_df, per_genotype_units = aggregate_by_genotype(
                per_plant_df, units
            )
            write_per_genotype_csv(
                out_dir / "per_genotype" / "traits_per_genotype.csv",
                per_genotype_df,
                per_genotype_units,
                run_metadata,
            )

        if no_plots:
            logger.info("plotting disabled (--no-plots)")
        else:
            import matplotlib

            matplotlib.use("Agg", force=True)
            from sleap_roots.circumnutation import plotting

            plotting.save_plots(inputs, out_dir=out_dir, enabled=True)

        click.echo(f"Analysis complete -> {out_dir}")
    except (ValueError, FileNotFoundError) as exc:
        raise click.ClickException(str(exc))
    finally:
        # Restore global logging state (both logger levels + the tagged handler) so
        # the CLI's config does not leak into a host process / the test session.
        _restore_logging()
