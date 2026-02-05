"""CLI commands for the HTML prediction viewer."""

import sys
from pathlib import Path
from typing import Optional

import click


@click.command()
@click.argument("predictions_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--images",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory containing source images (optional).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("viewer.html"),
    help="Output HTML file path.",
)
@click.option(
    "--max-frames",
    type=int,
    default=10,
    help="Maximum frames to render per scan (default: 10, 0 for all).",
)
@click.option(
    "--no-limit",
    is_flag=True,
    default=False,
    help="Disable 1000 total frame limit for large datasets.",
)
def viewer(
    predictions_dir: Path,
    images: Optional[Path],
    output: Path,
    max_frames: int,
    no_limit: bool,
) -> None:
    """Generate an HTML viewer for SLEAP prediction visualization.

    PREDICTIONS_DIR is the directory containing .slp prediction files.

    By default, samples 10 frames per scan for fast QC. Use --max-frames 0
    to include all frames (may produce large files).
    """
    from sleap_roots.viewer.generator import (
        FrameLimitExceededError,
        ViewerGenerator,
    )

    def progress_callback(scan_name: str, frames_done: int, total_frames: int) -> None:
        """Display progress during generation."""
        # Use carriage return to update in place
        sys.stdout.write(
            f"\rRendering: {scan_name} [{frames_done}/{total_frames} frames]"
        )
        sys.stdout.flush()

    generator = ViewerGenerator(predictions_dir, images_dir=images)
    try:
        click.echo("Discovering scans...")
        generator.generate(
            output,
            max_frames=max_frames,
            no_limit=no_limit,
            progress_callback=progress_callback,
        )
        # Clear progress line and show completion
        sys.stdout.write("\r" + " " * 60 + "\r")
        click.echo(f"Viewer generated: {output}")
    except FrameLimitExceededError as e:
        raise click.ClickException(str(e))
