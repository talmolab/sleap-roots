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
@click.option(
    "--render",
    is_flag=True,
    default=False,
    help="Pre-render overlays with matplotlib to disk files (slower, for sharing).",
)
@click.option(
    "--embed",
    is_flag=True,
    default=False,
    help="Embed images as base64 in HTML (single self-contained file).",
)
@click.option(
    "--format",
    "image_format",
    type=click.Choice(["jpeg", "png"], case_sensitive=False),
    default="jpeg",
    help="Image format for H5 extraction and --render mode (default: jpeg).",
)
@click.option(
    "--quality",
    type=click.IntRange(1, 100),
    default=85,
    help="JPEG quality for H5 extraction and --render mode (1-100, default: 85).",
)
@click.option(
    "--zip",
    "create_zip",
    is_flag=True,
    default=False,
    help="Create a ZIP archive containing the viewer and all required files.",
)
@click.option(
    "--timepoint",
    multiple=True,
    default=None,
    help="Filter scans by timepoint pattern (e.g., 'Day0*'). Can be used multiple times.",
)
def viewer(
    predictions_dir: Path,
    images: Optional[Path],
    output: Path,
    max_frames: int,
    no_limit: bool,
    render: bool,
    embed: bool,
    image_format: str,
    quality: int,
    create_zip: bool,
    timepoint: tuple,
) -> None:
    r"""Generate an HTML viewer for SLEAP prediction visualization.

    PREDICTIONS_DIR is the directory containing .slp prediction files.

    \b
    Output Modes:
      (default)   Client-render mode with external images (fast, portable)
      --render    Pre-render overlays to disk files (viewer_images/ directory)
      --embed     Embed images as base64 in HTML (single file, slower)

    \b
    Portability:
      --zip       Create a ZIP archive for easy sharing (includes all images)

    \b
    Filtering:
      --timepoint Filter scans by timepoint pattern (e.g., --timepoint 'Day0*')

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

    # Convert timepoint tuple to list (empty tuple means no filter)
    timepoint_patterns = list(timepoint) if timepoint else None

    generator = ViewerGenerator(predictions_dir, images_dir=images)
    try:
        click.echo("Discovering scans...")
        generator.generate(
            output,
            max_frames=max_frames,
            no_limit=no_limit,
            progress_callback=progress_callback,
            render=render,
            embed=embed,
            image_format=image_format,
            image_quality=quality,
            create_zip=create_zip,
            timepoint_patterns=timepoint_patterns,
        )
        # Clear progress line and show completion
        sys.stdout.write("\r" + " " * 60 + "\r")
        click.echo(f"Viewer generated: {output}")

        # Show images directory info for render mode
        if render:
            images_dir = output.parent / f"{output.stem}_images"
            click.echo(f"Images directory: {images_dir}")

        # Show ZIP path if created
        if create_zip:
            zip_path = output.with_suffix(".zip")
            click.echo(f"ZIP archive: {zip_path}")
    except ValueError as e:
        raise click.ClickException(str(e))
    except FrameLimitExceededError as e:
        raise click.ClickException(str(e))
