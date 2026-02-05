"""Command-line interface for sleap-roots."""

import click

from sleap_roots.viewer.cli import viewer


@click.group()
@click.version_option()
def main():
    """Analysis tools for SLEAP-based plant root phenotyping."""
    pass


main.add_command(viewer)


if __name__ == "__main__":
    main()
