## Why

Scientists need to validate SLEAP prediction quality when processing experiments with new or unsupported species (e.g., Amaranth using Canola models). Currently this requires opening SLEAP GUI for each .slp file and manually navigating frames - slow and cumbersome for experiments with 30+ scans x 72 frames each.

## What Changes

- Add new `sleap_roots/viewer/` module with HTML report generator
- Add `sleap-roots viewer` CLI command using Click
- Generate self-contained HTML files with prediction overlays
- Support scan overview (thumbnail grid) and frame drill-down navigation
- Add keyboard navigation (arrow keys, Enter, Esc)
- Display confidence scores and instance counts per frame
- Add user documentation in `docs/guides/`

## Impact

- Affected specs: New `viewer` capability (no existing specs affected)
- Affected code:
  - New module: `sleap_roots/viewer/` (cli.py, generator.py, renderer.py, templates/)
  - Entry point: `pyproject.toml` (add viewer CLI command)
  - Documentation: `docs/guides/viewer.md`
- Dependencies: Jinja2 for HTML templating (already available via Click)
- No breaking changes to existing functionality