# sleap-roots Documentation

This directory contains the source files for the sleap-roots documentation site.

## Building Locally

To build and preview the documentation locally:

```bash
# Install dependencies (from repo root)
pip install -e .[dev]

# Serve docs locally with live reload
mkdocs serve

# Open http://127.0.0.1:8000 in your browser
```

The docs will auto-rebuild when you edit files.

## Building for Production

```bash
# Build static site
mkdocs build

# Output will be in site/
```

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/            # Installation, quickstart, SLEAP intro
├── guides/                     # User guides
│   ├── pipelines/             # Pipeline-specific docs
│   ├── data-formats/          # File format documentation
│   ├── trait-reference.md     # Complete trait reference
│   └── batch-processing.md    # Batch workflow guide
├── dev/                       # Developer documentation
├── cookbook/                  # Examples and recipes
├── changelog.md               # Release history
├── gen_ref_pages.py          # Auto-generates API reference
├── stylesheets/              # Custom CSS
└── javascripts/              # Custom JS (MathJax)
```

## Auto-Generated Content

### API Reference

The API reference under `/reference/` is auto-generated from Python docstrings using mkdocstrings.

The generation script is `docs/gen_ref_pages.py`, which runs automatically during `mkdocs build`.

### Trait Reference

**Coming soon:** Auto-generate trait reference from `TraitDef` objects in the codebase.

## Writing Documentation

### Style Guide

- Use American English spelling
- Use sentence case for headings
- Include code examples with syntax highlighting
- Add type hints to code examples
- Use admonitions for tips, warnings, etc.

### Admonitions

```markdown
!!! note
    This is a note

!!! tip
    This is a helpful tip

!!! warning
    This is a warning

!!! info
    This is informational
```

### Code Blocks

````markdown
```python
import sleap_roots as sr

series = sr.Series.load(...)
```
````

### Math Equations

Inline: `\(E = mc^2\)`

Display:
```
\[
E = mc^2
\]
```

## Versioning

Documentation is versioned using [mike](https://github.com/jimporter/mike):

- **latest** – Latest development version (main branch)
- **stable** – Latest stable release
- **v0.1.4**, **v0.1.3**, etc. – Specific versions

The default version is `latest`.

## Deployment

Documentation is automatically deployed to GitHub Pages via GitHub Actions:

- **On push to main:** Deploy as `latest` and `dev`
- **On release:** Deploy as version tag (e.g., `v0.1.4`) and update `stable`

Manual deployment:

```bash
# Deploy current version as latest
mike deploy --push latest dev

# Deploy specific version
mike deploy --push v0.1.4 stable

# Set default version
mike set-default --push latest
```

## Contributing

When contributing documentation:

1. Create a feature branch
2. Edit/add markdown files in `docs/`
3. Test locally with `mkdocs serve`
4. Ensure all links work
5. Submit PR

## Configuration

Main configuration is in `mkdocs.yml` at the repository root.

Key settings:

- **theme:** Material theme with plant-themed green colors
- **plugins:** search, mkdocstrings, gen-files, mike
- **markdown_extensions:** admonitions, code highlighting, math, mermaid diagrams

## Troubleshooting

### "ModuleNotFoundError" when building

Install dev dependencies:
```bash
pip install -e .[dev]
```

### Navigation not updating

Restart `mkdocs serve` after changing `mkdocs.yml` navigation.

### API reference not generating

Check `docs/gen_ref_pages.py` and ensure it has no syntax errors.

### Math not rendering

Ensure MathJax is configured in `docs/javascripts/mathjax.js` and included in `mkdocs.yml`.

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [mkdocstrings](https://mkdocstrings.github.io/)
- [mike - Multi-version docs](https://github.com/jimporter/mike)