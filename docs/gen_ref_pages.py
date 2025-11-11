"""Generate API reference pages and navigation.

This script auto-generates API documentation from the sleap_roots package
using mkdocstrings. It creates a reference page for each Python module.
"""

from pathlib import Path

import mkdocs_gen_files

# Root of the package to document
nav = mkdocs_gen_files.Nav()
root = Path(__file__).parent.parent
src = root / "sleap_roots"

# Iterate through all Python files in the sleap_roots package
for path in sorted(src.rglob("*.py")):
    # Get module path relative to package root
    module_path = path.relative_to(src.parent).with_suffix("")
    doc_path = path.relative_to(src.parent).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    # Get parts for navigation
    parts = tuple(module_path.parts)

    # Skip __init__.py files but include in nav
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_") and parts[-1] != "__init__":
        # Skip private modules
        continue

    # Add to navigation
    nav[parts] = doc_path.as_posix()

    # Create the documentation file
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        # Convert module path to import identifier
        ident = ".".join(parts)
        # Write mkdocstrings directive
        fd.write(f"::: {ident}")

    # Set edit path for GitHub integration
    mkdocs_gen_files.set_edit_path(full_doc_path, path.relative_to(root))

# Write navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())