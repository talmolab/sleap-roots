"""CI-enforced package boundary: the ``sleap_roots`` library never imports contracts.

The ``trait_extractor`` service depends on ``sleap-roots-contracts``, but the published
``sleap_roots`` library must not — so ``pip install sleap-roots`` stays pure. This guard
AST-scans the library source (robust against docstrings/strings/comments that merely
mention the name) and fails if any module imports ``sleap_roots_contracts``.
"""

import ast
from pathlib import Path

import sleap_roots

_LIBRARY_ROOT = Path(sleap_roots.__file__).parent


def _imported_modules(tree: ast.AST):
    """Yield every module name imported by an AST (``import`` + ``from`` forms)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                yield node.module


def test_library_source_never_imports_contracts():
    """No module under ``sleap_roots/`` imports ``sleap_roots_contracts``."""
    offenders = []
    for path in _LIBRARY_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _imported_modules(tree):
            if module == "sleap_roots_contracts" or module.startswith(
                "sleap_roots_contracts."
            ):
                offenders.append(path.as_posix())
    assert not offenders, (
        "sleap_roots library must not import sleap_roots_contracts "
        f"(found in: {offenders})"
    )
