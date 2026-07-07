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


_EXTRACTOR_ROOT = Path(__file__).resolve().parents[2] / "trait_extractor"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _imports_module(root: Path, prefix: str) -> list:
    """Return files under ``root`` that import a module named/prefixed ``prefix``."""
    offenders = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for module in _imported_modules(tree):
            if module == prefix or module.startswith(prefix + "."):
                offenders.append(path.as_posix())
    return offenders


def test_library_source_never_imports_contracts():
    """No module under ``sleap_roots/`` imports ``sleap_roots_contracts``."""
    offenders = _imports_module(_LIBRARY_ROOT, "sleap_roots_contracts")
    assert not offenders, (
        "sleap_roots library must not import sleap_roots_contracts "
        f"(found in: {offenders})"
    )


def test_extractor_never_imports_predict_at_runtime():
    """No module under ``trait_extractor/`` imports ``sleap_roots_predict``."""
    offenders = _imports_module(_EXTRACTOR_ROOT, "sleap_roots_predict")
    assert not offenders, (
        "trait_extractor must not import sleap_roots_predict at runtime "
        f"(found in: {offenders})"
    )


def test_packaging_config_excludes_the_extractor():
    """The wheel/sdist discovery config keeps trait_extractor out of the package."""
    import tomllib

    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    find = pyproject["tool"]["setuptools"]["packages"]["find"]
    # Allowlist includes only the library package, so a top-level sibling is never
    # discovered; the explicit exclude is belt-and-suspenders.
    assert find["include"] == ["sleap_roots"]
    assert "trait_extractor*" in find["exclude"]


def test_packaging_config_declares_the_extractor_extra():
    """The slim ``extractor`` extra pins contracts so the container image can install it.

    The image runs ``uv sync --frozen --no-dev --extra extractor``; if this extra is removed
    or its pin drifts, that install path breaks with no other CI signal (``ci.yml`` triggers
    on ``pyproject.toml``, so this guard catches it here).
    """
    import tomllib

    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    extractor = pyproject["project"]["optional-dependencies"]["extractor"]
    assert any(
        # Split off the marker and match the requirement exactly (not startswith, so a
        # superstring version like ==0.1.0a30 cannot slip through).
        dep.split(";")[0].strip() == "sleap-roots-contracts==0.1.0a3"
        and "python_version >= '3.11'" in dep
        for dep in extractor
    ), (
        "extractor extra must pin sleap-roots-contracts==0.1.0a3 with the "
        f"python_version >= '3.11' marker; got {extractor}"
    )
