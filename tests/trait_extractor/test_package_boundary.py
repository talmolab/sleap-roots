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
    assert (
        "extractor" in pyproject["project"]["optional-dependencies"]
    ), "pyproject must declare a [project.optional-dependencies] extractor group"
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
    # pyyaml is declared explicitly (trait_extractor's pipeline_chooser imports it directly);
    # guard it so a future edit can't silently drop it back to transitive-only.
    assert any(
        dep.split(";")[0].strip() == "pyyaml" for dep in extractor
    ), f"extractor extra must declare pyyaml explicitly; got {extractor}"


def test_image_bakes_traits_code_sha_for_provenance():
    """The Docker image + workflow bake the build commit into ``SRT_TRAITS_CODE_SHA``.

    Guards the traceability wiring the image exists to provide: emitted envelopes carry a
    non-empty ``provenance.traits_code_sha`` (an idempotency-key input). A dropped ``ARG``/
    ``ENV`` or ``build-args`` line would ship images stamping ``""`` — a *valid* contract value
    that passes all other CI, so nothing else would catch it (the CI docker job only
    build-validates; it never runs the entry).
    """
    dockerfile = (_REPO_ROOT / "trait-extractor.Dockerfile").read_text(encoding="utf-8")
    assert "ARG SRT_TRAITS_CODE_SHA" in dockerfile
    assert "ENV SRT_TRAITS_CODE_SHA=${SRT_TRAITS_CODE_SHA}" in dockerfile
    workflow = (
        _REPO_ROOT / ".github" / "workflows" / "docker-trait-extractor.yml"
    ).read_text(encoding="utf-8")
    assert "SRT_TRAITS_CODE_SHA=${{ github.sha }}" in workflow


def test_docker_workflow_config_invariants():
    """CI-guard the trait-extractor workflow's load-bearing, otherwise-eyeball-only invariants.

    A regression here (wrong image identity, a re-introduced ``release:`` trigger, or a dropped
    ``sleap_roots/**`` path) would ship a mis-identified or stale image while staying CI-green.
    """
    import yaml

    raw = (
        _REPO_ROOT / ".github" / "workflows" / "docker-trait-extractor.yml"
    ).read_text(encoding="utf-8")
    # Distinct service identity — NOT the library's ${{ github.repository }}.
    assert "images: ghcr.io/talmolab/sleap-roots-trait-extractor" in raw
    assert "ghcr.io/${{ github.repository }}" not in raw
    parsed = yaml.safe_load(raw)
    on = parsed[True]  # YAML 1.1 parses the `on:` key as the boolean True.
    # Decoupled from the PyPI library release (build.yml owns `release:`).
    assert "release" not in on
    # Same filter on both triggers, and the library source is a build input (baked in).
    assert on["push"]["paths"] == on["pull_request"]["paths"]
    assert "sleap_roots/**" in on["push"]["paths"]
    assert parsed["jobs"]["docker"]["permissions"]["packages"] == "write"
