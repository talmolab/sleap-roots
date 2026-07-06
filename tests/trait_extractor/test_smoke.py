"""Smoke test: the service package imports."""


def test_trait_extractor_imports():
    """The ``trait_extractor`` package imports from the repo root."""
    import trait_extractor

    assert trait_extractor.__doc__
