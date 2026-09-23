"""Shared fixtures for the trait_extractor tests."""

import pytest


@pytest.fixture(autouse=True)
def _no_argo_workflow_name(monkeypatch):
    """Clear ``ARGO_WORKFLOW_NAME`` so no test inherits a run identity from its shell.

    ``extract_batch`` resolves its run identity from this variable when no
    ``pipeline_run_id`` is passed, and a known identity makes a missing run manifest
    fail loud. A developer or CI shell that happens to export it must not flip every
    existing test into that mode. Subprocess CLI tests build their environment from
    ``os.environ`` at call time, so they inherit the cleared value too. Tests that need an
    identity set it explicitly.
    """
    monkeypatch.delenv("ARGO_WORKFLOW_NAME", raising=False)
