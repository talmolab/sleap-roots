"""Tests for skip-if-done: idempotency_key + contract_version comparison."""

import json
from pathlib import Path

import trait_extractor.envelope
import trait_extractor.extractor
from sleap_roots_contracts import ResultEnvelope

from trait_extractor.extractor import extract_batch

_FIXTURE_TREE = Path("tests/data/rice_3do_pipeline_output")


def test_second_call_skips_unchanged_scan(tmp_path):
    """A second identical run skips (not recomputes) and leaves output unchanged."""
    first = extract_batch(_FIXTURE_TREE, tmp_path)
    assert first.ok
    assert set(first.succeeded) == {"scan0K9E8BI", "scanYR39SJX"}
    assert first.skipped == []

    before = {
        scan_key: (tmp_path / f"{scan_key}.result.json").read_bytes()
        for scan_key in first.succeeded
    }

    second = extract_batch(_FIXTURE_TREE, tmp_path)
    assert second.ok
    assert set(second.skipped) == {"scan0K9E8BI", "scanYR39SJX"}
    assert second.succeeded == []

    for scan_key, content in before.items():
        assert (tmp_path / f"{scan_key}.result.json").read_bytes() == content


def test_changed_traits_code_sha_forces_recompute(tmp_path):
    """A different traits_code_sha on the second call forces recomputation."""
    extract_batch(_FIXTURE_TREE, tmp_path, traits_code_sha="sha-one")
    out = tmp_path / "scan0K9E8BI.result.json"
    first_key = ResultEnvelope.model_validate_json(
        out.read_text()
    ).provenance.idempotency_key

    result = extract_batch(_FIXTURE_TREE, tmp_path, traits_code_sha="sha-two")
    assert "scan0K9E8BI" in result.succeeded
    assert "scan0K9E8BI" not in result.skipped

    second_key = ResultEnvelope.model_validate_json(
        out.read_text()
    ).provenance.idempotency_key
    assert second_key != first_key


def test_contract_version_change_forces_recompute_even_if_idempotency_key_matches(
    tmp_path, monkeypatch
):
    """A contract_version-only change forces recompute, even if idempotency_key matches.

    idempotency_key's hash does not include contract_version, so comparing
    idempotency_key alone would wrongly skip this case.
    """
    extract_batch(_FIXTURE_TREE, tmp_path)
    out = tmp_path / "scan0K9E8BI.result.json"
    first_envelope = ResultEnvelope.model_validate_json(out.read_text())

    monkeypatch.setattr(
        trait_extractor.envelope, "contract_version", lambda: "9.9.9-different"
    )
    result = extract_batch(_FIXTURE_TREE, tmp_path)

    assert "scan0K9E8BI" in result.succeeded
    assert "scan0K9E8BI" not in result.skipped
    second_envelope = ResultEnvelope.model_validate_json(out.read_text())
    assert second_envelope.provenance.contract_version == "9.9.9-different"
    assert (
        second_envelope.provenance.idempotency_key
        == first_envelope.provenance.idempotency_key
    )


def test_corrupt_prior_output_is_recomputed_not_crashed(tmp_path, caplog):
    """A corrupt pre-existing output does not crash the batch and is recomputed."""
    (tmp_path / "scan0K9E8BI.result.json").write_text(
        "{not valid json", encoding="utf-8"
    )

    with caplog.at_level("WARNING"):
        result = extract_batch(_FIXTURE_TREE, tmp_path)

    assert result.ok
    assert "scan0K9E8BI" in result.succeeded
    assert "scan0K9E8BI" not in result.skipped
    ResultEnvelope.model_validate_json(
        (tmp_path / "scan0K9E8BI.result.json").read_text()
    )
    assert "scan0K9E8BI" in caplog.text


def test_skip_does_not_invoke_expensive_steps(tmp_path, monkeypatch):
    """Skipping avoids Series loading, not just the write.

    A full, correct recompute would also produce byte-identical output, so output
    equality alone doesn't prove the expensive steps were skipped -- this test counts
    calls to load_series directly: the first run must invoke it, the second (skipped)
    run must not.
    """
    real_load_series = trait_extractor.extractor.load_series
    calls = []

    def _counting_load_series(*args, **kwargs):
        calls.append(1)
        return real_load_series(*args, **kwargs)

    monkeypatch.setattr(trait_extractor.extractor, "load_series", _counting_load_series)

    first = extract_batch(_FIXTURE_TREE, tmp_path)
    assert first.ok
    assert len(calls) == 2  # one per fixture scan

    calls.clear()
    second = extract_batch(_FIXTURE_TREE, tmp_path)
    assert second.ok
    assert set(second.skipped) == {"scan0K9E8BI", "scanYR39SJX"}
    assert calls == []
