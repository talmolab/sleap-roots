"""A3-traits trait-extractor service.

Consumes ``sleap-roots-predict``'s per-scan output (a ``{scan_key}.predictions.json``
manifest + named per-root ``.slp``) plus a ``{scan_key}.scan_metadata.json`` sidecar,
computes root traits with the ``sleap_roots`` pipelines, and emits a per-scan
``sleap-roots-contracts`` ``ResultEnvelope`` as JSON for Bloom write-back.

This is a service package that lives at the repository root and is deliberately
excluded from the published ``sleap-roots`` wheel; it depends on
``sleap-roots-contracts`` (a dev/test/container dependency, never a runtime dependency
of the ``sleap_roots`` library).
"""
