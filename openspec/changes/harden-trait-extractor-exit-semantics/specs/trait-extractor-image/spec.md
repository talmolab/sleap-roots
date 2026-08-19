## MODIFIED Requirements

### Requirement: Container image runs the trait extractor over predict outputs

The repository SHALL provide a `trait-extractor.Dockerfile` at the repo root that builds an
image which runs the merged `trait_extractor` service as its entry point. The built image
SHALL execute `python -m trait_extractor <input_dir> <output_dir>` such that, given a
directory of predict per-scan outputs (a `{scan_key}.predictions.json` manifest, a
`{scan_key}.scan_metadata.json` sidecar, and the referenced `.slp` artifacts), it writes one
`{scan_key}.result.json` `ResultEnvelope` per scan into the output directory. The image SHALL
contain the importable `sleap_roots` library, the `sleap_roots_contracts` package, and the
`trait_extractor` package.

The emitter's parsing, pipeline-selection, and per-scan emission behavior is owned by the
`result-envelope-output` capability; this requirement asserts only that the built image
executes that behavior end-to-end at the container boundary, including propagating the batch
driver's three-way process exit code (`0` full success, `3` partial, `1` crash — see
`result-envelope-output`'s "Batch driver and module CLI" requirement) unchanged, which Argo reads
for DAG-node success/retry decisions. Coverage of this container-boundary propagation is currently
via the driver-level subprocess tests as a proxy (the exec-form `ENTRYPOINT` has no intermediary
process to diverge from the driver's own exit code); no automated test runs the built image itself
end-to-end (a pre-existing gap, not introduced by this requirement).

#### Scenario: Real entry emits a valid envelope over the committed fixture

- **WHEN** the image is built and run as
  `docker run -v <tests/data/rice_3do_pipeline_output>:/in -v <out>:/out
  ghcr.io/talmolab/sleap-roots-trait-extractor /in /out`
- **THEN** a `scan0K9E8BI.result.json` and a `scanYR39SJX.result.json` are written to the
  output directory
- **AND** each file parses as a `sleap-roots-contracts` `ResultEnvelope` whose
  `provenance.scan_key` matches its manifest and whose `provenance.contract_version == "0.1.0a7"`
- **AND** the process exits `0` (all scans succeeded)

#### Scenario: Required packages are importable inside the image

- **WHEN** `python -c "import sleap_roots; import sleap_roots_contracts; import trait_extractor; import yaml"`
  is run inside the built image
- **THEN** all four imports succeed with a `0` exit code

#### Scenario: A failing scan yields the partial container exit code without discarding good envelopes

- **WHEN** the image is run over an input tree containing one valid scan and one scan whose
  manifest references a nonexistent `.slp`
- **THEN** the valid scan's `{scan_key}.result.json` is still written to the output mount
- **AND** the container exits `3` — the exec-form `ENTRYPOINT` propagates the batch driver's exit
  code unchanged (a shell-form entry would swallow it), and A4's Argo template can key off `3`
  specifically to treat this as a completed run with partial failures rather than retrying the
  whole batch (the `retryStrategy`/`continueOn` wiring itself is a separate, pipeline-repo change —
  see design.md)

#### Scenario: SIGTERM during preemption terminates the pod promptly

- **WHEN** Argo sends `SIGTERM` to the running container (preemption or cancellation) while a
  batch is in progress
- **THEN** the process exits promptly with code `143` instead of waiting out
  `terminationGracePeriodSeconds` before SIGKILL
- **AND** any `{scan_key}.result.json` already written to the output mount before the signal is
  unaffected — per-scan writes are atomic (temp→rename) and the batch is idempotent on retry
