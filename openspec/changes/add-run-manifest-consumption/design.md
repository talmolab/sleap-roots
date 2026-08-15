## Context

Cross-repo initiative `talmolab/sleap-roots-pipeline#37` ("Cross-repo correctness: manifest-scoped
processing") fixes two real contamination incidents seen during A4 cluster testing by having each
pipeline stage scope itself to exactly the `scan_keys` a run was given, instead of directory-wide
scanning. `sleap-roots-contracts` (v0.1.0a7, PR #30) and `bloomctl` (bloom #653/#655, merged
2026-08-14) are done. This is the `sleap-roots` (traits) leg of that chain, running in parallel with
an equivalent change in the separate `sleap-roots-predict` repo.

Two load-bearing facts, both verified directly against the installed/checked-out packages, not
assumed from docs:

1. **`RunManifest` (contracts 0.1.0a7) is scoping-only.** Its 3 fields are `schema_version`,
   `pipeline_run_id`, `scan_keys: list[str]` — **no `idempotency_key` field**. That's an explicit
   non-goal of its design doc ("this manifest only carries the scoping data, not the comparison
   logic"). The idempotency key that already exists — `Provenance.idempotency_key`, auto-derived
   from `scan_key + images_checksum + model versions/weights + param_hash + predict_code_sha +
   traits_code_sha` — lives on a *different*, already-in-use model. `trait_extractor`'s own
   `envelope.py:build_provenance()` already builds this exact `Provenance` for every scan; it's just
   never compared against anything before overwriting. So "consume RunManifest" (scoping) and "add
   skip-if-done" (idempotency comparison) are two independent mechanisms this change delivers
   together, not one.
2. **The CLI entrypoint is frozen at 2 positional args.** `trait_extractor/__main__.py` takes exactly
   `input_dir output_dir`; sleap-roots-pipeline's design doc confirms a 3rd argparse arg would
   hard-fail. The manifest must be discovered by well-known filename
   (`sleap_roots_contracts.run_manifest.RUN_MANIFEST_FILENAME`, `"run_manifest.json"`) inside
   `input_dir`, never via a new CLI arg.

**Confirmed during review (2026-08-14), not assumed:** Bloom's deployed `insert_cyl_result_envelope`
write-back RPC hard-pins `Provenance.contract_version` to exactly `0.1.0a3` (bloom PR #399, merged to
`staging`, closing bloom#393 — the RPC is "prefix-tolerant" only for a leading `v`, and its own test
suite explicitly asserts `0.1.0a30`/`0.1.0a2`/etc. are rejected, not just `v`-mismatches). Bumping
this repo's pin to `0.1.0a7` means every envelope emitted after this change ships will carry
`contract_version = "0.1.0a7"` — which that RPC, as it stands today, **will reject**. See "Risks /
Trade-offs" below; this is a real, confirmed pre-deploy blocker, not a hypothetical one, and is
**not fixable inside this proposal** (it requires a Bloom-side PRs mirroring #399's pattern for the
new version).

## Goals / Non-Goals

- Goals: scope scan discovery to `RunManifest.scan_keys` when present; add a real
  idempotency-key-based skip-if-done (not existence-only); copy the manifest forward so `write-back`
  can see it; preserve backward compatibility when no manifest is present.
- Non-Goals: `sleap-roots-predict`'s own consumption of `RunManifest` (separate repo/handoff);
  `write-back`'s unscoped-glob bug; any lock/lease mechanism for reading the manifest (this is a pure
  read-then-skip-or-compute decision, no concurrent writers on the trait_extractor side); any CLI
  signature change.

## Decisions

- **Decision: manifest-absent fallback is the unscoped `rglob`, unchanged.** Confirmed with user.
  Preserves every existing test and any non-pipeline caller. Scoping/skip-if-done simply don't
  activate without a manifest to scope against — this is intentionally the same posture
  `sleap-roots-pipeline`'s design doc describes ("a file the readers simply don't look for yet" is
  forward-compatible; requiring it outright is not, since nothing writes it for non-pipeline runs).
  - Alternative considered: treat absence as a batch-level error. Rejected — would break every
    existing test and local dev workflow for no correctness benefit (there's nothing to scope
    against if bloomctl never ran).

- **Decision: a manifest-declared `scan_key` with no matching `{scan_key}.predictions.json` is a
  per-scan failure**, appended to `BatchResult.failed`, not silently dropped. Confirmed with user.
  Mirrors the existing missing-sidecar failure path (extractor.py's `FileNotFoundError` for a
  missing sidecar) — an upstream gap (predict didn't produce something it was asked to) should
  surface, not vanish silently.
  - Alternative considered: silently ignore. Rejected — that would mask a real upstream problem as
    if the batch fully succeeded.

- **Decision: skip-if-done gets its own `BatchResult.skipped: List[str]` bucket**, additive to the
  existing `succeeded`/`failed` fields; `BatchResult.ok` is unchanged (skipped is not a failure).
  Confirmed with user. Mirrors predict's existing "Skipping %s (manifest exists)" logging intent and
  gives observability into reuse vs fresh computation without breaking any existing consumer of
  `BatchResult` (additive field, existing `succeeded`/`failed` semantics untouched).
  - Alternative considered: fold skipped scans into `succeeded`. Rejected — loses the
    computed-this-run vs. reused-from-prior-run distinction that's the whole point of surfacing
    skip-if-done for observability/debugging.

- **Decision: compute the "expected" `Provenance` early, before the expensive steps, and reuse it.**
  `build_provenance(manifest, sidecar, params, ...)` only needs `manifest`, `sidecar`,
  `params = sidecar.to_resolved_params()`, and the batch-level `traits_code_sha`/
  `traits_container_digest` — all available before `load_series`/`choose_pipeline`/
  `check_pipeline_compatible`/`compute_scan_traits` run. So `extract_scan` builds `Provenance` right
  after resolving `params`, checks it against any pre-existing output's stored `idempotency_key`,
  and only proceeds to the expensive steps if they differ (or nothing exists yet) — reusing the
  already-built `Provenance` for the final envelope rather than rebuilding it after computation. This
  is what makes skip-if-done actually skip the expensive work, not just skip the write.
  - Alternative considered: run the full pipeline every time and diff the *output* envelope before
    writing. Rejected — defeats the entire purpose (the expensive trait computation would still run
    on every "skipped" scan).

- **Decision: a corrupt/unparseable pre-existing `{scan_key}.result.json` is treated as "not done",
  never crashes the batch, but logs a warning naming the scan.** Reading `idempotency_key` off a
  pre-existing output is a best-effort check, not a hard dependency — if the file is missing,
  truncated, or fails `ResultEnvelope`/JSON validation, `trait_extractor` proceeds exactly as if
  nothing existed there yet (full recompute + overwrite), rather than raising. Added during review:
  a `logger.warning` naming the scan_key is emitted in this case — costs nothing, doesn't compromise
  "never crash the batch," and gives a researcher auditing results later a trace that something
  unexpected was on disk (a prior crashed run, a corrupted file, someone else's process) rather than
  silent overwrite with zero record.
  - Alternative considered: raise and record the scan as failed. Rejected — a stale/corrupt output
    from a prior crashed run shouldn't block a fresh, valid recomputation; that's precisely the kind
    of contamination this initiative is trying to eliminate, not reproduce in traits.

- **Decision: skip-if-done compares BOTH `idempotency_key` AND `contract_version`, not
  `idempotency_key` alone.** Added during review, closing a real gap found by adversarial review:
  `idempotency_key`'s hash payload (`scan_key`, `images_checksum`, model info, `param_hash`,
  `predict_code_sha`, `traits_code_sha`, `predict_output_params` — confirmed by reading
  `sleap-roots-contracts`' `identity.py` directly) does **not** include `contract_version`. Before
  this change, that was harmless (every run unconditionally overwrote), but skip-if-done is the
  first thing that ever *acts on* a stale pre-existing key — so without this fix, the very
  `pyproject.toml` pin bump this proposal ships (`0.1.0a3` -> `0.1.0a7`) could cause an old
  `contract_version: "0.1.0a3"` envelope to be judged "done" forever after rebuild (its
  `idempotency_key` alone still matches, since none of that hash's inputs changed) and never get
  re-stamped with the new `contract_version` — silently perpetuating a stale contract version
  indefinitely, exactly the kind of drift the existing `test_envelope.py` pin-literal assertion
  exists to catch, but which skip-if-done would let slip through in production, not in tests. Fix:
  skip only when the pre-existing output's `idempotency_key` **and** `contract_version` both equal
  the freshly-computed expected values; a `contract_version` mismatch alone forces recomputation even
  if `idempotency_key` matches. This also generalizes correctly to any future contracts pin bump.
  - Alternative considered: leave `idempotency_key` alone as sufficient. Rejected once traced through
    `identity.py` — demonstrably insufficient the moment a contracts pin bumps without a corresponding
    change to any of the hash's own inputs, which is exactly what this proposal itself does.

- **Decision: `trait_extractor/__main__.py` is updated to report skipped scans, not just
  succeeded/failed.** Added during review: `main()`'s existing loop only prints `ok`/`FAIL` lines and
  a `"{n} succeeded, {n} failed"` summary; a skipped scan would otherwise print nothing at all,
  making a fully-skipped batch look indistinguishable from a batch that discovered nothing — a real
  regression from today's behavior (every discovered scan currently prints at least `ok`), and
  directly undermines the "observability into reuse vs fresh computation" rationale for
  `BatchResult.skipped` in the first place, since the CLI's stdout/stderr is the only audit trail
  that survives into Argo pod logs in production.
  - Alternative considered: leave `__main__.py` untouched, since `BatchResult.skipped` alone
    satisfies the dataclass-level goal. Rejected — the dataclass is not what an operator watching
    container logs sees; the stated observability goal is not delivered without this.

- **Decision: manifest-scoped resolution reuses the same `seen: Dict[str, Path]` duplicate-scan_key
  guard the unscoped path already has**, rather than a separate/simpler lookup. Added during review:
  a naive "first `rglob` match for this scan_key" resolution would silently pick whichever of two
  candidate directories happens to sort first if a stale leftover directory (the literal
  contamination scenario motivating this whole proposal) coexists with the correct one for the same
  in-scope `scan_key` — fixing "wrong scan_key entirely" while leaving "right scan_key, wrong/stale
  file" completely unfixed. Concretely: the scoped path keeps the existing single `rglob` loop and
  `seen` dict structurally unchanged, adding only a `continue` for out-of-scope stems as the first
  check inside the loop — NOT a separate per-`scan_key` search — so the existing duplicate-collision
  check (raise into `BatchResult.failed` naming the collision) applies unmodified; two candidates for
  the same in-scope scan_key are refused the same way today's unscoped duplicate-across-manifests
  case already is.
  - Alternative considered: a simpler "first match wins" resolution. Rejected — reintroduces the
    exact contamination class this proposal exists to close.

- **Decision: a new `trait_extractor/run_manifest.py` module, not reused/mixed into
  `trait_extractor/manifest.py`.** `manifest.py` already owns `PredictionManifest` +
  `ScanMetadata` — predict's *per-scan* manifest, an unrelated concept from `RunManifest`'s
  *per-run* scoping shape. Keeping them in separate modules avoids the naming collision risk
  ("manifest" meaning two different things) that a prior research pass flagged as a real point of
  confusion.

## Risks / Trade-offs

- **CONFIRMED PRE-DEPLOY BLOCKER, not fixable inside this proposal: Bloom's live write-back RPC will
  reject every envelope this change emits, until Bloom's side is updated too.** Verified directly
  against bloom PR #399 (merged to `staging`, closed bloom#393): `insert_cyl_result_envelope` compares
  `contract_version` after stripping only a single leading `v`, pinned to the literal `0.1.0a3` —
  its own test suite explicitly asserts near-misses (`0.1.0a2`, `0.1.0a30`, `V0.1.0a3`, `vv0.1.0a3`,
  trailing whitespace) are all rejected, not just `v`-prefix mismatches. This proposal's pin bump
  means every envelope emitted after this change ships will carry `contract_version = "0.1.0a7"`,
  which that RPC will reject exactly as it once rejected `0.1.0a3` before #399. **This must be
  resolved on the Bloom side (a new PR mirroring #399's pattern, re-pinning the RPC's accepted
  literal to `0.1.0a7`) before this repo's image is rebuilt and redeployed to the pipeline** — merging
  this proposal's code and tests in `sleap-roots` is safe (nothing here talks to the live RPC), but
  deploying the resulting image without that Bloom-side companion change would silently break write-back
  for 100% of scans, not a partial degradation. Recommend filing a tracking issue in
  `Salk-Harnessing-Plants-Initiative/bloom` referencing this proposal and #399's pattern (confirmed
  during review: no such issue exists yet), and sequencing the image rebuild/rollout after that Bloom
  PR merges — a decision for the user/deploy owner, not something resolved by this change alone.
  Worth also noting on `talmolab/sleap-roots-pipeline#37` itself, since its own tracking comments log
  each newly-discovered gap in this chain and this RPC-version-rejection risk isn't among them yet.
  **Observable consequence if sequencing is violated anyway (added during second-pass review, since
  the fact alone doesn't tell an operator what they'd see):** `trait_extractor` itself is unaffected
  and unaware — it writes `{scan_key}.result.json` to `output_dir` and does its own skip-if-done
  bookkeeping entirely from its own output files, never from Bloom's ingestion state, so envelope data
  is **not destroyed**; it sits in `output_dir` and is presumably re-ingestable once the Bloom-side fix
  lands and write-back is re-run over it (whether write-back itself retries a rejected RPC call or
  drops it with no record is write-back's own contract, outside what this proposal can inspect — an
  open question for whoever owns that sequencing decision, not resolved here). The sharper risk is a
  **false-green audit trail**: this same proposal's CLI-reporting fix (§ "`__main__.py` is updated to
  report skipped scans") means an operator watching container/Argo logs will see `ok`/`skip` lines for
  every scan exactly as if nothing were wrong, because `trait_extractor` never calls the write-back RPC
  and has no visibility into its outcome — a fully green `trait_extractor` run can coexist with 100% of
  envelopes silently failing to reach Bloom's database downstream, and nothing in this proposal's own
  observability surface would reveal that.
- Fixed-filename manifest with no run-scoping in its path is a known, accepted residual from
  contracts' own design doc (a concurrent-run race on `run_manifest.json` itself) — out of scope
  here; `bloomctl`'s write side owns that risk, and `trait_extractor` only ever reads it once per
  batch, never mutates it in place.
- Skip-if-done correctness depends entirely on `idempotency_key`'s (and, after this change,
  `contract_version`'s) inputs actually capturing every behavior-affecting factor. Confirmed by
  reading `Provenance`'s full field list against `identity.py`'s hash payload: beyond the two fields
  this change explicitly compares, several other `Provenance` fields are excluded from both — a
  traits-code change landing without bumping `traits_code_sha`/`SRT_TRAITS_CODE_SHA` would be
  invisible and wrongly skipped; likewise a `sleap_roots` library version bump
  (`traits_sleap_roots_version`) without a corresponding `traits_code_sha` bump; and, not previously
  called out, `predict_container_digest`, `traits_container_digest`, and the full
  `predict_inference_config` (as opposed to the hashed `predict_output_params` subset) could each
  independently drift (e.g. a rebuilt base image / security patch changing a container digest with no
  code_sha bump) without forcing recomputation. This is an existing, accepted limitation of the
  idempotency-key design itself (shared with predict's parallel fix), not something this change
  attempts to close beyond the one field (`contract_version`) that this proposal's own pin bump makes
  immediately consequential.
- The `sleap-roots-contracts` pin jump from `0.1.0a3` to `0.1.0a7` spans four alpha releases. Verified
  during review (by diffing the actual published a3 and a7 wheels) that every contracts export
  `trait_extractor` currently imports (`ModelRef`, `InputRef`, `ResolvedParams`, `Provenance`,
  `TraitValue`, `ResultEnvelope`, `compute_idempotency_key`) is byte-identical across that range — the
  new surface (`RunManifest`, `RUN_MANIFEST_FILENAME`, `PredictionManifest`, `LabelCard`, etc.) is
  additive only. This is verified by direct wheel diff, not merely asserted; the full regression
  suite (task 1.3/6.1) is the ongoing safety net for anything this diff missed.

## Migration Plan

No data migration within `sleap-roots` itself. Purely additive/behavioral: existing callers without a
`run_manifest.json` see byte-identical behavior to today. Callers that do have one (via the pipeline,
once `sleap-roots-predict`'s parallel change starts copying it forward) get scoping + skip-if-done for
free once both `pyproject.toml` pin bumps and this change are merged and the image is rebuilt.

**Deploy-sequencing requirement (see Risks above):** the image rebuild + pipeline rollout for this
change MUST be sequenced after a companion Bloom-side PR re-pins `insert_cyl_result_envelope`'s
accepted `contract_version` literal to `0.1.0a7` — otherwise every real write-back call fails from the
moment the new image is live, even though every test in this repo passes.

### Why a second delta spec (`trait-extractor-image`) instead of just `result-envelope-output`?

Discovered during implementation (Section 1.2, updating pin-literal references), not anticipated when
this proposal was first drafted: the **deployed** `trait-extractor-image` capability
(`openspec/specs/trait-extractor-image/spec.md`) independently hardcodes the `0.1.0a3` literal in its
own requirement text and scenarios ("Slim contracts install via an extractor extra" and "Real entry
emits a valid envelope over the committed fixture") — a second, unrelated copy of the same fact this
proposal's `result-envelope-output` delta already updates. Left alone, archiving this change would have
left `trait-extractor-image`'s deployed spec silently contradicting both the actual pinned version and
the updated `test_package_boundary.py` guard. Added a MODIFIED delta for that capability alongside the
original two (see `specs/trait-extractor-image/spec.md`), updating the same literal in both places —
no new behavior, purely closing a latent spec/reality drift this proposal's own pin bump would
otherwise have introduced.

### Why a fix landed after implementation, during pre-PR self-review

A 5-subagent self-review of the finished diff (before opening the PR) found one real, genuine bug this
design missed: `copy_run_manifest_forward`'s unconditional `shutil.copyfile` raises `SameFileError`
when `input_dir` and `output_dir` resolve to the same path (a plausible dev/testing invocation, not
just a pipeline scenario), and that call sat outside `extract_batch`'s per-scan isolation boundary —
propagating uncaught and discarding the already-computed, already-durably-written `BatchResult`. Fixed
by (1) making `copy_run_manifest_forward` a no-op when source and destination resolve to the same
file, and (2) wrapping the copy-forward call in `extract_batch` in a `try/except OSError` that logs a
warning rather than crashing the batch — copy-forward is best-effort infrastructure for the next
pipeline stage, not part of this batch's own computed result. The same review pass also found two
`docstring`/`except`-clause references to `json.JSONDecodeError` that were dead code (pydantic v2's
`model_validate_json` parses and validates in one step, raising `pydantic.ValidationError` for both
malformed JSON and schema violations) — corrected in `envelope.py` and `run_manifest.py`. Tests added
for all of the above (`test_run_manifest.py`'s same-file and overwrite-a-different-manifest cases,
`test_batch.py`'s `input_dir == output_dir` case); the two manifest-invalidity tests that previously
asserted bare `Exception` were tightened to `pydantic.ValidationError` specifically, per the same
review pass's finding that a bare `Exception` assertion could mask an unrelated bug as if the intended
validation contract were being exercised.
