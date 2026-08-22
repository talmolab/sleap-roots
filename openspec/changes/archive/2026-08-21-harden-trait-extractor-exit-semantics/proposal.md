## Why

`trait_extractor`'s batch driver (`trait_extractor/__main__.py` / `extractor.py`, from #254) is
packaged as a GHCR image in #257 and will be driven by A4's Argo `trait-extractor` template (in
`sleap-roots-pipeline`). Tracked as sleap-roots#259, three driver-side behaviors need hardening for
that runtime — none is a defect in the current emitter or image, they're the sleap-roots-side
changes A4's request-driven-pipeline design (`sleap-roots-pipeline`'s
`docs/superpowers/specs/2026-07-06-a4-request-driven-pipeline-design.md` §8) assumes:

1. **Exit-code convention vs Argo `retryStrategy`.** Today `python -m trait_extractor <in> <out>`
   returns `1` if any scan failed (per-scan envelopes are still written; failure is isolated). A4's
   design wants an isolated "poison" scan to yield a `partial` run (continue, don't fail the step),
   while the Argo step carries `retryStrategy: {limit: 2}`. As-is, one bad scan → non-zero exit →
   Argo retries the whole batch twice, then marks the step failed — the opposite of "mark that scan
   failed, continue → partial". This was flagged and explicitly deferred to "the traits-wiring step"
   by both the archived `add-trait-extractor-image` design (Open Questions) and the current
   `trait-extractor-image` spec's "A failing scan yields a non-zero container exit without discarding good envelopes" scenario.
2. **Empty-input guard.** `extract_batch`'s unscoped fallback (no `run_manifest.json` present)
   recursively globs `input_dir` for `*.predictions.json`; when it finds none, `BatchResult.ok` is
   `True` and the batch writes nothing. A wrong or empty `-v …:/in` mount therefore produces a
   **green** Argo node that emitted zero envelopes — a silent no-op. (Confirmed narrower than it
   first looks: when a `run_manifest.json` **is** present, an in-scope `scan_key` with no matching
   file is already recorded as a per-scan failure today — `BatchResult.ok` is already `False`. The
   actual gap is only the no-manifest, zero-manifests-found case.)
3. **SIGTERM / PID-1 handling for graceful preemption.** The image's exec-form `ENTRYPOINT
   ["python","-m","trait_extractor"]` makes `python` PID 1, and the driver installs no SIGTERM
   handler. Per Linux PID-namespace semantics, a signal sent to PID 1 of a namespace from outside
   that namespace (kubelet/Argo preemption) is delivered only if a handler is explicitly installed
   — otherwise the pod ignores SIGTERM and waits out `terminationGracePeriodSeconds` before
   SIGKILL. This is a preemption-**latency** issue, not data loss: per-scan writes are already
   atomic (temp→rename, `envelope.py`) and the batch is idempotent on retry, so a SIGKILL loses no
   completed envelope — confirmed by the archived `add-trait-extractor-image` design's own
   "PID-1 / SIGTERM (deferred)" note, which this change closes.

**Cross-repo coordination note:** items 1 and 2 are meant to be reconciled uniformly with
`talmolab/sleap-roots-predict#26`, predict's identical hardening. As of this proposal,
`sleap-roots-predict#26` has no branch or PR yet, so there is no existing decision to reconcile
against. This proposal makes the call independently (below) and the resulting convention is called
out explicitly so the predict-side session mirrors it rather than inventing its own.

## What Changes

- **Exit-code convention (new, driver-owned):** `python -m trait_extractor <in> <out>` returns:
  - `0` — every discovered scan succeeded or was skipped (`BatchResult.ok`, no failures).
  - `3` — **partial**: `extract_batch` returned normally but `BatchResult.failed` is non-empty
    (one or more scans failed inside the existing per-scan isolation boundary — the `except
    Exception` in `extract_batch`'s discovery loop). A4's Argo template should treat this as a
    completed run with partial failures, not retry the whole batch.
  - `1` — **crash**: an exception escaped `extract_batch` entirely (invalid `run_manifest.json`,
    the new empty-input guard below, or any other bug). This is a real pod-level failure; Argo's
    `retryStrategy` should retry it.
  - `2` — **unchanged, not part of this convention**: `argparse` already exits `2` on a CLI usage
    error (missing/extra positional args), *before* `extract_batch` ever runs. Reviewed and
    confirmed empirically (`python -m trait_extractor` with no args exits `2` today). `2` is
    deliberately **not** reused for "partial" precisely to avoid colliding with this pre-existing,
    unrelated meaning — see design.md Decision 1.

  This maps exit codes onto a distinction the code already makes structurally (caught-inside-the-
  loop vs. escaped-the-loop) rather than introducing new failure-classification logic. It is a
  **driver-only** change — no A4 Argo template change is required to land this proposal;
  `retryStrategy` wiring that acts on the new codes is separate, tracked pipeline-repo work.

  **BREAKING (informational):** any existing caller that treats "any nonzero exit" as "the batch
  failed" (the only convention that has ever existed for this driver) still works unchanged — `1`
  and `3` are both nonzero. A caller that additionally assumed nonzero *always* means `1`
  specifically would break; no such caller is known to exist yet (the A4 template that will consume
  this hasn't been wired), but this is asserted, not proven, so it's called out explicitly here
  rather than left silent.
- **Empty-input guard:** `extract_batch` raises `RuntimeError` when no `run_manifest.json` is
  present at the top level of `input_dir` AND zero `*.predictions.json` files are discovered
  anywhere under it — before any envelope is written. This mirrors the existing precedent that an
  invalid `run_manifest.json` already raises uncaught and aborts the batch before processing
  anything (`test_invalid_manifest_aborts_batch`). Under the new exit-code convention this
  correctly surfaces as exit `1` (crash), not `3` (partial) — a misconfigured/empty mount is an
  operator error, not an isolated per-scan failure.
- **SIGTERM handling:** `trait_extractor/__main__.py`'s `main()` installs a `signal.signal(signal.
  SIGTERM, ...)` handler before running the batch. The handler logs and exits promptly with `143`
  (`128 + SIGTERM`, standard shell convention), distinct from the `0`/`1`/`3` batch-outcome codes
  above. No changes to `envelope.py`'s write path — writes are already atomic.
- Update `result-envelope-output`'s "Batch driver and module CLI" requirement and
  `trait-extractor-image`'s "Container image runs the trait extractor over predict outputs"
  requirement to specify the three driver-owned exit codes instead of "exits non-zero" and
  "non-zero container exit", and add scenarios for the empty-input guard and SIGTERM handling.
- Update `docs/dev/trait-extractor-service.md` (currently documents the old two-way "exits
  non-zero if any scan failed" convention) to describe the three-way codes, and add a
  `docs/changelog.md` `[Unreleased]` entry, matching the convention the recent
  `add-run-manifest-consumption` change followed.
- No CLI/argparse signature change — the entrypoint stays exactly `python -m trait_extractor
  <input_dir> <output_dir>`.

## Impact

- **Affected specs:** `result-envelope-output` (MODIFIED: "Batch driver and module CLI");
  `trait-extractor-image` (MODIFIED: "Container image runs the trait extractor over predict
  outputs").
- **Affected code:** `trait_extractor/__main__.py` (exit-code mapping, SIGTERM handler),
  `trait_extractor/extractor.py` (`extract_batch`'s empty-input guard).
- **Affected docs:** `docs/dev/trait-extractor-service.md` (exit-code prose), `docs/changelog.md`
  (`[Unreleased]` entry).
- **Known, pre-existing, unaddressed gap (not introduced by this change):** the
  `trait-extractor-image` spec's scenarios describe container-boundary behavior (exit-code
  propagation, SIGTERM delivery to the real PID-1 process), but no automated test runs the built
  image itself — `docker-trait-extractor.yml` only builds/pushes it. Coverage here is via the
  driver-level subprocess tests as a proxy (reasonable given the exec-form `ENTRYPOINT` has no
  intermediary process), not a true container-boundary test. Flagged, not fixed, by this proposal.
- **Out of scope:** the A4 Argo template's `retryStrategy`/`continueOn` wiring that acts on the new
  exit codes, and re-pinning the Argo template's image digest to the rebuild this change triggers
  (tracked in `sleap-roots-pipeline`'s A4 wiring, per design.md) — the new exit codes only reach a
  live pod once that re-pin happens; any change to `sleap-roots-predict` (tracked separately in
  `sleap-roots-predict#26`, which should mirror this proposal's exit-code convention — see
  design.md).
