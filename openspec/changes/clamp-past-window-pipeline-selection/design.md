## Context

`choose_pipeline` (`trait_extractor/pipeline_chooser.py`) filters `cards` by `species`, `mode` and
an inclusive `age_min <= age <= age_max` window, then raises on zero or more than one match.
bloom#971 phase 1 makes ages above every window for a species + mode match the highest window
instead. sleap-roots-predict's `choose_models` gets the same rule in a parallel change. The two
matchers share no code (`PipelineCard` is in-tree here; sleap-roots-contracts#14 is open), so
parity rests on both repos testing the same inputs (tasks.md "Shared case list").

## Goals / Non-Goals

- Goals: past-window scans select the highest window's pipeline; the scan's real age stays in
  `params`, provenance and the idempotency key; in-window scans select exactly what they do today.
- Non-Goals: younger-than-window scans (bloom#994), gaps between windows, choosing models per
  root type (bloom#897), a shared matcher in contracts (#13/#14), any new provenance or envelope
  field, and the pre-existing `cards or load_pipeline_cards()` fallback in `extract_scan` (an
  empty `cards` list there still means "use the packaged cards").

## Decisions

- **Shape: mirror predict.** Predict's draft keeps `choose_models` silent and adds a public
  `past_window_age(params, cards, overrides)` helper, and its batch logs the warning with the scan
  key. Traits takes the same shape, so phase 2's shared matcher (contracts#13) can lift one design
  from both repos:
  - `past_window_age(params, cards) -> Optional[int]` (pure, logs nothing);
  - `choose_pipeline(params, cards, override=None)`, signature unchanged, matching at that age and
    logging nothing;
  - `extract_scan` logs the warning, since it holds the scan key.

  Traits' helper has no `overrides` argument: predict's overrides are per root type, while traits'
  `override` replaces the whole selection, and `extract_scan` never passes one. An earlier draft of
  this change had `choose_pipeline` log the warning through an optional keyword-only `scan_key`.
  It was dropped for parity, since it changes no selection result.

  Two smaller differences from predict's helper, neither of which matters in phase 1, are noted
  for contracts#13:
  - Predict's helper validates and coerces `params` itself (a named error on a missing key; a
    string age is converted to an `int`). Traits' helper reads the already-canonical `int` age,
    as `choose_pipeline` does.
  - Predict keeps its helper module-level and unexported. Traits' is "public" only in that tests
    and `extract_scan` import it from `pipeline_chooser`; `trait_extractor/__init__.py` exports
    nothing.
- **`past_window_age`.** Let `same` be the cards whose `species` and `mode` equal the scan's.
  Return `max(c.age_max for c in same)` when `same` is non-empty and `age` is greater than that
  maximum; otherwise return `None`. If the age is above the maximum it can't match any card in
  `same`, so this only fires when the real age matches nothing.
- **`choose_pipeline` algorithm.** After `override` (which returns first, as today):
  1. `matched_age = past_window_age(params, cards)`, or the real age if that is `None`.
  2. Match the cards on `species`, `mode` and `matched_age`.
  3. Zero matches: raise `No pipeline matches species=... mode=... age=<real age>`, worded as
     today. This catches every unmatched case: no card for the species + mode, an age below its
     lowest window, and a gap between its windows. A clamped age always matches the card holding
     the maximum, so this never happens after a clamp.
  4. More than one match: raise the existing
     `Ambiguous pipeline selection (<n> cards match) for species=... mode=... age=<real age>`.
     After a clamp, append ` matched as age=<N>`, so both ages appear (as predict's ambiguity error
     does). A tie at the highest `age_max` lands here, so the clamp never breaks a tie.
  5. Resolve the class; an unknown name raises.
- **`extract_scan`.** After the skip-if-done check, resolve the cards once
  (`cards or load_pipeline_cards()`, as today) and pass that same list to both calls:
  `pipeline_cls = choose_pipeline(params, cards)`. Then, if `past_window_age(params, cards)` is
  not `None`, log the warning, and only then run `check_pipeline_compatible`. Logging after
  `choose_pipeline` returns means a raise logs nothing. Logging before the grain guard means a
  past-window multiplant scan still records that it was clamped before it is rejected.
- **Clamp the matching age only.** `params.values` is read, never written. `extract_scan` builds
  provenance from `params` before selection, so the idempotency key comes from the real age
  whatever the chooser does.
- **Warning format.** One `logger.warning` on `trait_extractor.extractor`:

  `past-window age: scan_key=scan0K9E8BI species='rice' mode='cylinder' age=18 matched as age=10 -> OlderMonocotPipeline`

  Exactly: `f"past-window age: scan_key={scan_key} species={species!r} mode={mode!r} age={age}
  matched as age={matched_age} -> {pipeline_cls.__name__}"`. Species and mode use `%r`, as the
  existing errors do.
  - The batch CLI configures no logging handler. The first such lines reach stderr bare,
    through Python's last-resort handler. After `sleap_roots/convhull.py`'s module-level
    `logging.debug` first fires, it implicitly runs `logging.basicConfig()`, so later lines
    carry a `WARNING:trait_extractor.extractor:` prefix. That's why the message carries its
    own fixed `past-window age:` marker and every field, and why the docs say to search for
    the marker anywhere in the line. Configuring logging in the CLI was considered and left
    out of this change; see "Implementation notes".
  - Pod logs are readable under the `bloom-pipeline` kubeconfig (sleap-roots-pipeline
    `docs/cluster-identities.md`) but aren't archived. The warning is for diagnosis; the
    envelope's real age is the lasting record.

## Risks / Trade-offs

- **Older roots measured with the highest window's pipeline.** Every species except rice has a
  single cylinder window, so a clamped scan gets the pipeline it would get in-window. Rice has two
  windows (YoungerMonocot 2–5, OlderMonocot 6–10); past day 10 it gets OlderMonocot (crown only),
  matching predict's clamped `rice-older-crown`. The traits on these older scans are not
  validated: bloom#971 chose to run them anyway, with a warning in Bloom's confirm dialog.
  The changelog and `docs/dev/trait-extractor-service.md` say so where users will read it.
- **Multiplant cylinder and plate** past-window scans clamp to `MultipleDicotPipeline` /
  `MultipleDicotPlatePipeline`, which the scan-grain guard still rejects, the same as in-window
  scans of those modes. Plate cards exist only on the traits side: predict's catalog has none,
  so predict resolves no models for a plate scan and no manifest ever reaches traits. The
  plate clamp is therefore unreachable in production today, and a parity audit should treat
  it as N/A rather than a mismatch.
- **Parity drift with predict.** Mitigated by the shared case list, checked against predict's PR
  before merge (tasks.md 0.1).
- **Deploy order.** If predict's clamp goes live before this one, past-window scans run GPU
  inference, write predictions, then fail at traits (`No pipeline matches`, exit 3). The traits
  template retries that twice (`retryStrategy` `limit: 2`, `retryPolicy: Always`). Bloom gets
  nothing bad (write-back reads envelopes only), but compute is wasted. Re-pinning both together,
  or traits first, avoids it. The reverse order changes nothing visible: predict still fails
  those scans before traits sees them.
- **Rollback.**
  - Operational: re-pin sleap-roots-pipeline's traits template to the previous digest. No library
    release or contracts change is involved.
  - Code: `git revert` of the squash commit. It's clean because no schema, envelope field or
    contracts pin changes.
  - What a rollback doesn't undo: envelopes already written for past-window scans stay in the
    output tree and in Bloom. The old image has a different `traits_code_sha`, so those scans'
    keys no longer match and are not skipped; they are re-attempted and fail with
    `No pipeline matches` as before. As on any deploy, the old image's `traits_code_sha` also
    changes every in-window scan's key, so a rollback recomputes everything in scope. Removing
    the past-window results means deleting the Bloom rows, which is out of scope here.

## Implementation notes

### Why two envelope tests instead of one?

tasks.md 1.7 put the skip-if-done re-run inside `test_past_window_envelope_keeps_real_age`. The
implementation moves it to its own `test_past_window_rerun_is_skipped_without_warning`, so a
skip failure is reported separately from a provenance or trait failure. The assertions are the
ones 1.7 lists; nothing was dropped. Both tests were red before the clamp and green after it,
as 1.9 and 2.5 predicted for the combined test.

### Why more tests than tasks.md listed?

The pre-PR `/review-pr` pass mutation-tested the branch: 18 of 21 mutants were caught. Two
of the survivors were real gaps, and the third can't be caught by any test: a zero-match only
happens when nothing was clamped, so the matched and real ages are equal there. Tests were
added after the fact for the two real gaps; they pin behaviour the code already had, so they
were green on arrival rather than red first:
- `test_warning_uses_the_cards_passed_in`: injected cards whose highest window (12) differs
  from the packaged one (10). This pins "the cards are resolved once and shared"; a warning
  computed from the packaged cards survived before.
- Every chooser test now asserts that nothing at all is logged, at DEBUG on the root logger.
  Before, only past-window rows were checked, and only for the chooser's own logger name.

The same pass also:
- folded the injected rows' expected outcomes into `INJECTED_CASES`, so one test selects or
  raises per row;
- pinned the in-window ambiguity message (no `matched as` suffix);
- added a plate row to the extractor's warn-then-reject test.

### Why document the log prefix instead of fixing it?

Found while reviewing predict#50: the `past-window age:` line can change format partway
through a batch (see "Warning format"). The author chose to correct the docs rather than
configure logging in the CLI. It's cosmetic: envelopes, exit codes and the recommended grep
are unaffected, and the CLI test already matches on a substring. Fixing the root cause, a
library calling module-level `logging` functions, belongs in a separate sleap_roots change.

## Open Questions

- bloom#994 (open, no comments as of 2026-10-01): whether younger-than-window scans should clamp
  down to the lowest window. A future change would MODIFY this same requirement, so archive this
  change first (tasks.md 6.1).
