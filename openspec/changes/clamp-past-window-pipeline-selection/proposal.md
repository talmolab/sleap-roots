## Why

Scans older than every window for their species can't run today. The bloom#971 decision comment
counts 21,902 such cylinder scans on staging.
- sleap-roots-predict's `choose_models` uses the same per-species windows, so predict finds no
  model for them and raises `no models resolved`.
- Even with predictions, `trait_extractor`'s `choose_pipeline` would raise `No pipeline matches`.

bloom#971 settled phase 1 ([decision](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/971#issuecomment-5937500723),
[update](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/971#issuecomment-5938030166)):
a scan older than its species' window runs with that species' **highest-age** window, with no
upper limit. The rule lives directly in predict's `choose_models` and traits' `choose_pipeline`;
a shared matcher in sleap-roots-contracts (#13/#14) is phase 2.

This change is the traits half. The predict half is a parallel change to `choose_models`. As of
2026-10-01 it is drafted locally as `update-past-window-model-selection`, not yet pushed or opened
as a PR. The two changes test the same inputs (tasks.md "Shared case list").

## What Changes

- **`choose_pipeline` clamps past-window ages.** Ages above every window for the scan's species +
  mode match that species + mode's highest window. The normative rule is in the spec delta; the
  cases are in tasks.md.
- **Only the matching age is clamped.** `params` is not mutated, so provenance and the
  idempotency key keep the scan's real age (`build_provenance` runs before selection in
  `extract_scan`). Nothing new is stamped in the envelope or provenance. Per bloom#971, saved
  `params` + `predict_models` + the wandb windows are enough to tell a past-window result apart.
- **A public `past_window_age(params, cards)` helper** returns the age to match at (the
  species + mode's highest `age_max`) for a past-window scan, else `None`. `choose_pipeline` uses
  it, keeps its signature, and logs nothing. This mirrors predict's draft, which adds the same
  helper beside a silent `choose_models`, so phase 2's shared matcher can take one shape from
  both repos.
- **One WARNING per clamped scan, from `extract_scan`**, naming the scan key. The batch CLI's
  per-scan `ok`/`skip`/`FAIL` lines print only after the whole batch finishes, so a warning
  without the key couldn't be traced to its scan.
- **Unchanged:**
  - in-window selection;
  - younger-than-window scans still raise (bloom#994, open and undecided as of 2026-10-01);
  - ages in a gap between windows, and species + modes with no card, still raise
    `No pipeline matches`;
  - `override` still wins;
  - ambiguity still raises, including a tie at the highest window;
  - `check_pipeline_compatible` is untouched.
- **Existing test rewritten.** `test_no_match_raises` asserts rice age 99 raises; it now clamps to
  `OlderMonocotPipeline`. The no-match coverage moves to younger-than-window, no-card and
  wrong-mode cases.

## Impact

- Affected spec: `result-envelope-output`, MODIFIED requirement "Pipeline selection by species,
  mode, and age".
- Affected code:
  - `trait_extractor/pipeline_chooser.py` (new `past_window_age`, `choose_pipeline` and the
    module docstring);
  - `trait_extractor/extractor.py` (logs the clamp warning);
  - `trait_extractor/pipeline_selection.yaml` (header comment only; no card changes).
- Tests: `tests/trait_extractor/test_pipeline_chooser.py`, new
  `tests/trait_extractor/test_past_window.py`.
- Docs: `docs/changelog.md`, `docs/dev/trait-extractor-service.md`.
- `trait_extractor/` is excluded from the published wheel, so the published `sleap_roots` library
  is unaffected.
- **Runs:** past-window scans that failed before will now produce envelopes; they have no prior
  envelope, so skip-if-done doesn't skip them. The clamp changes no in-window scan's pipeline and
  no scan's key inputs. As with any image bump, the new image's `traits_code_sha` changes every
  scan's idempotency key, so the first run after deploy recomputes every in-scope scan.
- **Deploy:**
  - Merging to main builds and pushes a new trait-extractor image (`docker-trait-extractor.yml`).
    sleap-roots-pipeline then re-pins `sleap-roots-trait-extractor-template.yaml`'s `image:`
    digest and `SRT_TRAITS_CONTAINER_DIGEST`. `SRT_TRAITS_CODE_SHA` is baked into the image, not
    set in the template.
  - That template on sleap-roots-pipeline main pins `sha-e373b0f` (sleap-roots#269). As of
    2026-10-01, no commit touching the image's inputs has landed on sleap-roots main since, so the
    new image differs from that pin only by this change. It bumps no contracts version, so no
    Bloom re-pin is needed first. This describes the repo's pin; the cluster may run something
    else.
- **Order:**
  - This and the predict change can merge in either order. A past-window scan still fails until
    both images are live.
  - sleap-roots-pipeline re-pins both templates in one PR. If the re-pins must be split, re-pin
    traits first. With predict's clamp live and traits' not, every past-window scan runs GPU
    inference and then fails at traits with exit 3, which the template retries twice.
  - The past-window warning in Bloom's confirm dialog ships last. bloom#965 added that dialog; the
    bloom#971 decision specifies the warning.
- **Out of scope:** younger-than-window scans (bloom#994), choosing models per root type (bloom#897,
  phase 2), species with no cards (bloom#993), and the shared contracts matcher
  (sleap-roots-contracts#13/#14).
