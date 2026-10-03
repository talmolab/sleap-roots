## Why

Wheat and sorghum are going to production (talmolab/sleap-roots-pipeline#118, step 3;
talmolab/sleap-roots#276). Predict will get their model cards from
talmolab/sleap-roots-training#72. `trait_extractor`'s packaged `pipeline_selection.yaml` has no
row for either species, so every wheat or sorghum scan would run GPU inference and then fail at
traits with `No pipeline matches`.

The rows come from past production runs on hpi_dev, which forced the class through
`pipeline_params.json`:
- **Wheat:** EDPIE ran `OlderMonocotPipeline` on the wheat crown model at days 5, 11 and 14. That
  model labels seminal roots as `crown`. Wheat users get crown traits only.
- **Sorghum:** all five past sorghum runs used `DicotPipeline` on primary + lateral.

The data change alone would be a configuration change, but it falsifies the live spec. The
"Unmatched scans still raise" scenario of `result-envelope-output` uses `sorghum cylinder age 30`
as its no-card example. With a sorghum row, that scan clamps to 14 and resolves to
`DicotPipeline`. So the requirement is MODIFIED here.

## What Changes

- **Two packaged cards** in `trait_extractor/pipeline_selection.yaml`:
  - `wheat` / `cylinder` / 5–14 / `OlderMonocotPipeline`;
  - `sorghum` / `cylinder` / 3–14 / `DicotPipeline`.
- **The windows must equal the predict cards' windows** in sleap-roots-training#72 (wheat
  `age: "5, …, 14"`, sorghum `age: "3, …, 14"`, read 2026-10-03). Nothing checks this across repos
  (a follow-up in #118). A new test pins the two cards exactly, and its docstring names the
  cross-repo copy. The coupling is also recorded in `docs/dev/trait-extractor-service.md`.
- **Spec:**
  - the no-card example becomes `alfalfa` (no row, none planned);
  - wheat age 4 and sorghum age 2 join the below-window examples, and wheat plate age 10 joins
    the wrong-mode examples;
  - new scenarios cover the two cards, selection by window (boundaries and past-window clamp),
    the scan-grain guard, and a crown-only wheat scan through `extract_scan`.
- **Tests:** `alfalfa` replaces `sorghum` in `test_pipeline_chooser.py`'s shared case list and in
  `test_past_window.py::test_unmatched_scan_raises_without_warning`. Otherwise, sorghum 30 clamps
  in the first, and in the second it clamps, warns and fails the root-type guard instead.
- **Unchanged:** `choose_pipeline`, `past_window_age`, `check_pipeline_compatible` and
  `PIPELINE_REQUIRED_ROOTS`. `OlderMonocotPipeline` already needs only `crown`, and
  `DicotPipeline` needs `primary` + `lateral`, so no code changes.
- **Docs:**
  - **`docs/guides/index.md`:** a note under the Quick Reference table. It says production
    selection comes from `pipeline_selection.yaml` and differs from the table. The table's
    "≤7 days" row assigns wheat 5–7 to `YoungerMonocotPipeline` (primary + crown); production runs
    all wheat ages on `OlderMonocotPipeline`, crown only. The table's dicot row omits sorghum, a
    monocot that production runs on `DicotPipeline`. Production rice 6–7 already differs from
    the table too.
  - **`docs/dev/trait-extractor-service.md`:** a "window coupling" bullet.
  - **`docs/changelog.md`:** an entry.

## Impact

- Affected spec: `result-envelope-output`, MODIFIED requirement "Pipeline selection by species,
  mode, and age".
- Affected code: `trait_extractor/pipeline_selection.yaml` (data and header comment only).
- Tests: `tests/trait_extractor/test_pipeline_chooser.py`,
  `tests/trait_extractor/test_compatibility.py`, `tests/trait_extractor/test_past_window.py`.
- Docs: `docs/guides/index.md`, `docs/dev/trait-extractor-service.md`, `docs/changelog.md`.
- `trait_extractor/` is excluded from the published wheel, so the `sleap_roots` library is
  unaffected. No contracts bump, so no Bloom re-pin.
- **Runs:**
  - Wheat scans aged 5–14 and sorghum scans aged 3–14, and older ones via the clamp, now select a
    pipeline instead of failing.
  - Younger scans still raise (bloom#994).
  - The new `traits_code_sha` changes every scan's idempotency key, so the first run after deploy
    recomputes every in-scope scan.
- **Deploy order (safety-critical; done outside this repo):**
  1. This PR merges. `docker-trait-extractor.yml` pushes the image (tags `latest`, `main`,
     `sha-<7-char sha>`; digest in the run summary), and `docs.yml` publishes the docs site. No
     service picks the image up: production pins by digest.
  2. sleap-roots-pipeline#119 re-pins and **deploys** that digest.
  3. Only then does sleap-roots-training#72 link the wheat and sorghum cards to `production`.
     Predict reads the registry live, so linking is a deploy, and it reaches staging and prod at
     once. Linked before step 2, every wheat and sorghum scan burns GPU time and then fails
     `No pipeline matches`.
  - **Concurrency:** `docker-trait-extractor.yml` cancels an in-progress build on the same ref.
    Don't merge another PR that touches the image's paths until this merge's build has succeeded
    with a digest. If it was cancelled, rerun it rather than pinning a later sha.
- **Rollback, in reverse:**
  1. Unlink the cards first.
  2. Then re-pin traits to the previous digest. That alone is the operational rollback; no
     revert here is needed. It also rolls back any later traits change.
  - **If a code revert is wanted:** revert the whole squash commit, not just the rows; reverting
    only the rows leaves the new tests red. After the archive (task 5.2), a revert also needs an
    OpenSpec change, because the live spec would then name the cards.
  - Envelopes already written to Bloom stay.
- **Independent of** open PR #140: it touches only `openspec/changes/add-fourier-shape-descriptors/`.
- **Out of scope:**
  - wheat lateral traits (one scan-level pipeline per scan can't do a second pass);
  - younger-than-window scans (bloom#994);
  - the cross-repo window check (#118 follow-up);
  - predict's copy of the shared cases. sleap-roots-predict's `tests/test_model_selection.py`
    uses `sorghum cylinder 30` as a no-card row (lines 351 and 410), checked against a
    test-local `production_cards()` catalog, not the live registry. Those rows don't break, but
    they go stale if that catalog gains sorghum. This goes to #118 as a note for predict#51;
  - `docs/api/index.md:39` and `docs/api/core/pipelines.md:234`. They name wheat as a
    `YoungerMonocotPipeline` example with no age cutoff, so they don't contradict production.
    Only the guides table states a cutoff.
