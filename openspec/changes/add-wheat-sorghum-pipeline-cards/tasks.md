## Commit plan

One PR, on branch `add-wheat-sorghum-pipeline-cards` from `main` `1d532cc`.

### Rules for every commit
- **Green locally first.** Run `uv run pytest tests/`, `uv run black --check sleap_roots tests
  trait_extractor` and `uv run pydocstyle --convention=google sleap_roots trait_extractor`. CI
  runs only on the pushed HEAD, so local runs are the per-commit evidence.
- **The tree matches the commit.** Before running those checks, `git status` shows nothing
  unstaged or untracked beyond what the commit stages, except `.worktrees/`.
- **Staging:** by explicit path; never `git add -A` or `git add .`.
- **Red output:** save it to the session scratchpad, outside the worktree. Paste 1.6's red
  summary into the commit-2 message draft as soon as it runs (the scratchpad is session-only).
- **Messages:** write to a file and commit with `git commit -F <file>`. End each with the
  `Co-Authored-By` trailer.
- **Ticks and fixes:** task ticks ride in the commit that does the matching work. Commit 1 is
  never amended. Trailing commits for review fixes and ticks are fine; never force-push.
- **Push/PR:** pushing and opening the PR happen only with the author's go-ahead.

### Squash merge
- The repo squash-merges: the PR title (commit 2's subject) becomes the subject, and every commit
  body is concatenated into the body. So the deploy note and the `Refs`/`Closes` lines go in
  commit 3's body.
- The PR body repeats `Closes #276` so the issue links before merge. It also asks that the
  merger keep GitHub's default squash message.

### Commits
1. **`openspec: propose add-wheat-sorghum-pipeline-cards`**
   - Stage `openspec/changes/add-wheat-sorghum-pipeline-cards`.
   - `openspec validate add-wheat-sorghum-pipeline-cards --strict` passes.
2. **`traits: add wheat (OlderMonocot) and sorghum (Dicot) pipeline cards (#276)`** (groups 1–2)
   - Stage `trait_extractor/pipeline_selection.yaml`, the three test files and the `tasks.md`
     ticks.
   - The tests and the rows land together.
   - The body quotes 1.6's red output (test ids plus first error line), the 2.2/2.3 green counts,
     and the date of the 0.1 window check.
3. **`docs: wheat and sorghum production pipeline cards`** (group 3)
   - Stage `docs/guides/index.md`, `docs/dev/trait-extractor-service.md`, `docs/changelog.md`,
     the `tasks.md` ticks and any 4.1 drift edits. Name each drift edit in the body.
   - The body carries a `Deploy note:` (proposal "Deploy order" and "Rollback", in short), plus:
     `Refs talmolab/sleap-roots-pipeline#118`, `Refs talmolab/sleap-roots-pipeline#119`,
     `Refs talmolab/sleap-roots-training#72`, `Closes #276`.

## 0. Before tests

- [ ] 0.1 Re-read sleap-roots-training#72's body and confirm wheat is cylinder 5–14 and sorghum is
      cylinder 3–14. If either differs, stop and ask the author; don't change either side here.

## 1. Tests first (red)

- [ ] 1.1 `test_pipeline_chooser.py`, `SHARED_CASES`:
      - Replace `_shared("sorghum", "cylinder", 30, None, None)` with `_shared("alfalfa",
        "cylinder", 30, None, None)`: a no-card species with no row. Green before and after the
        rows.
      - Update the comment above `SHARED_CASES`: the wheat, sorghum and alfalfa rows come from
        this change (#276), and predict's copy doesn't have them yet.
- [ ] 1.2 `test_pipeline_chooser.py`, `SHARED_CASES`: add rows (spec "Wheat and sorghum select by
      window", "Unmatched scans still raise"):
      - wheat cylinder: 5, 10, 14 → `OlderMonocotPipeline`, `None`; 15, 20 →
        `OlderMonocotPipeline`, 14; 4 → `None`, `None`;
      - wheat plate 10 → `None`, `None` (wrong mode);
      - sorghum cylinder: 3, 8, 14 → `DicotPipeline`, `None`; 15, 17 → `DicotPipeline`, 14; 2 →
        `None`, `None`.

      `_select` routes them into `test_in_window_selection_unchanged`,
      `test_past_window_age_selects_highest_window` and `test_unmatched_scans_still_raise`.
      `test_past_window_age_shared_cases` takes every row. Together these cover the selected
      class, the matched-as age, unchanged `params`, no logging, and `No pipeline matches` with
      the real age.

      **Red (14 ids):**
      - the six in-window rows in `test_in_window_selection_unchanged`, on `No pipeline matches`;
      - the four past-window rows in `test_past_window_age_selects_highest_window`, on
        `No pipeline matches`;
      - the four past-window rows in `test_past_window_age_shared_cases`, on `assert None == 14`.

      **Green before the rows:** wheat 4, sorghum 2 and wheat plate 10 everywhere, and the
      in-window rows in `test_past_window_age_shared_cases`. These rows guard the lower bound and
      the mode only after the rows exist. 1.3 is what catches a wrong `age_min`.
- [ ] 1.3 `test_pipeline_chooser.py`, `test_in_window_selection_unchanged`: snapshot
      `before = dict(params.values)` and assert `params.values == before` after
      `choose_pipeline`, as `test_past_window_age_selects_highest_window` does. This is the
      in-window half of the spec's "`params.values` is unchanged" clause. Green for existing
      rows.
- [ ] 1.4 `test_pipeline_chooser.py`, new `test_wheat_and_sorghum_packaged_cards` (spec "Wheat and
      sorghum packaged cards"):
      - It asserts that the `wheat` and `sorghum` cards from `load_pipeline_cards()` equal exactly
        `[_card("wheat", "cylinder", 5, 14, "OlderMonocotPipeline")]` and
        `[_card("sorghum", "cylinder", 3, 14, "DicotPipeline")]`.
      - Its docstring says the windows must equal sleap-roots-training#72's predict cards.
      - Red: there are no such cards.
- [ ] 1.5 `test_compatibility.py`: new `test_wheat_selection_accepts_crown_only_series` and
      `test_sorghum_selection_accepts_primary_lateral_series` (spec "Wheat and sorghum
      selections pass the scan-grain guard").
      - **Imports:** `loaded_root_types`, plus `choose_pipeline` and `load_pipeline_cards` from
        `trait_extractor.pipeline_chooser`, and `ResolvedParams` from `sleap_roots_contracts`.
      - **Selection:** each test resolves its class with `choose_pipeline` over
        `load_pipeline_cards()` (wheat 10, sorghum 8), asserts it is `OlderMonocotPipeline` or
        `DicotPipeline`, then calls `check_pipeline_compatible`.
      - **Series:**
        - wheat: `Series.load(series_name="0K9E8BI",
          crown_path="tests/data/rice_10do/0K9E8BI.crown.predictions.slp")`;
        - sorghum: the existing `_canola_series()`.
      - **Precondition:** each test first asserts that `loaded_root_types(series)` is exactly
        `{"crown"}` or `{"primary", "lateral"}`. `Series.load` prints and leaves the labels
        `None` for a missing path, so without this the test could pass vacuously.
      - **Stand-in data:** no wheat or sorghum `.slp` exists in `tests/data`. The guard reads only
        which root types loaded.
      - Red: selection raises `No pipeline matches`.
- [ ] 1.6 `test_past_window.py`:
      - **`test_unmatched_scan_raises_without_warning`:** `species="alfalfa"` instead of
        `"sorghum"` (directory `tmp_path / "alfalfa"`). Green before and after the rows. Sorghum
        would now clamp, warn, and fail the root-type guard.
      - **New `test_past_window_crown_only_wheat_scan_emits_envelope`** (spec "Past-window
        crown-only wheat scan emits an envelope with one warning"):
        - **Crown-only scan:** a helper copies `scan0K9E8BI.model123.rootcrown.slp` into a tmp
          scan directory. It writes the rice manifest there with `artifacts` filtered to
          `root_type == "crown"`, then writes wheat sidecars with `_write_sidecar`.
        - **Run:** `extract_scan` with the packaged cards, at age 20 and at age 14, into separate
          output directories.
        - **Asserts:** the age-20 envelope's `provenance.params.values["age"] == 20`, and its
          trait values equal the age-14 envelope's.
        - **Warning:** under `caplog.at_level(logging.WARNING, logger=_EXTRACTOR_LOGGER)`,
          `[w.getMessage() for w in _clamp_warnings(caplog)]` for the age-20 run equals exactly
          `["past-window age: scan_key=scan0K9E8BI species='wheat' mode='cylinder' age=20
          matched as age=14 -> OlderMonocotPipeline"]`.
        - Red: `No pipeline matches`.
- [ ] 1.7 Run `uv run pytest tests/trait_extractor -q` and save the output to the scratchpad.
      - **Expect exactly 18 red ids:** the 14 from 1.2, plus 1.4, the two from 1.5, and the new
        test from 1.6.
      - **Each fails on `No pipeline matches`**, except the four
        `test_past_window_age_shared_cases` past-window ids (`None == 14`) and 1.4 (empty card
        list).
      - **Green:** 1.1, 1.3 and the alfalfa swap in 1.6.

## 2. Implementation (green)

- [ ] 2.1 Append the two rows from #276 verbatim to `trait_extractor/pipeline_selection.yaml`,
      after the rice rows. Extend the header comment: wheat and sorghum come from past hpi_dev
      production runs (#276), not the legacy table, and their windows must equal
      sleap-roots-training#72's predict cards.
- [ ] 2.2 Run `uv run pytest tests/trait_extractor -q`: all green.
- [ ] 2.3 Run the full suite, `black --check` and `pydocstyle`, as in the commit rules: all green.

## 3. Docs

- [ ] 3.0 Before any docs edit, record a baseline: `uv run mkdocs build 2>&1 | grep -c WARNING`
      (`docs.yml` runs a non-strict `mkdocs build` after `pip install -e .[dev]`).
- [ ] 3.1 `docs/guides/index.md`: add a `!!! note "Production trait-extractor"` admonition under
      the Quick Reference table, and leave the table unchanged. The note:
      - says the table is a rule of thumb for library users, and that the production
        trait-extractor chooses by species, mode and age from
        [`trait_extractor/pipeline_selection.yaml`](https://github.com/talmolab/sleap-roots/blob/main/trait_extractor/pipeline_selection.yaml),
        so its choices can differ;
      - gives two examples. Wheat runs on `OlderMonocotPipeline` at every age it supports: the
        wheat model labels seminal roots as `crown`, so wheat gets `crown_*` (and whole-network)
        traits only, with no primary or lateral traits. Sorghum runs on `DicotPipeline`
        (primary + lateral);
      - links to `../dev/trait-extractor-service.md`;
      - states no age windows (the yaml holds them).
- [ ] 3.2 `docs/dev/trait-extractor-service.md`, "Notes & follow-ups": add a "Predict/traits
      window coupling" bullet.
      - It says each species + mode's windows in `pipeline_selection.yaml` must equal its predict
        model cards' windows, which sleap-roots-training manages.
      - Nothing checks this across repos (follow-up in talmolab/sleap-roots-pipeline#118).
      - `test_wheat_and_sorghum_packaged_cards` pins the wheat and sorghum cards.
- [ ] 3.3 `docs/changelog.md` `[Unreleased]` → `### Added`, after the past-window clamp entry.
      One entry of about 100 words, opening "**`trait_extractor` selects pipelines for wheat and
      sorghum**". It covers:
      - the OpenSpec change, #276 and pipeline#118;
      - both cards and their windows;
      - wheat crown-only;
      - the training#72 coupling;
      - a one-sentence **Deploy note** (#119 deploys before training#72 links `production`) and
        the recompute.
- [ ] 3.4 Re-run the 3.0 count: it does not rise, and the yaml link renders.

## 4. Validation and reconciliation

- [ ] 4.1 Re-read `proposal.md`, the spec delta and this file against the diff. Fix any drift in
      the change, with a `### Why N instead of M?` note in `proposal.md`.
- [ ] 4.2 Run `openspec validate add-wheat-sorghum-pipeline-cards --strict` (after 4.1, and again
      after any later edit to the change).
- [ ] 4.3 PR pre-merge checklist:
      - repeat 0.1;
      - check that the PR's `CI` run (lint + 3-OS tests) and `Docker Trait-Extractor Build and
        Push` run are green. Branch protection requires no checks, so check by hand. A
        `benchmark-pr` failure on this data-only change is noise; note it, don't chase it.

## 5. After merge (tracking only; no service is deployed from this repo)

- [ ] 5.1 Find the `docker-trait-extractor.yml` run for the merge commit (event `push`, branch
      `main`).
      - **Cancelled:** rerun it (`gh run rerun <id>`) rather than pinning a later sha.
      - **Succeeded:** record on #276 and talmolab/sleap-roots-pipeline#119 the merge commit, the
        tag `sha-<7-char short sha>` and the `sha256:` digest, both copied verbatim from the run
        summary. The pin is `ghcr.io/talmolab/sleap-roots-trait-extractor@sha256:…`. The
        envelopes' `traits_code_sha` will be the full sha.
      - Posting needs the author's go-ahead.
- [ ] 5.2 With the author's go-ahead, comment on talmolab/sleap-roots-pipeline#118: predict's
      `tests/test_model_selection.py` uses `sorghum cylinder 30` as a no-card row, which goes
      stale if its test catalog gains sorghum.
- [ ] 5.3 Archive this change (`openspec archive add-wheat-sorghum-pipeline-cards`) in a
      follow-up PR.
