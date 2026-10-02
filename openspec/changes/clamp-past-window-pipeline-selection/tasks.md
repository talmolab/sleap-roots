## Commit plan

One PR. Rules for every commit:
- Each commit is green locally before committing. CI runs only on the pushed HEAD, so local runs
  are the per-commit evidence.
- Stage by explicit path; never `git add -A` or `git add .`. Save red test output outside the
  worktree, so it can't be staged by accident.
- Write messages to a file and commit with `git commit -F <file>`.
- Task ticks and drift fixes to this change ride in the commit that does the matching work.
  Commit 1 is never amended. Trailing commits for review fixes and ticks are fine.
- Pushing and opening the PR happen only with the author's go-ahead.

The repo squash-merges with the PR title as the subject and every commit body concatenated as the
body. So the deploy note and the `Refs` line go in a commit body (commit 4), not only in the PR
body.

1. `openspec: propose clamp-past-window-pipeline-selection`
   - Stage: `openspec/changes/clamp-past-window-pipeline-selection`.
   - `openspec validate clamp-past-window-pipeline-selection --strict` passes. `ci.yml`: no path
     match (`docs.yml` still builds).
2. `traits: clamp past-window ages in choose_pipeline (bloom#971)` (groups 1–2, chooser parts)
   - Stage: `trait_extractor/pipeline_chooser.py`, `trait_extractor/pipeline_selection.yaml`
     (header comment), `tests/trait_extractor/test_pipeline_chooser.py`, and this change's
     `tasks.md` ticks.
   - The `test_no_match_raises` rewrite must land here, with the clamp; either one alone is red.
   - `test_past_window.py` exists locally but stays uncommitted until commit 3, and is still red
     here (`extract_scan` logs no warning yet). Check with
     `uv run pytest tests/ --ignore tests/trait_extractor/test_past_window.py`. It must also pass
     `black --check`, since black scans untracked files.
   - The body quotes 1.9's red output for the chooser tests. `ci.yml` path match; expected
     green.
3. `traits: log one warning per past-window scan in extract_scan` (groups 1 and 3, extractor parts)
   - Stage: `trait_extractor/extractor.py`, `tests/trait_extractor/test_past_window.py`, and
     `tasks.md` ticks.
   - The body quotes `test_past_window.py`'s two reds: before group 2 (1.9, `No pipeline matches`
     and exit 3) and after it (2.5, no warning logged). `ci.yml` path match; expected green.
4. `docs: past-window pipeline selection` (group 4)
   - Stage: `docs/changelog.md`, `docs/dev/trait-extractor-service.md`, `tasks.md` ticks, and any
     5.4 drift edits.
   - The body carries the `Deploy note:` (the proposal's Deploy and Order bullets, in short) and
     `Refs Salk-Harnessing-Plants-Initiative/bloom#971`. `ci.yml`: no path match (`docs.yml`
     still builds).

## Shared case list

The inputs are aligned with predict's draft change `update-past-window-model-selection` (its
tasks 1.3, 2.1–2.7, 3.1 and 4.4, read 2026-10-01). One predict input is left out: its gap case
(age 7) runs on predict's own synthetic card, which isn't spelled out, and the gap row below
covers the same rule. Outputs differ by repo: a pipeline class here, a model set there. The
matched-as age must agree, and both repos expose it the same way: the "Clamp" column is what
`past_window_age` returns ("no" means `None`). Two raise rows still return a value, because
`choose_pipeline` raises only after clamping: the tie returns 14 and the unknown-class row
returns 10.
- **P** marks inputs predict's draft already tests.
- **T** marks inputs added here, which tasks.md 0.1 suggests predict add.
- Two predict cases don't apply to traits: string ages (traits reads an already-canonical `int`)
  and per-root-type overrides (traits takes one pipeline override).

**Packaged cards** (`pipeline_selection.yaml`):

| Case | Expected here | Clamp | |
|---|---|---|---|
| soybean · cylinder · 10 | `DicotPipeline` | as 8 | P |
| canola · cylinder · 14 | `DicotPipeline` | as 13 | P |
| pennycress · cylinder · 15 | `DicotPipeline` | as 14 | P |
| pennycress · cylinder · 20 | `DicotPipeline` | as 14 | P |
| arabidopsis · cylinder · 28 | `DicotPipeline` | as 14 | P |
| rice · cylinder · 18 | `OlderMonocotPipeline` | as 10 | P |
| arabidopsis · cylinder · 365 | `DicotPipeline` | as 14 (no upper limit) | P (predict's draft doesn't name the species; confirm in 0.1) |
| rice · cylinder · 99 | `OlderMonocotPipeline` | as 10 (the old no-match test) | T |
| rice · cylinder · 11 | `OlderMonocotPipeline` | as 10 (first clamped age) | T |
| soybean · cylinder · 9 | `DicotPipeline` | as 8 (first clamped age) | T |
| arabidopsis · multiplant cylinder · 28 | `MultipleDicotPipeline` | as 14 | T |
| arabidopsis · plate · 20 | `MultipleDicotPlatePipeline` | as 14 | T |
| arabidopsis · cylinder · 10 | `DicotPipeline` | no | P |
| arabidopsis · cylinder · 14 | `DicotPipeline` | no (highest `age_max`) | P |
| rice · cylinder · 4 | `YoungerMonocotPipeline` | no | P |
| rice · cylinder · 8 | `OlderMonocotPipeline` | no | P |
| canola · cylinder · 13 | `DicotPipeline` | no (highest `age_max`) | P |
| soybean · cylinder · 8 | `DicotPipeline` | no (highest `age_max`) | P |
| rice · cylinder · 10 | `OlderMonocotPipeline` | no (highest `age_max`) | T |
| rice · cylinder · 3 | `YoungerMonocotPipeline` | no | T |
| canola · cylinder · 0 | raises `No pipeline matches` | no (younger) | P |
| arabidopsis · cylinder · 1 | raises `No pipeline matches` | no (younger) | P |
| rice · cylinder · 1 | raises `No pipeline matches` | no (younger) | P |
| arabidopsis · plate · 3 | raises `No pipeline matches` | no (below this mode's window) | T |
| sorghum · cylinder · 30 | raises `No pipeline matches` | no (no card) | P |
| canola · multiplant cylinder · 20 | raises `No pipeline matches` | no (no card for this mode) | P |
| soybean · plate · 10 | raises `No pipeline matches` | no (no card for this mode) | T |

**Injected cards.** Each card in a case gets a distinct class, so a clamp to the wrong window
can't pass:

| Cards | Case | Expected | Clamp | |
|---|---|---|---|---|
| canola `cylinder` 2–13 `DicotPipeline`; canola `multiplant cylinder` 2–20 `MultipleDicotPipeline` | canola cylinder 15 | `DicotPipeline` | 13 | P |
| same | canola multiplant cylinder 15 | `MultipleDicotPipeline` | no | T |
| `x`/`cylinder` 2–10 `DicotPipeline`, 2–14 `OlderMonocotPipeline` | age 28 | `OlderMonocotPipeline`, no ambiguity | 14 | P |
| `x`/`cylinder` 2–5 `YoungerMonocotPipeline`, 8–10 `OlderMonocotPipeline` | age 6 | raises `No pipeline matches` (gap) | no | T |
| same | age 11 | `OlderMonocotPipeline` | 10 | T |
| arabidopsis `cylinder` 2–14 `DicotPipeline`, 10–14 `OlderMonocotPipeline` | age 28 | raises ambiguity naming ages 28 and 14 | 14 | P |
| rice `cylinder` 2–5 `YoungerMonocotPipeline` | age 9 | `YoungerMonocotPipeline` | 5 | P |
| canola `cylinder` 5–13 `DicotPipeline`; pennycress `cylinder` 2–14 `OlderMonocotPipeline` | canola age 3 | raises `No pipeline matches` (lowest window is per species) | no | P |
| no cards (`cards=[]`) | rice cylinder age 100 | raises `No pipeline matches` | no | P (predict's draft doesn't name the species) |
| `x`/`cylinder` 2–10 `NopePipeline` | age 12 | raises `Unknown pipeline class` | 10 | T |

In `test_pipeline_chooser.py`, keep the packaged rows as one module-level `SHARED_CASES` list of
`pytest.param(species, mode, age, expected_class_or_None, past_window_age_or_None,
id="<species>-<mode with _ for spaces>-<age>")`, in this table's order, so predict can diff it
row for row. Derive each test's rows by filtering. Keep the injected rows in a separate
`INJECTED_CASES` list.

## 0. Coordination (no code)

- [ ] 0.1 Send the predict change's author a note (via this change's author) with:
      - the shared case list above, including the **T** rows to consider adding (the
        boundaries, rice 99, arabidopsis plate 3, soybean plate 10, the past-window multiplant
        and plate rows), and the two P rows whose species predict leaves unnamed (365 and
        `cards=[]`);
      - that traits mirrors predict's shape (`past_window_age` helper, silent chooser,
        warning logged where the scan key is known). The one difference is that traits' helper
        takes no `overrides`, because traits has no per-root-type overrides;
      - the deploy order: re-pin both templates together, or traits first.

      Before this PR merges, read predict's PR test table (read-only) and confirm the inputs and
      matched-as ages match row for row. Any comment on sleap-roots-contracts#13 is posted only
      with the author's go-ahead.

## 1. Tests first

Write every test in this group before any of group 2.
- `past_window_age` doesn't exist yet. Reference it as `pipeline_chooser.past_window_age` inside
  each test, not through a module-level import, so its tests fail one by one instead of breaking
  collection of the whole module.
- Assert on messages, not bare `ValueError`: `pytest.raises(ValueError, match=...)`, with
  `re.escape` on species and mode.
- Check that the chooser never logs: under
  `caplog.at_level(logging.DEBUG, logger="trait_extractor")`, no records come from
  `trait_extractor.pipeline_chooser`.
- Assert the warning through
  `caplog.at_level(logging.WARNING, logger="trait_extractor.extractor")`, checking `record.name`,
  `record.levelno == logging.WARNING`, and that `record.getMessage()` **equals** the exact format
  in design.md ("Warning format").

- [x] 1.1 `past_window_age` on every packaged and injected row returns the table's "Clamp"
      value, with "no" meaning `None`. The tie (14) and unknown-class (10) rows return a value
      even though `choose_pipeline` raises.
- [x] 1.2 Past-window packaged rows, except multiplant and plate. `choose_pipeline`:
      - returns the expected class;
      - leaves `params.values` equal to a `dict(params.values)` copy taken before the call;
      - leaves `params.param_hash == compute_param_hash(params.values)` (from
        `sleap_roots_contracts`). The stored hash doesn't change on in-place mutation, so it has
        to be recomputed;
      - logs nothing.
- [x] 1.3 In-window packaged rows, including the highest-`age_max` boundaries: expected class.
- [x] 1.4 Raise rows. **Replaces `test_no_match_raises`**: rice 99 now clamps, and it's in 1.2.
      Each raises `match=rf"^No pipeline matches species={re.escape(repr(s))}
      mode={re.escape(repr(m))} age={real}$"`.
- [x] 1.5 Injected rows through `choose_pipeline`:
      - canola per-mode, lower window, gap age 11, rice 2–5 at age 9 and canola multiplant 15:
        the expected class;
      - gap age 6, canola 3 (per-species lowest window) and `cards=[]`: as 1.4;
      - tie: `match=r"^Ambiguous pipeline selection \(2 cards match\) for species='arabidopsis'
        mode='cylinder' age=28 matched as age=14$"`;
      - unknown class: `match="^Unknown pipeline class"`.
- [x] 1.6 Multiplant and plate rows return `MultipleDicotPipeline` / `MultipleDicotPlatePipeline`,
      and each class is in `compatibility.MULTI_PLANT_PIPELINES` (the grain guard itself is
      already covered by `test_compatibility.py`). `override="DicotPipeline"` on rice 18 still
      returns `DicotPipeline`.
- [x] 1.7 Extractor level, new `tests/trait_extractor/test_past_window.py`:
      - `monkeypatch.delenv` `SRT_TRAITS_CODE_SHA` and `SRT_TRAITS_CONTAINER_DIGEST`
        (`raising=False`).
      - A helper writes a copy of
        `tests/data/rice_3do_pipeline_output/scan0K9E8BI/scan0K9E8BI.scan_metadata.json` into its
        own `tmp_path` subdirectory with given `params`, using `write_text(..., encoding="utf-8")`.
      - Run `extract_scan(<original manifest>, <sidecar>, tmp_path / "out<n>")`.
      - In `test_past_window_envelope_keeps_real_age` (age 8 and age 18), assert:
        - `env18.provenance == build_provenance(manifest, sidecar18, sidecar18.to_resolved_params())`;
        - `env18.provenance.params.values["age"] == 18`;
        - `env18.provenance.idempotency_key` differs from `build_provenance` over the same params
          with `age` 10;
        - `env18.traits == env8.traits`, and at least one `TraitValue.value` is not `None`, so the
          equality isn't vacuous;
        - a second age-18 `extract_scan` into `out18` returns `None`, leaves the file bytes
          unchanged, and logs no clamp warning. (Implemented as its own
          `test_past_window_rerun_is_skipped_without_warning`; see design.md "Implementation
          notes".)
      - In separate warning tests (kept apart so 2.5 shows the envelope test already passes):
        - age 18: exactly one WARNING whose message equals
          `past-window age: scan_key=scan0K9E8BI species='rice' mode='cylinder' age=18 matched as
          age=10 -> OlderMonocotPipeline`;
        - age 8: no clamp warning;
        - arabidopsis `multiplant cylinder` age 28: exactly one WARNING (matched as 14,
          `MultipleDicotPipeline`), then `extract_scan` raises "not supported for scan-grain
          emission";
        - sorghum cylinder age 30: raises `No pipeline matches`, with no clamp warning;
        - rice cylinder age 12 with injected
          `cards=[PipelineCard(species="rice", mode="cylinder", age_min=2, age_max=10,
          pipeline_class="NopePipeline")]`: raises `Unknown pipeline class`, with no clamp
          warning. This pins "log only after `choose_pipeline` returns".
- [x] 1.8 Batch CLI, in `test_past_window.py`:
      - Copy `tests/data/rice_3do_pipeline_output/scan0K9E8BI/` to `tmp_path / "in" /
        "scan0K9E8BI"` and set its sidecar's age to 18 (`encoding="utf-8"`).
      - Run `python -m trait_extractor <in> <out>` as a subprocess, the same way
        `test_batch.py`'s CLI tests do. Write a local helper rather than importing their private
        one.
      - Assert return code 0, `ok    scan0K9E8BI` in stdout, and
        `past-window age: scan_key=scan0K9E8BI` in stderr. Put the stderr assertion last.
- [x] 1.9 Run group 1 against the unchanged code and save the output (outside the worktree) for
      the commit bodies. Expected:
      - **red:**
        - 1.1 (every row: `AttributeError`, no `past_window_age`);
        - 1.2;
        - 1.5's canola per-mode, lower window, gap age 11, rice 2–5 at age 9, tie and
          unknown-class cases (today they raise `No pipeline matches`);
        - 1.6's multiplant and plate rows;
        - 1.7's envelope test, age-18 and multiplant warning tests, and the unknown-class test
          (each raises `No pipeline matches` today);
        - 1.8 (exit 3).
      - **green already:**
        - 1.3;
        - 1.4;
        - 1.5's canola multiplant 15, gap age 6, canola 3 and `cards=[]`;
        - 1.6's override;
        - 1.7's age-8 and sorghum warning tests.

      Any other outcome means a test is wrong; fix it before group 2.

## 2. Implement the clamp (`trait_extractor/pipeline_chooser.py`)

- [x] 2.1 Add `past_window_age(params, cards) -> Optional[int]` with a Google-style docstring
      (match the module's existing conventions). Update the `choose_pipeline` docstring and the
      module docstring (lines 1–8) to say that ages above every window for a species + mode
      select its highest window (bloom#971).
- [x] 2.2 Implement design.md's `past_window_age` and `choose_pipeline` algorithm, including the
      ambiguity suffix. `choose_pipeline`'s signature is unchanged. Never write to
      `params.values`, and don't log.
- [x] 2.3 Add one pointer line to the `pipeline_selection.yaml` header comment: ages above a
      species + mode's highest window match that window; see `past_window_age`. No card changes.
- [x] 2.4 1.1–1.6 pass. `test_compatibility.py`, unchanged, passes.
- [x] 2.5 Re-run `test_past_window.py`. The envelope test and the no-warning tests now pass,
      including unknown-class. Only the age-18 and multiplant warning tests and 1.8's stderr
      assertion are red (no warning logged). Save that output for commit 3's body.

## 3. Log the warning (`trait_extractor/extractor.py`)

- [x] 3.1 Add the `logger.warning` in `extract_scan` per design.md's "`extract_scan`" decision.
      It goes after `choose_pipeline` returns and before `check_pipeline_compatible`, and fires
      only when `past_window_age(params, cards)` is not `None`. Pass the same resolved cards to
      both calls, and update the `extract_scan` docstring. 1.7 and 1.8 pass.
- [x] 3.2 `uv run pytest tests/trait_extractor`. `test_envelope.py`'s golden regression and the
      skip-if-done tests pass unchanged.

## 4. Docs

- [ ] 4.1 `docs/changelog.md` `[Unreleased]` → Added, appended after the existing trait_extractor
      entries (oldest-first in that group). Format, matching those entries:
      - a bold lead;
      - the parenthetical (OpenSpec change `clamp-past-window-pipeline-selection`; phase 1 of
        [bloom#971](https://github.com/Salk-Harnessing-Plants-Initiative/bloom/issues/971));
      - 3–4 sentences with a `**Deploy note:**` (predict's change must also be live; re-pin both
        templates together, or traits first).
- [ ] 4.2 `docs/dev/trait-extractor-service.md`:
      - in the `params.age` row, note that ages above the species + mode's highest window match
        that window;
      - under "Notes & follow-ups", one bullet on the `past-window age:` WARNING (stderr, one per
        clamped scan, logged by `extract_scan`, names the scan key) and the bloom#994 follow-up.

## 5. Verification and PR

- [ ] 5.1 `openspec validate clamp-past-window-pipeline-selection --strict` (local only; no
      workflow runs it).
- [ ] 5.2 CI's lint commands (`ci.yml`): `uv run black --check sleap_roots tests trait_extractor`
      and `uv run pydocstyle --convention=google sleap_roots trait_extractor`.
- [ ] 5.3 CI's test command: `uv run pytest tests/`.
- [ ] 5.4 Re-read the proposal, design, spec and tasks against the implementation. Record any
      deviation in design.md under a `### Why N instead of M?` heading.
- [ ] 5.5 Before pushing, check that the branch is up to date: run `git fetch`, then
      `git merge-base --is-ancestor origin/main HEAD`. If it's behind, rebase onto `origin/main`
      and re-run 5.1–5.3; `docs/changelog.md` is the likely conflict. Force-push
      (`--force-with-lease`) only with the author's go-ahead.
- [ ] 5.6 Write the PR body to a file and open the PR with `gh pr create --body-file` (only with
      the author's go-ahead). Title: `traits: clamp past-window ages to the highest pipeline
      window (bloom#971)`. Body sections:
      - Summary;
      - a before/after behavior table;
      - test evidence (full-suite counts; the 1.9 and 2.5 reds, summarized);
      - the OpenSpec change path and its review rounds;
      - the deploy note;
      - trait computation changes (rice past day 10 is measured with `OlderMonocotPipeline`,
        crown only);
      - follow-ups (group 6, bloom#994, contracts#13/#14, the Bloom dialog warning);
      - `Refs Salk-Harnessing-Plants-Initiative/bloom#971` (not `Closes`, and never a bare
        `#971`).

## 6. Post-merge (follow-up PRs in this and other repos)

- [ ] 6.1 `openspec: archive clamp-past-window-pipeline-selection after PR #N merge`:
      `openspec archive … --yes`, then `openspec validate --all --strict`. Do this before any
      bloom#994 change is proposed, since that change would MODIFY the same requirement.
- [ ] 6.2 Find the main-push `docker-trait-extractor.yml` run for the squash commit. A later
      merge can cancel it (`cancel-in-progress`), so check that the recorded digest's
      `org.opencontainers.image.revision` equals the squash SHA. Optional local smoke on that
      digest:
      `docker run --entrypoint python <digest> -c "…past_window_age and choose_pipeline on
      rice/cylinder/18…"`.
- [ ] 6.3 sleap-roots-pipeline, re-pinning traits together with predict's re-pin (or traits
      first, if they must be split):
      - In `sleap-roots-trait-extractor-template.yaml`, update the `image:` line's `:sha-<short>`
        tag and digest, and `SRT_TRAITS_CONTAINER_DIGEST`.
      - Re-check that digest's revision against GHCR with the template's own curl recipe.
      - Rewrite the template's "Bumped" comment block: past-window scans now produce envelopes,
        and every scan recomputes because `traits_code_sha` changes.
      - Add a roadmap entry in `docs/bloom-integration/roadmap.md`.
      - Run `bash scripts/check_all.sh`, which runs `check_manifests.py`. No CI runs it.
- [ ] 6.4 Deploy (the author only): run `scripts/check_cluster_drift.sh` before
      `argo template update`, keeping that output as the rollback snapshot, and again after. The
      namespace serves both Bloom staging and prod.
- [ ] 6.5 After deploy, confirm one live past-window scan produced an envelope and that the
      write-back accepted it.
- [ ] 6.6 Update bloom#971, only with the author's go-ahead.
