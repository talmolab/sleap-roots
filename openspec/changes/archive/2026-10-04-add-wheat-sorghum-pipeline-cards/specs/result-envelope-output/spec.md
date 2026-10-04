## MODIFIED Requirements

### Requirement: Pipeline selection by species, mode, and age

The `trait_extractor` package SHALL select exactly one `sleap_roots` `Pipeline` subclass via
`choose_pipeline(params, cards, override=None)`, modeled on predict's `choose_models` (which is
NOT importable here — the matcher and age-coercion are authored in-tree). `cards` is an
injectable list of a public, test-constructible `PipelineCard` type (`{species, mode, age_min,
age_max, pipeline_class}`); production cards load from a packaged `pipeline_selection.yaml`. A
card matches when `species` and `mode` are equal and `age_min <= age <= age_max` (contiguous
inclusive window), reading `age` from the already-canonicalized `params.values` (an `int`).
`choose_pipeline` MUST NOT mutate `params.values`. An explicit `override` pipeline-class name
wins and bypasses matching.

**Past-window clamp (bloom#971 phase 1).** A public `past_window_age(params, cards)` SHALL
return the highest `age_max` among the cards with the scan's `species` and `mode` when at least
one such card exists and `age` is greater than every such `age_max`; otherwise it SHALL return
`None`. When it returns a value, `choose_pipeline` SHALL match as if `age` were that value, with
no upper limit on how far past. The maximum SHALL be taken over that species + mode only. Only the
age used for matching is clamped: `params`, and therefore provenance and the idempotency key,
SHALL keep the real age, and no field SHALL be added to the envelope or provenance.
`past_window_age` and `choose_pipeline` SHALL NOT log.

**Clamp warning.** For each scan whose selection used a clamped age, `extract_scan` SHALL log
exactly one WARNING on the `trait_extractor.extractor` logger. It logs after `choose_pipeline`
returns and before the scan-grain compatibility check. Its message SHALL start with
`past-window age:` and name the manifest's `scan_key`, species, mode, real age, matched-as age,
and selected class name. A scan whose `choose_pipeline` call raises SHALL NOT log a clamp
warning, and neither SHALL an in-window scan or one skipped as already done. A clamped selection
that the scan-grain guard then rejects has already logged its warning.

**Errors.** Any selection that still matches zero cards SHALL raise a `ValueError` starting
`No pipeline matches` and reporting the real age. That includes no card for the scan's species +
mode, an age below that species + mode's lowest `age_min` (bloom#994), and an age in a gap
between that species + mode's windows. More than one match SHALL raise a `ValueError` starting
`Ambiguous pipeline selection` that reports the real age and, after a clamp, the matched-as age.
That includes two cards tied at the highest `age_max`. An unknown `pipeline_class` name SHALL
raise a `ValueError`.

#### Scenario: Rice age selects younger vs older monocot by window

- **WHEN** `choose_pipeline` is given canonical `params.values = {species: "rice", mode:
  "cylinder", age: 3}` and again `age: 8` against the packaged cards
- **THEN** it returns `YoungerMonocotPipeline` for age 3 and `OlderMonocotPipeline` for age 8, and
  `params.values` is unchanged by the call

#### Scenario: Arabidopsis plate resolves the plate pipeline (legacy gap fixed)

- **WHEN** `choose_pipeline` is given `{species: "arabidopsis", mode: "plate", age: 10}`
- **THEN** it returns `MultipleDicotPlatePipeline` (the class the legacy chooser table named but
  could not resolve) — selection resolves the class; scan-grain support is guarded separately

#### Scenario: Wheat and sorghum packaged cards

- **WHEN** the packaged cards are loaded with `load_pipeline_cards()`
- **THEN** the only `wheat` card is `{species: "wheat", mode: "cylinder", age_min: 5, age_max: 14,
  pipeline_class: "OlderMonocotPipeline"}`
- **AND** the only `sorghum` card is `{species: "sorghum", mode: "cylinder", age_min: 3,
  age_max: 14, pipeline_class: "DicotPipeline"}`

#### Scenario: Wheat and sorghum select by window

- **WHEN** `choose_pipeline` and `past_window_age` are given, against the packaged cards, wheat
  cylinder ages 5, 10, 14, 15 and 20, and sorghum cylinder ages 3, 8, 14, 15 and 17
- **THEN** `choose_pipeline` returns `OlderMonocotPipeline` for every wheat age and
  `DicotPipeline` for every sorghum age
- **AND** `past_window_age` returns `None` for wheat 5, 10 and 14 and sorghum 3, 8 and 14, and
  14 for wheat 15 and 20 and sorghum 15 and 17
- **AND** `params.values` is unchanged by each call, and neither function logs

#### Scenario: Wheat and sorghum selections pass the scan-grain guard

- **WHEN** the class `choose_pipeline` returns for wheat cylinder age 10 is checked by
  `check_pipeline_compatible` against a `Series` that loaded `crown` labels only, and the class it
  returns for sorghum cylinder age 8 against a `Series` that loaded `primary` and `lateral` labels
- **THEN** the selected classes are `OlderMonocotPipeline` and `DicotPipeline` respectively, and
  `check_pipeline_compatible` returns without raising for both

#### Scenario: Past-window crown-only wheat scan emits an envelope with one warning

- **WHEN** `extract_scan` runs on a scan whose manifest lists only a `crown` artifact and whose
  sidecar says wheat cylinder age 20
- **THEN** it writes an envelope whose `provenance.params.values["age"]` is 20, and whose trait
  values equal those `extract_scan` produces for the same crown-only scan with a wheat age-14
  sidecar
- **AND** exactly one WARNING on `trait_extractor.extractor` is logged, whose message starts
  `past-window age:` and names the scan key, species `wheat`, mode `cylinder`, ages 20 and 14, and
  `OlderMonocotPipeline`

#### Scenario: Past-window age matches the species' highest window

- **WHEN** `choose_pipeline` and `past_window_age` are given, against the packaged cards,
  cylinder scans of soybean age 10, canola age 14, pennycress age 15, arabidopsis age 28, rice
  age 18, and arabidopsis age 365
- **THEN** `choose_pipeline` returns `DicotPipeline` for the soybean, canola, pennycress and
  arabidopsis scans, and `OlderMonocotPipeline` (never `YoungerMonocotPipeline`) for rice
- **AND** `past_window_age` returns 8, 13, 14, 14, 10 and 14 respectively
- **AND** `params.values` is unchanged by each call, and neither function logs

#### Scenario: Window boundaries

- **WHEN** `choose_pipeline` and `past_window_age` are given rice cylinder age 10 and soybean
  cylinder age 8 (each the highest `age_max`), and rice cylinder age 11 and soybean cylinder age 9
- **THEN** ages 10 and 8 return `OlderMonocotPipeline` and `DicotPipeline` with
  `past_window_age` `None`, and ages 11 and 9 return the same classes with `past_window_age` 10
  and 8

#### Scenario: Highest window is taken per species and mode

- **WHEN** injected cards give canola `cylinder` the window 2–13 (`DicotPipeline`) and canola
  `multiplant cylinder` the window 2–20 (`MultipleDicotPipeline`), and both functions are given
  canola `cylinder` age 15 and canola `multiplant cylinder` age 15
- **THEN** the cylinder scan returns `DicotPipeline` with `past_window_age` 13, and the
  multiplant scan returns `MultipleDicotPipeline` in-window with `past_window_age` `None`

#### Scenario: Clamping never selects through a lower window

- **WHEN** injected cards give species `x` mode `cylinder` the windows 2–10 (`DicotPipeline`) and
  2–14 (`OlderMonocotPipeline`), and `choose_pipeline` is given age 28
- **THEN** it returns `OlderMonocotPipeline` (matched as 14), with no ambiguity

#### Scenario: A gap below the highest window doesn't block the clamp

- **WHEN** injected cards give species `x` mode `cylinder` the windows 2–5
  (`YoungerMonocotPipeline`) and 8–10 (`OlderMonocotPipeline`), and `choose_pipeline` is given
  age 11
- **THEN** it returns `OlderMonocotPipeline` (matched as 10)

#### Scenario: Past-window multiplant and plate scans clamp but are still rejected at scan grain

- **WHEN** `choose_pipeline` is given arabidopsis `multiplant cylinder` age 28 and arabidopsis
  `plate` age 20 against the packaged cards
- **THEN** it returns `MultipleDicotPipeline` and `MultipleDicotPlatePipeline` respectively
  (matched as 14), both of which the scan-grain guard rejects, the same as in-window scans of
  those modes
- **AND** `extract_scan` on such a scan logs exactly one clamp WARNING and then raises the
  "not supported for scan-grain emission" `ValueError`

#### Scenario: Past-window scan emits an envelope carrying its real age

- **WHEN** `extract_scan` runs on a rice cylinder scan whose sidecar says age 18
- **THEN** it writes an envelope whose trait values equal those `extract_scan` produces for the
  same scan with an age-8 sidecar, and whose provenance equals `build_provenance` over the age-18
  params (so `provenance.params.values["age"]` is 18 and the idempotency key is not the one from
  age-10 params)
- **AND** exactly one WARNING on `trait_extractor.extractor` is logged, whose message starts
  `past-window age:` and names the scan key, species, mode, ages 18 and 10, and
  `OlderMonocotPipeline`
- **AND** the same scan with an age-8 sidecar logs no clamp warning
- **AND** a second `extract_scan` of the age-18 scan into the same output directory is skipped
  (returns `None`, file unchanged) without logging a clamp warning

#### Scenario: Past-window scan succeeds through the batch CLI

- **WHEN** `python -m trait_extractor <in> <out>` runs on an input tree whose only scan is a rice
  cylinder scan with a sidecar age of 18
- **THEN** it exits 0, reports the scan as `ok`, and prints the `past-window age:` warning naming
  the scan key to stderr

#### Scenario: Explicit override wins

- **WHEN** an `override` of `"DicotPipeline"` is supplied
- **THEN** `choose_pipeline` returns `DicotPipeline` regardless of species/mode/age

#### Scenario: Unmatched scans still raise

- **WHEN** `choose_pipeline` is given any of these:
  - canola cylinder age 0, rice cylinder age 1, arabidopsis cylinder age 1, wheat cylinder age 4 or
    sorghum cylinder age 2 (below every window);
  - arabidopsis plate age 3 (below that mode's window, though other arabidopsis modes start at 2);
  - alfalfa cylinder age 30 (no card);
  - soybean plate age 10, canola multiplant cylinder age 20 or wheat plate age 10 (species has
    cards, but not for this mode);
  - an empty `cards` list at age 100;
  - age 6 against injected `x`/`cylinder` cards with windows 2–5 and 8–10 (a gap)
- **THEN** it raises a `ValueError` starting `No pipeline matches` that reports the real age, and
  `past_window_age` returns `None` for each
- **AND** `extract_scan` on an unmatched scan raises that error without logging a clamp warning

#### Scenario: Ambiguous or invalid selection raises

- **WHEN** more than one card matches (including two cards tied at the highest `age_max` for a
  past-window age), or a matched/override `pipeline_class` name is not a known pipeline (including
  a highest-window card reached by clamping)
- **THEN** `choose_pipeline` raises `ValueError`; a tie after a clamp reports both the real age
  and the matched-as age
