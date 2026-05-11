# Preliminary kinematic characterization of root circumnutation in rice CMTN data

*Working draft, 2026-05-07. Author: Elizabeth Berrigan. Tags: circumnutation, sleap-roots, kinematics, preliminary. Feasibility analysis for the sleap-roots circumnutation pipeline.*


## Summary

We characterize the kinematics of root circumnutation in tracked SLEAP predictions from a 6-plant rice (*Oryza sativa* cv. Nipponbare) plate, imaged at 5-min cadence over 47.9 hr. The dominant nutation period is **~3333 s (55.5 min)**, the lateral peak-to-peak amplitude after centerline detrending is **~10 px (~0.20 mm at 1200 DPI scan resolution)**, and the longitudinal tip-growth speed is **~4.3 px/frame ≈ 1.0 mm/hr**, consistent with rice primary-root growth rates in the literature. Empirical SLEAP localization noise on proofread predictions is **~2 px (~40 µm)**, giving a per-cycle SNR of ~5× and a coherent spectral SNR of ~35–40× across the ~57 cycles captured per track. Both temporal and spatial wavelet analyses are feasible from these data: ~11 samples per nutation cycle (well above Nyquist) and ~52 spatial wavelengths along the cumulative tip trail. A pixel-to-millimeter scale ambiguity (TIFF EXIF claims 400 DPI but the implied 4.4 mm/hr growth rate is too fast for rice) remains to be resolved with the experimenter; the 1200 DPI working assumption is biology-corroborated but not source-of-truth.

## 1. Background

### 1.1 Biological framing

Plant circumnutation is a self-sustaining oscillatory growth movement, observable as a helical or quasi-circular trajectory of the organ apex. In roots, the period and amplitude depend on species, hormonal state, and mechanical loading. Two complementary theoretical frames currently inform pipeline design:

- **Auxin morphogen / Cholodny–Went oscillator.** Asymmetric lateral redistribution of auxin (IAA) across the root diameter, controlled by PIN efflux carriers, breaks the radial symmetry of the organ at the organ scale and drives differential growth. Periodic redistribution produces nutation. Reviewed in [Meroz, 2026](https://arxiv.org/abs/2604.21763), §"Signal processing".
- **Mechanical asymmetric-growth wave.** [Rivière et al., 2025](https://doi.org/10.1017/qpb.2025.10013) (preprint: [bioRxiv 2022.02.22.481493](https://www.biorxiv.org/content/10.1101/2022.02.22.481493v1)) show in carambola leaves that nutation proceeds via a steady traveling wave of asymmetric growth, with cell-wall-stiffness varying along the propagation direction. The phenomenology is consistent with the Cholodny–Went oscillator in dynamics (∂κ/∂t = -β sin(θ-θ_p) - γκ; [Bastien & Meroz, 2016](https://arxiv.org/abs/1603.00459)) but identifies the mechanical field as the substrate of memory rather than auxin alone.

### 1.2 Methodological lineages

- **Time-domain pipeline** (rice circumnutation): [Taylor et al., 2021](https://doi.org/10.1073/pnas.2018940118), supplementary R code at [Duke Research Data Repository, 10.7924/r4b27x11m](https://doi.org/10.7924/r4b27x11m). LOESS centerline → drift correction → natural-spline smoothing → peak-finding for amplitude. Local copies in `external_code/taylor_2021_pnas_loess_spline/` (provenance documented in `external_code/README.md`).
- **Time-frequency pipeline** (Derr unpublished): Continuous Wavelet Transform (CWT) via `pywavelets`, applied to 5-min-cadence rice tip trajectories from Suyash Patil's pilot data. The reference outputs (sample data, time-averaged amplitude spectrum, CWT scaleogram + Fourier transform) are at `external_code/derr_wavelets/sept_2025_outputs/`.

## 2. Data

### 2.1 Source

Experiment ID: `20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT`

| Property | Value |
|---|---|
| Genotype | *Oryza sativa* cv. Nipponbare |
| Media | ½ MS, 0.6–0.8% phytagel |
| Cadence | 5 min |
| Frames per plate | 575 |
| Run duration | 47.9 hr |
| Plants per plate | 6 |
| Plates analyzed | 1 (MOCK control) of 4 in experiment |
| Other plate treatments (not yet analyzed) | 2.5 µM GA₄ (plate 002), MOCK rep (003), 0.2 µM TZT (004) |

NetApp working copy: `\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT\`

Original imaging source (groot-data): `\\pbiob-centos.salk.edu\groot-data\Suyash\AA_Suyash_MDATA\Circumnutation project\Images every 5 minutes\Nipponbare_0.8PG_GA4vsTZT\`

Coordination metadata: [`Circumnutation_Experimets_Directory.xlsx`](https://salkinstitute.box.com/file/1987750239743) (Box, shared by Suyash 2025-09-16).

### 2.2 SLEAP predictions

- **Inference:** `sleap-track` (SLEAP 1.4.1a2), top-down models `250820_082118.centroid.n=377` + `250820_104319.centered_instance.n=377`, max 6 instances per frame, simplemaxtracks tracker, `track_window=5`.
- **Tracked + proofread output:** `runs/run_20250917_201037/plate_001_greyscale.tracked_proofread.slp` (the file analyzed below).
- 575 frames × 6 tracks = 3450 expected instances; 3906 actual (some frames have proofread *and* original predictions co-stored). 456 user-corrected points across 76 frames per track (13% manual override rate).

## 3. Methods

### 3.1 Data extraction

Direct HDF5 reads of `*tracked_proofread.slp` via `h5py` (analysis scripts at `external_code/analysis_2026-05-07/` once finalized; current working copies at `C:\Users\Elizabeth\AppData\Local\Temp\analyze_*.py`). For each frame and track, the tip coordinate $(x, y)$ is the first node of the (single-node) skeleton. Where an `instance_type=0` (user-corrected) and `instance_type=1` (predicted) instance exist for the same frame and track, the user-corrected value takes precedence.

### 3.2 Kinematic statistics

For each track, with the trace $\{(x_i, y_i)\}_{i=1}^{N}$ at frame indices $i$:

- **Frame-to-frame displacement:** $\Delta_i = (x_{i+1}-x_i,\ y_{i+1}-y_i)$
- **Step magnitude:** $s_i = \|\Delta_i\| = \sqrt{\Delta x_i^2 + \Delta y_i^2}$
- **Path length:** $L = \sum_i s_i$
- **Net displacement:** $D = \|\mathbf{x}_N - \mathbf{x}_1\|$
- **Growth axis:** $\hat{\mathbf{u}}_g = (\mathbf{x}_N - \mathbf{x}_1) / D$ (unit vector from start to end)
- **Lateral axis:** $\hat{\mathbf{u}}_\ell = R_{90°}\hat{\mathbf{u}}_g$ (perpendicular)
- **Longitudinal step:** $\Delta_i^g = \Delta_i \cdot \hat{\mathbf{u}}_g$
- **Lateral step:** $\Delta_i^\ell = \Delta_i \cdot \hat{\mathbf{u}}_\ell$

The growth axis is defined by net displacement rather than by PCA of $\{\Delta_i\}$: PCA-on-diffs picks the direction of maximum *variance*, which for a steady-growth + isotropic-noise signal is the noise/oscillation axis, not the growth direction. The net-displacement definition is robust as long as the cumulative drift is much larger than per-frame noise, which holds here ($D \approx 2400$ px ≫ noise $\sim 2$ px).

### 3.3 SLEAP localization noise

Two independent estimators for the localization noise standard deviation $\sigma$:

**(a) Savitzky–Golay residual.** Fit a degree-3 polynomial in a 5-frame sliding window (window length much shorter than the nutation period $T_n \approx 11.1$ frames, but long enough to suppress white noise) to $x(t)$ and $y(t)$ separately:
$$\sigma_\text{SG}^2 = \mathrm{Var}(x_\text{raw} - x_\text{smooth}) + \mathrm{Var}(y_\text{raw} - y_\text{smooth})$$

**(b) Second-difference noise.** For i.i.d. noise $\varepsilon \sim (0, \sigma^2)$ added to a smooth signal $s(t)$:
$$\Delta^2 x(t) = x(t+1) - 2x(t) + x(t-1)$$
$$\mathrm{Var}(\Delta^2 \varepsilon) = 6\sigma^2 \quad \Rightarrow \quad \sigma \approx \frac{\mathrm{std}(\Delta^2 x)}{\sqrt{6}}$$

The two estimators make different smoothness assumptions; agreement is the cross-check.

### 3.4 Nutation amplitude (corrected method)

The lateral bounding-box span $\max(x) - \min(x)$ along the lateral axis is **not** a reliable amplitude proxy: it conflates per-cycle oscillation with hour-scale drift of the centerline. Correct estimate:

1. Detrend the lateral coordinate $\ell(t)$ with a long-window Savitzky–Golay filter (window = 23 frames ≈ 2 nutation periods, polynomial order 3). This suppresses the oscillation and retains slow centerline drift.
2. Compute residuals $\ell_\text{res}(t) = \ell(t) - \ell_\text{drift}(t)$.
3. Report:
   - $2\sigma_{\ell_\text{res}}$ — Gaussian peak-to-peak estimate
   - Median peak-to-peak between successive maxima/minima from `scipy.signal.find_peaks` (minimum spacing = $T_n / 2 \approx 5.5$ frames)

### 3.5 Period (independent confirmation)

For external validation, peak-finding in the detrended lateral residual. Number of detected positive peaks per track, divided by the run duration, gives an empirical period estimate to compare against Derr's spectral peak.

## 4. Results

### 4.1 Tip kinematics (median across 6 tracks)

| Quantity | Pixels | mm @ 1200 DPI |
|---|---|---|
| Mean per-frame step (Euclidean) $\langle s \rangle$ | **5.83** | 0.123 |
| Median per-frame step | 6.93 | 0.147 |
| Std of per-frame step | 3.39 | 0.072 |
| Mean longitudinal step $\langle \Delta^g \rangle$ | **4.29** | 0.091 |
| Mean lateral step $\langle |\Delta^\ell| \rangle$ | 2.75 | 0.058 |
| Longitudinal:lateral ratio | 1.56 | — |
| Path length $L$ | 3328 | 70.5 |
| Net displacement $D$ | 2445 | 51.8 |
| Path/net inflation $L/D$ | 1.36 | — |
| Mean tip growth speed (longitudinal) | 51.5 px/hr | **~1.03 mm/hr** ✓ |
| Outlier rate ($s > 2 \cdot \mathrm{median}$) | 0.87% | — |

The longitudinal growth speed of ~1.0 mm/hr is in the literature range for *Oryza sativa* primary roots ([Taylor et al., 2021](https://doi.org/10.1073/pnas.2018940118) and references therein), supporting the working pixel-scale assumption (§4.6).

### 4.2 SLEAP localization noise

| Estimator | Median across 6 tracks (px) |
|---|---|
| Savitzky-Golay residual (xy) | 1.83 |
| Second-difference estimate (xy) | 2.67 |
| Mean prediction confidence | 0.998 |

Both estimators converge on a **localization noise floor of ~2 px (~40 µm at 1200 DPI)**. The SG estimate is slightly lower because it absorbs some genuine high-frequency motion into the smoother; the second-difference estimate is slightly higher because it is sensitive to all frame-to-frame change, including real curvature. The agreement at $\sim$1.5–3 px is the cross-check that this is dominated by localization noise rather than method-specific signal contamination.

### 4.3 Nutation amplitude (corrected)

Initial measurement using the bbox lateral span (median 209 px) overestimated the per-cycle nutation amplitude by ~20×, because the span captures total drift over the 48-hr run rather than oscillation around the drifting centerline. After centerline detrending:

| Quantity | Median across 6 tracks (px) | mm @ 1200 DPI |
|---|---|---|
| $2\sigma_{\ell_\text{res}}$ (Gaussian peak-to-peak) | 12.7 | 0.27 |
| Median peak-to-peak | **9.9** | **0.21** |
| Half-amplitude (zero-to-peak) | 4.9 | 0.10 |
| ~~Bbox lateral span (deprecated)~~ | ~~209~~ | ~~4.4~~ |

The two principled estimators (2σ Gaussian, direct peak-to-peak) agree at ~10–13 px.

### 4.4 Period

Peak-finder applied to the detrended lateral residual: median 57 oscillation cycles per track over 575 frames. This corresponds to one cycle per 575/57 ≈ 10.1 frames = **~50.4 minutes**, consistent with Derr's spectral-peak measurement of **~3333 s = 55.5 min** (~10% discrepancy attributable to peak-finder vs. spectral-peak methodology). Reference: `external_code/derr_wavelets/sept_2025_outputs/5minutes_average_period=3333s.pdf`.

### 4.5 Signal-to-noise ratios

| Quantity | Value |
|---|---|
| Per-cycle peak-to-peak amplitude / SLEAP noise | $\sim 10/2 = 5$ |
| Per-cycle zero-to-peak amplitude / SLEAP noise | $\sim 5/2 = 2.5$ (marginal) |
| Cycles per 47.9-hr run | ~57 |
| Coherent-sum SNR boost factor across cycles | $\sqrt{57} \approx 7.5$ |
| Effective spectral-peak SNR | **$\sim 35–40$** (clean) |
| Per-frame step / SLEAP noise | $\sim 5.8/2 \approx 3$ (single-frame velocity is noisy) |

Implication: the wavelet *scaleogram* will reproduce Derr's reference outputs cleanly. Per-cycle amplitude estimates from raw data will be noisy and benefit from smoothing, longer runs, or ensembling across plants. Single-frame velocity estimates are unreliable; smooth across ~3–5 frames before differentiating.

### 4.6 Pixel-to-millimeter scale (unresolved)

TIFF EXIF metadata of `_set1_day1_20250913-165722_001.tif` (raw scan):
- Image size: 6608 × 6614 px
- Resolution tag: 400 DPI → mpp = 25.4/400 = **0.0635 mm/px**
- Implied physical size: 419.6 × 420.0 mm

The implied 400 DPI scan would yield a longitudinal growth rate of ~4.4 mm/hr, which is too fast for rice primary roots (literature range 1–2 mm/hr at standard conditions). The scanner firmware is likely writing a nominal DPI to the tag rather than the actual scan resolution.

A scan resolution of **1200 DPI** (mpp = 0.0212 mm/px) gives:
- Longitudinal growth rate: 1.03 mm/hr ✓ in literature range
- Physical plate dimension: ~140 × 140 mm (consistent with a square Petri plate)

The 1200 DPI value is the working assumption for all mm conversions in this document, but is not source-of-truth. Confirmation from Suyash's coordination spreadsheet or direct measurement (known plate dimension / image-pixel dimension) is pending. **All mm numbers in this document should be treated as scale-tentative.**

## 5. Implications for pipeline design

### 5.1 Cadence is sufficient

**Temporal Nyquist:** 11.1 samples per nutation period at 5-min cadence — well above the Nyquist limit of 2 samples/period and inside the conservative ~5–10 samples/period buffer typical for spectral / wavelet work.

**Spatial Nyquist (for the spatial wavelet branch):** per-frame step / spatial wavelength = 5.8 / 64 = **9% per sample**, equivalently 11 samples per spatial wavelength along the cumulative trail.

5-min cadence is comfortable. 10-min cadence (5.5 samples/cycle) would be borderline. 30-min cadence (1.8 samples/cycle, sub-Nyquist) would alias.

### 5.2 Spatial wavelet branch is feasible

| Window | Trail length (px) | n_wavelengths | Wavelength (px) |
|---|---|---|---|
| 14-hr (Derr's reference cut) | ~975 | ~15 | ~64 |
| 47.9-hr (full run) | ~3328 | ~52 | ~64 |

Even the truncated Derr window contains ~15 spatial wavelengths along the trail, well above the rule-of-thumb minimum of 5 cycles for stable CWT decomposition. Treating the centerline-detrended lateral residual as a spatial signal indexed by arc length is justified by the trajectory aspect ratio (longitudinal:lateral bbox ~12:1, so arc length ≈ longitudinal coordinate to first order).

### 5.3 When detrending matters (and when it doesn't)

The pre-detrending lateral bbox span (~209 px) is a ~20× overestimate of the true per-cycle nutation amplitude (~10 px peak-to-peak). The bbox is dominated by hour-scale drift of the centerline, not by oscillation around it. This conflation is the single largest methodological pitfall identified in this analysis.

Detrending is **required** for:

- **Amplitude quantification**, in either the time domain (peak-to-peak, RMS, zero-crossings) or the wavelet domain ($|W(s,t)|$ at the nutation scale). Drift contamination biases all of these. The 20× factor above is the magnitude of the bias when the bbox span is used naively.
- **Time-domain kinematic traits** (handedness, phase coherence, instantaneous period from zero crossings) — all defined relative to a centerline and meaningless without one.
- **The spatial wavelet branch**, intrinsically. The signal there *is* the lateral deviation from a centerline, so detrending is upstream of the wavelet rather than a preprocessing option.

Detrending is **helpful but not strictly required** for:

- **Time-domain Morlet CWT used purely to detect the dominant period.** Morlet at the nutation scale is band-pass with low DC response; the drift's spectral content at $\sim 3 \times 10^{-4}$ Hz is negligible. Empirically, [Derr's reference scaleogram](external_code/derr_wavelets/sept_2025_outputs/_5minutes_wavelets.png) shows a clean nutation band even though his input signal ([sample data](external_code/derr_wavelets/sept_2025_outputs/5minutes_sample_data.pdf)) retains the drift. So *period detection* via Morlet CWT is robust without detrending. Caveats: wavelets with larger DC sensitivity (Mexican hat, Haar) are more fragile; the cone-of-influence edge contamination is somewhat larger with drift present; auto-scaled scaleogram visualizations can be visually dominated by low-frequency drift content (Derr appears to clip the colorscale).

In summary: **for any quantitative trait, detrend first; for Morlet CWT scaleogram inspection of the period, you can get away without it but it's still good practice.**

### 5.4 SLEAP precision is not the bottleneck — but is also not infinite

Per-cycle SNR (~5×) is comfortable but not luxurious. The bottleneck for per-cycle traits is biology (real plant-to-plant variability), not localization noise. For spectral peaks summed across the full run (effective SNR ~40×), the scaleogram will be clean.

The 13% manual-correction rate on the proofread file is worth noting: raw model-only predictions would have a higher noise floor. If un-proofread runs become the standard, the per-cycle SNR could drop below the marginal threshold for amplitude work. Re-measure noise on raw `.tracked.slp` files before generalizing this margin to other species or imaging conditions.

### 5.5 Per-cycle vs. spectral-peak metrics

The pipeline should report two distinct classes of trait, with different detrending and noise sensitivities:

- **Spectral / integrated traits** (dominant period, mean amplitude, phase coherence over the run, drift in dominant period). High effective SNR (~35–40× after coherent summation across ~57 cycles). Period detection via Morlet CWT is robust to drift; mean amplitude requires detrending.
- **Per-cycle traits** (instantaneous period(t), per-cycle amplitude(t), handedness within cycle). Noisier — single-cycle measurements have SNR ~5×. All require detrending. Consumers should expect to smooth across 2–3 cycles or ensemble across plants for clean per-cycle reads.

## 6. Limitations

- **Single plate analyzed.** All numbers above are from plate 001 (MOCK control), 6 plants. Plates 002 (GA₄), 003 (MOCK replicate), 004 (TZT) are tracked + proofread but not yet processed. Plant-to-plant variability *within* this plate is captured; between-treatment effects are not yet quantified.
- **Single species, single growth condition.** Per-species period and amplitude must not be hardcoded in the pipeline. Soybean / canola / pennycress / Arabidopsis will have different values.
- **Pixel-to-mm scale unresolved.** All mm numbers depend on the working assumption mpp = 0.0212 mm/px (1200 DPI). Verify against Suyash's coordination spreadsheet or direct measurement before publication.
- **Proofread-only.** Noise floor is post-manual-correction; raw model-only predictions will have higher noise. Re-measure when/if proofreading is removed from the standard pipeline.
- **Drift detrending parameter choice.** The 23-frame detrend window (≈2 periods) is principled but somewhat arbitrary; sensitivity analysis should be part of the design spec.
- **No comparison to existing R pipeline (Taylor et al. 2021 / Elizabeth's adaptation).** A direct comparison run on the same data would validate the Python re-implementation.

## 7. Next steps

1. Resolve pixel-to-mm scale with Suyash (coordination spreadsheet or direct measurement).
2. Re-run the analysis on plates 002, 003, 004 to characterize MOCK plate-to-plate variability and treatment effects (GA₄, TZT).
3. Implement the time-domain pipeline (LOESS centerline → drift correction → spline smoothing → peak-finding) in Python and validate against the R reference output on the same data.
4. Implement the time-frequency pipeline (Morlet CWT via `pywavelets`) and compare scaleogram against `external_code/derr_wavelets/sept_2025_outputs/_5minutes_wavelets.png` as a sanity-check oracle.
5. Test on a synthetic signal with known period, amplitude, and noise to verify pipeline recovers ground truth.
6. Generalize to the un-tracked-yet 5-min experiments (Kitx vs. Hk1-3, Nipponbare PG concentration) once their tracking is finalized.

## 8. References

### Verified, used in this analysis

- **Meroz, Y. (2026).** Physics of computation and behavior in plants. *arXiv*, 2604.21763. [URL](https://arxiv.org/abs/2604.21763) · DOI: [10.48550/arXiv.2604.21763](https://doi.org/10.48550/arXiv.2604.21763). Local literature note: [`_sources/meroz2026-plant-computation.md`](../_sources/meroz2026-plant-computation.md).
- **Taylor, I., Lehner, K., McCaskey, E., Nirmal, N., Ozkan-Aydin, Y., Murray-Cooper, M., Jain, R., Hawkes, E. W., Ramani, P., Goldman, D. I., & Benfey, P. N. (2021).** Mechanism and function of root circumnutation. *Proceedings of the National Academy of Sciences*, 118(8), e2018940118. DOI: [10.1073/pnas.2018940118](https://doi.org/10.1073/pnas.2018940118). Open access via [PMC7923379](https://pmc.ncbi.nlm.nih.gov/articles/PMC7923379/). Code & data: Duke Research Data Repository, [10.7924/r4b27x11m](https://doi.org/10.7924/r4b27x11m). Local copy of supplementary R code: [`external_code/taylor_2021_pnas_loess_spline/`](external_code/taylor_2021_pnas_loess_spline/).
- **Rivière, M., Peaucelle, A., Derr, J., & Douady, S. (2025).** Plant nutation relies on steady propagation of spatially asymmetric growth pattern. *Quantitative Plant Biology*. DOI: [10.1017/qpb.2025.10013](https://doi.org/10.1017/qpb.2025.10013). Open access via [PMC12451244](https://pmc.ncbi.nlm.nih.gov/articles/PMC12451244/). Earlier preprint: [bioRxiv 2022.02.22.481493](https://www.biorxiv.org/content/10.1101/2022.02.22.481493v1).
- **Bastien, R., & Meroz, Y. (2016).** The Kinematics of Plant Nutation Reveals a Simple Relation Between Curvature and the Orientation of Differential Growth. *PLOS Computational Biology*. arXiv preprint: [1603.00459](https://arxiv.org/abs/1603.00459). Code (interactive simulator): [github.com/RnoB/Nutation-Simulator](https://github.com/RnoB/Nutation-Simulator) (Unity / C#, MIT). Live: [unred.org/nutation](http://unred.org/nutation).

### Reference outputs used as oracles

- **Derr, J. (unpublished, 2025-09-19).** Wavelet scaleogram analysis of Suyash Patil's 5-min cadence rice CMTN pilot data. Three reference outputs provided via personal communication, archived locally at [`external_code/derr_wavelets/sept_2025_outputs/`](external_code/derr_wavelets/sept_2025_outputs/):
  - `5minutes_sample_data.pdf` (input signal)
  - `5minutes_average_period=3333s.pdf` (time-averaged amplitude spectrum)
  - `_5minutes_wavelets.png` (CWT scaleogram + Fourier transform)

### Data sources

- Tracked + proofread predictions: `\\multilab-na.ad.salk.edu\hpi_dev\users\eberrigan\circumnutation\20250917_Suyash_Patil_CMTN_Nipponbare_0.8PG_GA4vsTZT\runs\run_20250917_201037\plate_001_greyscale.tracked_proofread.slp`
- Raw scans: `\\pbiob-centos.salk.edu\groot-data\Suyash\AA_Suyash_MDATA\Circumnutation project\Images every 5 minutes\Nipponbare_0.8PG_GA4vsTZT\`
- Coordination metadata: [Circumnutation_Experimets_Directory.xlsx](https://salkinstitute.box.com/file/1987750239743) (Box).

### Notion task pages

- Hub (forward-looking): [Circumnutation work (sleap-roots) — 2026 next phase](https://www.notion.so/3494a67a7667818db2aeee80d76efdd7)
- Operational task (run logs, Box deliveries): [Flat bed scanner circumnutation](https://www.notion.so/9b89bd279dab49d38ae26f5bd8059c65)
- Literature consolidation: [Consolidate circumnutation reference materials](https://www.notion.so/3594a67a766781e49483dea54f721a84)
- Methods review: [Review Meroz 2026 — circumnutation traits + modeling](https://www.notion.so/3574a67a7667814d8cead4fd7a4d4e99)

## Appendix: analysis scripts

Working copies (will be promoted to a tracked location in the sleap-roots repo or `external_code/analysis/` before publication):

- **Frame-to-frame kinematics:** `C:\Users\Elizabeth\AppData\Local\Temp\analyze_tip_growth.py` — per-track path length, net displacement, mean speed, bbox.
- **SLEAP localization noise:** `C:\Users\Elizabeth\AppData\Local\Temp\analyze_sleap_resolution.py` — Savitzky–Golay residual + second-difference estimator.
- **Nutation amplitude (corrected):** `C:\Users\Elizabeth\AppData\Local\Temp\analyze_amplitude.py` — bbox vs. detrended residual cross-check.
- **Per-frame step decomposition:** `C:\Users\Elizabeth\AppData\Local\Temp\analyze_step.py` — longitudinal/lateral split via net-displacement growth axis.

All scripts use `uv run --script` with inline `# /// script` dependency declarations (h5py, numpy, scipy). They read the proofread `.slp` HDF5 directly, without `sleap-io`, for environment portability.

## Document metadata

- Source: this session, 2026-05-07.
- Memory: relevant numbers cached at `C:\Users\Elizabeth\.claude\projects\c--vaults\memory\project_circumnutation_oracle_metrics.md`.
- Status: working draft. To be reviewed against published R pipeline and re-validated on additional plates before any external sharing or publication.
