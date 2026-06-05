# Circumnutation Theory and Trait Specification

**Repo path:** `docs/circumnutation/theory.md`
**Status:** Theoretical foundation for the `sleap_roots/circumnutation/` pipeline. Implementation spec is separate.
**Audience:** Pipeline authors (human and Claude). Self-contained — does not assume prior reading of the four source papers.

---

## Contents

1. [Purpose and scope](#1-purpose-and-scope)
2. [Conventions and calibration contract](#2-conventions-and-calibration-contract)
3. [Foundational framework — Bastien & Meroz 2016](#3-foundational-framework--bastien--meroz-2016)
4. [Spatiotemporal growth pattern — Rivière et al. 2022 / 2025](#4-spatiotemporal-growth-pattern--rivière-et-al-2022--2025)
5. [Modern synthesis — Meroz 2026 review](#5-modern-synthesis--meroz-2026-review)
6. [Translation to tip-only root data](#6-translation-to-tip-only-root-data)
7. [Trait specification](#7-trait-specification)
8. [Validation strategy](#8-validation-strategy)
9. [Appendix A — Cell-wall mechanism (out of pipeline scope)](#9-appendix-a--cell-wall-mechanism-out-of-pipeline-scope)
10. [Appendix B — Corrections from prior conversation drafts](#10-appendix-b--corrections-from-prior-conversation-drafts)
11. [References](#11-references)

---

## 1. Purpose and scope

This document fixes the theoretical content the circumnutation pipeline rests on. It states what the literature claims, with paper + section + equation references for each claim, and translates those claims into operational definitions for traits computed from SLEAP-tracked root tip trajectories.

**What the pipeline gets as input:** per-track time series of tip positions, $(t, x, y)$ in pixels at fixed cadence, plus an optional root cross-section radius `R_px` (pixels, for Tier 4 Bastien-Meroz). No midline, no multi-node skeleton, no destructive measurements. **The pipeline does NOT take `px_per_mm` as a parameter** — see the pure-pixel note in §2.3.

**What the pipeline produces:** per-plant scalar traits in pixel units for length-bearing quantities and calibration-independent units (hours, radians, dimensionless) elsewhere, written to a CSV plus a UTF-8 units sidecar JSON. Users who want millimeter output compose `sleap_roots.circumnutation.units.convert_to_mm()` downstream on the trait DataFrame; the pipeline itself never sees calibration. Diagnostic plots per plate are emitted alongside the CSV.

**What is explicitly out of scope:** any measurement requiring midline tracking along the organ, destructive sampling (AFM, immunolabelling), or 3D imaging. These are flagged in §9 as future biology validation, not pipeline work.

**Reference data used in feasibility numbers throughout:** plate 001 of the proofread CMTN Nipponbare dataset, **assumed** 1200 DPI imaging, 5-min cadence, 14-hour typical run, median 6 tip tracks per plate.

> ⚠ **DPI = 1200 is ASSUMED, NOT VERIFIED.** All `mm` and `µm` numbers throughout this document derive from this assumption (1 px = 21.17 µm, 1 mm = 47.24 px). Pixel-based numbers (per-frame steps, noise, trail length in px) and time-based numbers (period in seconds, frame counts) are independent of the DPI assumption and are reliable. If the actual DPI differs, all `mm` and `µm` numbers scale linearly by (true DPI / 1200), but the structural conclusions (Tier 3 feasibility, Nyquist comfort, $d/\lambda$ ratio) hold regardless of DPI. **Verify DPI before quoting any `mm`/`µm` number from this doc in any external writeup, paper, or grant.** Search this doc for `[†DPI]` to find every value that depends on this assumption.

Per-plate calibration: 1 px = 21.17 µm, so 1 mm = 47.24 px `[†DPI]`. Median per-frame total step ≈ 5.83 px (≈ 1.4 mm/hr `[†DPI]`); longitudinal component 4.29 px/frame (≈ 1.0 mm/hr `[†DPI]`); lateral component 2.75 px/frame (≈ 0.66 mm/hr `[†DPI]`). SLEAP detection noise σ ≈ 2 px (≈ 42 µm `[†DPI]`; estimated by two independent methods, see §7 QC). Nutation period from Derr's pilot CWT analysis on the same data: T ≈ 3333 s ≈ 55.5 min ≈ 0.926 hr (time only, no DPI dependence).

---

## 2. Conventions and calibration contract

### 2.1 Coordinate frame

Image-space coordinates with $x$ horizontal, $y$ vertical (image convention: $y$ increases downward). The growth axis is *not* assumed to be aligned with any image axis — the pipeline must derive it per-track. We define the "longitudinal" direction per track as the principal axis of net displacement from $t=0$ to $t=t_\text{end}$, and the "lateral" direction as its perpendicular in-plane. This makes traits rotation-invariant under camera-mounting variation.

### 2.2 Time

Cadence is fixed per dataset (5 min for the reference data). Time is in seconds internally; trait outputs use hours where convention demands (e.g., growth rate in mm/hr).

### 2.3 Calibration (pure-pixel pipeline)

**Architectural decision (2026-05-07; see `docs/circumnutation/roadmap.md` CC-3):** the pipeline is pure-pixel. It never accepts `px_per_mm` and never emits `[mm]` columns. Calibration is always a downstream concern — users who want millimeter output run the trait DataFrame through `sleap_roots.circumnutation.units.convert_to_mm(traits_df, units, px_per_mm)`. The advantage is that the unresolved DPI ambiguity (1200 vs 400; see §1) is fully decoupled from the pipeline's correctness: every pipeline output is bit-exact reproducible regardless of calibration source.

Every trait in §7 is flagged either:

- **`[mm-convertible]`** — emitted in pixel form by the pipeline (column suffix `_px`, `_px_per_hr`, `_px_per_frame`, `_px²`). Convertible to mm via `convert_to_mm()` downstream. Previously written as `[mm]` throughout the trait tables in §7; that flag is retained for cross-reference to literature and downstream documentation, but the column the pipeline writes uses the px form.
- **`[—]`** — calibration-independent (dimensionless ratios, time-only quantities, booleans, angles).

The pipeline contract is: input $(t_\text{seconds}, x_\text{px}, y_\text{px}, R_\text{px}^\text{opt})$. Internal CWT, ridge extraction, derivative computations, and trait emission all operate in pixels. There is no trait-emission-time physical-unit conversion inside the pipeline — that step is owned by the user, downstream.

### 2.4 Notation

| Symbol | Meaning |
|---|---|
| $s$ | arc length along the organ midline, base $\to$ apex |
| $s_a$ | reverse arc length, apex $\to$ base (used in Rivière et al.) |
| $\theta(s,t)$ | local tangent angle of the organ midline |
| $\kappa(s,t)$ | scalar curvature, $\kappa = \partial\theta/\partial s$ |
| $\kappa_\perp(s,t)$ | transverse curvature in the apical plane (top-view projected) |
| $\psi_c(s,t)$ | direction of principal curvature (azimuthal angle in 3D) |
| $\psi_g(s,t)$ | direction of maximal differential growth |
| $\dot{E}(s,t)$ or $\dot{\varepsilon}(s,t)$ | mean elongation rate along the median line |
| $\dot{\delta}(s,t)$ | differential elongation rate (between opposite sides) |
| $\boldsymbol{\Delta}$ | differential growth vector (3D) |
| $L_{gz}$ | length of growth zone |
| $L_c = \gamma/\beta$ | convergence length |
| $B = L_{gz}/L_c$ | balance number (dimensionless control parameter) |
| $R$ | organ cross-sectional radius |
| $T$ | nutation period (temporal) |
| $\lambda$ | nutation wavelength (spatial, along organ) |
| $v$ | apex propagation speed |
| $\Delta\phi$ | angular amplitude of nutation oscillation |

---

## 3. Foundational framework — Bastien & Meroz 2016

**Source:** Bastien R., Meroz Y. (2016) *The Kinematics of Plant Nutation Reveals a Simple Relation Between Curvature and the Orientation of Differential Growth.* PLoS Comp. Biol. 12(12):e1005238. arXiv:1603.00459.

This paper extends the in-plane tropism kinematic framework of Bastien-Douady-Moulia (2013, 2014) to 3D nutation by adding a perpendicular-curvature equation. It is the canonical deterministic nutation model that justifies tip-only tangent-angle analysis.

### 3.1 Geometric setup

The organ is a thin cylinder of constant radius $R$ along arc length $s$. At each $(s,t)$ it has a curvature magnitude $C(s,t)$ and a curvature direction $\psi_c(s,t)$ — the azimuthal angle of the bending plane in the cross-sectional frame [Bastien & Meroz 2016, §"Geometrical Description", Figure 2]. The differential growth around the cross-section has a maximal direction $\psi_g(s,t)$ and a magnitude $\Delta(\psi_g, s, t)$.

### 3.2 Coupled curvature dynamics

Decomposing differential growth into components parallel ($\Delta_\parallel$) and perpendicular ($\Delta_\perp$) to the current bending plane, the coupled dynamics are [Bastien & Meroz 2016, Eqs. 13 & 14]:

$$\frac{DC(s,t)R}{Dt} = \Delta_\parallel(s,t)\,\dot{E}(s,t)$$

$$\frac{D\psi_c(s,t)}{Dt} = \frac{\Delta_\perp(s,t)\,\dot{E}(s,t)}{C(s,t)\,R}$$

where $D/Dt = \partial/\partial t + v(s,t)\,\partial/\partial s$ is the material derivative co-moving with each tissue element [Bastien & Meroz 2016, Eq. 7]. The first equation describes magnitude of bending; the second describes rotation of the bending plane. Nutation is the rotation of $\psi_c$ around the cross-section, driven by oscillating $\psi_g$.

### 3.3 Slaving claim

The model's central claim is that the bending plane $\psi_c(s,t)$ is *slaved* to the direction of maximal differential growth $\psi_g(s,t)$ — the organ's curvature direction chases the growth direction with a delay. Nutation arises when $\psi_g$ rotates periodically in time [Bastien & Meroz 2016, §"Variation of the principal direction of the differential growth $\psi_g(s,t)$ as an internal oscillator"].

### 3.4 Hypotheses for tip-only analysis

To make any sense of apical-tip projection data (the only data we have for root tips), Bastien & Meroz state explicitly the hypotheses required [Bastien & Meroz 2016, §"Measurements of the apical tip in the horizontal plane", H1–H3]:

- **H1 — Spatial homogeneity:** dynamics do not depend on position along the organ; $\Delta(\psi, s, t) = \Delta(\psi, t)$.
- **H2 — Unique tip→shape map:** the 2D apical projection determines the 3D organ shape up to known constraints.
- **H3 — Elongation effects negligible:** organ length growth does not distort the projected pattern over one nutation period.

H1 is the most likely to fail — and in fact does fail along the Averrhoa rachis (see §4). For root tips in the regime where the elongation zone is short relative to the organ length and the apical tip is near the apical edge of the elongation zone, H1 is more defensible.

### 3.5 Tip-only extraction equations (load-bearing for the pipeline)

Under H1–H3, with the apical tip projected to coordinates $(x_a(t), y_a(t))$ in the horizontal plane, Bastien & Meroz derive [2016, Eqs. 20 & 21]:

$$\boxed{\psi_g(t) = \arctan\!\left(\frac{dx_a/dt}{dy_a/dt}\right)}$$

$$\boxed{\Delta(\psi_g, t)\,\dot{E}(t) = \frac{2R}{L}\sqrt{\dot{x}_a^2 + \dot{y}_a^2}}$$

These say: **(a)** the direction of maximal differential growth equals the tangent angle of the tip trajectory, and **(b)** the magnitude of differential growth times mean elongation rate is proportional to the tip speed, with prefactor $2R/L$.

This is the load-bearing physical justification for the pipeline's use of tangent angle as the primary spectral signal. $\psi_g(t)$ is not a numerically convenient detrended quantity — it is, *under H1–H3*, the direct estimator of the underlying differential-growth oscillator. CWT applied to $\psi_g(t)$ recovers the spectral content of that oscillator.

**Convention note:** Bastien & Meroz use $\arctan(dx/dy)$, returning values in $(-\pi/2, \pi/2)$. The pipeline must use $\text{atan2}(dx/dt, dy/dt)$ and unwrap the result to allow $\psi_g$ to grow continuously past $\pm\pi$ — otherwise wrap-around discontinuities create spurious high-frequency CWT content.

### 3.6 Regulation: proprioception term

To prevent the orbit from drifting away from the base, the parallel-curvature equation can be augmented with a curvature-dependent damping term [Bastien & Meroz 2016, Eq. 23]:

$$\Delta_\parallel(s,t) = -\gamma C + \Delta(\psi_g) \sin(\psi_g - \psi_c)$$

where $\gamma$ is the proprioceptive sensitivity. In the Meroz 2026 review framework (see §5), this $\gamma$ together with a misalignment sensitivity $\beta$ controls the convergence length $L_c$.

### 3.7 H1-failure signature: trochoid patterns

If two segments of the organ oscillate at different angular frequencies $\omega_1$ and $\omega_2$, the apical tip traces an epi- or hypotrochoid (spirograph) pattern, and equation 3.5(a) returns an *effective* $\psi_g$ dominated by the faster oscillator [Bastien & Meroz 2016, Figure 7, Movies S6–S7]. Trochoid signatures in the tip xy trajectory or multi-peak structure in the CWT spectrum are direct evidence H1 is failing — and motivate moving to spatially-resolved measurement.

---

## 4. Spatiotemporal growth pattern — Rivière et al. 2022 / 2025

**Sources:**
- Rivière M., Peaucelle A., Derr J., Douady S. (2022) *Spatiotemporal growth pattern during plant nutation implies fast dynamics for cell wall mechanics and chemistry: a multiscale study in Averrhoa carambola.* bioRxiv 2022.02.22.481493. ("Rivière 2022")
- Rivière M., Peaucelle A., Derr J., Douady S., Marmottant P. (2025) *Plant nutation relies on steady propagation of spatially asymmetric growth pattern.* Quantitative Plant Biology 6, e10013. doi:10.1017/qpb.2025.10013. ("QPB 2025")

The 2022 preprint is the multiscale paper containing both kinematic and cell-wall results. QPB 2025 is the published descendant focused on the kinematic narrative. The kinematic content overlaps; cell-wall content (AFM + immunolabelling) is in the 2022 version only — see §9 for context. Citations below give the 2022 paper for kinematic specifics with QPB 2025 cross-references where the same finding appears.

### 4.1 System and method

Time-lapse imaging of *Averrhoa carambola* compound leaves; top-view (2.5-min cadence) for curvature and node tracking, side-view (1-min cadence) for local elongation. Rachis midline obtained by thresholding fluorescent pigment + moving-median smoothing + local circle fit for $\kappa_\perp$. Local elongation rate measured by image-to-image correlation between successive frames [Rivière 2022, §"Kinematics: data analysis" and §"Kinematics: fine elongation measurements"].

Nutation period 1.5–4 hours (typically 2–3 h), angular amplitude ~25°, principal axis perpendicular to growth axis. Pendulum-like oscillation [Rivière 2022, §"Characterizing nutation", Figure 1].

### 4.2 Spatial localization of growth and bending

Mean elongation rate $\dot{E}$ measured per interfoliolar segment; differential elongation $\dot{\delta}$ envelope extracted from $\kappa_\perp(s,t)$ via Hilbert transform. Both quantities show a spatially localized profile relative to the apex [Rivière 2022, Figure 2, panels B and C; QPB 2025 Figure 2 same data].

**Headline kinematic finding [Rivière 2022, §"Differential elongation peaks where elongation drops"; QPB 2025 abstract and §"Spatial profile of differential growth"]:**

> The differential elongation rate is non-monotonous and its maximum coincides with the edge of the growing zone, where the mean elongation rate drops.

Quantitatively for Averrhoa: typical growth-zone length ~50 mm, no detectable growth past ~100 mm from apex [Rivière 2022, §"Differential elongation peaks where elongation drops"]. Mean elongation rate in the growth zone ~$10^{-2}$ h⁻¹. Maximum differential elongation rate ~$3 \times 10^{-2}$ h⁻¹ (transient, modulated in time).

### 4.3 Kinematic model

The paper's kinematic model has four components [Rivière 2022, §"Details and implementation of the model", Eqs. 2–5]:

**(a) Lateral asymmetry decomposition** [Eq. 2]:

$$\dot{\varepsilon} = \frac{\dot{\varepsilon}_R + \dot{\varepsilon}_L}{2}, \qquad \dot{\delta} = \frac{\dot{\varepsilon}_R - \dot{\varepsilon}_L}{2}$$

**(b) Mean elongation as sigmoid in reverse arc length** [Eq. 3]:

$$\dot{\varepsilon}(s_a) = \frac{\dot{\varepsilon}_0}{2}\left[1 - \tanh\!\left(\frac{s_a - L_{gz}}{\Delta L}\right)\right]$$

This is a sigmoid centered at $s_a = L_{gz}$ with transition width $\Delta L$. At the apex ($s_a \to 0$), $\dot{\varepsilon} \to \dot{\varepsilon}_0$. Past the growth zone ($s_a \gg L_{gz}$), $\dot{\varepsilon} \to 0$.

**(c) Differential elongation as derivative of mean profile, modulated in time** [Eq. 4]:

$$\dot{\delta}(s_a, t) = \dot{\delta}_0 \left[1 - \tanh^2\!\left(\frac{s_a - L_{gz}}{\Delta L}\right)\right] \sin(\omega t)$$

The factor $1 - \tanh^2$ is the derivative of $\tanh$, peaked at $s_a = L_{gz}$ with characteristic width $\Delta L$. This is the mathematical realization of the headline finding: $\dot{\delta}$ peaks exactly where the gradient of $\dot{\varepsilon}$ is largest — at the basal edge of the growth zone.

**(d) Curvature dynamics** [Eq. 5]:

$$\frac{\partial \kappa_\perp}{\partial t} \approx \frac{1 - R^2 \kappa_\perp^2}{R}\,\dot{\delta}$$

Material-derivative advection is neglected because the nutation period is much shorter than the elongation timescale. For small $R\kappa_\perp$, this reduces to $\partial\kappa_\perp/\partial t \approx \dot{\delta}/R$.

### 4.4 Apex angular amplitude relation (load-bearing for tip-only fitting)

From the model, the angular amplitude at the apex is constrained by [Rivière 2022, Eq. 1]:

$$\boxed{\Delta\phi = \frac{2\,\Delta L\,\dot{\delta}_0}{\omega R}}$$

Interpretation: $\dot{\delta}_0/\omega$ is the total differential elongation accumulated over half a nutation period; dividing by $R$ gives a curvature; integrating that curvature over the bending zone width $2\Delta L$ gives the apex deflection angle. **This is a closed-form relation between five observable quantities** — and four of them are extractable from tip data alone:

- $\Delta\phi$ from tip angular extent
- $\omega = 2\pi/T$ from temporal CWT
- $\Delta L$ from spatial decay of $|\kappa(s)|$ basal of peak (see §6)
- $R$ from external measurement (root cross-section)

So $\dot{\delta}_0$ is **derivable**: $\dot{\delta}_0 = \omega R\,\Delta\phi / (2\Delta L)$.

### 4.5 Best-fit parameters (Averrhoa)

For one Averrhoa rachis, fitting the kinematic model to the side-view apparent-elongation wavelet decomposition [Rivière 2022, Figure 3I–J caption]:

- $L_{gz} = 20.6$ mm
- $\Delta L = 12.2$ mm
- $\dot{\delta}_0 = 4.5 \times 10^{-3}$ h⁻¹
- $\dot{\varepsilon}_0 = 1.4 \times 10^{-2}$ h⁻¹ (fixed before fitting)
- $R = 0.26$ mm (fixed before fitting)
- $\Delta\phi = 8°$

Note $\dot{\delta}_0 < \dot{\varepsilon}_0$ but the ratio is high enough that local contractions occur transiently when the local mean rate is at the falling edge of its sigmoid.

### 4.6 Local contractions and the wavelet signature

The side-view apparent elongation rate $\dot{\varepsilon}(s_a, t)$ goes negative in places — actual local contraction, not just slower growth [Rivière 2022, Figure 3A]. CWT decomposition of this 2D field (using cgau2 mother wavelet — the second-derivative complex Gaussian) shows two dominant modes spatially separated [Rivière 2022, Figure 3B]:

- $\tau_{2f} \approx 1.2$ h near the apical end of the bending zone
- $\tau_f \approx 2.1$ h near the basal end

The model predicts that *whether both modes appear* depends on the relative magnitudes of $\dot{\delta}_0$ and $\dot{\varepsilon}_0$ [Rivière 2022, Figure 3C–H]: only when contractions are allowed (low $\dot{\varepsilon}_0$ relative to $\dot{\delta}_0$, panels C–D) does the apical-end frequency-doubled mode appear in the projection. The 2:1 ratio is therefore a side-view projection artifact under contraction conditions, *not* evidence of two separate oscillators.

**This matters for the pipeline because the analogous side-view projection effect is not present in our top-view root-tip data.** Whatever frequency-doubling signature we look for, it must be derived from the trail geometry directly, not from a side-view kymograph.

### 4.7 Steady traveling-wave hypothesis (QPB 2025 framing)

QPB 2025 sharpens the 2022 finding into a **steady traveling-wave** statement [QPB 2025, §"Steady propagation of growth pattern", Figure 4]: the spatial profile $\dot{\varepsilon}(s_a)$ and $\dot{\delta}(s_a, t)$ is fixed in the reference frame attached to the apex, and propagates basally as the apex grows forward. Under this picture:

$$\boxed{\lambda_\text{spatial} = v_\text{growth} \cdot T_\text{temporal}}$$

The spatial wavelength of the nutation pattern imprinted in the post-growth-zone tissue equals the apex propagation speed times the temporal period of nutation. This is a falsifiable equality and the central physical claim of the QPB paper.

---

## 5. Modern synthesis — Meroz 2026 review

**Source:** Meroz Y. (2026) *Physics of Computation and Behavior in Plants.* arXiv:2604.21763 [cond-mat.other].

This is a review framing plant tropisms and circumnutations as spatiotemporal dynamical systems with distributed sensing, embodied mechanics, and functional stochasticity. It does **not** present new nutation dynamics — the deterministic model used (Eq. 1 below) is presented as tropism dynamics; circumnutation appears in §6.2 under stochasticity-as-functional-resource. The relevance for our pipeline is the dimensionless control parameter framework that ports cleanly to nutation and gives a single scalar trait with cross-species literature baseline.

### 5.1 Tropism dynamics (the canonical Bastien-Douady-Moulia equation)

The minimal tropism model [Meroz 2026, Eq. 1; originally Bastien et al. 2013]:

$$\frac{\partial\kappa(s,t)}{\partial t} = -\beta \sin(\theta(s,t) - \theta_p) - \gamma\kappa(s,t), \qquad s > L - L_{gz}$$

In the mature zone ($s \le L - L_{gz}$), $\partial\kappa/\partial t = 0$ — curvature is passively advected by growth but not actively generated. $\beta$ is the tropic sensitivity (gravity, light); $\gamma$ is the proprioceptive sensitivity; $\theta_p$ is the preferred direction of the stimulus.

**Caveat from §2 of this doc:** This equation as written is a tropism equation, not a nutation equation. It does not produce sustained oscillations on its own — it produces relaxation toward $\theta = \theta_p$. The Bastien & Meroz 2016 extension (§3 here) is required to produce nutation, by adding the perpendicular-curvature equation and an oscillating $\psi_g$.

### 5.2 Convergence length

In the small-angle linearization, the steady-state shape is exponential [Meroz 2026, Eq. 5]:

$$\theta(s) = \theta_0 \exp\!\left(-\frac{\beta}{\gamma}s\right)$$

defining a characteristic length [Meroz 2026, Eq. 6]:

$$\boxed{L_c = \frac{\gamma}{\beta}}$$

This is the spatial extent over which curvature decays. Small $L_c$ means localized bending; large $L_c$ means distributed curvature.

### 5.3 Balance number

The dimensionless ratio of growth-zone length to convergence length [Meroz 2026, Eq. 7]:

$$\boxed{B = \frac{L_{gz}}{L_c} = \frac{L_{gz}\,\beta}{\gamma}}$$

For tropism dynamics:
- $B < 1$: bending insufficient to reach the vertical, overdamped relaxation.
- $B > 1$: organ reaches the vertical and may overshoot, with oscillatory modes.

Bastien et al. 2013 measured $B$ across 12 angiosperm species [Meroz 2026, §2.3 and Figure 2c; original data Bastien et al. 2013]: range 0.9–9.3, with overshoot mode count increasing with $B$. Wheat coleoptile is mode 0 (no overshoot, $B \approx 0.9$); *Impatiens glandulifera* is mode 2 ($B \approx 9.3$).

### 5.4 Where circumnutation sits

Circumnutation appears in [Meroz 2026, §6.2 "Noisy circumnutations facilitate exploration and sensing"] under the stochasticity-as-functional-resource framing, citing Nguyen, Dromi, Kempinski, Peleg, Meroz (2024, *Phys. Rev. X* 14:031027). The deterministic dynamical equations of §3 here remain the canonical model; the 2026 review adds a complementary stochastic-exploration interpretation.

---

## 6. Translation to tip-only root data

### 6.1 The tip-trail-as-midline identity

For an apically-growing organ where tissue past the elongation zone does not reshape (the "tissue freezing" assumption underlying §4.7), every past tip position is a material point still sitting where it was laid down. So if $\mathbf{p}(\tau) = (x(\tau), y(\tau))$ is the tip at time $\tau$, then the curve

$$\{\mathbf{p}(\tau) : 0 \le \tau \le t_\text{now}\}$$

is the organ midline at time $t_\text{now}$, for the part of the organ older than the growth zone. The time-to-arc-length conversion is:

$$s(\tau) = \int_0^\tau |v(\sigma)|\,d\sigma$$

Trajectory curvature in time equals midline curvature at arc length:

$$\kappa_\text{path}(\tau) = \kappa(s(\tau))$$

This identity — together with the QPB steady-traveling-wave duality $\lambda = vT$ — is what permits root-tip-only data to address spatial QPB-style findings without midline tracking. **It is the load-bearing assumption for Tier 3 below.** It can fail in three ways: (a) tissue continues to bend after exiting the growth zone (gravitropic re-orientation, elastic relaxation), (b) the growth zone is so long that the trail doesn't extend past it within the imaging window, (c) tip detection is so noisy that the reconstructed midline is dominated by jitter rather than real curvature. The pipeline must check for these.

### 6.2 Trajectory curvature

Tip trajectory curvature is computed directly from the time series [standard differential geometry]:

$$\kappa_\text{path}(t) = \frac{\dot{x}\,\ddot{y} - \dot{y}\,\ddot{x}}{(\dot{x}^2 + \dot{y}^2)^{3/2}}$$

Sign convention: positive = left turn, negative = right turn. The denominator $|\mathbf{v}|^3$ diverges when the tip momentarily stops; at 5-min cadence on rice this can occur on individual frames where displacement is below noise. The pipeline must guard with $|\mathbf{v}| > k\sigma_\mathbf{v}$ for some $k$ (recommended $k=2$, masking sub-noise frames).

Reparameterizing by arc length $s(\tau) = \int_0^\tau |\mathbf{v}|\,d\sigma$ gives $\kappa(s)$ — the midline curvature as a function of position along the organ.

### 6.3 Pipeline tier structure

Five tiers, each callable independently from a notebook; the full `CircumnutationPipeline` runs all of them.

```
Tier 0 — Raw kinematics (no spectral analysis)
  v_total, v_long, v_lat, long/lat ratio, path/displacement ratio,
  angular amplitude, growth-axis identification

Tier 1 — Derr-faithful temporal CWT
  CWT on x(t) (or one chosen coordinate) using cmor1.5-1.0 wavelet
  Ridge extraction → T(t), A(t), Fourier spectrum
  Regression-tested against Derr's Sept-2025 pilot output

Tier 2 — Bastien-Meroz ψ_g(t) CWT
  ψ_g(t) = atan2(dx/dt, dy/dt), unwrapped
  SG-DETRENDED (residual = raw − SG-smooth, window 23) before CWT
    (PR #7: detrend, not the literal "pre-smooth" — the residual is the
     oscillation a period-CWT needs; reuses Tier 1's primitive — see Appendix B)
  CWT on ψ_g(t) → T_psig (only conditioned trait); handedness/Δ_E/helix are raw
  Δ(ψ_g)·Ė(t) = (2R/L)|v(t)| → kinematic differential-growth proxy

Tier 3 — Tip-trail spatial CWT (the QPB branch)
  Reconstruct midline via tip-trail-as-midline (§6.1)
  Compute κ(s) along the trail (smoothing required, see §7 QC)
  |κ(s)| envelope → peak location estimates L_gz
  Spatial CWT on κ(s) basal of the peak → λ(s), L_c, decay length
  Steady-traveling-wave check: λ_spatial vs v·T

Tier 4 — Bastien-Meroz parametric fit
  From Tier 3 outputs: L_gz, L_c → B = L_gz/L_c
  From Eq. 4.4: δ̇₀ = ωR·Δφ / (2ΔL)
  From Eq. 5.2: L_c = γ/β; one constraint, two unknowns —
    needs an additional measurement (e.g., decay rate during transient)
    to disentangle β and γ individually. Phase 1 emits L_c only;
    β and γ are Phase 2 with stimulus experiments.

QC tier — runs alongside all above
  Tracking quality (SG residual, 2nd-diff noise, outlier step rate)
  Cone-of-influence masks per CWT
  Calibration presence flag
  Nutation-presence boolean (band power vs noise floor)
```

### 6.4 Feasibility numbers (reference data)

> All `mm` numbers in this section assume 1200 DPI per §1 — verify before quoting. The pixel ratios ($d/\lambda$, Nyquist ratio) and frame counts are DPI-independent, and so are the structural conclusions of this section.

Computing in pixels first (DPI-independent), then converting:

- Spatial wavelength along the trail: $\lambda \approx v_\text{px/frame} \cdot T_\text{frames} = 5.83 \times 11.1 \approx 65$ px ($\approx 1.37$ mm `[†DPI]`)
- Trail length over 14-hr run: $d \approx 5.83 \times 168 \approx 980$ px ($\approx 20.7$ mm `[†DPI]`)
- $d/\lambda \approx 15$ wavelengths imprinted in the trail (DPI-independent)

For Tier 3 to be core-trait-worthy, $d/\lambda > 3$ is required; 15 is well above threshold. **The reference data fully supports Tier 3 regardless of the exact DPI value.**

For the growth-zone region (most apical $\sim L_{gz} \approx 1$–5 mm in rice — *literature estimate, pending verification with rice expert; see §11 references*), the trail comprises $\sim L_{gz}/v \approx 0.7$–3.6 hours of imaging `[†DPI; rice $L_{gz}$ unverified]`, or 9–43 frames at 5-min cadence (DPI-independent if expressed in frames-per-elongation-zone-traverse, but the elongation zone length itself is in mm and so depends on both DPI calibration and biology). Peak-$|\kappa(s)|$ location is therefore resolvable but the spatial decay basal of the peak (decay length $L_c$) is the better-resolved trait, since it operates on the longer post-growth-zone trail.

#### Peak-$|\kappa(s)|$ resolvability for `L_gz_estimate`

The peak of the $|\kappa(s)|$ envelope sits at arc-length $L_{gz}$ from the current apex (Rivière 2022 Eq. 4). To localize it, the growth-zone region of the trail must span enough samples for stable peak/transition detection. The number of trail frames within the growth zone is:

$$\boxed{N_{gz} = \frac{L_{gz}}{v \cdot \Delta t}}$$

where $\Delta t$ is the frame cadence (in the same time units as $v$). Resolvability requires $N_{gz} \geq N_{\min}$, with $N_{\min} \in [5, 10]$ as a rule of thumb for stable peak/corner localization in 1D. Equivalently, the minimum measurable $L_{gz}$ for given imaging conditions is:

$$L_{gz}^{\min} = N_{\min} \cdot v \cdot \Delta t$$

**Substituted for the reference data:** $v = 5.83$ px/frame, $\Delta t = $ 5 min (so $v \cdot \Delta t = 5.83$ px/frame). With $N_{\min} = 5$: $L_{gz}^{\min} \approx 29$ px ≈ 0.62 mm `[†DPI]`. With $N_{\min} = 10$ (conservative): $L_{gz}^{\min} \approx 58$ px ≈ 1.23 mm `[†DPI]`.

**Conclusion:** any rice $L_{gz}$ above $\sim 1$ mm is comfortably resolvable. At the low end of the literature estimate ($L_{gz} = 1$ mm = 47 px = 8 frames), it sits in the resolvability window between $N_{\min} = 5$ and $N_{\min} = 10$ — measurable but with reduced confidence. At $L_{gz} \geq 2$ mm, the peak is robustly resolvable. **If the rice expert returns a value below ~1 mm, the `L_gz_estimate` trait should be flagged unreliable on the reference imaging conditions and either cadence increased or imaging resolution improved before relying on it.**

This criterion is DPI-independent when expressed in pixels: $N_{\min} \cdot v_\text{px/frame}$. The DPI assumption only enters when converting the threshold to mm for comparison with literature.

**Implementation note — $v$ in the formula is local, not instantaneous.** The $v$ in $N_{gz} = L_{gz}/(v \cdot \Delta t)$ is the tip growth speed in the trail region containing the candidate peak, *not* the instantaneous tip velocity at $t_\text{now}$. On a typical clean rice plate $v$ is approximately steady across the run (per the QPB steady-state assumption), so the distinction is moot. But on plates where growth speed varies substantially — early-growth ramp-up, late-run slowdowns, environmental perturbations, tracking dropouts — the resolvability criterion should use $\text{median}(v)$ computed over the most recent trail window of length $\sim L_{gz}$, not $v(t_\text{now})$ which can be biased by single-frame noise or transient slowdowns. For the QC trait `L_gz_resolvable`, recommend computing $v$ as the median per-frame step magnitude over the apical-most $L_{gz}/(v_\text{rough}\,\Delta t)$ frames, where $v_\text{rough}$ is a one-pass estimate using the median over the full run. One iteration of refinement is enough; do not iterate to convergence.

### 6.5 Cadence-Nyquist check

Per-frame step / spatial wavelength = 5.83 / 65 ≈ 9.0%, well below the conservative 25% threshold for spatial aliasing. 5-min cadence is comfortable; 10-min would still work; 30-min would alias the nutation. **This ratio is in pixels on both sides and so is DPI-independent** — true regardless of calibration verification. This becomes a gatekeep parameter for upstream cadence choices.

### 6.6 Fundamental limits of tip-only kinematics

This subsection clarifies what *cannot* be measured from tip-only data, regardless of imaging quality or pipeline cleverness. It determines which traits in §7 are upper bounds on what's possible from current data, and which would require an upstream pipeline change.

**Apex velocity is an integral over the growth zone, not a profile.** The instantaneous tip speed equals the integrated elongation rate along the growth zone:

$$v(t) = \int_0^{L_{gz}} \dot{\varepsilon}(s_a, t)\,ds_a$$

A single number ($v$) determined by an integral over a function ($\dot{\varepsilon}$) gives one equation with infinitely many unknowns. Two organs with very different $\dot{\varepsilon}(s_a)$ shapes — including different combinations of growth-zone length, sigmoid transition width, and amplitude — can have identical $v(t)$ as long as their integrals match. **Tip velocity tells us *total* growth rate, not *spatial distribution* of growth.**

**The trail-as-midline identity holds only past the growth zone.** Each past tip position $p(\tau)$ is a "fossil marker" — the apex's lab-frame location at time $\tau$. After deposition, the corresponding material element continues to move basally in the apex frame until it exits the growth zone, then becomes stationary in the lab frame. Our trail recording never updates: it preserves $p(\tau)$ at the deposition time. So:

- For $t - \tau > \tau_{gz}$ (post-growth-zone): trail position equals current material-element position. Trail = midline. ✓
- For $t - \tau < \tau_{gz}$ (within-growth-zone): trail position $\neq$ current material-element position. The element has been "pushed forward" in the lab frame by ongoing growth happening apical to it; the trail records its deposition position only.

Within the most recent $\sim L_{gz}/v$ time of trail data, the geometry is *deposition geometry*, not *current midline geometry*. Spatial CWT and curvature analysis in this region pick up the deposition-time structure, not the current-midline structure. **The recent trail cannot be inverted to recover $\dot{\varepsilon}(s_a)$** — the deposition spacing is degenerate with the elongation profile.

**What this means for the trait list:**

| Trait | Status from tip-only data |
|---|---|
| $L_{gz}$ via peak-$|\kappa(s)|$ location | ✓ Measurable (Tier 3) |
| $L_c$ via post-peak exponential decay | ✓ Measurable (Tier 3) |
| $B = L_{gz}/L_c$ | ✓ Measurable (Tier 4) |
| Apex velocity $v(t)$ | ✓ Measurable (Tier 0) |
| Mean elongation rate $\dot{\varepsilon}_0$ in growth zone (scalar) | ✓ Approximately, via $v / L_{gz}$ |
| Differential growth amplitude $\dot{\delta}_0$ (scalar) | ✓ Derivable via $\dot{\delta}_0 = \omega R \Delta\phi / (2\Delta L)$ — Rivière 2022 Eq. 1 |
| Spatial profile $\dot{\varepsilon}(s_a)$ — sigmoid shape, transition width $\Delta L$ | ✗ Not measurable — degenerate with $v(t)$ |
| Spatial profile $\dot{\delta}(s_a, t)$ shape | ✗ Not measurable for same reason |
| Local contractions within growth zone | ✗ Not measurable; would need multi-node tracking |
| $(\beta, \gamma)$ separately (Bastien-Meroz parameters) | ✗ Phase 2 — requires gravitropism stimulus |

**Required upstream change for the missing traits:** multi-node SLEAP skeleton tracking (e.g., 5–10 nodes equispaced along the root midline, or feature-point tracking of distinguishable surface marks). This would give $(x_i(t), y_i(t))$ for nodes $i = 1, \ldots, N$, from which $\dot{\varepsilon}(s_a, t)$ is computable via image-to-image correlation à la Bastien et al. 2016 (KymoRod) or Chavarría-Krauser et al. 2008. **This is a Phase 2+ pipeline upgrade and is outside the scope of this document.** The current pipeline, working from tip-only data, can correctly emit $L_{gz}$, $L_c$, $B$, $\dot{\delta}_0$, and integrated $\dot{\varepsilon}_0$, but cannot emit the spatial *profiles* of these quantities along the organ.

---

## 7. Trait specification

Each trait has: symbol, units, computation source, calibration flag, and literature anchor.

### 7.1 Tier 0 — Raw kinematics

> ⚙ **Units annotation (post-conversion):** under the pure-pixel pipeline convention (§2.3 / CC-3), the velocity traits are emitted by ``sleap_roots.circumnutation.kinematics.compute`` in ``px/frame`` units. The ``mm/hr`` annotation in the Units column below is the *post-conversion* form, produced downstream by composing ``sleap_roots.circumnutation.units.convert_to_mm()`` (PR #1 foundation) and a future ``convert_to_per_hour()`` utility on the trait DataFrame. The pipeline itself emits column names with the ``_px_per_frame`` suffix (e.g., ``v_total_median_px_per_frame``); the table below uses the short form for readability.

| Symbol | Units | Description | Source / anchor |
|---|---|---|---|
| `v_total_median` | mm/hr `[mm]` | Median per-frame total tip step magnitude (magnitude — always ≥ 0). Rotation-invariant; survives the growth-axis reliability gate. | This doc §1; Rivière 2022 §"Characterizing nutation" for context |
| `v_long_signed_median` | mm/hr `[mm]` | Signed median of per-frame longitudinal projection ``Δ_long_i = Δxy_i · û_g``. Typically > 0 for a growing plant, ~0 for pure jitter, < 0 for a retracting plant. Rotation-dependent; NaN'd when growth axis is unreliable. | This doc §2.1, §3.2 (prelim) |
| `v_long_abs_median` | mm/hr `[mm]` | Absolute median of per-frame longitudinal projection. Always ≥ 0. Rotation-dependent. | This doc §2.1 |
| `v_lat_signed_median` | mm/hr `[mm]` | Signed median of per-frame lateral projection ``Δ_lat_i = Δxy_i · û_lat``. Expected ≈ 0 by symmetry around the growth axis (itself a sanity check on the axis estimate). Rotation-dependent. | This doc §2.1 |
| `v_lat_abs_median` | mm/hr `[mm]` | Absolute median of per-frame lateral projection. Always ≥ 0. Rotation-dependent. | This doc §2.1 |
| `long_lat_ratio` | — `[—]` | ``v_long_abs_median / v_lat_abs_median``; uses **abs** versions (the ratio of signed quantities is dominated by the near-zero signed-lateral denominator and is rarely meaningful). NaN when ``v_lat_abs_median = 0``. Intuition: ratio ≈ 1 means strong nutation, ratio ≫ 1 means weak. Rotation-dependent. | This doc §1 reference data |
| `path_displacement_ratio` | — `[—]` | Total path length `L` / net base-to-end displacement `D`. NaN when `D = 0` exactly (closed loop). Rotation-invariant; survives the gate. | This doc §1; rice 1.36 reference |
| `angular_amplitude` | rad `[—]` | $\Delta\phi$, peak-to-peak angular extent of the unwrapped velocity-direction time series ``ψ_g(t)`` per Bastien-Meroz 2016 Eq. 20 / §3.5: ``max(ψ_g) − min(ψ_g)``. **Rotation-invariant under offset AND sign-flip** of ψ_g, so the trait's value is INDEPENDENT of the ``atan2`` argument-order convention. Survives the gate. | Rivière 2022 Eq. 1; Bastien-Meroz 2016 §3.5 |
| `principal_axis_angle` | rad `[—]` | Image-frame angle of the growth axis via STANDARD math ``atan2(y_N − y_1, x_N − x_1)``. NOT the same as ψ_g (different formula, different quantity, different convention — see §3.5 for the ψ_g convention warning). Under image-y-down (§2.1), a root growing image-down reads as ``+π/2``. Rotation-dependent. | Internal; rotation-invariance support |
| `growth_axis_unreliable` | bool `[—]` | True iff ``D < GROWTH_AXIS_RELIABILITY_K × sg_residual_xy_local`` (default `K = 10`); when True, Tier 0 NaN's the 6 rotation-dependent traits above. **Emitted by Tier 0 (PR #2), composed but not re-emitted by QC tier (PR #3).** See ``openspec/changes/archive/.../add-circumnutation-tier0-kinematics/design.md`` D2 for the cross-tier ownership rationale. | `roadmap.md` CC-5; this PR's design D2 |

### 7.2 Tier 1 — Derr-faithful temporal CWT

| Symbol | Units | Description | Source / anchor |
|---|---|---|---|
| `T_nutation_median` | hr `[—]` | Median of $T(t)$ from CWT ridge of one tip coordinate, COI-masked | Derr 2025 pilot; Bastien & Meroz 2016 §"Temporal linear variation of $\psi_g$" (linear oscillator) |
| `T_nutation_iqr` | hr `[—]` | Inter-quartile range of $T(t)$, indicates period drift | Rivière 2022 §"Elongation and bending are localized" mentions amplitude modulation |
| `A_nutation_envelope_max_px` | px `[—]` | Peak of $|C(t)|$ ridge envelope from CWT (px-units, calibration-independent for relative amplitude); `_px` suffix marks unit per program convention | Derr 2025 pilot |
| `band_power_ratio` | — `[—]` | Spectral power in $[0.5T, 2T]$ band / total spectral power | New trait, used for `is_nutating` boolean |
| `period_residual_vs_derr_reference` | — `[—]` | Fractional period residual `(T - DERR_EXPECTED_PERIOD_S) / DERR_EXPECTED_PERIOD_S` (positive = slower than Derr reference); `DERR_EXPECTED_PERIOD_S = 3333.0` is rice-specific (override via `ConstantsT` for other species) | Derr 2025 pilot (Sept-2025 oracle) |

### 7.3 Tier 2 — Bastien-Meroz $\psi_g$

| Symbol | Units | Description | Source / anchor |
|---|---|---|---|
| `T_psig_median_s` | s `[—]` | Median period from CWT ridge of $\psi_g(t)$, COI-masked. Emitted in **seconds** (PR #7; consistent with Tier 1 `T_nutation_median` — see Appendix B). | Bastien & Meroz 2016 Eq. 20 |
| `psig_long_consistency` | — `[—]` | Correlation between $T_\text{psig}$ and $T_\text{nutation}$ across CWT range; diagnostic for H1. Deferred to PR #13 Layer-3 (owns both this trait and the cross-tier test). | Bastien & Meroz 2016 §3.7 (trochoid signature) |
| `delta_E_amplitude_proxy_px_per_frame` | px·frame⁻¹ `[—]` | Median of $\sqrt{\dot{x}^2 + \dot{y}^2}$ in **px/frame** (PR #7: px/frame, not px·hr⁻¹ — Tier 0 velocity convention; see Appendix B); $\propto (L/2R)\Delta\dot{E}$ | Bastien & Meroz 2016 Eq. 21 |
| `handedness` | $\in \{-1, 0, +1\}$ `[—]` | Sign of the **net unwrapped $\psi_g$ rotation over all finite frames** (**COI-free** — PR #7; see Appendix B); $+1$ = $\psi_g$ increasing (positive mean $d\psi_g/dt$) = counterclockwise as displayed in the y-down image, $-1$ = clockwise, $0$ = no net rotation | Bastien & Meroz 2016 §"Constant principal direction of growth" |
| `helix_signed_area` | px² `[—]` | Signed area enclosed by tip trajectory (Shoelace formula); confirmation of handedness | Standard kinematic |

### 7.4 Tier 3 — Spatial wavelet (QPB branch)

| Symbol | Units | Description | Source / anchor |
|---|---|---|---|
| `L_gz_estimate` | mm `[mm]` | Distance from apex to peak of $|\kappa(s)|$ envelope along reconstructed midline | Rivière 2022 Eq. 4 (peak of $1-\tanh^2$ at $s_a = L_{gz}$); QPB 2025 Figure 4 |
| `L_gz_steady_state_residual` | mm `[mm]` | Median absolute deviation of `L_gz_estimate` measured at multiple sliding-window timepoints (recommended: at $t = $ 4, 8, 12 hr into the run, with trail truncated at each). Steady-traveling-wave hypothesis (QPB 2025) predicts the apex-frame peak position is time-invariant. **Threshold:** residual $< 0.2 \times$ `L_gz_estimate` is steady; larger residual flags hypothesis violation and reduces confidence in `L_gz_estimate`. Adds ~3 spatial-CWT calls per plant. | This doc §6.6 (steady-state assumption); QPB 2025 §"Steady propagation"; falsification check |
| `L_gz_resolvable` | bool `[—]` | True iff `L_gz_estimate` $\geq N_{\min} \cdot v \cdot \Delta t$ with $N_{\min} = 5$. False values mean the growth zone spans too few trail frames for reliable peak detection — see §6.4. | This doc §6.4 |
| `L_c_estimate` | mm `[mm]` | Exponential decay length of $|\kappa(s)|$ basal of peak: fit $|\kappa(s)| = A\exp(-s'/L_c)$ for $s' = s - L_{gz}$, $s' > 0$ | Meroz 2026 Eq. 5 (steady-state shape); Eq. 6 ($L_c = \gamma/\beta$) |
| `B_balance_number` | — `[—]` | $L_{gz} / L_c$ | Meroz 2026 Eq. 7; Bastien et al. 2013 cross-species baseline |
| `lambda_spatial_median` | mm `[mm]` | Median of dominant spatial wavelength from CWT of $\kappa(s)$ basal of $L_{gz}$, using `cgau2` mother wavelet (matches Rivière 2022) | Rivière 2022 §"Kinematics: fine elongation measurements"; QPB 2025 §"Spatial Fourier decomposition" |
| `traveling_wave_residual` | — `[—]` | $\|\lambda_\text{spatial} - v\,T_\text{nutation}\| / (v\,T_\text{nutation})$ | QPB 2025 §4.7 (this doc); falsifiable test of steady-traveling-wave hypothesis |
| `delta_dot_0_estimate` | hr⁻¹ `[mm-required-via-R]` | $\dot{\delta}_0 = \omega R\,\Delta\phi / (2\Delta L)$, with $\omega = 2\pi/T_\text{nutation}$, $R$ from external root-radius measurement, $\Delta L$ from spatial decay (= $L_c$ in linearization), $\Delta\phi$ from Tier 0 | Rivière 2022 Eq. 1 |
| `apex_basal_period_consistency` | — `[—]` | Whether spatial CWT shows uniform $\lambda(s)$ along trail; large variation = H1 violation | Bastien & Meroz 2016 §3.7; QPB 2025 §"Steady propagation" |

### 7.5 Tier 4 — Parametric fit (Phase 1 partial)

| Symbol | Units | Description | Source / anchor |
|---|---|---|---|
| `gamma_over_beta` | mm `[mm]` | $L_c$ alias, restated as $\gamma/\beta$ for explicit parametric reading | Meroz 2026 Eq. 6 |
| `beta` | h⁻¹ `[—]` | Tropic sensitivity. **Phase 2** — requires gravitropism stimulus experiment to disentangle from $\gamma$ | Meroz 2026 Eq. 1; Bastien et al. 2013 |
| `gamma` | h⁻¹ `[—]` | Proprioceptive sensitivity. **Phase 2** — same as above | Bastien & Meroz 2016 Eq. 23 |
| `theta_p` | rad `[—]` | Preferred (gravity) direction. **Phase 2** | Meroz 2026 Eq. 1 |

### 7.6 QC tier (runs unconditionally)

**Cross-tier ownership note (revised with PR #3):** the ``growth_axis_unreliable`` flag (see §7.1) is emitted by **BOTH** Tier 0 and the QC tier. The two emissions are element-wise equal as ``bool`` dtype by construction, because both tiers compute the flag via the same gate formula on the same inputs through the shared helper ``sleap_roots.circumnutation._noise.compute_sg_residual_xy`` after applying the same ``to_numpy(dtype=float)`` cast. PR #14 pipeline composition may coalesce or drop one column (either choice is safe because they are equal). This revises PR #2's earlier wording that QC SHALL NOT re-emit; the change is documented in PR #3 design.md D5, with the canonical formula now living in ``openspec/specs/circumnutation/spec.md`` Requirement: Growth-axis reliability gate.

**Methodological note on noise estimators.** The two tracking-noise traits (`sg_residual_xy` and `d2_noise_xy`) are standard signal-processing noise estimators, not pipeline-novel inventions. SG-residual estimation appears in Press et al. *Numerical Recipes* and is standard in time-series analysis; the second-difference variance method derives from the textbook noise-propagation rule $\text{Var}(\Delta^2 x) = 6\sigma^2$ for white noise. Neither is the *canonical* method in plant kinematics literature — most root-tracking papers (e.g., KymoRod, Bastien et al. 2016 *Plant J.* 88:468) report a single empirically-estimated tracking accuracy without formal characterization, and Rivière 2022 doesn't formally characterize tip noise. The closest formal-noise-characterization tradition is **single-particle tracking (SPT)** in cell biology — Berglund 2010 (*Phys. Rev. E* 82:011917) for camera-based localization uncertainty and Michalet 2010 (*Phys. Rev. E* 82:041914) for the MSD-extrapolation method. Two-estimator agreement (`sg_d2_agreement` near 1) is a stronger validation than either method alone, which is why the pipeline computes both. **For publication-grade defensibility, MSD extrapolation could be added as a third independent estimator** — see implementation note below the table.

| Symbol | Units | Description | Source / anchor |
|---|---|---|---|
| `sg_residual_xy` | px `[—]` | Std of residuals after Savitzky-Golay (degree 3, window 5) smoothing of $(x,y)$, summed in quadrature | Press et al. *Numerical Recipes* (any ed.); standard signal processing |
| `d2_noise_xy` | px `[—]` | $\text{std}(\Delta^2 x)/\sqrt{6}$ in quadrature with $y$, where $\Delta^2 x_t = x_{t+1} - 2x_t + x_{t-1}$. Independent estimator from white-noise propagation $\text{Var}(\Delta^2 x) = 6\sigma^2$. | Standard finite-difference noise propagation |
| `sg_d2_agreement` | — `[—]` | $\max(\text{sg}, \text{d2}) / \min(\text{sg}, \text{d2})$. Should be $\le 1.5$ for clean tracks. Two-estimator agreement is stronger validation than either alone. | This doc §1 reference data shows 1.46 |
| `frac_outlier_steps` | — `[—]` | Fraction of per-frame total steps $>$ 2 × median | This doc §1 reference (0.87% on plate 001) |
| `worst_step_ratio` | — `[—]` | Max per-frame step / median step | Reference 2.5 on plate 001 |
| `track_is_clean` | bool `[—]` | All of: `sg_d2_agreement < 1.5`, `frac_outlier_steps < 0.05`, `worst_step_ratio < 5` | Composite QC flag |
| `coi_fraction_t1` | — `[—]` | Fraction of CWT scaleogram inside cone of influence (Tier 1). >50% → unreliable. | Standard CWT practice (Torrence & Compo 1998 *Bull. Amer. Meteor. Soc.* 79:61) |
| `coi_fraction_t3` | — `[—]` | Same for spatial CWT (Tier 3) | Same |
| `is_nutating` | bool `[—]` | `band_power_ratio > 3 × noise_floor_estimate` | Phase 1 sanity check |

*(Note: an earlier draft of this table included `calibration_present` as a QC trait. The pure-pixel pipeline decision in §2.3 removes calibration from the pipeline entirely, so this trait was dropped. If downstream code records calibration provenance, that lives in the run-metadata sidecar's `_constants_snapshot`/`calibration_source` fields the user populates when calling `convert_to_mm()`, not as a pipeline-emitted QC trait.)*

**Optional Phase 1+ trait — `msd_noise_xy`.** A third noise estimator from the MSD-extrapolation method standard in single-particle tracking biology. Compute the mean squared displacement of detrended residuals at small lag $\tau$:

$$\text{MSD}(\tau) = \langle (x(t+\tau) - x(t))^2 + (y(t+\tau) - y(t))^2 \rangle$$

For pure i.i.d. noise on a stationary signal, $\text{MSD}(\tau \to 0) = 4\sigma^2$ (in 2D), so $\sigma = \sqrt{\text{MSD}(\tau \to 0)/4}$. Implementation: detrend with the same SG filter, then compute MSD at lag 1 frame. Adds a third independent method whose assumptions differ from both SG and d2 — useful if you ever need stronger publication defensibility against a reviewer who knows the SPT literature. **Not recommended for Phase 1 unless requested** — the existing two-estimator agreement at 1.46 already establishes data quality, and a third estimator is incremental rather than structural.

### 7.7 Per-genotype aggregation

The above are per-track. Per-plant aggregation is median across the 6 tracks of the same plant. Per-genotype aggregation is median ± IQR across plants, with explicit `n_plants_passing_qc` count. Plants where `track_is_clean = False` for all tracks are excluded from aggregation but flagged in the trait CSV with reason.

---

## 8. Validation strategy

Three independent test layers, each catching a different failure mode.

**Layer 1 — Synthetic data with known parameters.** Generate a tip trajectory by integrating Eq. 4.3 forward with chosen $L_{gz}$, $\Delta L$, $\dot{\delta}_0$, $\dot{\varepsilon}_0$, $\omega$, $R$. Apply Gaussian noise at $\sigma = 2$ px to the resulting $(x, y)$. Run the full pipeline. Recovered $T_\text{nutation}$, $L_{gz}$, $L_c$ (= $\Delta L$ in the linear regime), handedness, and $\dot{\delta}_0$ should match inputs within tolerances (recommend $\pm 5\%$ for $T$, $\pm 15\%$ for spatial quantities given the noise level). This catches algorithmic bugs.

**Layer 2 — Regression test against Derr's pilot.** Run Tier 1 on the exact tip coordinate Derr used (5-min cadence, 14-hr run from the original CMTN dataset). The output scaleogram should match the PNG oracle in band location and intensity within visual tolerance; the Fourier peak should match $T = 3333$ s within $\pm 2\%$. This catches drift from the established pilot output.

**Layer 3 — Cross-tier consistency.** $T_\text{nutation}$ (Tier 1) and $T_\text{psig}$ (Tier 2) should agree within $\pm 5\%$ on plates where H1 holds. $\lambda_\text{spatial}$ (Tier 3) and $v \cdot T_\text{nutation}$ should agree within $\pm 10\%$ if the steady-traveling-wave hypothesis holds. Disagreement is *itself* a result, not a bug — but the pipeline should flag it explicitly via `traveling_wave_residual` rather than silently passing.

Validation runs as part of CI; any layer failing fails the build.

---

## 9. Appendix A — Cell-wall mechanism (out of pipeline scope)

[Rivière 2022, §"Asymmetric rigidity and cell wall composition in the bending zone", Figure 4] — included for context only; not reproducible from tip-tracking data.

In the Averrhoa rachis bending zone, AFM mapping of transverse rachis sections shows the inner (concave) face has cell walls 17% softer on average than the outer (convex) face (3 biological replicates, std 6%). Multitarget immunolabelling reveals that this mechanical asymmetry coincides with biochemical asymmetry: highly methylesterified homogalacturonan (HG, antibody LM20) and low-methylesterified HG (antibody 2F4) are both ~5–13% more abundant on the outer (growing) face than the inner. Other cell-wall components (crystalline cellulose, amorphous cellulose, mannans, xyloglucans) show no statistically significant asymmetry [Rivière 2022, §"Asymmetric rigidity..." final paragraph and Figure 4D].

The interpretation [Rivière 2022, §"Correlation between growth and cell wall properties"] is that HG demethylesterification triggers cell-wall softening and expansion (consistent with the Peaucelle lab's nanofilament expansion model — Haas et al. 2020, *Science* 367:1003), and this process must cycle on the nutation period (~30 minutes for a half-cycle). This is *fast cell-wall remodeling*, not the static-state assumption.

**Why this matters for the pipeline (and why it doesn't):** the bending zone identified by `L_gz_estimate` corresponds, in the Averrhoa picture, to a region of asymmetric HG distribution and reduced wall elasticity on the bending side. If a future biology collaborator wants to validate this on rice roots, the predicted assay is: section the root at the pipeline-identified `L_gz` distance from the tip during a known phase of the nutation cycle, and probe with LM19/LM20 antibodies for HG methylation state asymmetry, plus AFM for elasticity asymmetry. **Pipeline outputs the target zone; pipeline does not measure cell-wall properties.**

---

## 10. Appendix B — Corrections from prior conversation drafts

The conversation that produced this doc contained two interpretive errors that should not propagate to the implementing session. Recording them here so the implementing Claude inherits the corrected understanding.

**(1) The "2:1 mode ratio" along the rachis is not two oscillators at different positions.** Earlier framing suggested QPB found a 2:1 ratio between basal $T_\text{nutation}$ and apical $T_\text{projection}$ representing two distinct physical oscillators. Reading the actual paper [Rivière 2022 §"The elongation profile in the growth zone is compatible with local contractions", Figure 3]: the two periods $\tau_f \approx 2.1$ h (basal) and $\tau_{2f} \approx 1.2$ h (apical) appear in the wavelet decomposition of *side-view-projected apparent elongation rate* $\dot{\varepsilon}(s_a, t)$, and the model demonstrates that the apical-end frequency-doubling is a projection artifact that appears only when local contractions occur (i.e. when $\dot{\delta}_0 > \dot{\varepsilon}_0/2$ in the model). It is **diagnostic of contraction**, not of two separate oscillators. The pipeline's top-view tip data does not have this projection structure; we cannot reproduce or look for this specific 2:1 signature directly.

**(2) Bastien-Douady-Moulia (2013, 2014) vs. Bastien-Meroz (2016).** The earlier conversation initially attributed the curvature-with-proprioception equation $\partial\kappa/\partial t = -\beta\sin(\theta - \theta_p) - \gamma\kappa$ to "the Bastien-Meroz framework" and discussed fitting nutation traits to it. That equation is a **tropism** equation from Bastien-Douady-Moulia, restated in [Meroz 2026, Eq. 1] as such. The Bastien-Meroz 2016 paper extends it to nutation by adding the perpendicular-curvature equation [Eq. 14] and an oscillating $\psi_g$ — that is the actual nutation extension, and equations 20–21 from BM2016 are what give us the load-bearing $\psi_g(t) = \arctan(\dot{x}_a/\dot{y}_a)$ tip-only extraction. Phase 1 of the pipeline emits $L_c$ (= $\gamma/\beta$) only; full $(\beta, \gamma, \theta_p)$ identification is Phase 2 and requires a gravitropism stimulus experiment.

**(3) Tier 2 §7.3 trait definitions corrected during PR #7 implementation.** Three §7.3 entries were revised (originals preserved here for provenance / reproducibility of any analysis citing the prior definitions):

- **`handedness` is COI-free.** Original: *"Sign of mean $d\psi_g/dt$ over COI-masked range; $+1$ = counterclockwise (left-handed in image frame), $-1$ = clockwise."* Corrected: sign of the **net unwrapped $\psi_g$ rotation over all finite frames** (no COI mask). The COI is a CWT-edge-reliability concept derived from the SG-detrended ridge; COI-masking would (a) couple a raw kinematic sign to the conditioned signal and the CWT min-length floor, and (b) the per-frame `ridge.in_coi` interior is non-contiguous, so an endpoint difference across a masked gap can report the wrong sign. A raw `atan2`-velocity displacement has no edge contamination; §7.3 itself omits COI for the sibling `delta_E`, and `angular_amplitude` (§7.1) is a COI-free raw-angle precedent. The sign is anchored on the $d\psi_g/dt$ sign, not the frame-ambiguous word "counterclockwise" (which only holds in the y-down image sense).
- **`delta_E_amplitude_proxy` is px/frame, not px·hr⁻¹.** Original units px·hr⁻¹ ("× (frames/hr)"). Corrected to **px/frame** (drops the cadence factor) to match Tier 0's cadence-independent velocity convention (`px/s` is also absent from the pipeline unit vocabulary). As a per-track amplitude proxy the prefactor-/cadence-free magnitude preserves the shape and relative magnitude of $\Delta\dot{E}$.
- **`T_psig_median` is emitted in seconds (`T_psig_median_s`).** Original units "hr". Corrected to seconds for consistency with Tier 1 `T_nutation_median` (a future `psig_long_consistency` correlation then needs no unit conversion).

Additionally, the §6.3 Tier 2 conditioning line was corrected from "Pre-smoothed via Savitzky-Golay" to **SG-detrended** (the residual = raw − SG-smooth): the residual is the oscillation component a period-extracting CWT requires (smoothing-only would retain gravitropic drift that biases $T_\text{psig}$), and reuses the exact `_noise.compute_sg_detrended` primitive Tier 1 uses. Full rationale + the 3-round review reconciliation: `docs/superpowers/specs/2026-06-05-add-circumnutation-tier2-psi-g-design.md` §13 and the OpenSpec change `add-circumnutation-tier2-psi-g`.

---

## 11. References

Bastien, R., Bohr, T., Moulia, B., Douady, S. (2013). Unifying model of shoot gravitropism reveals proprioception as a central feature of posture control in plants. *PNAS* 110(2):755–760.

Bastien, R., Douady, S., Moulia, B. (2014). A unifying modeling of plant shoot gravitropism with an explicit account of the effects of growth. *Frontiers in Plant Science* 5:136.

Bastien, R., Meroz, Y. (2016). The kinematics of plant nutation reveals a simple relation between curvature and the orientation of differential growth. *PLoS Computational Biology* 12(12):e1005238. arXiv:1603.00459. → **§3, §6, §10 cite specific equations.**

Bastien, R., Guayasamín, O., Douady, S., Moulia, B. (2016). KymoRod: a method for automated kinematic analysis of rod-shaped plant organs. *The Plant Journal* 88(3):468–475. → **§6.6 multi-node midline tracking method (Phase 2+ upgrade); §7.6 noise-method context for plant kinematics.**

Berglund, A.J. (2010). Statistics of camera-based single-particle tracking. *Physical Review E* 82:011917. → **§7.6 SPT localization-uncertainty methodology.**

Haas, K.T., Wightman, R., Meyerowitz, E.M., Peaucelle, A. (2020). Pectin homogalacturonan nanofilament expansion drives morphogenesis in plant epidermal cells. *Science* 367(6481):1003–1007. → **§9 mechanism.**

Iijima, M., Kato, J. (2007). Combined soil physical stress of soil drying, anaerobiosis and mechanical impedance to seedling root growth of four crop species. *Plant Production Science* 10(4):451–459. → **§6.4 rice $L_{gz}$ estimate.** ⚠ **PROVISIONAL CITATION — NOT VERIFIED.** This paper was selected from memory as a plausible source for "rice elongation zone ~1–5 mm" but I have not confirmed the paper actually contains this measurement. **Action required:** either confirm the value is in this paper, substitute a verified rice-specific paper (rice expert consultation pending), or replace with explicit `# TODO measure on representative plate` in §6.4 and remove this reference. Beemster & Baskin 1998 (*Plant Physiology* 116:1515–1526) gives ~0.7 mm for *Arabidopsis* primary root and is the canonical Arabidopsis-elongation-zone paper, but is not rice-specific.

Meroz, Y. (2026). Physics of computation and behavior in plants. arXiv:2604.21763 [cond-mat.other]. → **§5 cites Eqs. 1, 5, 6, 7, and §6.2.**

Michalet, X. (2010). Mean square displacement analysis of single-particle trajectories with localization error: Brownian motion in an isotropic medium. *Physical Review E* 82:041914. → **§7.6 MSD-extrapolation noise-estimation method (optional Phase 1+ trait).**

Nguyen, C., Dromi, I., Kempinski, A., Gall, G.E.C., Peleg, O., Meroz, Y. (2024). Noisy circumnutations facilitate self-organized shade avoidance in sunflowers. *Phys. Rev. X* 14:031027. → **§5.4 stochastic-circumnutation framing.**

Press, W.H., Teukolsky, S.A., Vetterling, W.T., Flannery, B.P. (any edition). *Numerical Recipes: The Art of Scientific Computing.* Cambridge University Press. → **§7.6 Savitzky-Golay residual noise estimation; standard reference.**

Rivière, M., Peaucelle, A., Derr, J., Douady, S. (2022). Spatiotemporal growth pattern during plant nutation implies fast dynamics for cell wall mechanics and chemistry: a multiscale study in *Averrhoa carambola*. bioRxiv 2022.02.22.481493. → **§4 cites Eqs. 1–5 and Figs. 2, 3, 4.**

Rivière, M., Peaucelle, A., Derr, J., Douady, S., Marmottant, P. (2025). Plant nutation relies on steady propagation of spatially asymmetric growth pattern. *Quantitative Plant Biology* 6:e10013. doi:10.1017/qpb.2025.10013. → **§4.7 traveling-wave duality.**

Torrence, C., Compo, G.P. (1998). A practical guide to wavelet analysis. *Bulletin of the American Meteorological Society* 79(1):61–78. → **§7.6 cone-of-influence and CWT methodology; standard reference.**

---

**End of theory.md.** Implementing session: read top-to-bottom before writing the first line of code. Trait list in §7 is the contract; everything else exists to justify it.
