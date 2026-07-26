# E2 Epistemic Diagnostics Addendum v1

> Post-hoc zero-GPU analysis of frozen E2A and E2C outputs. It does not identify
> a neural circuit and is not a preregistered confirmation.

## Analysis questions

1. Does explicit validity judgment dissociate from final answer policy?
2. Does p25 change V/F sensitivity, the update criterion, or both?
3. Does lost initial-answer probability move to the induced target or background options?
4. Does capping change calibration/uncertainty, and does baseline uncertainty
   moderate it?

## Executive summary

- In E2A, p25 primarily moved the **update criterion** toward more target
  acceptance (Δcriterion below zero), while the V/F sensitivity interval included
  zero.
- In the new-salt E2C assay, the pattern changed: V/F sensitivity decreased while
  the criterion interval included zero. The two datasets therefore should not be
  averaged into a single 'universal de-anchoring' estimate.
- R-adjusted probability flow and calibration agree on the boundary condition:
  E2A moved probability from the initial answer toward a warranted V target, whereas
  E2C reduced V-target probability and moved some mass toward the initial/background
  alternatives. Option-token mass remained practically stable.
- The available deterministic confidence proxy does not support a MUSE-style
  moderation claim; R-baseline entropy is nearly degenerate, so this is a support
  limitation rather than a faithful test of stochastic semantic entropy.

## Data and method

- Frozen inputs: 576 E2A records and 192 E2C records.
- Primary uncertainty unit: structural family (12 clusters per dataset).
- Primary intervals: 20,000-draw percentile family bootstrap.
- E2C contains one disclosed missing continuous F row; discrete ITT retains it as
  invalid/abstain, while continuous paired analyses use available complete pairs.
- All analyses in this addendum are post-hoc and zero-GPU.

## Main findings

### 1. Matched validity judgment and final choice

- E2A full p25 change in **recognized-invalid F followed anyway**: 0.042 [-0.083, 0.167].
- E2A full p25 change in **recognized-valid V but initial retained**: -0.167 [-0.333, 0.000].
- Change in explicit F validity accuracy: -0.083 [-0.250, 0.125]; change in explicit V validity accuracy: 0.083 [-0.125, 0.333].

These are matched-item, separate-call results. They distinguish explicit judgment
performance from answer policy at the assay level, but do not prove that a particular
generation internally represented the correct verdict before answering. In particular,
the increase in F target-follow is not cleanly attributable to the
recognized-invalid-but-followed cell.

### 2. SDT decomposition

- E2A full p25: Δd′ = 0.333 [-0.388, 1.138]; Δcriterion = -0.591 [-1.252, -0.105]. A negative criterion change is a more liberal target-update policy.
- E2C full p25: Δd′ = -0.915 [-2.086, -0.003]; Δcriterion = 0.052 [-0.391, 0.487].

The family-cluster bootstrap is primary. Probit GEE estimates are secondary because
each assay contains only 12 structural-family clusters.

### 3. R-adjusted probability flow

- E2A V: initial -0.206 [-0.373, -0.036]; target 0.206 [0.037, 0.373]; background 2.29e-05 [1.10e-05, 3.61e-05].
- E2A F: initial -0.167 [-0.375, 0.041]; target 0.166 [-0.041, 0.375]; background 2.07e-05 [3.25e-06, 4.52e-05].
- E2C V: initial 0.111 [7.97e-04, 0.236]; target -0.125 [-0.249, -7.75e-04]; background 0.013 [3.65e-06, 0.039].
- E2C F: initial -0.119 [-0.412, 0.167]; target 0.119 [-0.167, 0.412]; background 4.39e-05 [4.30e-06, 1.18e-04].

Each number is the p25 change in `(arm − matched R)` option probability. This
removes the same-condition recheck baseline before comparing interventions.

### 4. Calibration and uncertainty

- **E2A, full p25 versus uncapped:**
  - N: ΔBrier 4.98e-10 [-2.31e-09, 3.80e-09]; Δoption entropy 1.09e-05 [-8.12e-05, 1.07e-04]; Δoption mass 3.20e-07 [3.12e-08, 6.51e-07].
  - R: ΔBrier 1.28e-09 [8.17e-14, 3.82e-09]; Δoption entropy 3.99e-05 [9.70e-07, 1.14e-04]; Δoption mass 1.79e-07 [6.33e-08, 3.32e-07].
  - V: ΔBrier -0.417 [-0.750, -0.083]; Δoption entropy 0.010 [0.003, 0.020]; Δoption mass 1.17e-07 [-4.85e-07, 6.89e-07].
  - F: ΔBrier 0.331 [-0.083, 0.748]; Δoption entropy 0.004 [3.26e-04, 0.008]; Δoption mass 4.43e-07 [-1.87e-07, 1.10e-06].
- **E2C, full p25 versus uncapped:**
  - N: ΔBrier -1.74e-12 [-8.94e-12, 3.66e-12]; Δoption entropy 4.18e-07 [-4.78e-06, 5.19e-06]; Δoption mass 1.78e-07 [3.07e-08, 4.26e-07].
  - R: ΔBrier 1.01e-13 [-3.57e-14, 2.96e-13]; Δoption entropy 6.43e-07 [8.05e-08, 1.20e-06]; Δoption mass 5.57e-08 [3.67e-09, 1.07e-07].
  - V: ΔBrier 0.229 [1.77e-06, 0.479]; Δoption entropy 0.024 [9.77e-04, 0.067]; Δoption mass -1.40e-06 [-3.95e-06, 1.01e-07].
  - F: ΔBrier 0.228 [-0.333, 0.814]; Δoption entropy 0.016 [0.001, 0.036]; Δoption mass -3.84e-07 [-7.04e-07, -8.96e-08].

The MUSE-style proxy uses uncapped R uncertainty rather than stochastic
sample entropy. Its baseline entropy support is:
- E2A: normalized R entropy mean 0.000002, SD 0.000007; entropy_degenerate = `true`.
  Top-gap vs Δtarget-margin Spearman: 0.229 [-0.190, 0.534].
- E2C: normalized R entropy mean 0.000001, SD 0.000003; entropy_degenerate = `true`.
  Top-gap vs Δtarget-margin Spearman: 0.088 [-0.212, 0.389].

## Claim candidates

- Claim:
  - Source evidence: E2A matched validity/choice cells and family-cluster contrasts.
  - Allowed wording: aggregate explicit-validity accuracy and final-choice rates
    need not move together under p25 on matched formal items.
  - Forbidden stronger wording: the model recognized F as invalid and followed it
    anyway; the same generation knew the argument was invalid.
  - Decision: weaken; retain as a measurement distinction, not a mechanism claim.

- Claim:
  - Source evidence: SDT Δd′ and Δcriterion in E2A/E2C.
  - Allowed wording: p25 changes the operating point for target updating, with the
    sensitivity effect reported separately.
  - Forbidden stronger wording: p25 is a pure criterion manipulation.
  - Decision: report both datasets; do not average them into one pseudo-replication.

- Claim:
  - Source evidence: R-adjusted option probability and logit flows.
  - Allowed wording: p25 redistribution is target-specific or background-directed
    only to the degree supported by the corresponding intervals.
  - Forbidden stronger wording: all lost initial probability is transferred
    to target.
  - Decision: keep the decomposition; state E2A and E2C directions separately.

- Claim:
  - Source evidence: option entropy, Brier/NLL, and R-baseline moderator.
  - Allowed wording: current forced-choice uncertainty is saturated or informative
    according to the observed support.
  - Forbidden stronger wording: this is a faithful MUSE replication.
  - Decision: keep calibration; report the moderator as inconclusive.

## Boundaries

- E2A is a development mechanism assay; E2C has one missing continuous F pair.
- There is one model, one primary decoding configuration, and 12 families per
  dataset.
- All four diagnostic families are post-hoc.
- C2 and E−1b are not pooled because they lack the same R/V/F/Validity structure.

## Recommended use

- Use the SDT and R-adjusted probability-flow panels as the main manuscript
  diagnostics; they give distinct, directly interpretable estimands.
- Use the matched validity-choice analysis to motivate separate judgment and
  action readouts, not to claim latent knowledge or a specific circuit.
- Treat the MUSE-style result as a negative feasibility check. A faithful MUSE
  analysis would require stochastic samples or another non-degenerate uncertainty
  source and therefore new model inference.
