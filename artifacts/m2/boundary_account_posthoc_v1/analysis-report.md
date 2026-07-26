# CEDAR Boundary Account post-hoc analysis

> Post-hoc diagnostic after observing all source results; not a preregistered 
> confirmation and not causal-mechanism evidence.

## Outcome

- Claim gate: **Gate B**
- Allowed wording: The results are consistent with a decision-boundary account, but the post-hoc evidence is not conclusive.
- Input audit: `passed`

## RQ1: uncapped top-gap and any flip

| Source | n | flips | slope | OR | 95% bootstrap CI (slope) | AUC(-gap) | status |
|---|---:|---:|---:|---:|---:|---:|---|
| E2A | 96 | 15 | 0.908 | 2.480 | [0.190, 2.891] | 0.622 | cluster_bootstrap |
| E2C | 95 | 14 | 0.415 | 1.514 | [-0.210, 1.361] | 0.633 | cluster_bootstrap |
| C2 | 480 | 1 | -1.939 | 0.144 | [-2.598, -1.539] | 0.438 | descriptive_only_fewer_than_5_flips |
| E-1b | 900 | 37 | -4.601 | 0.010 | [-6.541, -3.622] | 0.948 | cluster_bootstrap |
| Pooled | 1571 | 67 | -0.923 | 0.397 | [-1.457, -0.588] | 0.569 | cluster_bootstrap |

## What the evidence changes

- The pooled model tests whether pre-intervention proximity predicts later 
  discrete instability after holding source strata fixed.
- C2 is a deliberately separate cross-readout check. Its direct predictor 
  and free-generation outcome are not algebraically identical, but its event 
  count limits inference.
- E−1b is forced choice and therefore provides geometry consistency rather 
  than independent free-generation validation.
- Removing E−1b changes the pooled slope to 0.564; therefore the pooled negative association   is not evidence that the generative assays share the same boundary effect.

## Five required answers

1. **E2A:** No robust concentration after arm stratification. The frozen    fixed-stratum slope is 0.908    with CI [0.190, 2.891], in the    opposite direction, despite a favorable unadjusted AUC.
2. **E2C:** Not directionally consistent in the fixed-stratum model; slope    0.415, CI [-0.210, 1.361].
3. **C2:** Direct margins do not establish prediction of free-generation    flips: 1/480    events and AUC 0.438.
4. **E−1b:** The small-margin pattern is strong but primarily forced-choice    boundary consistency (AUC 0.948);    it cannot supply independent free-generation evidence.
5. **Wording:** Gate B permits only `consistent with`; `reconciles`    is not supported.

## Claim Candidates

- Claim:
  - Allowed wording: The results are consistent with a decision-boundary account, but the post-hoc evidence is not conclusive.
  - Source evidence: `BOUNDARY_ANALYSIS.json`, RQ1 and claim gate.
  - Forbidden stronger wording: the boundary account is a proven internal mechanism.
  - Uncertainty: post-hoc design, heterogeneous readouts, and low C2 free-flip count.
  - Decision: weaken
