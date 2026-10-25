# Figure catalog

All figures generated using standard Matplotlib from saved numerical artifacts. PDF and PNG pairs. Local PNGs visually inspected; plots have zero references, labelled units, and family-level uncertainty. They are internal diagnostic figures, not a camera-ready packaging claim.

## 01_primary_endpoints
Purpose: display the four frozen tests without hiding nonsignificant outcomes.
Source: SUMMARY.json primary, independently verified at closure.
Caption: 12 family means; 20,000 bootstrap replicates; descriptive 95% intervals; frozen same-prompt scaling; exact sign flips/Holm4. Fixed trajectory.
Notice: all CIs cross zero; specific-common is a weak negative pattern, not significant support.
Decision: do not upgrade C1 to a mechanism result.
Check: common response is not full capping main effect; interaction is not raw V/F difference.

## 02_output_response_controls
Purpose: inspect whether the learned direction produces special output magnitude.
Source: details.json metrics; family means of absolute raw changes.
Caption: gray points are families, diamonds equal-family means, bars descriptive CIs; random vectors average within matched prompts; post hoc.
Notice: true and random response magnitudes overlap; full local swap is also small.
Decision: absolute output motion alone cannot validate a semantic direction.
Caveat: interval overlap is not an equivalence test; signed response and paired absolute diagnostics are reported separately. Replacement has much greater dose and is not a fair direction comparator.

## 03_family_heterogeneity
Purpose: distinguish mean cancellation from uniform effects.
Source: frozen primary family_means.
Caption: cells are family means, fixed symmetric color range; 24 bases in 12 families.
Notice: signs 6/6,6/6,3/9,6/6; specific-common LOFO stays negative.
Decision: retain the weaker hypothesis as uncertain; do not select families after seeing signs.
Caveat: families not independently certified novel structures; this map is not a new subgroup significance search.

## 04_local_swap_context
Purpose: relate position sensitivity to the existing full-capping difference.
Source: noop/raw record logits and local replacement.
Caption: only last prompt token at block54 input replaced; other tokens/cache and future condition persist; descriptive family CI.
Notice: whole-condition broad gap +2.0781 vs local P-in-U +0.0174 and U-in-P +0.0007.
Decision: this location/operation does not yield a strong bidirectional broad rescue.
Caveat: not a mediation fraction; differences in intervention scope preclude causal proportion claims; replacement has small secondary specific response described in report.
