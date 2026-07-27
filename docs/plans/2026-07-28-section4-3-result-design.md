# Section 4.3 Result-Writing Design

## Scope

Write only Section 4.3, `Warranted and Unwarranted Updating Both Increase`, in synchronized English and Chinese. The subsection is confined to the Qwen3-32B E2A development assay. New-salt, natural-task, and cross-model evidence remain in Sections 4.4--4.5.

## Narrative structure

1. Connect the broad continuous displacement in Section 4.2 to unchanged selected answers in the `N/R` controls.
2. Report the realized target-following changes in `V/F` and their common update component.
3. Report the imprecise `V-F` selectivity change, bound the post-hoc decision-boundary account, and transition to transport evidence in Section 4.4.

The subsection follows the chain `latent continuous displacement -> realized warranted and unwarranted updating -> selectivity inference`.

## Frozen evidence

- In `N` and `R`, all 48 capped--uncapped pairs retain the supplied initial answer despite positive arm-level changes in `m_general`.
- `V` target following: `70.83% -> 91.67%`, a change of `+20.83pp`.
- `F` target following: `29.17% -> 45.83%`, a change of `+16.67pp`.
- Mean change across `V/F`: `+18.75pp`, 95% family-cluster bootstrap CI `[+4.17,+33.33]`.
- `V-F` selectivity change: `+4.17pp`, 95% family-cluster bootstrap CI `[-20.83,+29.17]`.
- The post-hoc decision-boundary analysis reaches Gate B only: some migrations are consistent with thresholded realization near a decision boundary, but the pattern is not uniform across datasets.

## Claim boundaries

- `N/R` show that a continuous displacement can remain latent at the selected answer; they are not unaffected in the continuous readout.
- Both warranted and unwarranted target following increase in E2A. Reporting either arm alone would misstate the result.
- The selectivity interval does not establish equivalence, zero effect, or the absence of reason sensitivity.
- The boundary analysis is post hoc and source-dependent. It cannot provide a unified causal explanation or mediation claim.
- E2A remains a development/mechanism assay, not held-out confirmation.

## Presentation

- Use three compact paragraphs and reference Figure 2B only for the discrete `V/F` result.
- Avoid introducing `CommonUpdate` as an undefined manuscript term; describe it as the mean of the two arm effects.
- Use direct result language and avoid repetitive `not X but Y`, slogan-like contrasts, and internal gate terminology in the manuscript.
- Route aggregate replication and heterogeneous behavioral allocation to Section 4.4.
