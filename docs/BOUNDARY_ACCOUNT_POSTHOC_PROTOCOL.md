# CEDAR Boundary Account Post-hoc Protocol v1

## Status and claim boundary

Status: **frozen before running the boundary analysis**. The source model outputs
were already observed and are immutable.

> This is a post-hoc boundary diagnostic conducted after observing the E2A,
> E2C, C2, and E−1b results. It is not a preregistered confirmation and cannot
> establish a causal mechanism.

The analysis uses existing option logits and existing discrete outputs only. It
does not run Qwen, use a GPU, generate text, modify an upstream result, repair a
parse, rerun a truncation, impute a missing logit, delete an item, or redefine a
margin after inspecting the result.

## Frozen questions

The diagnostic separates two questions:

1. whether p25 produces a continuous option-space displacement; and
2. whether discrete changes concentrate among instances that were close to the
   uncapped decision boundary.

The primary pooled question is whether a larger uncapped top-1/top-2 option gap
predicts a lower probability of an intervention-associated discrete change.
All predictors in the primary model are measured under uncapped or direct
uncapped conditions.

## Frozen sources

All blobs are loaded with `git show COMMIT:PATH`, not from mutable working-tree
copies. Exact SHA-256 values are recorded in
`docs/BOUNDARY_ACCOUNT_ANALYSIS_PLAN.json`.

- E2A: commit `0c8517bb5fb4640f7d5e4d51356d0cf167046643`; Answer assay only;
  `uncapped` and `full_p25`; N/R/V/F; 96 prompt pairs.
- E2C: commit `ca729de0e6b55952266f1b32169b8fc2cd53005c`; 96 prompt pairs;
  `CEDAR-E2C-022-F::uncapped` remains truncated and missing logits.
- C2/V8: commit `8ea7025f2a726997d78ad8919455a8756e3c192e`; direct uncapped logits
  predict matched free-generation uncapped-to-p25 changes; AO and S are
  separate, with domain fixed strata and `normalized_stem_id` clusters.
- E−1b: commit `880aab2250eefc69f81c595dc0283555b91ca6b4`; original and swapped
  orderings are retained and clustered by `pair_id`; sample seed is a fixed
  stratum.

The C2 margin-choice CSV named in the task specification does **not** exist in
commit `8ea7025f`. It was accepted later at commit
`364af5f797ee1eccf21c99e67570f59cbe052e5b`. The raw direct/free records at the
requested C2 anchor remain the sole analytical inputs. The later CSV is pinned
only as a provenance cross-check and cannot replace or filter raw records.

## Input audit and stopping rule

Before estimation, the script writes `INPUT_AUDIT.json` and checks:

1. every pinned commit and blob is accessible and has the frozen SHA-256;
2. source runtime/delivery hashes agree where a manifest hash is available;
3. IDs and condition cells are unique and pair one-to-one;
4. prompt identity, base, arm, family/stem, initial, target, candidate-token
   mapping, and answer-position metadata are invariant within a pair;
5. E2A contains only the frozen Answer subset in the analytical table;
6. E2C has exactly one missing uncapped-logit cell, the frozen truncated item;
7. C2 direct/free matching uses the exact 480 shared IDs and preserves AO/S,
   domain, normalized stem, and token mappings;
8. E−1b has exactly two conditions for each `pair_id × variant`, with original
   and swapped treated as repeated observations;
9. retries/repairs remain zero where those fields exist;
10. logit argmax versus parsed/generated choice agreement is reported, not
    silently required to be perfect across different readouts;
11. thinking generation, free generation, direct forced readout, and E−1b
    forced choice remain distinct readout labels; and
12. no upstream source file changes during the analysis.

A failed source hash, major pairing invariant, candidate-token mapping, or
answer-position check stops estimation. Expected E2C missingness and
cross-readout C2 disagreement are reported limitations, not audit failures.

## Frozen variables

For uncapped option logits, let `i` be the initial option, `t` the target, and
`w` the option argmax:

```text
g_initial = logit(i) - max(logit(j) for j != i)
g_top     = largest option logit - second-largest option logit
g_target  = logit(w) - logit(t), evaluated for target-entry eligible rows

m_general  = mean(noninitial option logits) - logit(i)
m_specific = logit(t) - mean(other two noninitial option logits)
```

`g_top`, not `abs(m_general)`, is the primary distance for any flip.
`g_initial` is used for initial exit and `g_target` for target entry.

Choices are mapped into the mutually exclusive transitions:

```text
stable_initial, initial_to_target, initial_to_other,
target_to_initial, target_to_other, other_transition,
invalid_or_missing
```

The binary outcomes are `any_flip`, `initial_exit`, and `target_entry` exactly
as defined in the analysis plan. `initial_to_other` is never counted as target
follow.

For C2 main analyses, the predictor is from direct uncapped logits and the
outcome is the matched strict free-generation answer. Direct argmax-to-argmax
results are a deterministic boundary-consistency check only. For E−1b, the
two-option gap is algebraically close to the forced-choice outcome and is
reported as supporting geometry, not independent behavioral validation.

## Frozen statistical analysis

### Primary model (RQ1)

```text
logit P(any_flip) = stratum fixed intercept + beta * z(g_top)
```

The expected direction is `beta < 0`. Gap z-scores are computed once on the
full valid analysis set within `dataset × arm/readout` and held fixed through
bootstrap resampling. Strata are:

- E2A/E2C: dataset × arm;
- C2 free: arm × domain;
- E−1b: sample seed × original/swapped variant.

Fixed-effect strata with no outcome variation remain in descriptive tables but
are omitted from fixed-effect slope estimation because they contain no
within-stratum likelihood information. A source with fewer than five flips or
complete separation receives no standalone logistic significance claim.

### RQ2 and RQ3

RQ2 fits `initial_exit ~ z(g_initial) + stratum` only where the uncapped choice
is initial. E2A/E2C V/F are the focal data; zero-event N/R strata are shown but
not forced into a separated model.

RQ3 fits `target_entry ~ z(g_target) + stratum` only where uncapped is not the
target.

### RQ4

The secondary crossing model adds the observed post-intervention
`delta_m_general` for E2A/E2C and matched direct C2. Because this is a
post-intervention variable closely tied to the outcome, it is descriptive and
is never called mediation. E−1b instead reports the change in the signed lead
of the uncapped winner as an explicitly algebraic consistency check.

### Bootstrap and descriptive statistics

- 10,000 deterministic cluster-bootstrap draws; seed `20260726`.
- E2A/E2C source-specific clusters: `family_id`.
- Pooled E2 resampling couples matching F01–F12 structural-family slots across
  E2A and E2C as one upper cluster.
- C2 clusters: `normalized_stem_id`; domain remains a fixed stratum.
- E−1b clusters: `pair_id`, preserving original/swapped repetition.
- Pooled resampling samples E2 upper families, C2 stems, and E−1b pairs
  independently, then combines them.

Report slope, odds ratio, percentile 95% CI, valid bootstrap draw count, median
gap by flip status, median difference, AUC using `-gap`, and nearest-minus-
farthest quantile flip-rate difference. Four source-specific slope p-values,
when estimable, receive Holm correction; the single pooled diagnostic does not.

Quantiles use uncapped gaps only. They are assigned deterministically within
`dataset × readout`, with stable ID breaking ties. Use quartiles for at least 40
valid rows and halves otherwise. Raw counts are always reported.

## Gate

Gate A requires all six conditions specified in the task: a negative pooled
slope with CI below zero; at least three of four source point estimates in the
negative direction; no source CI fully supports the reverse; higher nearest-
than-farthest flip rate; directionally concordant C2 cross-readout evidence;
and stable leave-one-cluster-out direction.

Gate B is directional but incomplete or uncertain evidence. Gate C is no
support or material directional conflict. No gate permits causal-mechanism
wording.

## Outputs and stopping

The script writes the exact artifacts and PNG/PDF figures listed in the task,
using deterministic data ordering and metadata. It may not edit a paper. After
tests, Ruff, deterministic rerun, manifest verification, commit, and push, work
stops.
