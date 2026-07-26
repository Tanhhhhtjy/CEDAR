# E2 Epistemic Diagnostics Addendum v1

Status: **post-hoc explanatory analysis, frozen before implementation**

This addendum does not generate model outputs, alter prompts, select items, or modify any
pre-existing result. It analyzes two completed formal assays:

| Dataset | Rows | Scope | Frozen SHA-256 |
|---|---:|---|---|
| E2A phase-validity-choice | 576 | development mechanism assay | `5ba0116b51f677f72498fd5df4a46cae4e53084715cb9905150866ef6bb28358` |
| E2C new-salt confirmation | 192 | new-salt directional confirmation | `e9fc1c502b45d76e55432e5491d3a285f58a36ff373a65bb44f2164890383c8f` |

E2A contains 24 bases in 12 structural families, four Answer arms (`N/R/V/F`), a matched
Validity assay for `V/F`, and four conditions (`uncapped`, `prefill_only`, `decode_only`,
`full_p25`). E2C contains 24 new-salt bases in 12 structural families, four Answer arms, and
two conditions (`uncapped`, `full_p25`).

The family is the resampling and clustering unit. Bases and arms within a family are not
treated as independent replications.

## D1. Validity–choice gap

E2A Validity rows are matched to their Answer rows by
`(source_answer_id, condition) -> (answer.id, condition)`.

For `F`:

- **recognized-and-yielded**: explicit Validity verdict is correct (`insufficient`) while the
  final Answer selects the induced target;
- **misjudged-and-yielded**: explicit Validity verdict is incorrect while the final Answer
  selects the induced target.

For `V`:

- **recognized-and-updated**: explicit Validity verdict is correct (`sufficient`) and the
  final Answer selects the induced target;
- **recognized-but-refused**: explicit Validity verdict is correct while the final Answer
  remains at the initial answer.

All joint rates use the full 24-item arm denominator. Conditional rates among correctly judged
items are secondary. Contrasts are condition minus uncapped with a family-cluster bootstrap.

Because Validity and Answer are separate model calls, the analysis can support a
**matched-item judgment/action dissociation**, not the stronger claim that the same generation
internally represented the correct validity verdict before choosing.

## D2. SDT decomposition

For each dataset and cap condition:

- signal-present trials are `V`;
- signal-absent trials are `F`;
- a yes response is selecting the induced target;
- hit rate is `P(target | V)`;
- false-alarm rate is `P(target | F)`.

Hautus log-linear correction is applied:

`rate = (count + 0.5) / (n + 1)`.

The descriptive equal-variance SDT parameters are:

`d_prime = Z(hit) - Z(false_alarm)`

`criterion = -0.5 * (Z(hit) + Z(false_alarm))`

A lower criterion means a more liberal tendency to update to the target. Changes are
`cap - uncapped`. Uncertainty is estimated by resampling structural families with replacement.
A family-clustered probit GEE is reported as a convergent model, but the cluster bootstrap is
primary because each dataset contains only 12 families.

## D3. R-baseline probability flow

For every base, condition, and `V/F` arm, the matched `R` arm defines the no-new-evidence
reconsideration baseline. The analysis uses option-conditional probabilities over `A/B/C/D`.

For each option role (`initial`, `target`, `background`), evidence assimilation is:

`flow(arm, condition) = probability(arm, condition) - probability(R, condition)`.

The cap effect on assimilation is the difference-in-differences:

`[flow(arm, cap) - flow(arm, uncapped)]`.

The same contrast is computed in logit space:

- target-vs-initial;
- mean-background-vs-initial.

This distinguishes target-specific assimilation from probability diffusion toward unrelated
alternatives. Conditional option probabilities and full-vocabulary option mass are kept
separate.

## D4. Calibration, uncertainty, and MUSE-style moderation

For every complete Answer row:

- multiclass Brier score: `sum_k (p_k - y_k)^2`;
- negative log-likelihood: `-log p(gold)`;
- option-conditional entropy, normalized by `log(4)`;
- maximum option confidence;
- signed confidence gap: `confidence - correctness`;
- full-vocabulary option mass;
- full-vocabulary entropy.

Paired cap-minus-uncapped changes are reported by arm with family-cluster bootstrap intervals.
ECE is not a primary metric because 24 bases per arm do not support stable binning.

The MUSE-style moderator is explicitly a proxy rather than a reproduction of sample entropy.
Baseline uncertainty is the uncapped `R` option-conditional entropy for the same base. For
`V/F`, we test whether this baseline uncertainty moderates:

- the cap-induced change in target-vs-initial margin;
- the cap-induced change in target following.

The inferential moderator model is a family-clustered GEE. Family-bootstrap Spearman
correlations are reported descriptively. No value of zero entropy is interpreted as absolute
certainty.

## Missing data and multiplicity

E2A is expected to be complete. E2C contains one disclosed truncated `F::uncapped` row without
answer logits. Continuous paired analyses use complete pairs and report the missing ID; no
imputation or rerun is allowed. Discrete ITT outcomes retain the row as `invalid_or_abstain`.

The four diagnostic families are post-hoc and explanatory. Exact estimates and 95% intervals
are primary; Holm correction is applied within each family of related confirmatory contrasts
where p-values are available. Exploratory subgroups are labeled as such.

## Allowed and forbidden claims

Allowed if supported:

- p25 shifts the update criterion with limited change in V/F sensitivity;
- explicit validity judgment and final answer policy dissociate on matched items;
- lost initial-answer probability moves preferentially toward the target or toward background
  alternatives;
- uncertainty moderates or fails to moderate the cap effect in this assay.

Forbidden from this addendum alone:

- the model “knew internally” that an argument was invalid;
- a neural circuit has been identified;
- MUSE has been replicated;
- any result generalizes beyond Qwen3-32B and these assays;
- post-hoc intervals constitute preregistered confirmation.
