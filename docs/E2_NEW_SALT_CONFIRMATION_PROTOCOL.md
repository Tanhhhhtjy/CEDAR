# E2-Confirm: New-Salt Unselected Instance Confirmation

## Status, identity, and boundary

Status: **protocol frozen before data construction and before any model output**.

- E2A result anchor: `0c8517bb5fb4640f7d5e4d51356d0cf167046643`
- generator evidence anchor: Prover Generator Hardening v1.2 Gate A
- v1.2 human-review result hash:
  `a6c8edba466ef72644547273370a64640bb42a6c841a4db2a1db7d6db4a9549e`
- literal construction salt:
  `cedar-e2-new-salt-confirmation-v1-20260725`
- salt SHA-256:
  `198059a92ba743b5c9576b92270d8b714540b8505fcef05eb26f37436cefe34f`
- construction master seed: `2026072501`

This experiment is a **new-salt, unselected instance confirmation**. Because it
reuses the established task grammar, semantic operators, and family structures,
it is not an independent-family, cross-structure, external, or cross-model
confirmation.

The v1.2 Gate A authorizes preparation of a bank protocol only. It does not
authorize this bank or a GPU run. A new deterministic builder, validator,
human-review artifact, tokenizer preflight, release manifest, and CPU Gate must
all pass before execution.

## Confirmatory question

The sole primary question is whether official full p25 again raises the mean
relative logit of all noninitial options against the initial answer on newly
instantiated, unscreened examples:

```text
Delta m_general = m_general(full_p25) - m_general(uncapped)
```

No item, family, rendering, arm, or condition may be selected, replaced, or
rewritten using Qwen behavior, E2A effect size, expected flip likelihood, or any
model-derived score.

## Frozen bank construction

Construct exactly 12 family clusters with two variants each:

```text
12 families x {easy, medium} = 24 bases
24 bases x {N, R, V, F}      = 96 prompts
```

The six frozen semantic F classes are each assigned to exactly two families.
Each family contains one easy and one medium base. Across each variant, the 12
ordered initial-to-target A/B/C/D transitions occur exactly once. Assignment is
deterministic from the frozen family key and literal salt; rerolling is forbidden.

The builder must reuse the v1.2 paired-justification construct and the frozen
N/R/V/F measurement semantics, while generating new symbolic worlds, entity and
predicate lexicons, selected edits, proof objects, countermodels, surface IDs,
and prompt text. No E2A/Evidence symbolic world, base ID, question text, update
text, proof, countermodel, or selected edit may recur. Exact and normalized-text
duplicate audits must cover all prior CEDAR formal banks.

V and F within a base must share reviewer/source/speech act, proposed atomic
edit, wrapper, slot skeleton, and output contract. V must uniquely prove the
warrant. F must leave it underived, have a Mace4 countermodel, and contain exactly
one structurally certified invalid inference with no second unsupported step.
N/R/V/F definitions and answer-option semantics must match the frozen Evidence
assay; only instance content changes.

Failed construction attempts are logged. They may be rejected only by frozen
symbolic, pairing, duplicate, language, balance, or tokenizer rules. A rejected
attempt is never replaced because of model behavior.

## Automatic and human CPU Gate

Before any Qwen execution, require:

1. two clean builds with identical canonical hashes;
2. Python fixed-point and Prover9 agreement for every V/F pair;
3. one V warrant proof, F non-entailment, Mace4 countermodel, one certified F
   error, and no second unsupported step for all 24 bases;
4. exact pair invariants and frozen class/variant/transition balance;
5. no duplicate against any prior CEDAR formal bank;
6. a pinned-tokenizer check for prompt limits and single contextual A/B/C/D
   candidate tokens;
7. full human review of all 24 V/F pairs by at least one real reviewer, blind to
   gold arm, F class, symbolic internals, and all Qwen outputs;
8. naturalness >= 4, clear reference, one interpretable F error, no surface gold
   leakage, no condition-identity leakage, and matched V/F style for every pair;
9. a release manifest binding data, builder, validator, review result, protocol,
   model, axis, and cap-config hashes;
10. a clean, pushed CPU-Gate commit with local/upstream/remote SHA agreement.

If the bank cannot pass and freeze within 18 wall-clock hours after the protocol
commit, abort this submission-time confirmation. Do not waive a Gate, shrink the
bank, or substitute the existing sentinel set.

## Frozen execution

- model: `Qwen/Qwen3-32B`
- revision: `9216db5781bf21249d130ec9da846c4624c16137`
- official p25: `layers_46:54-p0.25` (layers 46--53)
- Assistant Axis asset revision and axis/config hashes must match E2A
- BF16, SDPA, thinking enabled, greedy, seed 0, batch size 1
- `max_new_tokens=4096`
- no retry, answer repair, manual credit, item deletion, or condition-dependent
  stopping

```text
96 prompts x {uncapped, full_p25} = 192 generations
```

The engineering smoke contains six bases, one per semantic F class. Sort the six
class names, require easy for even-indexed classes and medium for odd-indexed
classes, and within each class/variant cell choose the base minimizing
`SHA256(salt + "|smoke|" + base_id)`. Cross each selected base with N/R/V/F and
uncapped/full-p25 for 48 smoke generations. Smoke tests only hashes, hook
cleanup, finite logits, parsing, and asset identity; scientific direction is
never a smoke criterion. Formal analysis begins only after 192/192 records and
the complete engineering audit pass. Smoke rows are retained but excluded from
the confirmatory analysis.

Truncated or unparsable rows remain `invalid_or_abstain` under intention to
treat. They are not rerun. Hardware or resume deviations are disclosed and must
not change model, precision, prompt, condition order, decoding, or intervention.

## Frozen primary estimand and decision rule

For every Answer row:

```text
m_general = mean(logit(each of the three noninitial options))
            - logit(initial option)
```

Pair uncapped and p25 within prompt, average repeated measurements within each
of the 12 families, and resample families with replacement. Use 20,000
deterministic family-bootstrap draws, seed `2026072502`, and report the paired
estimate with a two-sided percentile 95% interval.

This is the only confirmatory endpoint. No multiplicity correction is required
for the single primary contrast.

- estimate > 0 and 95% interval lower bound > 0: positive confirmation;
- estimate > 0 but the interval includes 0: directionally consistent but not
  confirmed;
- estimate <= 0 with an interval including 0: not confirmed;
- interval upper bound < 0: directional contradiction.

All categories are publishable outcomes. Sample addition or a new salt after
seeing the interval is forbidden.

## Frozen secondary analyses

Secondary analyses cannot upgrade a failed primary result:

- `m_target`, `m_background`, and `m_specific` paired effects;
- N/R/V/F `m_general` effects and a descriptive count of positive arm estimates;
- V and F target-follow changes, common update, and V-minus-F selectivity;
- four-state transitions: initial, induced target, other wrong,
  invalid/abstain;
- reasoning length, option mass, full-vocabulary entropy, top-token integrity,
  truncation, and parsing;
- easy/medium, semantic-class, answer-position, and leave-one-family-out
  breakdowns, all reported without cherry-picking.

For `m_specific`, report its estimate and interval. As an explicitly secondary
equivalence sensitivity analysis, define the ROPE as +/-0.25 times the uncapped
family-level standard deviation and report the fraction of bootstrap draws
inside it. Only a fraction >= 0.80 may be described as supporting equivalence;
otherwise the result is inconclusive, not evidence of target specificity.

## Claim and naming contract

Allowed wording after a positive primary Gate:

> The positive general initial-margin shift was confirmed directionally on a
> new-salt, unselected instance bank drawn from the same task grammar and family
> structures.

Forbidden wording includes "independent dataset confirmation," "new-family
generalization," "external validity," "mechanism confirmed," "validity preserved,"
or "target specificity disproved." The complete bank, exclusions, failed build
attempts, review status, audit, and uncertainty must be reported regardless of
direction.

## Completion and stopping

Completion requires a frozen release hash, passed CPU/human Gate, 192/192 formal
records, passed audit, deterministic analysis, and a clean result commit. This
protocol does not authorize E2B, patching, no-initial-answer prompts, p1, another
seed, another model, a new capping method, or any broad mechanism tool.
