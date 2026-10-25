# CEDAR Prover-Generator Feasibility Protocol

Status: frozen exploratory protocol  
Research anchor: `364af5f797ee1eccf21c99e67570f59cbe052e5b`  
Branch: `agent/prover-generator-feasibility`  
Master seed: `20260725`  
Access/freeze date: `2026-07-23`

## 1. Research question

This candidate-only study compares two independently implemented construction styles:

- `provergen_style`: construct a proof graph for the requested labels first, then realize it
  as a finite symbolic world.
- `reviseqa_style`: construct a valid initial world first, then search its permitted atomic
  one-record neighbors for the requested revision.

The study asks which style is better suited to stable generation of strictly paired,
difficulty-controlled, automatically recomputable CEDAR N/R/V/F formal-domain items. It is
not a reproduction of either upstream pipeline and its outputs are not a formal held-out set.

## 2. Prohibited actions and hard stop

This study must not run Qwen, use a GPU, run capping, create an S arm, select items using model
behavior, modify C2/V8 or another existing experiment, or promote candidates into a held-out
bank. Failed attempts remain in route output. The study ends after the route comparison and
decision report.

Work stops before generation if a critical upstream source cannot be inspected, its intended
use is not licensed, its required data license is unresolved, method and implementation are
materially contradictory, or useful inspection requires a broad upstream rewrite. An upstream
source without a declared code license may be studied as a published method, but its code and
data must not be copied or incorporated.

## 3. Upstream-use boundary

The two source repositories are evidence for construction patterns only. No upstream data,
generated examples, prompts, or source code are copied into CEDAR. The `provergen_style` and
`reviseqa_style` names are deliberately lowercase style labels; neither result is called an
official reproduction.

Both styles use one external Prover9 executable only as a subprocess. The executable remains
outside the repository and is neither vendored nor committed. The exact executable version and
hash are recorded in the manifest. The independent Python evaluator shares no parsing,
inference, or subprocess code with the Prover9 adapter.

## 4. Unified logic semantics

The common language has finite named constants, function-free Horn rules, explicit negation,
and open-world semantics. Failure to derive a literal never licenses its negation. A world is
closed only under its stated Horn rules.

For query `q`, let `opposite(q)` be the same ground atom with its explicit-negation sign
reversed. Labels are:

- `true`: `q` is derivable and `opposite(q)` is not;
- `false`: `opposite(q)` is derivable and `q` is not;
- `unknown`: neither is derivable;
- `invalid`: both are derivable.

An invalid world is a failed candidate. It cannot enter valid output.

Every label is computed independently by:

1. Prover9 Dec-2007, invoked separately for the query and its opposite; and
2. a Python least-fixed-point Horn evaluator over signed ground atoms.

Disagreement is a hard record failure. No manual gold override exists.

## 5. Shared family and specification contract

Eight new family blueprints each have one easy and one medium variant, yielding sixteen common
symbolic specifications. Both routes must attempt all sixteen. The structures jointly cover:
fact replacement; fact withdrawal; rule replacement; rule withdrawal/undercut; conjunction;
multi-hop relations; two candidate support paths; and a local evidence change with a global
query consequence.

Across the sixteen specifications, each of the six directed transitions appears at least twice:
`true_to_false`, `false_to_true`, `true_to_unknown`, `unknown_to_true`,
`false_to_unknown`, and `unknown_to_false`.

Easy specifications require a decisive-state shortest proof depth of 2--3 and 2--4 distractor
records. Medium specifications require depth 4--6, 6--10 distractors, and at least one extra
state-reconstruction step. For a transition with an unknown endpoint, proof depth refers to the
unique proof of the determinate endpoint. Easy and medium variants share a structure signature;
only depth, distractor volume, constants, and parameters differ.

## 6. N/R/V/F symbolic arms

- N is an exact world-preserving neutral continuation. It contains no new proposition and its
  future naturalization must not say `No new information is provided.`
- R leaves the world unchanged and requests only a fresh check. It contains no target mention,
  evidence, authority cue, or social pressure.
- V adds, deletes, or replaces exactly one complete fact or rule record. Replacement stores
  `before`, `after`, and `edit_count: 1`. Initial and updated worlds must be consistent, have
  unique labels, and have different labels. Removing V's edit must eliminate the updated label,
  and no undisclosed support path may independently establish the updated target.
- F leaves the initial world unchanged while presenting a candidate argument for exactly the
  target established by V. Exactly one step is invalid. All other steps must be entailed.
  Each F stores its fatal step, error class, countermodel or opposite proof, a repair witness,
  and an explicit check that no second failure point exists.

The four arms are symbolic state/argument records. No model-facing prompt is generated in the
formal comparison.

## 7. Fairness and search

Both routes receive identical common specifications, schema, solver adapter, independent
evaluator, difficulty checks, validity checks, failure taxonomy, and a maximum of 100 candidate
worlds per specification. Item seeds are derived as:

`stable_hash(20260725, route, family_id, variant)`.

Route-specific generation is the only intended independent variable. A common family amendment
requires a written contradiction or design-error explanation, retention of old results, and a
full rebuild of both routes.

## 8. Sentinels and freeze gate

Before the sixteen formal attempts, both routes must pass two unscored toy sentinels: a fact
replacement and a rule withdrawal. Each sentinel must pass dual label validation, V atomicity,
F countermodel/opposite-proof validation, and deterministic rebuild.

The sentinel gate freezes canonical family-card, schema, builder, validator, and solver-version
identifiers. A failed sentinel stops formal generation.

## 9. Determinism and timing

Canonical JSON uses parsed values, sorted keys, UTF-8, LF newlines, no absolute paths, and compact
JSONL serialization. `canonical_content_sha256` is the cross-platform hard gate;
`raw_file_sha256` is provenance only.

Route records store a conservative deterministic wall-clock bucket rather than a fabricated
exact duration. Exact observed elapsed times are recorded once in the experiment ledger and are
not used to select or regenerate candidates. Two complete logical builds must have identical
canonical hashes. Timing never affects a route's valid/failed status.

## 10. Minimal naturalization

After all symbolic hard gates pass, four representative bases (fact/rule edit by easy/medium)
are rendered once per route with deterministic English templates, for eight unscored records.
No external LLM may polish or rewrite them. The audit checks gold leakage, reference clarity,
single-error preservation, template burden, and unambiguous reverse mapping.

## 11. Evaluation and decision

A valid record requires zero solver-label error, 100% agreement between validation paths,
atomic V, a verified single-error F, and deterministic rebuild. Valid yield is interpreted as:

- 15--16/16: pass;
- 13--14/16: boundary;
- 0--12/16: unsuitable for direct scaling.

At least seven of eight families must have medium proof depth greater than easy proof depth.
After hard gates, route scores use: yield/gold 35, V/F pairing 25, difficulty 15,
determinism 10, integration/maintenance 10, and naturalization 5. A hard-gate failure cannot be
outscored. A difference below five points, or complementary strengths, yields
`human_checkpoint / hybrid_candidate`. The only permitted decisions are ProverGen-style,
ReviseQA-style, hybrid, or neither.

## 12. Completion boundary

Completion means all contracted artifacts, tests, route attempts, failure records, comparison
metrics, hashes, and the decision report have been committed and pushed. The study then stops;
it does not extend the bank, add S, run a model, generate held-out items, or begin capping.
