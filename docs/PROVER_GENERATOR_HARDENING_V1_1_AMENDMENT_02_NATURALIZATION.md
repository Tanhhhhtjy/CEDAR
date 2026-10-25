# Prover Generator Hardening V1.1 Amendment 02: Naturalization

Status: frozen before any real human review and before any model run  
Scope: natural-language renderer, review fields, and human-review finalizer only  
Protected anchor: `06fd18863503734c32c754defa411172c2756297`

## 1. Timing and preservation

This amendment is made before a real reviewer has completed the workbook and before any Qwen,
GPU, capping, S-arm, or other model run. Commit `06fd18863503734c32c754defa411172c2756297`
and its original review package remain permanently available in Git history. The original files
are not overwritten; the revised package uses `_v2` filenames.

Original package bindings:

```text
naturalization_review12_bilingual.xlsx raw SHA-256:
aa574ae7f4f2670165b522b265ab74f94995edb10b6399afb646dff9778526f2

original workbook canonical-content SHA-256:
78755a2da9595f62e50d15ae4ae6bdb39b103b7ca75830f38caad0ad17f3cd20

original naturalization canonical SHA-256:
27c81b69b60d500169f1b630a68917450642c2169fe63d1e146d3371dfd56b06
```

## 2. Reasons for revision

The pre-review audit found four renderer/protocol defects:

1. all 12 F renderings omitted the grammatically required phrase `the claim that`;
2. some complete-record replacement sentences were syntactically awkward;
3. V and F used fixed condition-specific wording that could become a superficial identity cue;
4. the protocol did not freeze a quantitative, item-level human passing rule.

These are naturalization and review-governance defects. They are not symbolic, solver, gold,
atomic-edit, semantic-F, or candidate-enumeration defects.

## 3. Permitted changes

Amendment 02 may change only:

- the deterministic natural-language renderer;
- the frozen V/F wrapper-template set and its hash-only assignment;
- automatic condition-identity leakage diagnostics;
- visible reviewer fields and the bilingual review guide;
- a read-only completed-workbook importer and deterministic human-gate finalizer;
- v2 naturalization artifacts, tests, report bindings, and manifest entries.

It must not change any of the 48 symbolic records, initial or updated worlds, queries, gold labels,
selected V edits, F certificates, candidate traces or ranks, solver proofs, countermodels, family
blueprints, seeds, or formal validation results. The canonical SHA-256 of
`hybrid_stress48.jsonl` is frozen as:

```text
9199c8259f53467d5cc10903dcdbb1a2260d74703e5fcfd0a6281ec84deff1ab
```

Any protected-artifact hash change invalidates this renderer-only amendment.

## 4. Renderer contract

- Every embedded F proposition uses `treats the claim that {claim} as established`.
- Every complete-record or rule replacement uses quotation marks in the form
  `replace the record ‘{before}’ with ‘{after}’`.
- At least three frozen equivalent wrapper pairs are used.
- A deterministic hash selects the wrapper. Selection cannot read family, difficulty, F class,
  answer transition, gold, or model behavior.
- Each pair shares its review scene, opening, closing, and authentication language.
- `authenticated`, if present, occurs in the shared scene rather than only in V.
- V states that the registry directly provides the formal revision. F states that a reviewer
  proposes the same revision from one invalid inference.
- The renderer must not add `No new information`, `which may be wrong`, or an equivalent hint.

## 5. Automatic condition-leakage report

The v2 automatic report must include:

- fixed token n-grams present in every V and no F;
- fixed token n-grams present in every F and no V;
- every item-level V/F length ratio;
- wrapper counts by family, difficulty, and F class;
- evidence that wrapper selection used only the frozen source hash.

This diagnostic does not replace human judgment. The workbook adds
`condition_identity_leakage`, defined as whether fixed wording alone, without analyzing the logic,
makes V versus F easy to identify.

## 6. Frozen human passing rule

Every one of the 12 rows must satisfy all six conditions:

```text
language_naturalness >= 4
clear_reference = yes
single_interpretable_error = yes
surface_gold_leakage = no
condition_identity_leakage = no
v_f_style_matched = yes
```

One failed row makes the result Gate C. Means, totals, or majority votes cannot override an item
failure. Gate C permits renderer revision only and does not permit symbolic edits.

## 7. Human finalizer contract

The finalizer opens a completed workbook read-only and requires all 12 rows to have every required
rating. It compares the completed file with the frozen blank v2 template and rejects changes to
IDs, source hashes, English or Chinese prompts, V/F text, sheet structure, and frozen metadata.
Only rating cells and notes may differ. The agent must not populate human ratings.

For an intact, complete workbook, the finalizer writes `human_review_result.json` and updates the
final gate deterministically:

- all 12 rows pass: Gate A, `step_1_complete`, development-bank protocol may be prepared;
- any row fails: Gate C, `naturalization_failed`, renderer revision only.

## 8. Stop boundary

After generating and verifying the blank v2 review package, work stops at real human review. This
amendment does not authorize Qwen, GPU, capping, S, development-bank expansion, held-out creation,
or agent-authored review ratings.
