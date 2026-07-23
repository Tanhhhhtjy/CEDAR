# CEDAR Prover-Generator Hardening V1.1 Protocol

Status: frozen exploratory protocol  
Research anchor: `4712b0d555ae2b07f73781c6a11b7caa660ef96a`  
Branch: `agent/prover-generator-hardening-v1-1`  
Architecture: `proof_graph_plus_atomic_edit_certification`  
Master seed: `20260726`

## 1. Objective and boundary

Phase 1.1 tests whether the v1 formal infrastructure can become a scalable candidate generator.
It must generate 48 fresh attempts without exposing a known edit to enumeration, enforce one
unique transition-producing edit, instantiate six genuinely different F inference operators,
and pass a minimum deterministic naturalization audit.

This is candidate-only development work. It is not a development bank, held-out set, model
evaluation, or evidence about capping. It must not add S, run Qwen, use a GPU, run capping, select
by model behavior, or promote the 48 stress records into a confirmation set.

The following v1 paths are immutable:

```text
datasets/m2_2b_formal/prover_generator_feasibility_v1/
explore_outputs/prover_generator_feasibility_v1/
docs/PROVER_GENERATOR_FEASIBILITY_PROTOCOL.md
```

All v1.1 artifacts use new `prover_generator_hardening_v1_1` paths.

## 2. Three-interface architecture

### 2.1 `proof_graph_builder`

Inputs are family blueprint, easy/medium variant, item seed, and desired transition. Outputs are
only an initial world, query, explicit opposite, route-neutral candidate vocabulary, desired
transition, permitted edit type, and formal difficulty contract.

The interface must not output, cache, log, attach, or secretly retain a correct V edit, updated
world, correct record location, or correct candidate rank. Family-specific graph constructors are
permitted, but all constants, predicates, depths, distractors, and parameters must derive from the
item seed rather than a seed-specific branch.

### 2.2 `atomic_edit_candidate_enumerator`

The enumerator sees only the builder's public output. It generates the permitted one-record
neighborhood from the initial world and route-neutral vocabulary. It may enumerate fact/rule
addition, deletion, or replacement only when the family contract permits that operation.

It must never accept an intended edit argument or append a known answer. Candidate ordering is a
deterministic shuffle derived from the item seed. Each item has at most 100 candidates; at least
90% of the 48 attempts must have at least 20 legal candidates. Every candidate operation and
validation result is retained. A valid item has exactly one candidate that realizes the desired
transition. Zero or multiple successes are failures, not occasions for manual selection.

### 2.3 `semantic_validator`

The validator computes every initial and candidate label using the independent Python fixed-point
evaluator. The unique selected candidate and both final states are then independently rerun with
Prover9. Invalid worlds, solver disagreements, wrong transitions, alternate successful edits,
or target proofs surviving removal of V fail the attempt.

## 3. Stress matrix

The formal run contains exactly:

```text
8 families × 2 variants × 3 replicates = 48 attempts
```

Item seeds are `stable_hash(20260726, family_id, variant, replicate_id)`. The eight v1 structure
families may be retained only as abstract blueprints. Concrete worlds, names, predicates,
parameters, distractors, and candidate order must be newly generated. Copying a v1 world and
renaming symbols is prohibited.

Easy requires shortest decisive proof depth 2--3 and 2--4 distractor records. Medium requires
depth 4--6, 6--10 distractors, and an explicit reconstruction step. Difficulty claims are limited
to formal proof depth. For every family, median medium depth must exceed median easy depth.

All 48 formal attempts remain in `hybrid_stress48.jsonl`, including failures. A formal-run bug
requires an amendment and retention of the failed run, followed by a full 48-attempt rebuild.
Individual repair, seed replacement, and post-run deletion are prohibited.

## 4. Semantic F operators

The generic v1 `unsupported_atomic_state_claim` is prohibited. Phase 1.1 implements exactly six
operator contracts, balanced at eight assigned attempts each:

- `cross_entity_transfer`;
- `converse_error`;
- `conjunction_weakening`;
- `unsupported_bridge`;
- `scope_transfer`;
- `support_path_generalization`.

Every F certificate stores entailed premises, one invalid inference with its premises and claimed
conclusion, valid downstream steps, induced target, `fatal_error_count: 1`, a structural semantic
certificate, countermodel/opposite proof, repair witness, and `contains_social_cue: false`.

The validator derives the operator class from structure. It must ignore the declared class while
classifying, then require derived and declared classes to agree. Each operator has independent
positive and negative tests. Adding a second unproved step must fail.

For every F:

1. pre-error premises are entailed;
2. the claimed conclusion is not entailed;
3. temporarily adding that conclusion makes all downstream steps valid;
4. the induced V target follows under that assumption;
5. removing the invalid inference removes the F target;
6. no second step is unsupported;
7. the true world is unchanged;
8. F and V targets are identical;
9. counterevidence exists.

## 5. F sentinel freeze

Before the stress run, six unscored sentinels, one per operator, must pass dual-solver gold,
operator-specific semantic validation, one-error validation, repair, counterevidence, no-social
checks, and deterministic rebuild.

The freeze records canonical family hash, F-operator source hash, candidate-enumerator source
hash, schema hash, validator hash, solver version/executable hash, and master seed. Formal stress
generation is prohibited until all six pass.

## 6. Formal hard gates

Any wrong gold causes `fundamental_failure`. Valid records require 100% Python/Prover9 agreement,
consistent initial and updated worlds, the contracted transition, one complete-record V selected
from general enumeration, exactly one successful candidate in the full neighborhood, target
dependence on V, operator-specific F validation, one fatal inference, counterevidence, and no
social/authority cues.

Yield interpretation:

- 44--48 valid: symbolic yield passes;
- 40--43 valid: boundary/human decision;
- below 40: generator failure.

In addition, every valid V and F check must be 100%, all six F operators need at least six valid
records, all eight family depth gradients must pass, and two clean complete builds must have
identical canonical hashes. Correctness gates cannot be relaxed to improve yield.

## 7. Naturalization and human checkpoint

Naturalization begins only after the symbolic gate passes. Twelve frozen records cover each F
operator at easy and medium difficulty. A deterministic renderer maps symbolic predicates to
neutral pseudo-domains without outside knowledge or per-item LLM rewriting.

Automated checks prohibit raw predicates, identifiers, variables, underscores, gold labels,
validity/error labels, `No new information is provided`, nested broken quotes, target mismatch,
and V/F length ratios outside 0.80--1.25. Reverse mapping must recover the source hashes.

The renderer also creates `naturalization_review12_bilingual.xlsx` and a Chinese guide. Review
rows contain full English tasks, Chinese assistance, V and F text, but no gold, F class, or
symbolic internals. The workbook fields are naturalness 1--5, reference clarity, single
interpretable error, surface leakage, V/F style match, and notes.

The generator cannot mark its own language review as passed. Until a real reviewer returns
labels, the only permitted passing-symbolic status is:

```text
symbolic_generator_passed
awaiting_human_naturalization_review
```

No kappa or independent review may be fabricated.

## 8. Determinism, manifest, and metrics

The canonical hard gate parses JSON, sorts keys, serializes UTF-8 with LF, excludes absolute
paths, and canonicalizes JSONL line by line. Raw hashes are provenance only. Two complete builds
must match file by file.

The manifest records this protocol's commit anchor, builder/validator/family/operator/solver
hashes, master seed, 48 input/output hashes, naturalization hash, all failures, and
`gpu_qwen_capping_run: false`.

Phase 1.1 reports yield, gold agreement, unique V rate, semantic F rate and class coverage, proof
depth, candidate-pool size, successful rank, deterministic rebuild, naturalization checks, human
status, and failure categories. It must not compute or repeat the v1 `100 vs 99` route score.

## 9. Final Gate and stop

- Gate A requires all symbolic gates, automatic naturalization, and real human approval; status
  `step_1_complete / may_prepare_development_bank_protocol`.
- Gate B means symbolic and automatic language gates pass but real review is absent; status
  `symbolic_generator_passed / awaiting_human_naturalization_review`.
- Gate C means symbolic generation passes but automatic or human naturalization fails; status
  `symbolic_generator_passed / naturalization_failed / renderer_revision_only`.
- Gate D covers yield below 40, any gold/F error, answer-dependent enumeration, non-unique V, or
  nondeterminism; status `generator_hardening_failed / do_not_expand_bank`.

After writing and pushing the final gate, work stops. It must not build a larger bank or held-out
set, add S, run Qwen/GPU/capping, adjust by model behavior, or reuse the 48 attempts as a
confirmation set.
