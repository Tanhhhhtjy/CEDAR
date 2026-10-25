# Prover-Generator Feasibility V1 Audit Correction

Status: corrective interpretation, frozen before v1.1 implementation  
Audited anchor: `4712b0d555ae2b07f73781c6a11b7caa660ef96a`  
Audit date: `2026-07-23`

## Corrected conclusion

V1 established that the shared formal infrastructure can represent finite signed Horn worlds,
compute labels independently with Python and Prover9, apply and audit one-record edits, preserve
counterevidence, and rebuild deterministically. These are useful infrastructure results.

V1 did **not** establish a scientific performance difference between two generator routes, did
not demonstrate blind discovery of valid edits, did not implement six semantically distinct F
errors, and did not establish model-ready natural-language quality. Its route-comparison result
is therefore corrected from a ranked/hybrid empirical result to:

```text
inconclusive
```

The hybrid architecture remains an approved engineering direction, not a scientific discovery
from the v1 comparison.

## Machine-confirmed audit findings

At the audited anchor, a direct row-by-row comparison of the two 16-record route files found:

- 16/16 `symbolic_world_initial` values identical;
- 16/16 `symbolic_world_updated` values identical;
- 16/16 `v_edit` values identical.

The route implementation explains this identity:

- the common template returns a precomputed `template["edit"]`;
- `reviseqa_style` builds its candidate list as `_wrong_edit_candidates(template) + [intended]`;
- the known intended edit is consequently guaranteed to occur in the candidate list.

This validates edit checking, but not edit discovery without answer leakage.

The route score also included non-empirical constants:

- `integration_score = 10.0` for ProverGen-style and `9.0` for ReviseQA-style;
- `naturalization_score = 5.0` for both routes.

The reported `100 vs 99` therefore must not be treated as measured generator performance and is
retired. Phase 1.1 must not compute a replacement route score.

The F audit found that all 32 formal route records used the same actual fatal-step kind:

```text
unsupported_atomic_state_claim
```

Although the records carried six different `error_class` strings and family-specific witnesses,
the operative invalid inference was always the assertion that the complete V edit already held.
V1 therefore demonstrated storage of error-class metadata, not six machine-validated semantic
fallacy operators.

Finally, v1 naturalization was a deterministic symbolic serialization smoke test. It retained
raw predicate-like language and was never reviewed as a usable model input. It cannot support a
natural-language quality claim.

## Claims that remain valid

The following v1 claims remain supported:

1. finite function-free Horn worlds with explicit negation can be evaluated under the intended
   open-world label scheme;
2. Python fixed-point and external Prover9 validation can agree on generated initial/updated
   worlds;
3. atomic add/delete/replace operations can be represented and audited;
4. opposite proofs and Mace4 countermodels can be preserved;
5. solver traces can be normalized without local paths or process telemetry;
6. canonical JSON/JSONL hashing supports deterministic rebuild checks.

## Corrective action

Phase 1.1 uses a single architecture named
`proof_graph_plus_atomic_edit_certification`. Its proof-graph builder may emit an initial world,
query, vocabulary, transition, and difficulty contract, but may not emit, retain, or transmit a
correct V edit. A separate enumerator must construct the permitted one-record neighborhood from
only those public inputs. A semantic validator must find exactly one successful edit and must
validate F structure independently of declared `error_class` strings.

All v1 files remain frozen. This document supersedes only the interpretation of v1; it does not
rewrite its data, metrics, reports, or commit history.
