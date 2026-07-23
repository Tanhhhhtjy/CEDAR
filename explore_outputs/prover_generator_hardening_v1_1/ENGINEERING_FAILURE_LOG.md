# Engineering Failure Log

New failures must be appended; prior entries must never be removed after a rebuild.

## EF-001: generated constant parsed as a Prover9 variable

- Stage: pre-freeze F sentinels; no formal stress attempt had begun.
- Symptom: the support-path-generalization sentinel had Python/Prover9 label agreement, but Mace4
  could not construct the expected countermodel for the unknown initial state.
- Root cause: a seeded entity such as `vek50` began with a Prover9 variable-prefix character and
  was normalized as a variable in the Mace4 clause set.
- General fix: all generated entity constants now use an explicit `c` prefix; generated graph
  nodes use the same convention. Predicates and true Horn variables are unchanged.
- Scientific effect: none. The seed, transition, proof structure, candidate algorithm, and gates
  are unchanged; this is a solver-language compatibility correction before sentinel freeze.
