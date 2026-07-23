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

## EF-002: globally balanced F schedule lacked within-difficulty coverage

- Stage: after the first committed 48-attempt run, at the frozen naturalization sampling gate.
- Symptom: overall F counts were exactly eight per class, but no medium source existed for the
  first three classes and no easy source existed for the last three.
- Root cause: the class-index formula reset at each family and coupled class groups to variant.
- Evidence preservation: commit `b4586f7cd601d3b7988ded56554fd8938c031734` and canonical
  stress hash `846fab323f4a06690361a140cf26ec948860886ff86efabaf028e27e51c82c5a`.
- General fix: Amendment 01 changes only the global class schedule, giving every class four easy
  and four medium attempts. All 48 records are rebuilt; no item or seed is selectively repaired.
- Scientific effect: F scaffolds change assignment across cells. Gold logic, desired transitions,
  candidate enumeration, family difficulty, solvers, and gates remain unchanged.
