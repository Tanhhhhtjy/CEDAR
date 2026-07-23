# Prover Generator Hardening V1.1 Amendment 01

Status: frozen before the replacement 48-attempt rebuild  
Scope: F-operator assignment schedule only  
Discovery stage: deterministic naturalization sampling gate

## Preserved pre-amendment result

The first formal stress run completed 48/48 structurally valid attempts, with zero gold, unique-V,
or semantic-F errors. It is preserved in Git commit
`b4586f7cd601d3b7988ded56554fd8938c031734`. Its `hybrid_stress48.jsonl` canonical SHA-256 is:

```text
846fab323f4a06690361a140cf26ec948860886ff86efabaf028e27e51c82c5a
```

This result is superseded for Phase 1.1 gating, not deleted or reinterpreted.

## Defect

The original balanced schedule assigned classes 1--3 only to easy attempts and classes 4--6 only
to medium attempts. Overall coverage was eight records per class, but the frozen naturalization
contract requires one easy and one medium example for each of the six classes. The required
12-cell sample therefore could not be drawn.

This is a global experimental-schedule defect. It is not a solver, gold, atomicity, or semantic-F
failure and does not justify selecting substitute seeds or editing individual instances.

## Approved correction

Retain the master seed, all 48 item seeds, family blueprints, proof graphs, candidate enumeration,
operator implementations, validators, solver versions, difficulty contracts, and hard gates.
Change only the class schedule to:

```text
class_index = (family_index * 3 + variant_index * 3 + replicate_index) mod 6
```

This yields exactly four easy and four medium attempts per class, while retaining exactly eight
attempts per class overall. Because the F scaffold is part of each initial world, all 48 attempts
must be rebuilt and revalidated. No pre-amendment output may be mixed with the replacement run.

## Required rerun

The replacement run must repeat all 48 constructions, dual-solver validation, unique-neighborhood
checks, semantic-F checks, family difficulty gradients, and two clean deterministic rebuilds before
naturalization resumes.
