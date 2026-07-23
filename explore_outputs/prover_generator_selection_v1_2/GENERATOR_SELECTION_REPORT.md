# Step 2 Generator Selection Report

## Decision

Select the hybrid architecture:

```text
proof graph construction
-> route-neutral one-record neighborhood enumeration
-> atomic edit certification
-> independent Python and Prover9 validation
-> operator-specific F validation and countermodel
```

Implementation name: `proof_graph_plus_atomic_edit_certification`.

## Why this decision is allowed

- The original v1 route ranking is retired as `inconclusive`; it is not selection evidence.
- Proof-graph construction remains the controllable interface for initial-world structure, unique gold, proof depth, and support-path design.
- Atomic-neighbor enumeration remains the auditable interface for add/delete/replace edits and for proving that exactly one transition-producing edit exists.
- The combined implementation passed 48/48 fresh Phase 1.1 attempts with zero gold errors, 100% unique atomic V, 100% one-error F validation, six semantic F operators, and deterministic rebuilding.
- Phase 1.2 then passed 12/12 paired automatic checks, 12/12 independent validation, and 12/12 item-level human naturalization review.

## Decision-rule mapping

The evidence matches the planned hybrid branch:

- proof-first world construction is stable and directly controls formal structure;
- edit discovery/certification benefits from a separate ReviseQA-style atomic neighborhood;
- neither v1 standalone route has valid evidence of clear superiority;
- the integrated architecture has already passed a stronger 48-attempt hardening test.

## Scope

This is an engineering selection for development-bank construction. It is not an official reproduction of ProverGen or ReviseQA, a route-performance result, a model-behavior result, or a claim about capping.

## Authorization

Step 2 is complete. The next permitted action is to freeze the 48-base development-bank protocol with the C2/V8-derived design requirements. No GPU is required or authorized at this point.
