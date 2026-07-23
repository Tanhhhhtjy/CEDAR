# Comparability Report

The comparison is route-internal and controlled: both styles receive the same common
specifications, semantics, Prover9 executable, independent evaluator, search budget, schema,
difficulty contract, validation gates, and scoring. Construction order is the intended variable.

It is not directly comparable to published ProverGen generation yield because CEDAR uses a
smaller Horn fragment, no upstream data assets, and no LLM naturalization. It is not directly
comparable to ReviseQA's reported behavior because CEDAR replaces stochastic, multi-record LLM
edits with deterministic one-record search and adds an F single-error contract.

The external prover is Dec-2007 rather than the ProverGen README's recommended 2009-11A. This
version deviation applies identically to both routes and is acceptable for the shared
function-free Horn fragment, but prevents a claim of strict upstream solver reproduction.
