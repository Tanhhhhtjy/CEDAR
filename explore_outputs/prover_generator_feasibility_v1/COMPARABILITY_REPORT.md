# Comparability Report

Both routes used the same 16 specifications, finite signed-Horn semantics, Prover9 executable,
Python evaluator, 100-candidate limit, schema, validity rules, and scoring. Construction order
was the intended variable. Canonical rebuild equality passed: `078c53d362abbf74d5702161ce5236ce6cbe2cb841cd1ff63ad3b1ad38f7a356`.

The result is not directly comparable to published ProverGen yield because no upstream assets,
rich FOL operators, or LLM translation were used. It is not directly comparable to ReviseQA
because its official stochastic multi-edit LLM proposal stage was replaced by deterministic
one-record neighbor search. Prover9 Dec-2007, rather than 2009-11A, is a shared compatibility
deviation. These boundaries prevent a strict-reproduction claim but preserve the internal route
comparison.
