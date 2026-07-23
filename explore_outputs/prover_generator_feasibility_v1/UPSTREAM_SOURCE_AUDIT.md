# Upstream Source Audit

Audit date: 2026-07-23  
CEDAR anchor: `364af5f797ee1eccf21c99e67570f59cbe052e5b`

## Decision

Proceed with independent style implementations. Do not run or copy either official generation
pipeline. The inspected repositories are sufficiently accessible to identify their construction
patterns, and no upstream dataset is needed. The study therefore does not rely on unresolved
dataset permissions.

## ProverGen

- URL: `https://github.com/opendatalab/ProverGen`
- Inspected commit: `1d8abd227912cee0b24819eb373ceba80979cb49`
- License: no `LICENSE`/`COPYING` file at this commit. The README says “open source” but does not
  grant a specific code license.
- Reuse decision: method-level scholarly reference only. No source, prompt, rule pool, names,
  WordNet data, generated examples, or training/evaluation data will be copied.
- Relevant inspected implementation: top-down goal-property sampling, proof-skeleton expansion,
  depth bands, distractor-rule generation, truth-table checks, and Prover9/NLTK integration in
  `logic_skeleton_generator.py`, `utils/logic_generator/{generator,rule_generator}.py`, and
  `utils/prover9_interface.py`.
- Official pipeline run: no. It would require Python 3.8-era pinned dependencies, NLTK plus a
  separately installed Prover9, repository rule/data assets, and LLM-backed translation for the
  later natural-language stages. Official execution is unnecessary for this style comparison.
- Issues: no declared repository code license; the Windows checkout displays mojibake for some
  logical symbols; the CLI uses `eval`; the Prover9 location is configured locally; and the
  official generator covers a richer FOL language than the unified Horn fragment used here.
- Patch classification: none. CEDAR does not patch or import this repository.

## ReviseQA

- URL: `https://github.com/ChadiHelwe/reviseqa`
- Inspected commit: `9a5778655a7a44793036cbe0a29e391ca8baa990`
- License: MIT (`LICENSE`).
- Reuse decision: method-level reference only, despite the permissive license. Independent code
  keeps the two routes comparable and prevents upstream LLM generation from entering the study.
- Relevant inspected implementation: KB extraction; positive/negative Prover9 checks; proof
  support extraction; repeated LLM proposals for `FLIP`, `UNCERTAIN`, and `INVARIANT` edits;
  solver-guided retry; edit-diff recording; and post-hoc FOL edit verification in
  `src/generate_reviseqa.py`, `src/prompt_engine.py`, `src/prover.py`, and
  `src/verification_fol.py`.
- Official pipeline run: no. It requires ProverGen-derived input, an OpenRouter API key,
  `deepseek/deepseek-chat-v3.1` at temperature 0.7, Instructor structured output, Prover9, and
  several unpinned dependencies.
- Issues: the repository contains a Windows-invalid result filename with `:`; Git objects and all
  critical code remain readable. The README presents systematic edits at a high level, while the
  implementation proposes non-simple multi-record edits through a stochastic external LLM. It
  does not natively enforce CEDAR one-record atomicity or a single-error F arm. This is a scope
  mismatch, not a material README/code contradiction.
- Patch classification: the failed Windows checkout is a platform packaging issue. No source
  patch was made; audit reads came directly from Git objects.

## Solver

Both routes require a real prover plus an independent Python fixed-point evaluator. No solver was
preinstalled. The official `LADR-2009-11A` source archive was accessible and declares GPL v2, but
this Windows host has no WSL distribution or C toolchain. The official Windows GUI distribution
was therefore installed silently into an external, repository-excluded directory solely to
obtain its bundled `prover9.exe` and `mace4.exe`.

Observed executable version: `Prover9 (32) version Dec-2007`. A bundled official sample returned
`THEOREM PROVED`. The executable is not copied into CEDAR and no PATH or project dependency is
changed. This older solver version is an engineering compatibility deviation, not a method
change; the compared fragment uses only finite function-free Horn formulas supported by both
versions. Exact hashes are frozen in the final manifest.

The legacy release uses `interpretation(...)` rather than a literal `MODEL FOUND` marker and
requires repeated direct quantifiers such as `all x all y all z`. These differences are handled
only in the route-neutral solver adapter. Fatal or unrecognized solver output is a hard failure,
never evidence of non-entailment.

## Hard-stop assessment

- Critical code inaccessible: no.
- Intended reuse prohibited by license: no, because ProverGen code/data reuse is explicitly zero.
- Required data permission unresolved: no required upstream data is used.
- Material method/code contradiction: no.
- Broad rewrite required: no; both candidate routes are clean-room style implementations.

The upstream gate passes with the named limitations above.
