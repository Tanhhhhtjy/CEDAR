# Generator Comparison Report

## Outcome

Both independently implemented styles passed the formal hard gates. `provergen_style` scored
100.0; `reviseqa_style` scored 99.0. The
difference is below five points, so the protocol requires `human_checkpoint / hybrid_candidate` rather
than a forced winner.

## Contract questions

1. **How many items were generated?** ProverGen-style: 16/16 valid;
   ReviseQA-style: 16/16 valid. All 16 attempt records remain in each file.
2. **Did any incorrect gold occur?** No. Gold error counts were 0 and
   0; Python fixed point and Prover9 agreed for every initial/updated world.
3. **Which structures failed most often?** No formal structure failed. In the world-first route,
   irrelevant atomic neighbors were rejected mainly because they preserved the initial label.
4. **Was V atomicity stable?** Yes: 16/16 and
   16/16 records passed the complete one-record V/F pairing checks.
5. **Did F remain a single-error argument?** Yes. Each F has one fatal state claim, all remaining
   recomputation is valid under that claim, and each target has an opposite proof or Mace4 model.
6. **Did difficulty reflect proof depth?** Yes. Medium exceeded easy in
   8/8 and 8/8 families.
7. **Which route was more deterministic?** Both had identical two-build canonical hashes. The
   proof-first route reached every specification in one construction attempt; world-first search
   had median/p95 attempts 3.5/
   6.0.
8. **Which route needed more special handling?** ReviseQA-style needed explicit neighbor
   enumeration and rejection bookkeeping. ProverGen-style needed direct family proof-skeleton
   constructors. Naturalization used the same generic renderer for both.
9. **Is a hybrid recommended?** Yes, as a human checkpoint only: proof-first construction gives
   direct depth/path control, while independent atomic-neighbor search supplies a useful V-edit
   certificate. This comparison does not implement the hybrid.
10. **Can formal development-bank construction begin?** Technically feasible, but only after a
    human approves the hybrid candidate. These 32 route records remain exploratory and cannot be
    promoted into held-out data.

## Score table

| Dimension | Weight | provergen_style | reviseqa_style |
|---|---:|---:|---:|
| Valid yield and gold | 35 | 35.0 | 35.0 |
| V/F pairing | 25 | 25.0 | 25.0 |
| Difficulty control | 15 | 15.0 | 15.0 |
| Determinism | 10 | 10.0 | 10.0 |
| Integration/maintenance | 10 | 10.0 | 9.0 |
| Minimal naturalization | 5 | 5.0 | 5.0 |
| **Total** | **100** | **100.0** | **99.0** |

## Scope boundary

This result supports a generator-mechanism decision only. It is not an official ProverGen or
ReviseQA reproduction, a language-model benchmark result, or evidence about Assistant Axis
capping. No Qwen, GPU, S arm, behavior-based selection, or capping was used.
