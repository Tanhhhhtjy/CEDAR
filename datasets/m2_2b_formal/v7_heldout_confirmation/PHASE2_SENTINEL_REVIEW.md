# C1/V7 Phase 2 sentinel construction review

Status: `awaiting_codex_sentinel_review`

Scope: four frozen, zero-GPU construction sentinels only. No remaining base, arm, blind-review packet, Qwen output, or capping output was generated.

## Summary

| base | family / variant | difficulty | migration | original gold | V gold | original steps | updated steps | unique changed node |
|---|---|---|---|---|---|---:|---:|---|
| C1V7-001 | C1V7-F01 / 1 | easy | A to B | 25 points | 18 points | 2 | 2 | Delta's active score record |
| C1V7-021 | C1V7-F11 / 1 | easy | B to A | fully contained | partial overlap | 2 | 2 | uncertainty half-width record |
| C1V7-033 | C1V7-F17 / 1 | easy | B to A | Omar gets Ruby | Omar gets Quartz | 3 | 3 | Nia's complete allowed-item record |
| C1V7-048 | C1V7-F24 / 2 | medium | C to B | `{R}` | `{Q}` | 5 | 5 | one functional-dependency record |

Answer texts were solved before their A/B/C positions were assigned. `C1V7-001` follows the protocol's A-to-B example; the other positions remain compatible with the eventual six-migration balance and do not alter any task logic.

## C1V7-001 — median after central-record correction

### Original recomputation

1. Sort the five active scores: `[12, 18, 25, 31, 44]`.
2. The frozen median rank is the third position, which contains `25`.

Unique original answer: A, `25 points`.

### V recomputation

V replaces only Delta's score record, `31 -> 16`. The roster and all other scores remain fixed.

1. Re-sort: `[12, 16, 18, 25, 44]`.
2. The third position now contains `18`.

Unique updated answer: B, `18 points`.

### Option verdicts

| option | before V | after V | reason |
|---|---|---|---|
| A: 25 | supported | contradicted | third before V; fourth after V |
| B: 18 | contradicted | supported | second before V; third after V |
| C: 22 | contradicted | contradicted | absent from both exact rosters |

Changed node count: one. External knowledge: none. Family alignment: exact order-statistic median with a rank-crossing record correction; no mean, weighting, or tie-break.

Ambiguity review: no ties occur and the median rank is stated explicitly. No repair was needed after solution-first construction.

## C1V7-021 — tolerance-interval containment

### Original recomputation

1. Construct `[50 - 1, 50 + 1] = [49, 51]`.
2. Both endpoints are inside the closed acceptance band `[48, 52]`.

Unique original answer: B, fully contained.

### V recomputation

V replaces only the uncertainty half-width record, `1 -> 3`; the center and band remain fixed.

1. Construct `[50 - 3, 50 + 3] = [47, 53]`.
2. The intervals intersect, but `[47, 53]` is not contained in `[48, 52]`.

Unique updated answer: A, partial overlap.

### Option verdicts

| option | before V | after V | reason |
|---|---|---|---|
| A: partial overlap | contradicted | supported | original interval is contained; corrected interval intersects without containment |
| B: fully contained | supported | contradicted | corrected endpoints extend beyond both band endpoints |
| C: fully excluded | contradicted | contradicted | both intervals have a nonempty intersection |

Changed node count: one. External knowledge: none. Family alignment: the whole uncertainty interval is classified; V changes width, not the center or an endpoint-inclusion convention.

Ambiguity review: the prompt explicitly defines closed intervals and makes the three relations exhaustive. This removes endpoint and terminology ambiguity.

## C1V7-033 — unique bipartite assignment

### Original recomputation

1. Nia's complete record `{Quartz}` forces `Nia-Quartz`.
2. Omar's remaining allowed item is Ruby, forcing `Omar-Ruby`.
3. Pia then takes Sable, yielding the sole complete matching.

Unique original answer: B, Ruby.

Exhaustive permutation check: exactly one matching exists,
`{Nia-Quartz, Omar-Ruby, Pia-Sable}`.

### V recomputation

V atomically replaces Nia's complete row from `{Quartz}` to `{Sable}`. Omar's and Pia's rows, the item universe, and the one-to-one rule remain unchanged.

1. Nia's replacement row forces `Nia-Sable`.
2. Pia's remaining allowed item is Ruby, forcing `Pia-Ruby`.
3. Omar then takes Quartz, yielding the sole corrected matching.

Unique updated answer: A, Quartz.

Exhaustive permutation check: exactly one matching exists,
`{Nia-Sable, Omar-Quartz, Pia-Ruby}`.

### Option verdicts

| option | before V | after V | reason |
|---|---|---|---|
| A: Quartz | contradicted | supported | assigned to Nia before V and to Omar after V |
| B: Ruby | supported | contradicted | assigned to Omar before V and to Pia after V |
| C: Sable | contradicted | contradicted | assigned to Pia before V and to Nia after V |

Changed node count: one complete eligibility-record node, not one edge. External knowledge: none. Family alignment: both answers require a global bijection and complete rematching.

Ambiguity and repair record: the Phase 1.1 impossibility was repaired in the frozen family card. This instance does not regress to edge addition/deletion; it replaces one complete row and preserves every other row.

## C1V7-048 — functional-dependency key closure

### Original recomputation

Under `R -> Q`, `Q -> P`, `P -> S`, and `Q -> T`:

1. `{R}+` adds Q.
2. Q adds P and T.
3. P adds S.
4. Thus `{R}+ = {P,Q,R,S,T}`.
5. The empty proper subset is not a key. `{P}+ = {P,S}` and `{Q}+ = {P,Q,S,T}`, so both other options omit attributes.

Unique original answer: C, `{R}`.

### V recomputation

V replaces only the dependency record `R -> Q` with `Q -> R`; all other dependencies and attributes remain fixed.

1. `{Q}+` adds R.
2. Q adds P and T.
3. P adds S.
4. Thus `{Q}+ = {P,Q,R,S,T}`.
5. The empty proper subset is not a key. `{P}+ = {P,S}` and `{R}+ = {R}`, so both other options omit attributes.

Unique updated answer: B, `{Q}`.

An exhaustive check over every attribute subset confirms that `{R}` is the sole minimal key before V and `{Q}` is the sole minimal key after V.

### Option verdicts

| option | before V | after V | reason |
|---|---|---|---|
| A: `{P}` | contradicted | contradicted | closure is `{P,S}` in both states |
| B: `{Q}` | contradicted | supported | misses R before V; reaches all attributes after V |
| C: `{R}` | supported | contradicted | reaches all attributes before V; has no outgoing dependency after V |

Changed node count: one functional-dependency record. External knowledge: none. Family alignment: the result requires iterative set closure plus proper-subset minimality, not merely reading the direction of one implication.

Ambiguity review: the closure algorithm, stopping condition, attribute universe, and key-minimality definition are all stated in the prompt. Exhaustive enumeration ruled out unlisted alternative minimal keys.

## Gate assessment

Machine validation:

```text
command: python datasets/m2_2b_formal/v7_heldout_confirmation/validate_phase2_sentinel4.py
status: passed
sentinels: 4
qwen_or_gpu_execution: false
draft SHA-256: 31e7be88f15d1b47378b2bfdc1191fea6e9ac437b787773518ddd37b43a0b95a
```

All four sentinels satisfy:

- one uniquely supported option before V;
- one different uniquely supported option after V;
- one directly stated changed node;
- distractor contradicted before and after V;
- no external knowledge or unstated convention;
- easy/medium derivation-step bounds;
- certificate/rationale consistency;
- no authored N/AO/S treatment fields;
- no model-output-based selection.

These author checks are construction evidence, not independent human review and not approval to generate the remaining 44 bases.
