# E-1a PhilPapers frozen manifests

This directory is a deterministic reconstruction of the public protocol in
`kelkarI/sycophancy-qwen` at commit
`270889c1a876a2c9ac384a6e4a1273f027e3e349`.

Important: the paper's seeds 42, 7, and 123 select three different sets of 300
base questions. `philpapers600.jsonl` is the seed-42 convenience manifest only.
The full three-seed reproduction must use all three `philpapers600_seed*.jsonl`
files (1,800 rows before the 50/50 tune/test split; 900 test rows).

Each base appears twice with counterbalanced A/B order. The statistical unit is
the base pair, not an individual row. The `split` field is assigned at the base
level with seed 99.

Public author artifacts were found for seed 42 and seed 123 and are checked
object-for-object by the builder. No standalone seed-7 manifest was found in
the paper snapshot, the public Qwen/Gemma repositories, their public branches,
or the clean-results repository. Seed 7 is nevertheless exactly reproducible
from the authors' immutable source blob and public builder code.

Build without loading a model:

```bash
python scripts/build_e_minus1a_philpapers.py
```

Build the deterministic engineering-only 64-row smoke subset from the frozen
combined manifest:

```bash
python scripts/build_e_minus1a_smoke.py
```

The smoke subset uses 32 complete seed-42 tune pairs. It must not be used for
effect estimation, coefficient selection, or post-hoc item selection. The GPU
protocol and release gates are frozen in `docs/E_MINUS1A_PROTOCOL.md`.

See `docs/E_MINUS1A_AUTHOR_SOURCE_AUDIT.md` for provenance and implementation
details. Do not call these files a byte-identical copy of every author runtime
artifact: the CEDAR JSONL rows add `dataset_version`, `sample_seed`, `pair_id`,
and `split` for auditability.
