# CEDAR anonymous supporting data

This archive contains the frozen data banks, row-level executed records,
analysis outputs, validation traces, and source code needed to inspect the
quantities reported in the paper and technical supplement.  It contains no
model weights and performs no training or checkpoint selection.

## Layout

    data/   certified Evidence inputs, Social inputs and review records,
            new-salt instances, proofs, solver traces and countermodels
    runs/   row-level model outputs, option logits, telemetry and analyses
    code/   builders, validators, runners, analyzers, plotting code and env files

`CLAIM_LEDGER.csv` maps each reported result family to its artifact and field.
`MANIFEST.csv` gives the full delivered SHA-256 and byte count for every file.
`SANITIZATION_LEDGER.csv` records source-to-delivery hash transformations.

## CPU-only checks

From the archive root:

    python code/analysis/analyze_scale_contraction.py --package-root .
    python code/audit_supporting_package.py --package-root .

The first command recomputes the post-hoc affine-contraction decomposition from
stored logits.  The second verifies anonymity, parsing, Python syntax, workbook
readability, and the delivery manifest.  Neither command loads a model, uses a
GPU, or executes capping.

## Omitted duplicates and large assets

Per-row `record_runtime.jsonl` files duplicate delivered `results.jsonl` files
and are omitted.  Per-record checkpoint directories, driver logs, generated
figures, caches, tensors, model weights and intervention vectors are also
omitted.  Frozen hashes and public third-party identifiers remain in the
manifests so those external assets can be matched independently.

## Licensing and review status

The Evidence generator and evaluation code are new research artifacts intended
for release with the paper.  The Social source block includes its source-data
license and the single-review audit record.  Review limitations and failed or
inconclusive checks are retained; the archive does not silently remove failed
records.
