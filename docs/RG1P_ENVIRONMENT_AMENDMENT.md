# RG1-P Stage A AutoDL Environment Amendment

## Scope and reason

The original RG1-P Stage A process was paused after 272 of 864 formal cells
because its four RTX 4090 GPUs had to be released. Those 272 records remain an
immutable infrastructure checkpoint under result commit
`0e939f1524955444779753dfa39dd23dece8bba2`. They are retained for provenance
but are excluded from the AutoDL scientific analysis.

AutoDL supplies four RTX 5090 GPUs (Blackwell, `sm_120`). The frozen
PyTorch 2.6.0+cu124 build does not support this architecture. Route A therefore
rebuilds all 72 smoke and 864 formal cells from zero with PyTorch 2.8.0+cu128.
Mixing the old 272 cells with new cells is prohibited.

## Frozen and amended components

The AutoDL run preserves:

- execution logic descended from `e50012a52f2aad17683cb79b68bcf4400f07f28d`;
- Qwen3-32B revision `9216db5781bf21249d130ec9da846c4624c16137`;
- Transformers 4.56.1, bfloat16, SDPA, batch size one, and four visible GPUs;
- all prompts, forced traces, conditions, random seeds, Assistant Axis assets,
  p25 thresholds, layer ranges, capture locations, and analysis seeds;
- no content retry, answer repair, item selection, or Stage B execution.

The only required execution-stack amendment is:

| component | original checkpoint | AutoDL Route A |
|---|---|---|
| GPU | 4 x RTX 4090 | 4 x RTX 5090 |
| PyTorch | 2.6.0+cu124 | 2.8.0+cu128 |
| CUDA runtime | 12.4 | 12.8 |

## Interpretation and replay diagnostics

RG1-P Route A is a self-contained within-run propagation diagnostic. All
scientific contrasts use records generated on the same AutoDL hardware and
software stack. It does not claim token-level numerical equivalence to frozen
E2A/E2B logits.

The frozen auditor remains byte-identical to the Stage A release. Route A uses
an additive policy wrapper:

```text
python scripts/audit_rg1p_stagea_autodl_route_a.py --scope <smoke|formal> ...
```

The wrapper preserves `*_AUDIT_STRICT_ANCHOR.json` before writing the canonical
Route A audit. Replay differences remain fully measured and reported, but do
not determine the Route A audit status. Tensor integrity, finite values, hook cleanup,
processor order, schedule matching, realized-norm matching, random sign pairs,
engineering no-op behavior, terminal alignment, and the within-run native-lens
anchor remain blocking gates.

The corresponding manuscript disclosure is:

> RG1-P was executed on a separate machine from the E2A/E2B runs. All RG1-P
> contrasts are within-run comparisons on identical hardware and software; we
> therefore do not claim token-level numerical equivalence between RG1-P and
> the frozen E2A traces.

## Model acquisition

The AutoDL instance may acquire Qwen3-32B through ModelScope because direct
Xet-backed transfer is unreliable. Before GPU execution, every required model
shard and configuration/tokenizer file must be checked against Hugging Face
metadata for revision `9216db5781bf21249d130ec9da846c4624c16137` (LFS SHA-256
or Git blob ID as applicable). A failed or incomplete comparison blocks GPU
execution.
