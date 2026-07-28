# RG1-P Stage A lightweight delivery

This Git delivery contains the analysis-ready outputs for the AutoDL Route A run:

- `RG1_SUMMARY.json`: compact preregistered estimates, family-bootstrap intervals,
  all 12 family values, four matched random-direction references, and layer 45--63
  plotting summaries;
- blocking and diagnostic audits;
- full derived analysis JSON/Markdown and Chinese summary;
- derived layer/propagation/plot tables;
- SVG, PDF, and PNG figures;
- execution, model-revision, delivery, and artifact manifests.

The following large intermediate data are intentionally excluded from Git:

- per-cell Formal and Smoke record/tensor directories;
- `hidden_states/` safetensor shards.

They are preserved in the separate archive `rg1p_route_a_artifacts.tar` with
SHA-256:

```text
df75bf81f4614e0a969447d05f05a74dcbd793db313d61cd3f12ebc15c8f7e56
```

The blocking Formal audit passed. The strict cross-environment replay audit is
retained as a diagnostic failure because this fresh Route A run used RTX 5090
and PyTorch 2.8/CUDA 12.8 rather than the original RTX 4090 and PyTorch
2.6/CUDA 12.4 stack. Claims are therefore within-run diagnostics, not exact
cross-environment replay equivalence.

Four matched random directions are a preregistered reference battery, not a
complete random null distribution. Stage B and patching were not run.
