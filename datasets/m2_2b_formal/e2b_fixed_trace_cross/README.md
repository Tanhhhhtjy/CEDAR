# E2B fixed-trace cross

This directory contains the CPU-frozen inputs for the controlled E2B
teacher-forcing diagnostic. It derives exactly 192 trace prefixes from the 96
E2A Answer prompts and their uncapped/full-p25 responses. No new text is
generated during E2B.

The final answer label and every token after it are excluded from each trace.
The pinned tokenizer, stored response-token hash, selected logit index, parsed
label, and character boundary jointly identify the cut point. The formal matrix
crosses both trace sources with uncapped/full-p25 forwards for 384 cells.

GPU execution is forbidden unless `CPU_GATE.json` has `gpu_authorized=true`
and the Gate commit is clean, pushed, and synchronized with its upstream.
