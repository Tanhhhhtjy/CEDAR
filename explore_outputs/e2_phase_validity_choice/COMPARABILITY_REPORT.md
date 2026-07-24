# Comparability Report

The Answer assay preserves all 96 model-visible messages from the Evidence
anchor. Its `uncapped` and `full_p25` cells are an engineering equivalence check
against `a0ad950e`, not a new scientific replication. Exact generated-token
hashes, parsed outcomes, answer boundaries, and numerical logits are audited.

The Validity assay is a new independent readout. It quotes the frozen V/F
material exactly, uses a separate system instruction and JSON contract, and
therefore is not numerically comparable to Answer accuracy. V/F comparisons
within Validity share the same wrapper and base-level label mapping.

`prefill_only` and `decode_only` differ only in the phase allowed to apply the
same official layer-46–53 p25 cap. `full_p25` retains the old cap path and adds a
read-only pre-cap phase monitor.
