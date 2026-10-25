# Public-source audit

This development PoC uses public work as protocol scaffolding, not as a pool of copied items.

## Sources inspected

1. **Contextual Belief Management / BeliefTrack**
   - Paper: arXiv:2605.30219v1.
   - Repository: <https://github.com/zjunlp/CBM>
   - Inspected commit: `6aead1b5dfee7058ec0048062a503d589d04dcd4`.
   - Ideas reused: finite candidate state, active-evidence recomputation, correction/retraction, and the Failed Stay / Failed Update / Failed Isolation distinction.

2. **DeltaLogic: Minimal Premise Edits Reveal Belief-Revision Failures in Logical Reasoning Models**
   - Paper: arXiv:2604.02733v1, CC BY 4.0 paper.
   - Ideas reused: support insertion/removal/replacement and True/False/Unknown revision episodes derived from formal proof graphs.
   - No official code repository was identified during this construction pass.

3. **Decomposing Factual Sycophancy in Language Models**
   - Repository: <https://github.com/Victordmz/decomposing-factual-sycophancy>
   - Inspected commit: `ba6a9f6d042ba354775e4eccadcebc020673b59c`.
   - The released table contains 158,312 response rows and graded first-person, authority, and bribery factors.
   - Only the idea of graded first-person belief pressure informed the S arm. No response row or prompt text was copied.

## Derivation and redistribution boundary

- All 36 scenarios and all wording in this directory are newly written for CEDAR.
- No BeliefTrack, PlausibleQA, FOLIO, ProofWriter, or sycophancy-response item is redistributed here.
- The builder has no runtime dependency on a third-party checkout.
- Source repositories were inspected in a system temporary directory and are not vendored into CEDAR.
- Public availability does not by itself establish redistribution rights. Any later plan to copy source records must separately verify the repository and dataset licenses.

## What remains original to CEDAR

- The N/R/V/F/S matched-arm wrapper.
- A shared target for V/F/S.
- The explicit F clarity contract and local fatal-flaw audit.
- Balanced answer transitions and fixed initial assistant transcript.
- The deliberative Qwen run contract and CEDAR-specific scoring contrasts.
