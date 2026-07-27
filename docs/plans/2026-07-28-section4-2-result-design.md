# Section 4.2 Result-Writing Design

## Scope

Write only Section 4.2, `Capping Broadly Compresses Initial-Answer Advantage`, in synchronized English and Chinese. Figure 2 is referenced at the panel level; its visual design remains a separate decision.

## Narrative structure

1. Derive a two-coordinate projection of the four-option logit space from `m_general` and `m_specific`.
2. Report the frozen E2A pooled estimate and the four arm-level estimates.
3. Use the pooled target/background decomposition to locate the shift and route discrete epistemic consequences to Section 4.3.

The subsection follows the chain `algebraic decomposition -> continuous result -> interpretation boundary`.

## Frozen evidence

- Assay: Qwen3-32B E2A development/mechanism assay, full p25 versus uncapped.
- Pooled `Delta m_general`: `+3.130`, 95% family-cluster bootstrap CI `[+1.834,+4.486]`.
- Arm estimates: N `+2.215`, R `+2.092`, V `+2.950`, F `+5.264`; all four family-cluster intervals lie above zero.
- Pooled `Delta m_background`: `+3.158`, CI `[+2.125,+4.250]`.
- Pooled `Delta m_target`: `+3.074`, CI `[+1.180,+5.038]`.
- Prespecified pooled target-specificity check, `Delta m_specific`: `-0.084`, CI `[-1.110,+0.953]`.

## Claim boundaries

- The algebraic identities are a decomposition of readouts, not a mechanism result.
- E2A is a development/mechanism assay, not held-out confirmation.
- Arm effects may be described as positive and directionally shared, not equal.
- Relative-margin changes do not establish which raw logits moved.
- The `m_specific` interval does not establish equivalence or the absence of target-specific movement.
- Discrete warranted/unwarranted revision and cross-setting transport belong to Sections 4.3--4.5.

## Presentation

- Keep the subsection to three paragraphs plus one compact displayed equation.
- Report the pooled interval in text; Figure 2A carries full arm intervals.
- Figure 2C carries the pooled background/target/specific decomposition.
- Maintain faithful English/Chinese semantics and avoid defensive claim lists in the manuscript.
