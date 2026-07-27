# Section 4 and Figure 2 Integration Design

Date: 2026-07-28  
Status: approved for autonomous drafting under the revised paper framework

## Purpose

This integration closes two presentation gaps in Section 4: the section currently begins directly with Section 4.1, and several subsections cite Figure 2 although the figure has no main-text interface. The change adds a two-sentence section lead and a page-aware Figure 2 placeholder with a self-contained caption. It does not add a result table or invent cross-model values.

## Section Lead

The lead will do two jobs in two sentences:

1. state that Section 4 applies CEDAR to the released Assistant-Axis steering and capping procedures;
2. preview the evidence sequence: source-style calibration, reason-sensitive Qwen assays, natural-task checks, and the matched Qwen--Llama comparison, with continuous margins and discrete answer changes kept separate.

The lead will not repeat the operator equations, list numerical results, or state a conclusion before the evidence subsections.

## Figure 2 Role

Figure 2 is the quantitative overview for Section 4. Its fixed four-panel contract is:

- **A — Broad revision geometry:** Qwen E2A arm-level capped--uncapped changes in `m_general` for N/R/V/F, with family-cluster intervals.
- **B — Warranted and unwarranted updating:** uncapped and p25 target-following under V/F, together with the capped--uncapped V--F selectivity contrast.
- **C — Location of the margin shift:** pooled changes in `m_background`, `m_target`, and the prespecified `m_specific` target-specificity check.
- **D — Model--configuration transport:** within-model `Delta m_general` for the regime-matched Qwen anchor and Llama under their released model-specific configurations.

Panel D contains only narrow TBD result slots until both matched runs pass the frozen gates. Raw logit levels are not compared across models.

## Placement and References

- Insert the double-column placeholder after Section 4.1 and before the Section 4.2 heading. This lets LaTeX place it at the next page top without putting its source after the first substantive citation.
- Give the figure a stable label and replace hard-coded references in Sections 4.2, 4.3, and 4.5 with label-based references.
- Keep the placeholder shallow enough to expose pagination pressure, while reserving a realistic double-column caption interface.
- Figure 3 retains its existing interface after Figure 2 increments the counter normally.

## Caption Contract

The caption must:

- identify A--C as Qwen development-assay results and D as a matched forced-choice comparison;
- define every plotted contrast at the level needed to read the figure without the main text;
- state that whiskers are 95% family-cluster bootstrap intervals over 12 structural families;
- distinguish continuous margins from discrete target-following rates;
- state that effects are estimated within assay and within model;
- route complete arm estimates, exposure, and output-integrity diagnostics to an appendix placeholder.

The caption must not:

- call the development bank held out or confirmatory;
- describe an interval crossing zero as equivalence;
- pool raw logits across models;
- imply that Figure 2 establishes a common mechanism;
- prefill a Llama direction or route.

## Table Decision

No new main-text table is added. Figure 2 carries the central geometry, discrete update, and cross-model result; a table would duplicate Section 3.4 and the figure. Complete coefficient and diagnostic tables remain in the appendix.

## Review and Verification

1. Draft synchronized English and Chinese leads and captions.
2. Review the caption against Sections 3.5--3.6 and all Figure 2 citations.
3. Compile both manuscripts and inspect page flow, float placement, undefined references, and overfull boxes.
4. Commit and push only the design, working draft, and Overleaf source changes.
