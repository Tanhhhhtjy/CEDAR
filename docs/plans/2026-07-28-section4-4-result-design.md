# Section 4.4 Result Design

Date: 2026-07-28  
Status: approved for paragraph-by-paragraph drafting

## Scope

Section 4.4 synthesizes within-Qwen evidence about variation across instances, tasks, and readouts. It does not make a cross-model claim and does not offer a causal account. Cross-model transport remains in Section 4.5; phase, boundary, and representation diagnostics remain in Section 5.

## Approved Title

**English:** Continuous Margin Shifts Accompany Heterogeneous Behavioral Changes  
**Chinese:** 连续 Margin 位移伴随异质的行为变化

## Central Result

The section distinguishes a recurring positive aggregate margin direction from heterogeneous discrete consequences. The three sources retain their native estimands and statistical units. They are not pooled, standardized into a common effect, or described as replications of one another.

## Evidence Ledger

### E2C new-salt instantiation

- Same task grammar and structural families as E2A, instantiated with a fresh salt.
- Complete capped--uncapped primary logits for 95 of 96 comparisons.
- Complete-case overall `Delta m_general = +1.538` logits.
- 95% family-cluster bootstrap CI `[+0.455,+2.741]`.
- V target-follow change: `-12.5` percentage points.
- F target-follow change: `+16.7` percentage points.
- V--F selectivity change: `-29.17` percentage points, 95% CI `[-58.33,+4.17]`.
- The completed tipping-point analysis retains the positive aggregate direction under the missing-logit sensitivity.
- Allowed identity: a new-salt instantiation of the same task grammar.
- Disallowed identity: independent-family, external, held-out-family, or independent replication.

### CEDAR-Social

- 600 paired SycoBench bases.
- AO target-margin change: `+1.26792` logits.
- S target-margin change: `+0.48167` logits.
- Cap-by-arm interaction, `Delta_cap(S-AO) = -0.78625` logits.
- 95% stratified base-bootstrap CI `[-0.86000,-0.71104]`.
- Both arm effects are positive; the negative interaction is a smaller increase under S than under AO.
- On the frozen free-generation subset, strict and completion-based SocialExcess are both zero, with no S-arm answer migration under p25.
- The result is bounded to the direct margin readout and does not establish a reduction in S choices, social pressure, or an internal sycophancy state.

### C3 auxiliary diagnostic

- Continuous and discrete readouts separate.
- Gate F is inconclusive.
- C3 receives one short sentence without headline numbers.
- C3 is not an independent confirmation and is not merged with the E2C or CEDAR-Social estimands.

## Paragraph Architecture

### Paragraph 1: E2C

Open with the result: the aggregate margin direction recurs in the new-salt instantiation while the arm-level answer changes differ from E2A. Report coverage, `Delta m_general`, its interval, the V and F discrete changes, and the V--F interval. Disclose the single missing logit and the completed tipping-point sensitivity without turning the paragraph into an audit log.

### Paragraph 2: CEDAR-Social

Introduce CEDAR-Social as a separate natural-task contrast. Report AO and S before the interaction so that the sign cannot be misread. Interpret the interaction as a reduction in the incremental social-framing margin contrast. Close with the frozen free-generation result, which limits the finding to the direct margin readout.

### Paragraph 3: C3 and synthesis

Use one short sentence for C3, followed by the section-level conclusion: continuous margin effects and discrete choices require separate interpretation across assays. Do not claim a common mechanism, cross-task replication, or uniform behavioral improvement.

## Figure and Appendix Allocation

- Section 4.4 does not require a new main-text panel if matched Qwen--Llama transport occupies Figure 2D.
- E2C missing-value sensitivity and complete CEDAR-Social/C3 results go to the appendix.
- If Section 4.5 cannot supply Figure 2D, E2C may occupy that panel as a forest plot without changing the prose identity of the study.

## Drafting and Review Sequence

1. Present Paragraph 1 in English and Chinese for author approval.
2. Present Paragraph 2 in English and Chinese for author approval.
3. Present Paragraph 3 in English and Chinese for author approval.
4. Synchronize the approved text to the working draft, `main.tex`, and `main_zh.tex`.
5. Compile both manuscripts, inspect the affected pages, run read-only bilingual and venue reviews, and resolve substantive issues with the author before committing.

