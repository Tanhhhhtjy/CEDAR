# Introduction, Abstract, and Conclusion Design

Date: 2026-07-28  
Status: approved for autonomous drafting under the revised paper framework

## Writing Order

Draft in reverse reader order:

1. Conclusion: fix the strongest evidence-bounded takeaway.
2. Introduction: motivate the question and lead to that takeaway.
3. Abstract: compress the now-stable paper into a short result sequence.

Section 6 is owned by a parallel author and is outside this edit.

## Narrative Choice

Three plausible emphases were considered:

- **Result-first audit:** CEDAR exposes a broad revision-geometry shift whose discrete consequences vary with reason condition and task.
- **Instrument-first:** the main contribution is the solver-certified reason-selectivity measurement framework.
- **Mechanism-first:** the fixed-trace and propagation analyses are the organizing center.

Use the result-first audit as the narrative spine, with CEDAR as the enabling contribution and fixed-trace replay as the diagnostic explanation. This matches the current title, the completed Qwen evidence, and the revised framework. It also keeps the paper valid if the matched Llama result supports only partial transport. The paper does not depend on a new controller or on RG1-P producing a positive result.

## Conclusion Contract

Target length: 110--140 English words; one paragraph.

Sentence functions:

1. Restate the evaluation question: activation control should be audited for sensitivity to the grounds for revision.
2. State the Qwen finding: official p25 capping shifts broad answer geometry, while selected-answer consequences vary and do not show stable reason selectivity.
3. State the controlled diagnostic: the continuous shift persists under token-for-token fixed visible reasoning.
4. Insert one narrow matched-Qwen/Llama route sentence only after its route is audited. RG1-P enters only if its strong route clears the bounded propagation claim; it does not replace the fixed-trace result.
5. End with the evidence-bounded implication that reason selectivity must be evaluated directly for activation-space control.

The conclusion does not introduce new numbers, RG1-P details, limitations, or controller proposals.

## Introduction Contract

Target length: 480--540 English words in five compact paragraphs.

### Paragraph 1: inference-time control and the missing audit target

Introduce activation steering/capping as lightweight runtime control. Existing evaluations combine target-behavior change with broad capability or benign-input checks, leaving sensitivity to reasons for answer revision unidentified. Cite representative activation-control work and the Assistant Axis.

### Paragraph 2: reason-sensitive intuition

Use the paired-target intuition: the same proposed answer can be supported by a valid argument, a single-fault fallacy, target exposure, or social endorsement. Define epistemic selectivity behaviorally as differential revision under these grounds. Point to Figure 1 without duplicating its caption.

### Paragraph 3: CEDAR and audit scope

Introduce CEDAR as a framework with two separately analyzed modules: the solver-certified CEDAR-Evidence paired dataset and the target-exposure-controlled CEDAR-Social evaluation protocol. State the discrete/continuous readouts and identify official model-specific Assistant-Axis p25 capping as the audit object.

### Paragraph 4: main empirical result

Use only headline evidence:

- Qwen E2A pooled `Delta m_general = +3.130`, 95% family-cluster CI `[+1.834,+4.486]`;
- V and F target-follow changes `+20.83` and `+16.67` percentage points, with the selectivity interval crossing zero;
- new-salt aggregate `+1.538 [0.455,2.741]` with different arm-level outcomes;
- CEDAR-Social only if space permits, framed as a direct-margin/discrete dissociation rather than behavioral improvement.

The paragraph concludes that the recurring object is the continuous revision geometry, while selected-answer consequences vary.

### Paragraph 5: diagnosis and contributions

State the fixed-trace result (`+2.078 [1.759,2.408]` on uncapped-source traces; `0/192` paired argmax changes) and use one narrow cross-model route slot. Then give three contributions in a compact `(i)--(iii)` paragraph:

1. epistemic selectivity, CEDAR, CEDAR-Evidence, and CEDAR-Social;
2. the bounded empirical audit of released Assistant-Axis p25 capping, including cross-model transport only if supported;
3. controlled phase, validity, trajectory, fixed-trace, and conditional propagation diagnostics.

Do not claim a new controller, universal activation-capping behavior, held-out-family replication, or causal mediation.

## Abstract Contract

Target length: 185--210 English words, six functional sentences.

1. Problem and evaluation gap.
2. CEDAR and the two separately analyzed modules.
3. Qwen broad-margin result and N/R implication.
4. New-salt heterogeneity plus one compact Social direct-margin/free-generation clause.
5. Fixed-trace diagnostic and one narrow matched-model route sentence.
6. Evidence-bounded takeaway about epistemic selectivity and activation-space control.

Numbers are included only when they materially identify the result. The abstract should normally retain `+3.130`, the V/F changes, `+1.538`, and `+2.078`; secondary calibration, boundary, and RG1-P numbers stay out. The matched-model sentence uses `[TBD-XM-ABSTRACT]` until the frozen route is known.

## Shared TBD Slots

- `[TBD-XM-INTRO-ROUTE]`: one sentence describing the audited cross-model route.
- `[TBD-XM-ABSTRACT]`: one compact abstract sentence or clause; retain only when the route materially improves or qualifies the abstract.
- `[TBD-XM-CONCLUSION-ROUTE]`: one conclusion sentence, retained only for an interpretable G-A/G-B result.
- `[TBD-RG1-INTRO]`: optional clause used only if RG1-P clears its claim gate; otherwise omitted.

The Abstract includes at most one optional result family beyond fixed trace: matched-model transport or RG1-P, not both. A G-C route may enter the Introduction as configuration heterogeneity but is not presented as a generalization contribution; G-D removes matched-model claims from all three high-level sections.

English and Chinese drafts use identical slot labels. No placeholder describes an experiment as unfinished.

## Style Rules

- Lead with concrete claims and verbs; avoid slogans and audit-log chronology.
- Avoid `not X but Y`, `whether rather than`, `never`, and unsupported absolutes.
- Do not repeat the Related Work argument in the Introduction.
- Keep `CEDAR-Evidence dataset`, `CEDAR-Social protocol`, and `CEDAR framework` distinct.
- Distinguish continuous margins from discrete answer changes in every summary.
- Use “two tested model--configuration pairs,” not “cross-model generalization,” until the route supports stronger language.
- Maintain sentence-level English/Chinese correspondence without forcing literal translations.

## Review and Verification

1. Draft and review Conclusion.
2. Draft and review Introduction against the fixed conclusion.
3. Draft and review Abstract against the full manuscript.
4. Run a claim--number audit, bilingual alignment review, citation check, and AI-style pass.
5. Compile both manuscripts, inspect page flow, and synchronize the working draft and Overleaf.
