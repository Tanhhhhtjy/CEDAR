# CEDAR Section 3 Integrated Rewrite Design

Date: 2026-07-27  
Status: approved for implementation

## Scope

Rewrite Section 3 as one coherent methods section while preserving its established order:

1. shared answer-change setting and controlled contrasts;
2. CEDAR-Evidence;
3. CEDAR-Social;
4. intervention and model scope;
5. behavioral outcomes and logit margins;
6. statistical inference and diagnostic checks.

Synchronize the result in:

- `docs/paper/cedar-aaai27-working-draft.md`;
- `E:\CEDAR-Overleaf\main.tex`;
- `E:\CEDAR-Overleaf\main_zh.tex`.

Section 4 and later result prose are outside this edit. Figure 2D remains undecided until the Llama results are available.

## Scientific Contract

### Shared notation and terminology

- `a_0` is the initial answer supplied in the dialogue context, not an answer sampled separately under every intervention condition.
- `a_t` is the paired target used to classify later outcomes.
- An **answer change** is a parseable final option different from `a_0`.
- **Target following** is a final answer equal to `a_t`.
- A **warranted revision** is target following under a follow-up that supplies sufficient evidence for `a_t`; in CEDAR-Evidence this normative status applies to `V`.
- Use these three terms consistently in both languages. Do not use `answer revision` as a generic synonym for any answer change.

The exposure contract must be explicit:

- `N` and `R` retain the paired target identity for analysis but do not expose `a_t` to the model;
- `V` and `F` present the same proposed update associated with `a_t`, with different formal validity;
- `AO` and `S` both expose `a_t`; only `S` adds social framing.

### Figure 1 placeholder

Keep the Figure 1 placeholder after Section 3.1 and before the module-specific subsections. The figure and caption must visually separate CEDAR-Evidence from CEDAR-Social. They may state that the modules have separate estimands and independent sampling units, but must not claim that all six conditions form one factorial design or that all Evidence arms expose the target.

The caption retains a generic appendix placeholder for additional examples and construction details. It does not lock an appendix letter.

### CEDAR-Evidence and CEDAR-Social

CEDAR is the reason-sensitive activation-control audit framework. CEDAR-Evidence is its new solver-certified paired dataset; CEDAR-Social is a separate evaluation protocol over SycoBench-600. The paper is not positioned as a dataset paper.

For CEDAR-Evidence, retain in the main text the formal task, dual-solver agreement, certified countermodel for `F`, one invalid inferential step, 24 bases per bank, 12 structural families, and family-level inference. Describe the new-salt bank as a second instantiation of the same task grammar and family structures, not as an independent-family replication.

For CEDAR-Social, retain the `AO/S` control logic, the high-level automated checks, and an output-blind two-researcher review. Agreement and adjudication values remain narrow `TBD` slots until inserted. Full templates, item-level records, and implementation checks belong in the appendix.

## Intervention and Model Scope

Express the released cap in the coordinate stored by the released configuration:

\[
\mathbf h'_{\ell,t}=\mathbf h_{\ell,t}-\mathbf v_\ell
[\langle\mathbf h_{\ell,t},\mathbf v_\ell\rangle-\tau_{\ell,c}]_+.
\]

The text states that the stored direction is opposite to the human-readable Assistant Axis, so the released minimum cap becomes an upper cap in the stored coordinate. Asset hashes, orientation cosines, and runner/monitor consistency checks move to the appendix.

Qwen3-32B uses the released `layers_46:54-p0.25` configuration. Llama-3.3-70B-Instruct uses its released model-specific axis and `layers_56:72-p0.25` configuration. Do not transfer or tune axes, layers, or thresholds across models.

The matched cross-model study uses the same frozen 24-base, 12-family `N/R/V/F` bank, reasoning disabled, deterministic forced-choice decoding, and uncapped versus p25 conditions. A regime-matched Qwen run anchors the Llama comparison. This is transport across two model--configuration pairs; it is not a pure model effect, universal generalization, or shared-mechanism claim.

## Outcomes and Estimands

Use `z(a)` for the next-option logit at the fixed answer position; `\ell` remains the layer index. Define:

\[
m_{\mathrm{general}}=\frac{1}{3}\sum_{a\ne a_0}z(a)-z(a_0),
\]

\[
m_{\mathrm{target}}=z(a_t)-z(a_0),
\qquad
m_{\mathrm{background}}=rac{1}{2}\sum_{a\notin\{a_0,a_t\}}z(a)-z(a_0),
\]

\[
m_{\mathrm{specific}}=z(a_t)-\frac{1}{2}\sum_{a\notin\{a_0,a_t\}}z(a).
\]

For outcome `Y`, condition `x`, and cap configuration `c`, define the within-condition cap effect:

\[
\delta_x^Y(c)=\mathbb E[Y\mid x,c]-\mathbb E[Y\mid x,0].
\]

The main estimands are:

\[
\Delta_{\mathrm{broad}}=\frac14\sum_{x\in\{N,R,V,F\}}
\delta_x^{m_{\mathrm{general}}},
\]

\[
\Delta_{\mathrm{sel}}^{(m)}=
\delta_V^{m_{\mathrm{target}}}-\delta_F^{m_{\mathrm{target}}},
\qquad
\Delta_{\mathrm{sel}}^{(p)}=
[p_V(c)-p_V(0)]-[p_F(c)-p_F(0)],
\]

where `p_x` is the target-following probability in Evidence condition `x`, and

\[
\Delta_{\mathrm{social}}=
\delta_S^{m_{\mathrm{target}}}-\delta_{AO}^{m_{\mathrm{target}}}
=[S-AO]_c-[S-AO]_0.
\]

A negative `Delta_social` means that capping reduces the additional target support attributable to social framing beyond target exposure. It does not imply that target following decreases within `S`; the `S` and `AO` component effects must be reported.

`m_general` measures broad movement away from the initial answer. The continuous `V-F` contrast on `m_target` measures epistemic selectivity. The discrete `V-F` target-following contrast provides its realized-choice counterpart. Pooled `m_specific` remains a prespecified target-specificity check; arm-wise `m_specific` decomposition is post hoc and belongs in the appendix.

## Statistical Inference and Diagnostics

- Each Evidence bank contains 24 bases nested in 12 structural families; the family is the independent unit.
- Paired capped--uncapped contrasts are averaged within family, and the 12 family effects receive equal weight.
- The primary interval uses the frozen family-cluster bootstrap. Report family effects and an equal-weight leave-one-family-out analysis for the headline margin.
- Add family-level `t_11` and studentized sign-flip intervals/tests as post-hoc small-cluster sensitivities. They do not replace the frozen primary analysis.
- CEDAR-Social resamples SycoBench questions with domain/difficulty stratification.
- Discrete analyses retain all scheduled evaluations. Truncated or unparsable outputs are coded as `invalid/abstaining` and are not selectively rerun because of content.
- Continuous contrasts use pairs with complete required option logits; incomplete coverage is disclosed with the affected result.
- Estimate banks, assays, and models separately. Do not pool the two Evidence banks or the two models.
- The development bank is exploratory. The new-salt bank and CEDAR-Social each have one primary contrast; remaining analyses are labeled prespecified secondary, sensitivity, or post hoc.
- Cross-model headline results are the two raw within-model effects and intervals. A family-paired interaction standardized by each model's uncapped scale is secondary and belongs in the appendix. A nonsignificant interaction does not establish equivalence.
- Diagnostics are the total full-vocabulary probability mass on the four option labels, full-vocabulary Shannon entropy, and whether the vocabulary-wide argmax is an option label.
- Initial/target/background logit decomposition is a substantive Section 3.5 analysis, not a quality-control item.
- Disclose material protocol amendments with the affected result; implementation details and complete sensitivities go to an appendix placeholder.

## Prose and Bilingual Style

- Use completed-study prose. Unknown numerical values and appendix locations remain narrow `TBD` slots.
- Avoid `not X but Y`, `whether ... rather than`, `never`, slogan-like parallelism, and internal audit-log language.
- Prefer ordinary research verbs such as `define`, `construct`, `estimate`, `report`, and `compare`.
- Keep English and Chinese semantically aligned without translating English syntax mechanically.
- Use `answer change / target following / warranted revision` and `答案改变（或改答）/目标跟随/有充分依据的改答` consistently.
- Do not describe the modules as a unified six-arm dataset or describe Llama as a repetition of the complete Qwen analysis.

## Verification

1. Review the English and Chinese Section 3 texts against this contract.
2. Confirm that Section 4 and later content is unchanged.
3. Compile `main.tex` and `main_zh.tex`.
4. Check references, encoding, overfull boxes, and caption placement.
5. Run technical-coherence, venue-structure, and bilingual-style reviews.
6. Inspect the rendered PDFs before committing the manuscript changes.

