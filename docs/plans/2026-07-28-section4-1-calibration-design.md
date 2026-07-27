# Section 4.1 Calibration Design

Date: 2026-07-28  
Status: approved for paragraph-by-paragraph drafting

## Scope

Section 4.1 calibrates the two released Assistant-Axis procedures that anchor the audit: continuous signed steering and the official layerwise p25 cap. It establishes how each procedure behaves on its source-style evaluation before Sections 4.2--4.4 examine broader and reason-sensitive effects. The subsection does not repeat the full operator definition from Section 3.3 and does not present a causal operator-only ablation.

## Approved Title

**English:** Calibrating Steering Direction and Cap Exposure  
**Chinese:** 校准 Steering 方向与钳制暴露

## Central Result

The released steering and capping procedures require separate empirical calibration. Signed addition at layer 32 reproduces the reported positive-direction margin shift, while the negative direction also lowers the same margin at high magnitude and changes output-distribution diagnostics. The official p25 cap across layers 46--53 activates broadly but produces a small adverse shift on its source-style benchmark. These observations motivate direct measurement of the cap rather than inferring its consequences from the semantic label of the axis.

## Paragraph Architecture

### Opening sentence: minimal operator distinction

Use one scalar sentence to recall the local difference between the procedures:

`s' = s + alpha` shifts every evaluated state, whereas `s' = min(s, tau)` changes only states above the threshold.

This sentence links Section 3.3 to the calibration evidence. It must not grow into a second derivation or imply that E-1a and E-1b differ only in operator choice.

### Paragraph 1: signed-steering calibration

- Identify E-1a as a source-style signed-steering calibration at layer 32.
- Report full coverage: 6,300 of 6,300 forward passes.
- Report the positive-direction result first: at `alpha = +200`, margin `-2.4312`, 95% CI `[-2.6248,-2.2407]`, reproducing the reported approximately `-2.41` shift.
- Report the exploratory negative direction: at `alpha = -200`, margin `-1.6757`, 95% CI `[-1.8506,-1.4971]`.
- State that both large-magnitude endpoints change top-token identity and entropy.
- Conclude narrowly: the sign of this benchmark margin cannot by itself be read as answer quality or selective behavioral control.
- Route the complete seven-dose curve to an appendix placeholder.

### Paragraph 2: cap-exposure calibration

- Identify E-1b as an audit of the official layerwise p25 procedure across layers 46--53.
- Report full coverage: 1,800 of 1,800 forward passes.
- Report exposure before outcome: every question triggers the cap at least once; the pooled token--layer trigger rate is `56.6628%`.
- Report the continuous outcome: margin change `-0.09375`, 95% CI `[-0.132761,-0.053826]`.
- Report the discrete outcome: sycophancy rate `+1.4444` percentage points, 95% CI `[0.1111,2.7778]`, comprising 25 non-sycophantic-to-sycophantic and 12 reverse migrations.
- Close by motivating the broader audit in Sections 4.2--4.4: broad exposure does not supply selectivity, and the cap's downstream effects must be measured in the target assays.

## Evidence Boundaries

- E-1a and E-1b use different intervention locations and released procedures. They are empirical calibrations, not a fixed-vector operator-only comparison.
- The negative steering direction is exploratory.
- Large-dose changes in token identity and entropy prevent interpreting a lower benchmark margin as improved answer quality.
- The p25 result does not establish reduced sycophancy; its discrete rate moves in the adverse direction on this benchmark.
- The p25 boundary diagnostic is post hoc and varies across datasets. It remains outside Section 4.1 and cannot support a mechanism claim.
- No result in this subsection licenses claims about a latent internal state or about activation capping in general.

## Presentation Rules

- Keep the subsection to two compact evidence paragraphs after the one-sentence operator bridge.
- Use direct result prose. Avoid slogan-like `not X but Y`, `whether rather than`, `never`, and audit-log narration.
- Give the positive-direction reproduction before the exploratory negative-direction result.
- Report component outcomes before interpretation.
- Keep the full steering curve and secondary diagnostics in an appendix placeholder whose final letter is not yet fixed.
- Maintain synchronized English and Chinese text.

## Drafting and Review Sequence

1. Present the operator bridge and signed-steering paragraph in English and Chinese for author approval.
2. Present the cap-exposure paragraph in English and Chinese for author approval.
3. Synchronize the approved text to the working draft, `main.tex`, and `main_zh.tex`.
4. Compile both manuscripts, inspect the affected pages, run read-only bilingual and venue reviews, and resolve substantive issues before committing and pushing.
