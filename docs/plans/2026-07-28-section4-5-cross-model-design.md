# Section 4.5 Cross-Model Design

Date: 2026-07-28  
Status: approved for autonomous drafting

## Scope

Section 4.5 tests whether the revision-geometry effect observed on Qwen appears under a second officially supported Assistant-Axis capping configuration. It compares a regime-matched Qwen3-32B anchor with Llama-3.3-70B-Instruct on the same frozen CEDAR-Evidence bank. Each model retains its released model-specific axis, layer range, and p25 threshold. The section therefore concerns transport across two model--configuration pairs; it does not isolate model identity, transfer one model's axis to another, or claim generality across activation capping.

## Approved Title

**English:** Transport Across Model-Specific Capping Configurations  
**Chinese:** 跨模型专属钳制配置的迁移

This title is preferred to “Cross-Model Transport and Heterogeneity” because model, axis, layer range, and threshold vary together in the released configurations.

## Evaluation Contract

- Models: Qwen3-32B and Llama-3.3-70B-Instruct.
- Data: the same frozen 24-base, 12-family CEDAR-Evidence bank, with N/R/V/F arms (96 prompts per model).
- Regime: reasoning disabled, deterministic forced choice, uncapped and official p25 conditions.
- Qwen configuration: released Qwen-specific Assistant Axis, `layers_46:54-p0.25` (layers 46--53).
- Llama configuration: released Llama-specific Assistant Axis, `layers_56:72-p0.25` (layers 56--71).
- Statistical unit: structural family; capped--uncapped effects are estimated within model using the frozen family-cluster bootstrap.
- Cross-model discipline: report both effects on their native scales and do not pool raw logits or infer a model difference from one significant and one nonsignificant interval.

## Primary Readouts

1. Overall capped--uncapped change in `m_general` for each model.
2. N/R/V/F arm decomposition.
3. Pooled `m_specific` and continuous V--F selectivity.
4. V/F target-follow changes and N/R option-argmax stability.
5. Cap exposure, option mass, entropy, tokenization, output integrity, and configuration checks.

## Result Routes

- **Shared transport:** both overall effects have the same direction, the Llama interval excludes zero, and V/F selectivity does not reliably improve.
- **Partial transport:** the Llama overall effect has the same direction but its interval includes zero, or the arm allocation differs materially.
- **Model--configuration specificity:** the matched Qwen anchor retains the main direction while Llama reverses, or falls inside a prespecified negligible-effect region if such a region was frozen.
- **Invalid comparison:** tokenization, cap exposure, output space, or another frozen gate fails.

The prose uses one narrow route slot and does not assume a route before the audited estimates are available.

## Paragraph Architecture

### Paragraph 1: matched regime and overall transport

Briefly identify the shared bank and forced-choice regime, then confirm the frozen integrity and exposure checks. Report the matched-Qwen `Delta m_general` and family-cluster interval, followed immediately by the Llama estimate and interval. The paragraph evaluates directional transport without comparing raw-logit levels or treating unequal point estimates as a model effect.

### Paragraph 2: selectivity and scope

Report the arm-level pattern, pooled target-specificity check, continuous V--F selectivity, necessary V/F discrete changes, and N/R stability for each model. Close with the frozen route and its scope: the result applies to the two tested model--configuration pairs. Full arm estimates, integrity diagnostics, and the standardized interaction remain in the appendix.

## Narrow TBD Slots

- `[TBD-XM-Q-BROAD]`: matched-Qwen estimate and interval.
- `[TBD-XM-L-BROAD]`: Llama estimate and interval.
- `[TBD-XM-TRANSPORT-ROUTE]`: one complete sentence selected from the frozen shared/partial/model-specific route.
- `[TBD-XM-SPECIFIC]`: pooled target-specificity estimates and intervals for both models.
- `[TBD-XM-VF-SELECTIVITY]`: continuous V--F selectivity estimates and intervals for both models.
- `[TBD-XM-DISCRETE]`: one complete sentence containing the necessary N/R stability and V/F target-follow results for both models.

The model/configuration gates are reported only after they pass; a failed gate routes the comparison out of the main text. Each remaining slot replaces one numerical package or result-dependent sentence. Identical labels are used in the English and Chinese drafts to prevent numerical drift. The surrounding prose remains completed-study prose and does not describe an experiment as unfinished.

## Figure and Table Allocation

- Figure 2D is the main-text display: a two-row forest plot of the within-model overall `Delta m_general` estimates and family-cluster intervals.
- If continuous V--F selectivity is also shown, it receives a separate facet and axis; logits and percentage-point effects are not placed on one scale.
- No new main-text table is added. A table would repeat the model configurations already defined in Section 3.4 and compete with the central result figure.
- The appendix holds the N/R/V/F decomposition, standardized family-paired interaction, cap exposure, trigger/correction, output integrity, model revisions, layer ranges, and asset hashes.

## Evidence Boundaries

- The comparison does not isolate model identity from the released model-specific configuration.
- The same axis vector, layer range, or raw-logit baseline is not transferred across models.
- The matched-Qwen estimate must come from the reasoning-off forced-choice anchor; the reasoning-enabled E2A estimate is not a cross-model comparator.
- A nonsignificant cross-model interaction does not establish equivalence.
- A same-direction point estimate with an interval crossing zero supports partial, not shared, transport.
- An interval crossing zero does not by itself establish a null or negligible effect.
- Results do not license claims about all Llama or Qwen models, all activation axes, or capping in general.
- Continuous margin and discrete answer outcomes remain separate readouts.

## Drafting and Review Sequence

1. Draft synchronized English and Chinese prose using only narrow result slots.
2. Review evidence identity, route logic, bilingual alignment, and nonformulaic style.
3. Synchronize the working draft, `main.tex`, and `main_zh.tex`.
4. Compile both manuscripts and inspect pagination, references, and overfull boxes.
5. Commit and push the section without staging unrelated user files.
