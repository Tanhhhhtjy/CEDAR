# CEDAR Paper Framework Guide

## 1. Paper positioning

CEDAR is an empirical evaluation and controlled-diagnosis paper. Its central contribution is a reason-sensitive audit of activation-space control: an intervention should be evaluated not only by whether an answer changes, but by whether revision remains responsive to the validity of the reason for changing.

The paper is not a dataset paper, although it contributes CEDAR-Evidence. It is not a controller paper, although Section 6 derives requirements for future selective control.

## 2. Contribution architecture

1. **Framework and evaluation assets.** We operationalize epistemic selectivity as an audit target through the CEDAR framework, the solver-certified CEDAR-Evidence paired dataset, and the separately analyzed CEDAR-Social protocol.
2. **Scoped empirical audit.** We audit Assistant-Axis threshold capping on Qwen and Llama, reporting each model-specific configuration and result separately. Cross-model evidence strengthens or limits transportability; it is not a separate conceptual contribution.
3. **Controlled diagnosis.** We use phase-localized interventions, validity-versus-choice analyses, fixed-trace replay, and conditional representation analysis to establish only the mechanism level licensed by the controls.

## 3. Argument and section order

| Section | Reader question | Function |
|---|---|---|
| 1. Introduction | Why is behavior-only evaluation insufficient? | Gap, same-target/different-reason example, finding, three contributions |
| 2. Related Work | What is already known, and what remains untested? | Distinguish steering from capping and position nearest reason-sensitive evaluations |
| 3. CEDAR | How is epistemic selectivity made identifiable? | Define the two modules, operator/model scope, readouts, and statistical discipline |
| 4. Auditing Assistant-Axis Capping | What changes under capping? | Continuous geometry, warranted/unwarranted updating, natural-task evidence, cross-model heterogeneity |
| 5. Diagnosing the Shift | Where and how does the shift arise? | Phase, validity/choice, hypothesis rejection, and fixed-trace replay; representation propagation is supplementary by default |
| 6. Implications, Scope, and Future Control | What follows for intervention design? | Representation-is-not-policy interpretation, controller requirements, evidence scope |
| 7. Conclusion | What should the reader retain? | One compact evidence-bounded conclusion |

The order is deliberate: measurement precedes effects; effects precede mechanism; mechanism precedes design implications.

## 4. Section 3 contract

Section 3 follows this order:

1. epistemic selectivity and within-module controlled contrasts;
2. CEDAR-Evidence as a solver-certified paired dataset;
3. CEDAR-Social as a target-exposure-controlled protocol;
4. the threshold-capping operator and model-specific configurations;
5. discrete migration and continuous margin readouts;
6. clustered uncertainty and audit discipline.

Figure 1 must visually separate CEDAR-Evidence from CEDAR-Social. A full method-contract table stays in the appendix unless the final figure cannot carry the distinction.

## 5. Seven-page target

| Component | Target pages |
|---|---:|
| Introduction | 0.70 |
| Related Work | 0.45 |
| CEDAR | 0.90 |
| Auditing Assistant-Axis Capping | 1.25 |
| Diagnosing the Shift | 1.00 |
| Implications, Scope, and Future Control | 0.40 |
| Conclusion | 0.15 |
| Three figures and captions | 1.55 |
| Layout buffer | 0.50 |
| **Total** | **7.00** |

Initial composition should target roughly 6.5 pages, preserving the 0.5-page buffer.

## 6. Figure plan

1. **Figure 1 — Same target, different reasons.** Two visibly separate modules and their controlled contrasts; no empirical values.
2. **Figure 2 — Epistemic non-selectivity.** Qwen margin effects, warranted versus unwarranted updating, target/background geometry, and matched model-specific result slots; the final panel becomes a compact within-model interval display when Qwen/Llama values are frozen.
3. **Figure 3 — Fixed-trace replay.** Source-conditioned broad shifts and valid-minus-fallacious contrast changes. Full arm decomposition and representation propagation remain supplementary unless their evidence gates pass and the seven-page budget remains intact.

No main-text table duplicates Figure 1. Extended schema, assay matrix, sensitivity analyses, model configuration, and protocol deviations belong in the appendix.

The phase/validity table is kept in the supplementary material. Section 5 retains all capped-minus-uncapped estimates and intervals in prose. No main-text table is reserved: complete absolute levels, arm-level estimates, method contracts, and model configurations remain supplementary.

The supplementary result interface uses three canonical tables: (i) development and new-salt CEDAR-Evidence, (ii) CEDAR-Social direct-margin and free-generation readouts, and (iii) matched within-model Qwen--Llama CEDAR-Evidence. The module-contract and phase/validity tables remain separate. Full strata and integrity fields live in machine-readable ledgers rather than widening these tables.

## 7. Drafting rule

Draft in completed-study prose. Missing values occupy their final semantic slots as `TBD`; they are not described as future work. The internal register may still mark their true evidence state so that no `TBD` is accidentally promoted into a factual claim.
