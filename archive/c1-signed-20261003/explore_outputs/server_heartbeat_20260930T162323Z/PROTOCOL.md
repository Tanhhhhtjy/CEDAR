# C1 signed-dose protocol — 2026-09-28

Prospective frozen execution specification for the user-authorized 52-cell engineering smoke and 3840-cell development C1 experiment. Written before any new GPU output. Informed by existing S1/P1 and failed C1 engineering attempts; not independent preregistration or H1 confirmation. The latest user instruction authorizes execution on the server and automatic formal continuation after engineering/resource gates, without conditioning on scientific effect direction or significance.

Local: D:/CEDAR/arr_c1_signed_20260928. Remote: /data4/WuMingrui/TianJinyu/CEDAR/c1_signed_20260928. Original root: /data4/WuMingrui/TianJinyu/CEDAR.

## Frozen scientific scope

Reuse original Qwen3-32B BF16, SDPA, official p25 caps, package versions, token hashes and cache-faithful historical run_one. Verify original manifests and 384 saved activation/logit arrays. The 96 uncapped-source prompts are the 24 existing bases, four arms N/R/V/F and 12 structural families. Both target conditions U=uncapped and P=full_p25 are measured on their unchanged original frozen trajectories. No new text generation, independent samples, additional layer, half-dose sweep, direction selection, or Jev API calls. Old S1/P1 results, v1 engineering failures and blocked v2 remain separate.

Patch the last prompt token at input to zero-based block 54, after the L53 cap, once during first prefill. The hook is called prefix-token-count+1 times but modifies one time only. Later decode tokens are untouched. Full local state replacement swaps this one vector with the opposite condition's same-prompt vector, not the earlier states or entire KV cache.

## Directions, signed doses and controls

Reuse the exact previously prepared, hash-verified P1 direction tensors and nuisance/scaler parameters, without fitting on new outputs. Each held-out family direction is learned from V/F in the other 11 families (44 prompts): unit-RMS states, initial/target identity nuisance regression, StandardScaler and balanced liblinear logistic C=1, dual=True, tol=1e-7, max_iter=10000, seed=20260927. Reconstructed prior P1 decision scores must agree to <=1e-6. Raw mapping d=normalize(coef/scaler.scale_). Both target conditions use the same U-trained d and training-family projection SD sigma (ddof=1); P-trained exports are provenance only, not an extra selected condition.

For q=h/RMS(h), doses alpha=-1,+1 add delta_h=RMS(h_target)*alpha*sigma*d in raw activation coordinates, then round to original BF16. Do not renormalize patched states. This is a mapped linear coordinate intervention, not exact spherical replacement. Log intended and realized raw norms, projection changes, normalized-state changes, and relative displacement.

Eight fixed isotropic Gaussian unit random vectors per held-out family use existing SHA-256 seeds (20260928,family,index). Reuse across arms, target conditions and signs. Crucially, random direction is alpha-sign times the fixed vector. A positive scalar bisection matches the true cell's realized BF16 displacement norm; relative error <=0.005, with zero-displacement no-op exact. No replacement random vectors, outcome-dependent matching, or tolerance relaxation. Random matching uses only cached activations, never output logits. Equal alpha is equal normalized training-defined dose, not exact equal raw BF16 displacement between U/P; report that difference.

## Matrix and engineering gates

Smoke: lexicographically first base, four arms * two targets * [noop, replacement, true -1, random0 -1, true +1, random0 +1] = 48; first V no-op and true +1 repeated in each target = 4. Total 52. Smoke checks numerical engineering only, not scientific support.

Formal: all 96 prompts * two targets * [noop + replacement + two signs * (true + eight random)] = 3840. All 12 families, all 24 bases included. The smoke does not substitute for or augment formal independent evidence.

Before GPU: synthetic sign/norm/shape/hook cleanup tests; estimator tests; frozen-input/direction checks; all 3892 planned patches CPU checked; rehash every pinned model and cap file. Preserve failures. Require fresh file stats unchanged at model load. Inference environment versions match historical manifest, device map has no CPU/disk offload. Historical token/prefix/concatenation checks remain active.

Every new no-op must match all original captured arrays, including full-vocabulary logits, max absolute error <=1e-4. Smoke repeats must match captured arrays <=1e-4. BF16 terminal logit reconstruction <=0.25. Require finite arrays, correct cached prepatch state, exactly one modification, correct total hook calls, and no leaked hooks. Save per-record JSON and NPZ with SHA256 and manifest fingerprint. Reject mixed-provenance resumes. Gate failure never enables formal; local full-state replacement need not reproduce the donor's whole output and no scientific effect is an engineering prerequisite.

## Four primary endpoints

Keep original margins: broad = mean(other three options) - initial option; specific = target - mean(two options excluding initial/target). For each base and arm, condition c and direction j, S(c,arm,j)=[margin(+1)-margin(-1)]/2. This is a finite difference over chosen doses, not an infinitesimal derivative.

G(c,arm)=S(c,arm,true)-mean_j S(c,arm,random_j).

1. Common response change: mean over V,F of G(P,arm)-G(U,arm).
2. Reason-arm interaction: [G(P,V)-G(P,F)]-[G(U,V)-G(U,F)].

Compute both for broad and specific: four primary comparisons. Freeze S1 same-prompt affine parameters before these outputs: U unchanged, P maps to (z-b)/a using identical locked coefficients for all patches. Raw and locked family-out scales are sensitivity analyses. Never refit patch-dependent scales as the primary endpoint. A separate full-vocabulary scale fit on existing non-option fit tokens and held-out errors diagnoses new perturbation scaling.

Report each condition/arm/direction, true-minus-random effects, all eight random responses, N/R controls, option ranking/argmax changes, actual dose telemetry and replacements. The random vectors are controls, not independent experimental units. Average bases within each family, then weight 12 family means equally. Report 20,000 family bootstrap resamples (seed20260928), exact 4096 family sign-flips and Holm across all four primary tests. These are exploratory due to small family count, overlapping cross-fit training and development reuse; symmetric independent family sign-flip assumptions are not established. A nonsignificant result alone is not equivalence.

Interpretation is limited to this direction/site/dose and frozen trajectory readout. A common response difference is not automatically validity-specific; interaction, N/R, random and scale checks matter. No universal claim about whether the model uses reasons, no natural-mediation claim, and no layer/dose search after nulls.

## Resources and recovery

Use physical GPUs 1,2,4,6, 18GiB model allocation limit each. Require at least22GiB free per selected GPU before load; wait at most12h without touching other jobs. Minimum disk free20GiB; projected remaining output <=20% of current free using2MB/record. Re-estimate formal time using all smoke runtimes relative to their historical baseline; never select scientific outcomes. Historical estimate: smoke1.385h and formal117.4h before actual overhead. Hard smoke bound6h, cumulative GPU bound10days, persisted across resumes. Stop affected runner if no state transition90min or budget/space guard fails; preserve everything for diagnosis.

Campaign and runner use independent exclusive flock. Check no surviving runner before campaign launch/resume; only task-owned child may be stopped. Credentials never appear in code, reports or logs. Network is unnecessary after local weights verification. Server snapshots every30min include PID+command+cwd, current cell/count, log size/growth, asset-verification timestamp, GPU memory/utilization/processes, and disk. Agent monitor compares snapshots every30min, notifies only significant milestones/failures/actions and pauses after complete integrity review with no remaining task processes.

H1 design and precision planning may proceed on CPU independently of C1 positivity. H1 GPU execution requires a concrete independently certified new-structure bank and frozen protocol; this campaign does not launch it. Jev remains future work.
