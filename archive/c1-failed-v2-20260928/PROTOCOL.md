# C1 v2 execution protocol — 2026-09-28

Status: prospective implementation specification for C1 v2, after the 32-cell v1 engineering attempt. This is development evidence, not independent confirmation or retrospective preregistration. The original S1/P1 PROTOCOL.md and all v1 results remain unchanged.

## Scope and scientific meaning
Use the same 192 frozen traces, pinned Qwen3-32B BF16 SDPA model, official p25 caps and cache-faithful historical run_one. No new generation. Main C1 uses 96 N/R/V/F prompts with uncapped-source trajectories, paired U/P forwards. Primary independent unit remains 12 structural families. Capped-source GPU repetition and H1 are not launched by this campaign.

Prompt L53 means the input to block 54 (zero based), after the L53 cap, final prompt token only, on the first prefill call. A hook is invoked on every decode step but modifies only that one position once. Full local state replacement changes neither earlier tokens nor the complete KV cache.

## Direction and coordinate mapping
For held-out family f, reproduce the P1 preprocessing using only V/F examples in the other 11 families: q=h/RMS(h); regress initial and target identities using the original nine-column design; StandardScaler; balanced liblinear logistic C=1, dual=True, tol=1e-7, max_iter=10000, seed=20260927. Export d=normalize(coef/scaler.scale_) in unit-RMS coordinates. Save regression coefficients, scaler and classifier parameters, training IDs and hashes. Reconstructed held-out P1 scores must match the saved P1 primary scores (max error <=1e-6).

The U-trained direction is the common primary direction for BOTH U-to-P and P-to-U, so dose comparisons change target capping condition without changing the candidate signal. P-trained directions are exported as a declared sensitivity reference, not selected based on C1 outcomes. This replaces the v1 smoke's raw-space, donor-condition-dependent directions; v1 and v2 cells are never pooled.

For same-item donor D and target T:
qD=hD/RMS(hD), qT=hT/RMS(hT);
component delta_h=RMS(hT) * dot(qD-qT,d) * d;
patched hT=hT+delta_h.
This is a linear intervention mapped back to raw activation coordinates, NOT an exact spherical normalized-state swap. Do not renormalize the patched state. Save realized raw and unit-RMS projection changes.

Signed dose endpoints use the same d and sigma_f=SD(dot(q_train-nuisance_train,d)), ddof=1, from U training-family V/F examples only. For alpha in {-1,-0.5,+0.5,+1}, delta_h=RMS(hT)*alpha*sigma_f*d. These doses remain informative when the paired donor component difference is tiny.

## Controls and BF16 numerical matching
Main: no-op, alpha=1 component transfer, eight random directions, full local state replacement, for all 96 prompts and both targets: 96*2*11=2112 forwards.
Random directions: eight isotropic Gaussian unit vectors per held-out family, deterministic SHA-256 seed from (20260928,family,index), reused across arms, targets and doses; no outcome-based selection.
For each random control match the REALIZED BF16 displacement norm of its true-direction cell, by a deterministic scalar bisection along the already fixed random vector. Absolute error must be <=1e-7 when target norm is zero; otherwise relative error <=0.005. Save scalar, intended/realized norms, cosine, vectors and actual patch states. Failure stops the run; never choose a replacement random vector or relax the tolerance after outcomes.
Signed-dose stage: V/F only, 48*2*4*(1+8)=3456 forwards, with main no-op references. It is queued after main only if the forecast stays inside the declared campaign time bound.
Primary matrix and doses together: 5568 forwards, excluding the technical smoke. Do not call main-only completion full C1 completion.

## Readout and inference
Freeze S1 same-prompt affine coefficients from the existing unpatched S1_DETAILS.json before v2 outcomes. U logits are unchanged; target-P logits map to (z-b)/a with the SAME locked a,b for no-op, true, random, replacement and doses. Save family-out fixed coefficients as sensitivity. Do not refit to patched outcomes for the primary contrast.
This controls the previously measured capping scale; it does not guarantee removal of every scale change caused by a new patch. Save full vocabulary logits and held-out non-option errors for that diagnostic.

Margins retain the original definitions: broad=mean(non-initial options)-initial; specific=target-mean(two options other than initial/target).
Per base, selectivity change is (V_patch-V_noop)-(F_patch-F_noop).
Primary direction-specific contrast is real selectivity change minus mean of eight random selectivity changes, after locked S1 correction, for broad and specific. Four comparisons (2 transfers*2 metrics) receive Holm adjustment on exact family sign flips, assuming symmetric independent family effects. Cross-fitted training sets overlap, so intervals/p-values remain exploratory and must be presented with that limitation. Raw and family-out-corrected contrasts, N/R changes, ranks/argmax and all individual random effects are secondary.
Prediction: U-to-P may move selection toward U, and P-to-U toward P; generic margin restoration alone is insufficient. A null is limited to this position/direction/dose and cannot imply no reason use anywhere.
Signed-dose contrasts use the central difference between +alpha and -alpha, compare U/P on the same d and training dose, and report real minus random. No layer/dose selection; descriptive secondary endpoints.

## Gates, resources, and persistence
Before v2: audit inputs and cached states; restore/compare P1 scores; add P1 identity endpoints, capped-source answer-boundary sensitivity, L53/L60 bridge, and S1 family-out non-option error using server CPU only.
Rehash all pinned model and cap files once before model load, and verify original formal manifest fingerprints. Use physical GPUs 1,2,4,6 only, 18 GiB model budgets each; require >=22 GiB free on each immediately before load. No CPU/disk model offload. Minimum disk free 20 GiB.
Use an exclusive flock on this campaign and checkpoint each JSON+NPZ with SHA-256; resumed fingerprints must match. Never overwrite failed records, logs, or gates. Runtime fails on any nonfinite array, unexpected token hash, mismatched cached prompt state, wrong call count or hook leak.
v2 smoke: first lexicographic base, N/R/V/F, both targets, no-op/component/random0/replacement (32); repeat no-op and component for first V in both targets (4); signed +/-1 real and random0 in both targets on that V (8). Total 44. Full-vocabulary no-op vs original <=1e-4; repeated captured arrays <=1e-4; terminal BF16 reconstruction <=0.25; patch only once; calls=prefix length+1; realized norm gates and hook cleanup required. Failure never enables formal.
Main and dose run only after all technical gates. Original v1 gate.json and gate_reaudited.json are retained as historical engineering artifacts, neither authorizes v2.
Forecast with historical per-trace seconds and current smoke throughput; record each stage forecast. Maximum total active GPU campaign duration 10 days, disk forecast <=20% of current free. If exceeded, stop before the affected stage and report; do not shorten traces or drop controls.
Progress snapshots every 30 minutes on the server; user-facing monitor should only notify at stage completion, actionable failure or required input. No credentials in scripts or artifacts.
H1 is deliberately pending C1 evidence, independent solver-certified structural design and precision planning.

Local root: D:/CEDAR/arr_c1_20260928.
Remote root: /data4/WuMingrui/TianJinyu/CEDAR/c1_20260928_v2.

