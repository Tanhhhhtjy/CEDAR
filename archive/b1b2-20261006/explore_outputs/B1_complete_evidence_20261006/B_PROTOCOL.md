# B history × terminal factorial v1 · frozen before acquisition

Authorized initial package: smoke24 then B1 192, automatically after technical/resource gates; no scientific-outcome gate. New development experiment, not independent confirmation. B2 192 is predetermined but excluded from this package's GPU budget. No C/D/H inference in this campaign.

## Sequence and intervention
Read the original192 frozen traces and use all96 uncapped-source traces. Within each of12 families sort base IDs: first base isB1, secondB2, retainNRVF. Smoke uses lexicographically first base:4 arms×00/10/01/11, then repeat00/11 for all4 arms, total24. Smoke not extra scientific samples; B1 reruns its planned192 cells.

Keep historical cache-faithful run_one and its raw-capture-before-forcer checks. With prefix lengthL>0, there areL+1 forwards: call0 prefill and calls1…L decode. Calls0…L-1 are history; callL consumes the last prefix token and emits raw answer logits. Never consume the forced answer label before measuring. Terminal cache length must be prompt_tokens+L-1, input length1, final input token=prefix[-1]. Log cache/position/mask dimensions and hash.

Use original ActivationCapper arithmetic, official caps at every original cap layer, original all-position setting. Per-layer call counter gates the unchanged cap hook by H on calls<L and T atL:00 neither,10 history,01 terminal,11 both. Historical cache is retained on mixed conditions. Full independent replay for every cell; no cache branching optimization. Total and enabled hook counts checked per layer. Frozen original model revision, BF16, SDPA, generation seed and forced tokens unchanged. model.generate is expected; natural_generation=false.

## Gates and evidence
Before GPU: CPU gating/boundary tests; deterministic split/matrix checks; original input/code/array hashes; model+cap hashes; freeze own scripts/protocol/dependencies. Model versions/device map match old runtime. Original00/11 captured arrays (full vocabulary logits and saved states) reproduce within1e-4; smoke repeats within1e-4; terminal head reconstruction<=0.25; all finite; no leaked hooks. Every completed cell has JSON+NPZ hash and immutable fingerprint. Reject mixed provenance; retain orphans/failures with timestamps. No scientific response required for gate.

Use GPUs1,2,4,6 with18GiB model max_memory; >=22GiB free each before model load; no CPU/disk offload. Output volume>=20GiB free; input volume>=2GiB free; projected new output<=20% free. Cumulative GPU loading+inference wall budget12h persisted across resumes; smoke<=3h, no-progress90min. After smoke forecast all B1 cells from original per-trace times×mean of all smoke timing ratios with conservative floor1, plus model loading overhead. Stop before exceeding budget; never expand automatically. Supervisor may terminate only its verified task child. Campaign and runner locks prevent duplicate launch.

## Outcomes and statistics
Store full vocabulary logits, four-option ranking and argmax, all original observed states, cap telemetry. broad=mean(other3)-initial; specific=target-mean(two excluding initial/target). Base outcomes: meanNRVF broad; V-minus-F specific.
For either outcome: total=y11-y00; EH=y10-y00; ET=y01-y00; I=y11-y10-y01+y00; ET_given_H=y11-y10; EH_given_T=y11-y01. Check total=EH+ET+I. Save paired base, family and structure results. Average bases within family then families within declared structure; equal8structure weights primary, equal12family weights sensitivity. Neither establishes independent sampling by itself.

B1 interim and B2 separate reports are descriptive, no significance-triggered selection. Once all24bases complete, six raw contrasts (EH/ET/I × broad/specific) form one Holm family, tested once by exact two-sided sign flips at8structure means (256 flips; minimump=.0078125, firstHolm=.046875). Exchangeable independent symmetric structure effects are assumptions, not established facts.20,000 cluster bootstrap draws seed20261005 give descriptive95% intervals; no equivalence claim. B1/B2/pooled always all reported, cannot select favorable view. No interim p value is used to advance work.

Secondary same-prompt affine correction uses old S1 token_split seed20260927, nonoption exclusions and equally weighted vocabulary strata. Fit each10/01/11 independently against00; freeze fitted coefficients for any future C/D patch. Report heldout errors and all nonpositive slopes without clipping or silent exclusion. Raw primary never replaced by corrected outcomes. Candidate branch ranking requires paired effects, uncertainty and heterogeneity; general readout recovery does not license reason-specific claims.

## Scope boundaries
Fixed trajectories may contain prior answer commitments. A annotation retains ambiguity and all traces; it never filters B. A symmetric numerator/denominator decomposition is arithmetic, not causal mediation. H preparation does not depend on B/C/D positivity. Old failed c1_20260928_v2 and completed C1/S1/P1 must never be restarted by this campaign.
