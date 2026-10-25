# C1 statistical appendix

Original estimand: G is the true signed finite-difference response minus mean of eight signed random controls. Four outcomes are condition change in the V/F common mean and V/F interaction for broad/specific margins, with frozen same-prompt S1 scaling.

Unit: 12 equally weighted family means, two bases each, 96 prompts total; 3840 dependent conditions and 52 engineering smoke, one frozen model/configuration. Cross-fit training overlaps; structural independence and symmetric family effects are not established. No parametric Gaussian test is introduced. Exact 4096 sign flips assume independent symmetric family effects. 20,000 percentile family-bootstrap replicates, seed 20260928, give descriptive rather than confirmatory sampling intervals. Holm correction applies to the four frozen tests.

| broad 共同响应变化 | -0.007331 | 0.059117 | [-0.039747, +0.023817] | 0.666016 | 1.000000 | 6/6 |
| broad V/F 交互 | +0.003185 | 0.069942 | [-0.033426, +0.042395] | 0.876953 | 1.000000 | 6/6 |
| specific 共同响应变化 | -0.021831 | 0.039717 | [-0.042865, +0.000202] | 0.084961 | 0.339844 | 3/9 |
| specific V/F 交互 | +0.009971 | 0.101916 | [-0.045128, +0.065529] | 0.745605 | 1.000000 | 6/6 |

Complete numeric family effects and standardized descriptive family effects are in diagnostics.json, family_endpoints.csv, primary_endpoints.csv. Standardized effect is mean/family SD, not a behavioral accuracy change. Family LOFO sensitivity uses existing means; no new training.

Post hoc: absolute logit shifts; absolute and signed odd/even responses; local replacement; donor-directed movement; token rank; patch scale; dose telemetry; BF16 spacing. Family aggregation first averages bases, arms, and matched controls according to each displayed scope. Bootstrap intervals for new diagnostics are unadjusted descriptive summaries. No new p-values have been constructed to upgrade a claim. Exploratory Holm16/Holm488 use saved secondary p-values only; these multiplicity families were chosen after completion and are not the frozen primary plan.

Raw/same/family correction variants are correlated sensitivity estimands; do not select a favorable variant or CI method. Family-out specific-common CI slightly below zero does not contradict its sign-flip p=0.083: these are different methods with coarse, non-Gaussian small-family distributions, and no matched decision rule was prespecified for that secondary contrast.

Every nonzero observed option delta is on the BF16 grid; smallest 0.0625. This is quantization, not a stochastic noise SD. Deterministic exact noop/repeat agreement does not establish arbitrarily fine resolution or general model reliability. No equivalence or power guarantee: no preregistered smallest relevant effect, only 12 reused families, limited confidence interval scope. A ratio of local swap to full-cap mean is not an identified mediation proportion.

Primary computation and source files were already fully audited at closure; this analysis rechecks all consumed input SHA256 against the transferred bundle manifest. Existing raw NPZ integrity evidence remains in the completion audit; this local diagnosis does not claim a second direct NPZ read.
