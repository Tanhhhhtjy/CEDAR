# Scientific Changelog

E2 separates candidate-reasoning sufficiency from final answer choice and
localizes the official p25 intervention across prefill and decode. The existing
24-base Evidence result becomes an exploratory mechanism anchor, not a new
confirmation set.

CPU construction identified a frozen design constraint: a shared base-level
A/B mapping cannot be exactly balanced within family, easy/medium variant, and
all 12 answer-position transitions simultaneously. Exhaustive search over all
4096 family-balanced assignments gives an optimum of 10/12 balanced
transitions. This limitation is recorded before GPU use and included as a model
covariate; it was not hidden or repaired by changing items.

After E2A completed, a hygiene-only analysis revision renamed the pooled
Validity `criterion` field to `overall_sufficient_rate` and added reasoning
length, answer-step telemetry, and four-state transition summaries under an
explicit `post_hoc_secondary` namespace. These additions do not alter the
frozen primary routing analysis or its Route C result.
