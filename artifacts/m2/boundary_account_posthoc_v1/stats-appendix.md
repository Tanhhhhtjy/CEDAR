# Statistical appendix: Boundary Account v1

## Design

- Bootstrap: 10,000 percentile draws, seed 20260726.
- Primary predictor: uncapped option top-1 minus top-2 logit gap.
- Gaps standardized within dataset × arm/readout on the full valid set.
- Fixed strata: E2 dataset/arm; C2 arm/domain; E−1b seed/variant.
- Natural clusters: E2 family, C2 normalized stem, E−1b pair.
- E2A/E2C use a coupled structural-family cluster in the pooled bootstrap.

## Source estimates

### E2A

- n/flips: 96/15
- slope: 0.908409; 95% CI [0.189713, 2.891150]
- odds ratio: 2.480374
- median gap (flip): 15.000000
- median gap (stable): 16.250000
- AUC using -gap: 0.622222; 95% CI [0.500495, 0.755476]
- nearest-minus-farthest flip rate: 0.125000
- status: `cluster_bootstrap`

### E2C

- n/flips: 95/14
- slope: 0.414668; 95% CI [-0.210066, 1.360976]
- odds ratio: 1.513867
- median gap (flip): 15.562500
- median gap (stable): 17.000000
- AUC using -gap: 0.633157; 95% CI [0.513209, 0.760717]
- nearest-minus-farthest flip rate: 0.208333
- status: `cluster_bootstrap`

### C2

- n/flips: 480/1
- slope: -1.938659; 95% CI [-2.597885, -1.539452]
- odds ratio: 0.143897
- median gap (flip): 5.625000
- median gap (stable): 5.000000
- AUC using -gap: 0.438413; 95% CI [0.392625, 0.479565]
- nearest-minus-farthest flip rate: 0.000000
- status: `descriptive_only_fewer_than_5_flips`

### E-1b

- n/flips: 900/37
- slope: -4.600949; 95% CI [-6.541242, -3.622144]
- odds ratio: 0.010042
- median gap (flip): 0.500000
- median gap (stable): 3.750000
- AUC using -gap: 0.947590; 95% CI [0.927562, 0.964757]
- nearest-minus-farthest flip rate: 0.164444
- status: `cluster_bootstrap`

### Pooled

- n/flips: 1571/67
- slope: -0.922642; 95% CI [-1.457222, -0.587704]
- odds ratio: 0.397468
- median gap (flip): 1.500000
- median gap (stable): 4.437500
- AUC using -gap: 0.568678; 95% CI [0.479699, 0.646856]
- nearest-minus-farthest flip rate: 0.114497
- status: `cluster_bootstrap`

## Leave-one-source-out diagnostic

- Omit C2: pooled slope -0.910427 (`estimated`).
- Omit E-1b: pooled slope 0.564213 (`estimated`).
- Omit E2A: pooled slope -1.702748 (`estimated`).
- Omit E2C: pooled slope -1.378791 (`estimated`).

## Multiplicity

Source-specific bootstrap sign p-values are Holm adjusted only where the 
source has at least five flips. The pooled post-hoc diagnostic is the single 
frozen primary test for this analysis bundle.

## Limitations

- E2C retains one truncated uncapped row with missing logits; it is not imputed.
- C2 direct/free readout disagreement is expected and explicitly measured.
- E−1b margin-to-choice association is partly algebraic.
- Delta-m models contain a post-intervention variable and are not mediation.
