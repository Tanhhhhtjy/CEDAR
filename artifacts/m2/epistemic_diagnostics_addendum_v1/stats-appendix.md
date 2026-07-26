# Statistical appendix

## Units and uncertainty

- Unit of pairing: base × arm × condition.
- Unit of resampling/clustering: structural family.
- Bootstrap draws: 20,000; seed: 20260726.
- Intervals: percentile family-cluster bootstrap 95% intervals.
- SDT extreme rates: Hautus log-linear correction.
- GEE: probit link with exchangeable within-family correlation.
- ECE omitted because 24 items per arm do not support stable binning.

## Missingness

- E2A missing answer-logit rows: [].
- E2C missing answer-logit rows: ['CEDAR-E2C-022-F'].
- Discrete ITT retains invalid/abstain rows; continuous metrics use complete pairs only.

## Multiple comparisons

- Bootstrap intervals and exact effect sizes are primary.
- Nominal GEE/OLS p-values are secondary.
- Holm adjustments are stored in the machine-readable analysis for SDT and moderator
  coefficient families.

## Interpretation rules

- Δd′ below zero: reduced V/F discrimination.
- Δcriterion below zero: more liberal target updating.
- Positive ΔBrier or ΔNLL: worse probabilistic accuracy.
- Positive Δconditional entropy: less concentrated A/B/C/D decision distribution.
- Negative Δoption mass: probability leaves the valid option-token space.
- Probability-flow contrasts are difference-in-differences against matched R.

## Model caveats

- Twelve family clusters are too few for asymptotic cluster-robust p-values to be
  definitive; the family bootstrap is emphasized.
- The SDT mapping is a measurement framework, not a literal cognitive model.
- The MUSE-style proxy is limited by nearly saturated R-baseline option distributions.
