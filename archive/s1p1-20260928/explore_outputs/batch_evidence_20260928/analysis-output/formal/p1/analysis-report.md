# P1 family-held-out transfer probes

Primary: prompt boundary, layer 53, unit-RMS activations, initial/target identity nuisance fitted on training families only.

| Transfer | Equal-family AUC | Descriptive 95% CI | Balanced accuracy | Paired-label permutation p |
|---|---:|---|---:|---:|
| U->U | 0.8542 | [0.7916666666666666, 0.9166666666666666] | 0.5833 | 0.0099 |
| U->P | 0.8333 | [0.7708333333333334, 0.8958333333333334] | 0.5833 | 0.0099 |
| P->U | 0.8125 | [0.7291666666666666, 0.8958333333333334] | 0.6042 | 0.0099 |
| P->P | 0.8125 | [0.7291666666666666, 0.8958333333333334] | 0.6042 | 0.0099 |

Predictive decoding is not causal evidence. Specific causal patches and independent new structures remain separate experiments.
