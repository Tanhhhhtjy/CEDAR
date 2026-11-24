# ACL-exp-c3.5 Stage 5 Strict Readout (`qwen3_14b`, M0 anchor IFEval=461, GSM8K=638)

## Dataset: `c3_blind_confirm_v2` (`900` Independent Units / `1080` Items)

| Method | Seed | 4-Cell (UW) | Joint T1∧4C | Row B | Pos `bef/aft` | IFEval / GSM8K | Wiki KL | Health |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `u1_base` | 42 | **50.17%** | **48.94%** | 40.8% | 41.7 / 48.3 | 461 / 642 | `0.001077` | PASS |
| `u2_corridor` | 42 | **51.83%** | **50.44%** | 46.7% | 43.3 / 50.0 | 462 / 636 | `0.001164` | PASS |
| `M3-v4-PureMid-MLP+` | 42 | **56.56%** | **53.78%** | 44.2% | 50.0 / 53.3 | 461 / 635 | `0.000814` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | 42 | **53.06%** | **50.50%** | 44.2% | 48.3 / 50.0 | 467 / 617 | `0.000826` | PASS |

### Paired Comparisons

| Comparison | Split | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | p |
|---|---|---:|---:|---:|
| `Corridor_vs_Base` | `seed_42` | **+1.67** `[0.89, 2.56]` | **+1.50** `[0.72, 2.39]` | 0.0000 |
| `M3-v4-PureMid-MLP+_vs_u2` | `seed_42` | **+4.72** `[3.33, 6.22]` | **+3.33** `[1.72, 4.94]` | 0.0000 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `seed_42` | **+1.22** `[0.11, 2.33]` | **+0.06** `[-1.22, 1.33]` | 0.0312 |
| `v4_MLPplus_vs_v4_RandMask` | `seed_42` | **+3.50** `[2.33, 4.78]` | **+3.28** `[2.11, 4.5]` | 0.0000 |

### Verdicts & Fine-Localization Bonus

```json
{
  "verdicts": {
    "M3-v4-PureMid-MLP+": {
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 4.72,
        "ci95_low_pp": 3.33,
        "joint_ci95_low_pp": 1.72,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 30.07,
        "health_passed": true,
        "path1_superiority": true,
        "path2_efficiency": true,
        "verdict": "PASS_PATH1_SUPERIORITY"
      }
    },
    "M3-v4-PureMid-MLP+-Random-Mask": {
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 1.22,
        "ci95_low_pp": 0.11,
        "joint_ci95_low_pp": -1.22,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 29.04,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      }
    }
  },
  "fine_localization_bonus": {
    "v4_MLPplus_vs_v4_RandMask": {
      "primary_diff_pp": 3.5,
      "primary_ci95_pp": [
        2.33,
        4.78
      ],
      "primary_ci_low_gt_0": true,
      "seed2_diff_pp": null,
      "seed2_same_direction": true,
      "fine_localization_bonus_passed": true
    }
  }
}
```

## Dataset: `c3_blind_confirm_v3` (`900` Independent Units / `1080` Items)

| Method | Seed | 4-Cell (UW) | Joint T1∧4C | Row B | Pos `bef/aft` | IFEval / GSM8K | Wiki KL | Health |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `u1_base` | 42 | **53.67%** | **53.22%** | 57.5% | 48.3 / 50.0 | 461 / 642 | `0.001077` | PASS |
| `u2_corridor` | 42 | **55.22%** | **54.56%** | 65.0% | 53.3 / 51.7 | 462 / 636 | `0.001164` | PASS |
| `M3-v4-PureMid-MLP+` | 42 | **59.67%** | **58.22%** | 61.7% | 56.7 / 58.3 | 461 / 635 | `0.000814` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | 42 | **58.28%** | **56.83%** | 60.8% | 55.0 / 56.7 | 467 / 617 | `0.000826` | PASS |

### Paired Comparisons

| Comparison | Split | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | p |
|---|---|---:|---:|---:|
| `Corridor_vs_Base` | `seed_42` | **+1.56** `[0.61, 2.56]` | **+1.33** `[0.33, 2.33]` | 0.0008 |
| `M3-v4-PureMid-MLP+_vs_u2` | `seed_42` | **+4.44** `[3.06, 5.89]` | **+3.67** `[2.11, 5.28]` | 0.0000 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `seed_42` | **+3.06** `[1.72, 4.44]` | **+2.28** `[0.83, 3.72]` | 0.0000 |
| `v4_MLPplus_vs_v4_RandMask` | `seed_42` | **+1.39** `[0.56, 2.28]` | **+1.39** `[0.56, 2.28]` | 0.0012 |

### Verdicts & Fine-Localization Bonus

```json
{
  "verdicts": {
    "M3-v4-PureMid-MLP+": {
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 4.44,
        "ci95_low_pp": 3.06,
        "joint_ci95_low_pp": 2.11,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 30.07,
        "health_passed": true,
        "path1_superiority": true,
        "path2_efficiency": true,
        "verdict": "PASS_PATH1_SUPERIORITY"
      }
    },
    "M3-v4-PureMid-MLP+-Random-Mask": {
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 3.06,
        "ci95_low_pp": 1.72,
        "joint_ci95_low_pp": 0.83,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 29.04,
        "health_passed": true,
        "path1_superiority": true,
        "path2_efficiency": true,
        "verdict": "PASS_PATH1_SUPERIORITY"
      }
    }
  },
  "fine_localization_bonus": {
    "v4_MLPplus_vs_v4_RandMask": {
      "primary_diff_pp": 1.39,
      "primary_ci95_pp": [
        0.56,
        2.28
      ],
      "primary_ci_low_gt_0": true,
      "seed2_diff_pp": null,
      "seed2_same_direction": true,
      "fine_localization_bonus_passed": true
    }
  }
}
```

## External Natural-Language Benchmark: Ma et al. EX-FEVER (`1,000` Official Test, `no_think`)

| Model | Turn-1 Acc | `n_correct / n_wrong` | `R_UY` (↓) | `R_RU` Evidence (↑) | `R_RU` User-Evid (↑) | Selectivity (`R_RU - R_UY`) |
|---|---:|---:|---:|---:|---:|---:|
| `M0` | 61.8% | 618 / 382 | **16.83%** (104/618) | **81.94%** (313/382) | 45.81% (175/382) | **65.11** |
| `u1_base@s42` | 59.5% | 595 / 405 | **7.56%** (45/595) | **72.84%** (295/405) | 37.28% (151/405) | **65.28** |
| `u2_corridor@s42` | 59.0% | 590 / 410 | **7.8%** (46/590) | **72.44%** (297/410) | 38.78% (159/410) | **64.64** |
| `M3-v4-PureMid-MLP+@s42` | 60.2% | 602 / 398 | **9.8%** (59/602) | **68.84%** (274/398) | 39.95% (159/398) | **59.04** |
| `M3-v4-PureMid-MLP+-Random-Mask@s42` | 60.1% | 601 / 399 | **8.32%** (50/601) | **70.43%** (281/399) | 36.34% (145/399) | **62.11** |

## External Natural-Language Benchmark: Ma et al. EX-FEVER (`1,000` Official Test, `think`)

| Model | Turn-1 Acc | `n_correct / n_wrong` | `R_UY` (↓) | `R_RU` Evidence (↑) | `R_RU` User-Evid (↑) | Selectivity (`R_RU - R_UY`) |
|---|---:|---:|---:|---:|---:|---:|
| `M0` | 66.2% | 662 / 338 | **7.1%** (47/662) | **64.2%** (217/338) | 27.81% (94/338) | **57.1** |
| `u1_base@s42` | 65.9% | 659 / 341 | **5.77%** (38/659) | **54.25%** (185/341) | 23.46% (80/341) | **48.48** |
| `u2_corridor@s42` | 65.4% | 654 / 346 | **3.98%** (26/654) | **57.51%** (199/346) | 23.7% (82/346) | **53.53** |
| `M3-v4-PureMid-MLP+@s42` | 64.3% | 643 / 357 | **5.6%** (36/643) | **55.74%** (199/357) | 22.97% (82/357) | **50.14** |
| `M3-v4-PureMid-MLP+-Random-Mask@s42` | 64.3% | 643 / 357 | **5.29%** (34/643) | **58.82%** (210/357) | 26.61% (95/357) | **53.53** |

