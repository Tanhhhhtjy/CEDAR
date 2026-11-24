# ACL-exp-c3.5 Stage 5 Strict Readout (`qwen3_8b`, M0 anchor IFEval=440, GSM8K=496)

## Dataset: `c3_blind_confirm_v2` (`900` Independent Units / `1080` Items)

| Method | Seed | 4-Cell (UW) | Joint T1∧4C | Row B | Pos `bef/aft` | IFEval / GSM8K | Wiki KL | Health |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `u1_base` | 42 | **46.39%** | **45.61%** | 15.0% | 45.0 / 41.7 | 443 / 538 | `0.001364` | PASS |
| `u1_base` | 1024 | **45.39%** | **44.94%** | 16.7% | 43.3 / 40.0 | 445 / 507 | `0.001350` | PASS |
| `u2_corridor` | 42 | **50.56%** | **49.89%** | 15.8% | 51.7 / 43.3 | 449 / 547 | `0.001362` | PASS |
| `u2_corridor` | 1024 | **48.44%** | **48.00%** | 20.8% | 48.3 / 41.7 | 445 / 540 | `0.001427` | PASS |
| `M3-v4-PureMid-MLP+` | 42 | **50.89%** | **50.17%** | 19.2% | 53.3 / 45.0 | 447 / 538 | `0.000954` | PASS |
| `M3-v4-PureMid-MLP+` | 1024 | **49.61%** | **49.00%** | 18.3% | 53.3 / 43.3 | 445 / 506 | `0.000959` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | 42 | **48.06%** | **47.06%** | 16.7% | 53.3 / 40.0 | 448 / 560 | `0.000947` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | 1024 | **48.61%** | **48.06%** | 17.5% | 55.0 / 45.0 | 445 / 563 | `0.000936` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask2` | 42 | **47.72%** | **46.83%** | 16.7% | 50.0 / 38.3 | 441 / 546 | `0.000962` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask2` | 1024 | **50.11%** | **49.56%** | 19.2% | 56.7 / 43.3 | 441 / 543 | `0.000933` | PASS |

### Paired Comparisons

| Comparison | Split | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | p |
|---|---|---:|---:|---:|
| `Corridor_vs_Base` | `seed_1024` | **+3.06** `[1.89, 4.33]` | **+3.06** `[1.89, 4.33]` | 0.0000 |
| `Corridor_vs_Base` | `seed_42` | **+4.17** `[2.89, 5.56]` | **+4.28** `[3.0, 5.61]` | 0.0000 |
| `Corridor_vs_Base` | `pooled_2seed` | **+3.61** `[2.64, 4.67]` | **+3.67** `[2.67, 4.7]` | 0.0000 |
| `M3-v4-PureMid-MLP+_vs_u2` | `seed_1024` | **+1.17** `[-0.33, 2.72]` | **+1.00** `[-0.5, 2.56]` | 0.1400 |
| `M3-v4-PureMid-MLP+_vs_u2` | `seed_42` | **+0.33** `[-1.0, 1.67]` | **+0.28** `[-1.06, 1.61]` | 0.6528 |
| `M3-v4-PureMid-MLP+_vs_u2` | `pooled_2seed` | **+0.75** `[-0.42, 1.92]` | **+0.64** `[-0.56, 1.83]` | 0.2196 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `seed_1024` | **+0.17** `[-1.33, 1.72]` | **+0.06** `[-1.5, 1.61]` | 0.8768 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `seed_42` | **-2.50** `[-3.83, -1.22]` | **-2.83** `[-4.22, -1.5]` | 0.0004 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `pooled_2seed` | **-1.17** `[-2.28, -0.08]` | **-1.39** `[-2.5, -0.31]` | 0.0376 |
| `M3-v4-PureMid-MLP+-Random-Mask2_vs_u2` | `seed_1024` | **+1.67** `[0.17, 3.22]` | **+1.56** `[0.06, 3.11]` | 0.0320 |
| `M3-v4-PureMid-MLP+-Random-Mask2_vs_u2` | `seed_42` | **-2.83** `[-4.22, -1.5]` | **-3.06** `[-4.45, -1.67]` | 0.0000 |
| `M3-v4-PureMid-MLP+-Random-Mask2_vs_u2` | `pooled_2seed` | **-0.58** `[-1.67, 0.47]` | **-0.75** `[-1.86, 0.33]` | 0.2860 |
| `v4_MLPplus_vs_v4_RandMask` | `seed_1024` | **+1.00** `[-0.22, 2.28]` | **+0.94** `[-0.28, 2.22]` | 0.1148 |
| `v4_MLPplus_vs_v4_RandMask` | `seed_42` | **+2.83** `[1.67, 3.95]` | **+3.11** `[1.94, 4.33]` | 0.0000 |
| `v4_MLPplus_vs_v4_RandMask` | `pooled_2seed` | **+1.92** `[0.97, 2.89]` | **+2.03** `[1.08, 3.03]` | 0.0000 |
| `v4_MLPplus_vs_v4_RandMask2` | `seed_1024` | **-0.50** `[-1.5, 0.5]` | **-0.56** `[-1.56, 0.44]` | 0.3380 |
| `v4_MLPplus_vs_v4_RandMask2` | `seed_42` | **+3.17** `[2.0, 4.33]` | **+3.33** `[2.17, 4.5]` | 0.0000 |
| `v4_MLPplus_vs_v4_RandMask2` | `pooled_2seed` | **+1.33** `[0.56, 2.14]` | **+1.39** `[0.61, 2.19]` | 0.0028 |

### Verdicts & Fine-Localization Bonus

```json
{
  "verdicts": {
    "M3-v4-PureMid-MLP+": {
      "seed_1024": {
        "candidate": "M3-v4-PureMid-MLP+@s1024",
        "vs_strongest_uniform": "u2_corridor@s1024",
        "diff_pp": 1.17,
        "ci95_low_pp": -0.33,
        "joint_ci95_low_pp": -0.5,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 32.8,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 0.33,
        "ci95_low_pp": -1.0,
        "joint_ci95_low_pp": -1.06,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 29.96,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "pooled_2seed": {
        "candidate": "M3-v4-PureMid-MLP+@2seed",
        "vs_strongest_uniform": "u2_corridor@2seed",
        "diff_pp": 0.75,
        "ci95_low_pp": -0.42,
        "joint_ci95_low_pp": -0.56,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 31.41,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      }
    },
    "M3-v4-PureMid-MLP+-Random-Mask": {
      "seed_1024": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@s1024",
        "vs_strongest_uniform": "u2_corridor@s1024",
        "diff_pp": 0.17,
        "ci95_low_pp": -1.33,
        "joint_ci95_low_pp": -1.5,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 34.41,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": -2.5,
        "ci95_low_pp": -3.83,
        "joint_ci95_low_pp": -4.22,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 30.47,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": false,
        "verdict": "HOLD"
      },
      "pooled_2seed": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@2seed",
        "vs_strongest_uniform": "u2_corridor@2seed",
        "diff_pp": -1.17,
        "ci95_low_pp": -2.28,
        "joint_ci95_low_pp": -2.5,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 32.48,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": false,
        "verdict": "HOLD"
      }
    },
    "M3-v4-PureMid-MLP+-Random-Mask2": {
      "seed_1024": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask2@s1024",
        "vs_strongest_uniform": "u2_corridor@s1024",
        "diff_pp": 1.67,
        "ci95_low_pp": 0.17,
        "joint_ci95_low_pp": 0.06,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 34.62,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask2@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": -2.83,
        "ci95_low_pp": -4.22,
        "joint_ci95_low_pp": -4.45,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 29.37,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": false,
        "verdict": "HOLD"
      },
      "pooled_2seed": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask2@2seed",
        "vs_strongest_uniform": "u2_corridor@2seed",
        "diff_pp": -0.58,
        "ci95_low_pp": -1.67,
        "joint_ci95_low_pp": -1.86,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 32.05,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      }
    }
  },
  "fine_localization_bonus": {
    "v4_MLPplus_vs_v4_RandMask": {
      "primary_diff_pp": 1.92,
      "primary_ci95_pp": [
        0.97,
        2.89
      ],
      "primary_ci_low_gt_0": true,
      "seed2_diff_pp": 1.0,
      "seed2_same_direction": true,
      "fine_localization_bonus_passed": true
    },
    "v4_MLPplus_vs_v4_RandMask2": {
      "primary_diff_pp": 1.33,
      "primary_ci95_pp": [
        0.56,
        2.14
      ],
      "primary_ci_low_gt_0": true,
      "seed2_diff_pp": -0.5,
      "seed2_same_direction": false,
      "fine_localization_bonus_passed": false
    }
  }
}
```

## Dataset: `c3_blind_confirm_v3` (`900` Independent Units / `1080` Items)

| Method | Seed | 4-Cell (UW) | Joint T1∧4C | Row B | Pos `bef/aft` | IFEval / GSM8K | Wiki KL | Health |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `u1_base` | 42 | **36.61%** | **36.11%** | 31.7% | 30.0 / 36.7 | 443 / 538 | `0.001364` | PASS |
| `u1_base` | 1024 | **35.89%** | **35.56%** | 30.0% | 31.7 / 36.7 | 445 / 507 | `0.001350` | PASS |
| `u2_corridor` | 42 | **38.61%** | **38.11%** | 31.7% | 35.0 / 36.7 | 449 / 547 | `0.001362` | PASS |
| `u2_corridor` | 1024 | **38.39%** | **38.06%** | 36.7% | 36.7 / 38.3 | 445 / 540 | `0.001427` | PASS |
| `M3-v4-PureMid-MLP+` | 42 | **39.83%** | **39.50%** | 36.7% | 36.7 / 41.7 | 447 / 538 | `0.000954` | PASS |
| `M3-v4-PureMid-MLP+` | 1024 | **40.39%** | **40.17%** | 32.5% | 38.3 / 45.0 | 445 / 506 | `0.000959` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | 42 | **38.61%** | **38.22%** | 33.3% | 35.0 / 38.3 | 448 / 560 | `0.000947` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask` | 1024 | **38.00%** | **37.78%** | 17.5% | 38.3 / 41.7 | 445 / 563 | `0.000936` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask2` | 42 | **38.72%** | **38.33%** | 35.0% | 36.7 / 36.7 | 441 / 546 | `0.000962` | PASS |
| `M3-v4-PureMid-MLP+-Random-Mask2` | 1024 | **39.33%** | **39.11%** | 23.3% | 38.3 / 41.7 | 441 / 543 | `0.000933` | PASS |

### Paired Comparisons

| Comparison | Split | 4-Cell Diff [95% CI] | Joint Diff [95% CI] | p |
|---|---|---:|---:|---:|
| `Corridor_vs_Base` | `seed_1024` | **+2.50** `[1.56, 3.56]` | **+2.50** `[1.5, 3.61]` | 0.0000 |
| `Corridor_vs_Base` | `seed_42` | **+2.00** `[1.11, 2.95]` | **+2.00** `[1.11, 2.94]` | 0.0000 |
| `Corridor_vs_Base` | `pooled_2seed` | **+2.25** `[1.5, 3.11]` | **+2.25** `[1.5, 3.08]` | 0.0000 |
| `M3-v4-PureMid-MLP+_vs_u2` | `seed_1024` | **+2.00** `[0.67, 3.33]` | **+2.11** `[0.78, 3.39]` | 0.0012 |
| `M3-v4-PureMid-MLP+_vs_u2` | `seed_42` | **+1.22** `[0.22, 2.28]` | **+1.39** `[0.33, 2.44]` | 0.0176 |
| `M3-v4-PureMid-MLP+_vs_u2` | `pooled_2seed` | **+1.61** `[0.67, 2.61]` | **+1.75** `[0.78, 2.75]` | 0.0004 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `seed_1024` | **-0.39** `[-1.72, 0.94]` | **-0.28** `[-1.61, 1.06]` | 0.5896 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `seed_42` | **+0.00** `[-1.0, 1.0]` | **+0.11** `[-0.89, 1.11]` | 1.0000 |
| `M3-v4-PureMid-MLP+-Random-Mask_vs_u2` | `pooled_2seed` | **-0.19** `[-1.17, 0.78]` | **-0.08** `[-1.06, 0.89]` | 0.6956 |
| `M3-v4-PureMid-MLP+-Random-Mask2_vs_u2` | `seed_1024` | **+0.94** `[-0.33, 2.28]` | **+1.06** `[-0.22, 2.39]` | 0.1776 |
| `M3-v4-PureMid-MLP+-Random-Mask2_vs_u2` | `seed_42` | **+0.11** `[-0.83, 1.0]` | **+0.22** `[-0.72, 1.11]` | 0.8660 |
| `M3-v4-PureMid-MLP+-Random-Mask2_vs_u2` | `pooled_2seed` | **+0.53** `[-0.36, 1.44]` | **+0.64** `[-0.25, 1.56]` | 0.2784 |
| `v4_MLPplus_vs_v4_RandMask` | `seed_1024` | **+2.39** `[1.5, 3.28]` | **+2.39** `[1.5, 3.28]` | 0.0000 |
| `v4_MLPplus_vs_v4_RandMask` | `seed_42` | **+1.22** `[0.39, 2.11]` | **+1.28** `[0.44, 2.22]` | 0.0044 |
| `v4_MLPplus_vs_v4_RandMask` | `pooled_2seed` | **+1.81** `[1.17, 2.47]` | **+1.83** `[1.19, 2.5]` | 0.0000 |
| `v4_MLPplus_vs_v4_RandMask2` | `seed_1024` | **+1.06** `[0.39, 1.78]` | **+1.06** `[0.39, 1.78]` | 0.0016 |
| `v4_MLPplus_vs_v4_RandMask2` | `seed_42` | **+1.11** `[0.44, 1.83]` | **+1.17** `[0.5, 1.89]` | 0.0000 |
| `v4_MLPplus_vs_v4_RandMask2` | `pooled_2seed` | **+1.08** `[0.58, 1.61]` | **+1.11** `[0.61, 1.67]` | 0.0000 |

### Verdicts & Fine-Localization Bonus

```json
{
  "verdicts": {
    "M3-v4-PureMid-MLP+": {
      "seed_1024": {
        "candidate": "M3-v4-PureMid-MLP+@s1024",
        "vs_strongest_uniform": "u2_corridor@s1024",
        "diff_pp": 2.0,
        "ci95_low_pp": 0.67,
        "joint_ci95_low_pp": 0.78,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 32.8,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 1.22,
        "ci95_low_pp": 0.22,
        "joint_ci95_low_pp": 0.33,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 29.96,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "pooled_2seed": {
        "candidate": "M3-v4-PureMid-MLP+@2seed",
        "vs_strongest_uniform": "u2_corridor@2seed",
        "diff_pp": 1.61,
        "ci95_low_pp": 0.67,
        "joint_ci95_low_pp": 0.78,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 31.41,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      }
    },
    "M3-v4-PureMid-MLP+-Random-Mask": {
      "seed_1024": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@s1024",
        "vs_strongest_uniform": "u2_corridor@s1024",
        "diff_pp": -0.39,
        "ci95_low_pp": -1.72,
        "joint_ci95_low_pp": -1.61,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 34.41,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 0.0,
        "ci95_low_pp": -1.0,
        "joint_ci95_low_pp": -0.89,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 30.47,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "pooled_2seed": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask@2seed",
        "vs_strongest_uniform": "u2_corridor@2seed",
        "diff_pp": -0.19,
        "ci95_low_pp": -1.17,
        "joint_ci95_low_pp": -1.06,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 32.48,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      }
    },
    "M3-v4-PureMid-MLP+-Random-Mask2": {
      "seed_1024": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask2@s1024",
        "vs_strongest_uniform": "u2_corridor@s1024",
        "diff_pp": 0.94,
        "ci95_low_pp": -0.33,
        "joint_ci95_low_pp": -0.22,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 34.62,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "seed_42": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask2@s42",
        "vs_strongest_uniform": "u2_corridor@s42",
        "diff_pp": 0.11,
        "ci95_low_pp": -0.83,
        "joint_ci95_low_pp": -0.72,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 29.37,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      },
      "pooled_2seed": {
        "candidate": "M3-v4-PureMid-MLP+-Random-Mask2@2seed",
        "vs_strongest_uniform": "u2_corridor@2seed",
        "diff_pp": 0.53,
        "ci95_low_pp": -0.36,
        "joint_ci95_low_pp": -0.25,
        "beats_rnd_iit_pp": null,
        "beats_no_iit_pp": null,
        "iit_controls_passed": true,
        "kl_reduction_vs_u_best_pct": 32.05,
        "health_passed": true,
        "path1_superiority": false,
        "path2_efficiency": true,
        "verdict": "PASS_PATH2_EFFICIENCY_PARETO"
      }
    }
  },
  "fine_localization_bonus": {
    "v4_MLPplus_vs_v4_RandMask": {
      "primary_diff_pp": 1.81,
      "primary_ci95_pp": [
        1.17,
        2.47
      ],
      "primary_ci_low_gt_0": true,
      "seed2_diff_pp": 2.39,
      "seed2_same_direction": true,
      "fine_localization_bonus_passed": true
    },
    "v4_MLPplus_vs_v4_RandMask2": {
      "primary_diff_pp": 1.08,
      "primary_ci95_pp": [
        0.58,
        1.61
      ],
      "primary_ci_low_gt_0": true,
      "seed2_diff_pp": 1.06,
      "seed2_same_direction": true,
      "fine_localization_bonus_passed": true
    }
  }
}
```

## External Natural-Language Benchmark: Ma et al. EX-FEVER (`1,000` Official Test, `no_think`)

| Model | Turn-1 Acc | `n_correct / n_wrong` | `R_UY` (↓) | `R_RU` Evidence (↑) | `R_RU` User-Evid (↑) | Selectivity (`R_RU - R_UY`) |
|---|---:|---:|---:|---:|---:|---:|
| `M0` | 62.7% | 627 / 373 | **35.73%** (224/627) | **79.09%** (295/373) | 47.45% (177/373) | **43.36** |
| `u1_base@s42` | 63.5% | 635 / 365 | **11.02%** (70/635) | **70.68%** (258/365) | 39.45% (144/365) | **59.66** |
| `u1_base@s1024` | 62.6% | 626 / 374 | **10.7%** (67/626) | **70.86%** (265/374) | 38.5% (144/374) | **60.16** |
| `u2_corridor@s42` | 63.2% | 632 / 368 | **9.02%** (57/632) | **68.21%** (251/368) | 38.86% (143/368) | **59.19** |
| `u2_corridor@s1024` | 62.1% | 621 / 379 | **8.37%** (52/621) | **70.45%** (267/379) | 37.2% (141/379) | **62.08** |
| `M3-v4-PureMid-MLP+@s42` | 64.0% | 640 / 360 | **11.41%** (73/640) | **68.06%** (245/360) | 41.39% (149/360) | **56.65** |
| `M3-v4-PureMid-MLP+@s1024` | 62.9% | 629 / 371 | **10.97%** (69/629) | **65.5%** (243/371) | 38.81% (144/371) | **54.53** |
| `M3-v4-PureMid-MLP+-Random-Mask@s42` | 63.8% | 638 / 362 | **10.5%** (67/638) | **66.85%** (242/362) | 39.78% (144/362) | **56.35** |
| `M3-v4-PureMid-MLP+-Random-Mask@s1024` | 63.1% | 631 / 369 | **11.25%** (71/631) | **67.75%** (250/369) | 40.92% (151/369) | **56.5** |
| `M3-v4-PureMid-MLP+-Random-Mask2@s42` | 63.1% | 631 / 369 | **10.62%** (67/631) | **69.11%** (255/369) | 39.84% (147/369) | **58.49** |
| `M3-v4-PureMid-MLP+-Random-Mask2@s1024` | 62.6% | 626 / 374 | **10.7%** (67/626) | **66.31%** (248/374) | 41.71% (156/374) | **55.61** |

## External Natural-Language Benchmark: Ma et al. EX-FEVER (`1,000` Official Test, `think`)

| Model | Turn-1 Acc | `n_correct / n_wrong` | `R_UY` (↓) | `R_RU` Evidence (↑) | `R_RU` User-Evid (↑) | Selectivity (`R_RU - R_UY`) |
|---|---:|---:|---:|---:|---:|---:|
| `M0` | 64.7% | 647 / 353 | **14.84%** (96/647) | **70.82%** (250/353) | 31.73% (112/353) | **55.98** |
| `u1_base@s42` | 64.6% | 646 / 354 | **12.07%** (78/646) | **66.95%** (237/354) | 25.42% (90/354) | **54.88** |
| `u1_base@s1024` | 64.5% | 645 / 355 | **12.4%** (80/645) | **61.13%** (217/355) | 27.89% (99/355) | **48.73** |
| `u2_corridor@s42` | 64.3% | 643 / 357 | **9.02%** (58/643) | **63.87%** (228/357) | 27.17% (97/357) | **54.85** |
| `u2_corridor@s1024` | 63.8% | 638 / 362 | **10.97%** (70/638) | **64.09%** (232/362) | 27.9% (101/362) | **53.12** |
| `M3-v4-PureMid-MLP+@s42` | 63.7% | 637 / 363 | **13.81%** (88/637) | **63.64%** (231/363) | 25.9% (94/363) | **49.83** |
| `M3-v4-PureMid-MLP+@s1024` | 64.7% | 647 / 353 | **14.37%** (93/647) | **60.62%** (214/353) | 24.93% (88/353) | **46.25** |
| `M3-v4-PureMid-MLP+-Random-Mask@s42` | 63.5% | 635 / 365 | **14.49%** (92/635) | **64.66%** (236/365) | 28.77% (105/365) | **50.17** |
| `M3-v4-PureMid-MLP+-Random-Mask@s1024` | 64.6% | 646 / 354 | **16.56%** (107/646) | **65.25%** (231/354) | 29.1% (103/354) | **48.69** |
| `M3-v4-PureMid-MLP+-Random-Mask2@s42` | 64.5% | 645 / 355 | **15.35%** (99/645) | **63.94%** (227/355) | 27.04% (96/355) | **48.59** |
| `M3-v4-PureMid-MLP+-Random-Mask2@s1024` | 66.0% | 660 / 340 | **13.94%** (92/660) | **61.18%** (208/340) | 25.59% (87/340) | **47.24** |

