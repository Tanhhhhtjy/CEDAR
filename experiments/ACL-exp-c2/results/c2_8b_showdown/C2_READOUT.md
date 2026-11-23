# ACL-exp-c2 · Verification-Gated `DAC-IIT` 2×2 因子因果对决统计汇总（Qwen3-8B）

- **最终判定**：**`HOLD`**
- **判定明细**：`{"flagship_healthy": true, "beats_rnd_iit_pool_3pp_ci_pos": true, "positive_2x2_interaction": true, "beats_or_pareto_unif_base": false, "beats_stealth_stress": true, "seed2_verified": false}`

## 一、通用能力、逐组无损保真与安全护盾（No-Regression Shield）

| 组别 | 逐组保真 | IFEval (≥430) | GSM8K (≥470) | Wiki KL (≤0.005) | `S` 拒 | `shell` 拒 | `P0/I0` 守 | `V_hes` 接 | `think` 失败 | 健康 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dac_iit` | ✅ | 451 | 495 | 0.000997 | 100.0% | 100.0% | 100.0%/100.0% | 96.7% | 0.1% | **PASS** |
| `dac_paired` | ✅ | 444 | 475 | 0.000997 | 100.0% | 100.0% | 100.0%/100.0% | 96.7% | 0.0% | **PASS** |
| `dac_base` | ✅ | 445 | 533 | 0.000989 | 100.0% | 100.0% | 100.0%/100.0% | 95.4% | 0.1% | **PASS** |
| `rnd1_iit` | ✅ | 439 | 514 | 0.001016 | 100.0% | 100.0% | 100.0%/100.0% | 97.5% | 0.0% | **PASS** |
| `rnd2_iit` | ✅ | 438 | 499 | 0.000981 | 100.0% | 100.0% | 100.0%/100.0% | 97.1% | 0.0% | **PASS** |
| `rnd1_base` | ✅ | 442 | 520 | 0.001014 | 100.0% | 100.0% | 100.0%/100.0% | 97.1% | 0.0% | **PASS** |
| `unif_paired` | ✅ | 449 | 479 | 0.001451 | 100.0% | 100.0% | 100.0%/100.0% | 95.4% | 0.1% | **PASS** |
| `unif_base` | ✅ | 443 | 507 | 0.001461 | 100.0% | 100.0% | 100.0%/100.0% | 97.1% | 0.1% | **PASS** |

## 二、三轴核心成绩表（`shared_eligible` 分母：`DC(V_hes, F_auth)` / `DC(V, F)`）

| 组别 | 轴1 `confirm_bidir` nt | 轴1 CE RowA / RowB nt | 轴1 `confirm_bidir` `think` | 轴2 后置深埋 nt / `think` | 轴2 古希腊诡辩 nt / `think` | 轴3 `tpl_v2` nt / `think` | `c1_dev` nt / `think` |
|---|---|---|---|---|---|---|---|
| `M0` | 0.0% (0.0%) | 0.0% (0.0%) / 0.0% (0.0%) | 92.2% (87.8%) | 0.0% (0.0%) / 100.0% (99.2%) | 1.8% (0.0%) / 72.4% (55.2%) | 0.0% (0.0%) / 47.5% (50.0%) | 0.0% (0.0%) / 91.2% (86.2%) |
| `dac_iit` | 82.8% (80.8%) | 100.0% (100.0%) / 56.5% (30.4%) | 98.9% (98.1%) | 46.5% (44.2%) / 100.0% (99.2%) | 90.7% (72.2%) / 86.2% (77.6%) | 76.9% (69.2%) / 80.0% (80.0%) | 77.8% (77.8%) / 95.0% (93.8%) |
| `dac_paired` | 84.5% (83.3%) | 100.0% (100.0%) / 60.9% (39.1%) | 99.4% (98.3%) | 44.2% (44.2%) / 99.2% (100.0%) | 90.7% (79.6%) / 87.9% (82.8%) | 76.9% (73.1%) / 81.7% (80.8%) | 82.2% (77.8%) / 93.8% (93.8%) |
| `dac_base` | 83.3% (82.8%) | 100.0% (100.0%) / 52.2% (34.8%) | 99.2% (98.1%) | 46.5% (46.5%) / 100.0% (100.0%) | 88.9% (53.7%) / 86.2% (77.6%) | 73.1% (73.1%) / 82.5% (79.2%) | 82.2% (77.8%) / 96.2% (93.8%) |
| `rnd1_iit` | 74.9% (67.4%) | 100.0% (100.0%) / 26.1% (13.0%) | 98.6% (97.2%) | 30.2% (18.6%) / 100.0% (100.0%) | 88.9% (55.6%) / 86.2% (84.5%) | 65.4% (61.5%) / 83.3% (81.7%) | 68.9% (57.8%) / 96.2% (96.2%) |
| `rnd2_iit` | 77.4% (67.0%) | 100.0% (100.0%) / 30.4% (17.4%) | 97.5% (97.8%) | 25.6% (14.0%) / 100.0% (100.0%) | 90.7% (53.7%) / 87.9% (79.3%) | 73.1% (61.5%) / 80.0% (77.5%) | 66.7% (62.2%) / 93.8% (96.2%) |
| `rnd1_base` | 83.7% (78.7%) | 100.0% (100.0%) / 39.1% (26.1%) | 97.5% (97.8%) | 46.5% (46.5%) / 100.0% (100.0%) | 88.9% (50.0%) / 87.9% (82.8%) | 73.1% (65.4%) / 81.7% (80.8%) | 77.8% (73.3%) / 96.2% (98.8%) |
| `unif_paired` | 89.1% (90.8%) | 100.0% (100.0%) / 87.0% (87.0%) | 99.2% (99.7%) | 58.1% (51.2%) / 100.0% (100.0%) | 88.9% (92.6%) / 91.4% (91.4%) | 76.9% (76.9%) / 85.8% (82.5%) | 82.2% (84.4%) / 95.0% (97.5%) |
| `unif_base` | 92.0% (92.0%) | 100.0% (100.0%) / 87.0% (78.3%) | 99.7% (99.2%) | 69.8% (55.8%) / 100.0% (100.0%) | 94.4% (88.9%) / 94.8% (86.2%) | 84.6% (80.8%) / 90.0% (86.7%) | 86.7% (86.7%) / 96.2% (97.5%) |

## 三、2×2 因子交互归因检验（`DC(V_hes, F_auth)`，簇配对 Bootstrap 95% CI）

| 评测切片 | 对比项 | Δ (pp) | 95% 簇 CI (pp) | p 值 |
|---|---|---:|---|---:|
| `c2_confirm_bidir|no_think|shared_eligible` | `dac_iit_vs_dac_base` | **-0.42** | [-2.55, +1.68] | McN=1.0000 |
| `c2_confirm_bidir|no_think|shared_eligible` | `dac_iit_vs_dac_paired` | **-1.67** | [-3.75, +0.00] | McN=0.2188 |
| `c2_confirm_bidir|no_think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **-8.79** | [-12.70, -5.08] | McN=0.0000 |
| `c2_confirm_bidir|no_think|shared_eligible` | `interaction_2x2_location_x_iit` | **+8.37** | [+4.26, +12.61] | boot=0.0000 |
| `c2_confirm_bidir|no_think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+7.95** | [+4.06, +12.08] | McN=0.0003 |
| `c2_confirm_bidir|no_think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+5.44** | [+2.11, +9.02] | McN=0.0044 |
| `c2_confirm_bidir|no_think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+6.69** | [+3.31, +10.32] | boot=0.0000 |
| `c2_confirm_bidir|no_think|shared_eligible` | `dac_iit_vs_unif_base` | **-9.21** | [-13.08, -5.51] | McN=0.0000 |
| `c2_confirm_bidir|no_think|shared_eligible` | `dac_iit_vs_unif_paired` | **-6.28** | [-10.20, -2.53] | McN=0.0026 |
| `c2_confirm_bidir|think|shared_eligible` | `dac_iit_vs_dac_base` | **-0.28** | [-1.34, +0.56] | McN=1.0000 |
| `c2_confirm_bidir|think|shared_eligible` | `dac_iit_vs_dac_paired` | **-0.56** | [-1.41, +0.00] | McN=0.5000 |
| `c2_confirm_bidir|think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **+1.11** | [-0.28, +2.75] | McN=0.2891 |
| `c2_confirm_bidir|think|shared_eligible` | `interaction_2x2_location_x_iit` | **-1.39** | [-3.27, +0.28] | boot=0.1630 |
| `c2_confirm_bidir|think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+0.28** | [-1.12, +1.70] | McN=1.0000 |
| `c2_confirm_bidir|think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+1.39** | [-0.27, +3.08] | McN=0.1797 |
| `c2_confirm_bidir|think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+0.83** | [-0.54, +2.21] | boot=0.2666 |
| `c2_confirm_bidir|think|shared_eligible` | `dac_iit_vs_unif_base` | **-0.83** | [-2.20, +0.28] | McN=0.3750 |
| `c2_confirm_bidir|think|shared_eligible` | `dac_iit_vs_unif_paired` | **-0.28** | [-1.69, +1.12] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `dac_iit_vs_dac_base` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `dac_iit_vs_dac_paired` | **+2.33** | [+0.00, +6.98] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **-16.28** | [-27.91, -6.98] | McN=0.0156 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `interaction_2x2_location_x_iit` | **+16.28** | [+6.98, +27.91] | boot=0.0010 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+16.28** | [+6.98, +27.91] | McN=0.0156 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+20.93** | [+9.30, +32.56] | McN=0.0039 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+18.60** | [+8.14, +30.23] | boot=0.0000 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `dac_iit_vs_unif_base` | **-23.26** | [-37.21, -11.63] | McN=0.0020 |
| `c2_stealth_stress:late_buried_depth3|no_think|shared_eligible` | `dac_iit_vs_unif_paired` | **-11.63** | [-23.26, -2.33] | McN=0.0625 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `dac_iit_vs_dac_base` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `dac_iit_vs_dac_paired` | **+0.83** | [+0.00, +2.50] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `interaction_2x2_location_x_iit` | **+0.00** | [+0.00, +0.00] | boot=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+0.00** | [+0.00, +0.00] | boot=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `dac_iit_vs_unif_base` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:late_buried_depth3|think|shared_eligible` | `dac_iit_vs_unif_paired` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `dac_iit_vs_dac_base` | **+1.85** | [-3.70, +7.41] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `dac_iit_vs_dac_paired` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **+0.00** | [-7.41, +7.41] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `interaction_2x2_location_x_iit` | **+1.85** | [-5.56, +9.26] | boot=0.8304 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+1.85** | [-7.41, +11.11] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+0.00** | [-9.26, +9.26] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+0.93** | [-7.41, +9.26] | boot=0.9068 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `dac_iit_vs_unif_base` | **-3.70** | [-11.11, +3.70] | McN=0.6250 |
| `c2_stealth_stress:classical_natural_sophisms|no_think|shared_eligible` | `dac_iit_vs_unif_paired` | **+1.85** | [-3.70, +7.41] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `dac_iit_vs_dac_base` | **+0.00** | [-8.62, +8.62] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `dac_iit_vs_dac_paired` | **-1.72** | [-5.17, +0.00] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **-1.72** | [-8.62, +5.17] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `interaction_2x2_location_x_iit` | **+1.72** | [-5.17, +8.62] | boot=0.8318 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+0.00** | [-5.17, +5.17] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **-1.72** | [-8.62, +5.17] | McN=1.0000 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **-0.86** | [-5.17, +3.45] | boot=0.8582 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `dac_iit_vs_unif_base` | **-8.62** | [-17.24, +0.00] | McN=0.1250 |
| `c2_stealth_stress:classical_natural_sophisms|think|shared_eligible` | `dac_iit_vs_unif_paired` | **-5.17** | [-12.07, +0.00] | McN=0.2500 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `dac_iit_vs_dac_base` | **+3.85** | [+0.00, +11.54] | McN=1.0000 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `dac_iit_vs_dac_paired` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **-7.69** | [-23.08, +7.69] | McN=0.6250 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `interaction_2x2_location_x_iit` | **+11.54** | [+0.00, +23.08] | boot=0.0850 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+11.54** | [+0.00, +23.08] | McN=0.2500 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+3.85** | [-7.69, +15.38] | McN=1.0000 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+7.69** | [-1.92, +19.23] | boot=0.2434 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `dac_iit_vs_unif_base` | **-7.69** | [-23.08, +7.69] | McN=0.6250 |
| `c1_proxy_template_v2|no_think|shared_eligible` | `dac_iit_vs_unif_paired` | **+0.00** | [-15.38, +15.38] | McN=1.0000 |
| `c1_proxy_template_v2|think|shared_eligible` | `dac_iit_vs_dac_base` | **-2.50** | [-6.67, +1.67] | McN=0.4531 |
| `c1_proxy_template_v2|think|shared_eligible` | `dac_iit_vs_dac_paired` | **-1.67** | [-5.83, +2.50] | McN=0.6875 |
| `c1_proxy_template_v2|think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **+1.67** | [-3.33, +6.67] | McN=0.7266 |
| `c1_proxy_template_v2|think|shared_eligible` | `interaction_2x2_location_x_iit` | **-4.17** | [-10.83, +2.50] | boot=0.2656 |
| `c1_proxy_template_v2|think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **-3.33** | [-7.50, +0.00] | McN=0.2188 |
| `c1_proxy_template_v2|think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+0.00** | [-4.17, +4.17] | McN=1.0000 |
| `c1_proxy_template_v2|think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **-1.67** | [-5.00, +1.67] | boot=0.3968 |
| `c1_proxy_template_v2|think|shared_eligible` | `dac_iit_vs_unif_base` | **-10.00** | [-15.83, -5.00] | McN=0.0005 |
| `c1_proxy_template_v2|think|shared_eligible` | `dac_iit_vs_unif_paired` | **-5.83** | [-10.00, -1.67] | McN=0.0156 |
| `c1_dev|no_think|shared_eligible` | `dac_iit_vs_dac_base` | **-4.44** | [-11.11, +0.00] | McN=0.5000 |
| `c1_dev|no_think|shared_eligible` | `dac_iit_vs_dac_paired` | **-4.44** | [-11.11, +0.00] | McN=0.5000 |
| `c1_dev|no_think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **-8.89** | [-20.00, +2.22] | McN=0.2188 |
| `c1_dev|no_think|shared_eligible` | `interaction_2x2_location_x_iit` | **+4.44** | [-6.67, +15.56] | boot=0.5408 |
| `c1_dev|no_think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **+8.89** | [-2.22, +22.22] | McN=0.2891 |
| `c1_dev|no_think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+11.11** | [-2.22, +24.44] | McN=0.1797 |
| `c1_dev|no_think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+10.00** | [-2.22, +22.22] | boot=0.1152 |
| `c1_dev|no_think|shared_eligible` | `dac_iit_vs_unif_base` | **-8.89** | [-17.78, -2.22] | McN=0.1250 |
| `c1_dev|no_think|shared_eligible` | `dac_iit_vs_unif_paired` | **-4.44** | [-15.56, +6.67] | McN=0.6875 |
| `c1_dev|think|shared_eligible` | `dac_iit_vs_dac_base` | **-1.25** | [-3.75, +0.00] | McN=1.0000 |
| `c1_dev|think|shared_eligible` | `dac_iit_vs_dac_paired` | **+1.25** | [+0.00, +3.75] | McN=1.0000 |
| `c1_dev|think|shared_eligible` | `rnd1_iit_vs_rnd1_base` | **+0.00** | [+0.00, +0.00] | McN=1.0000 |
| `c1_dev|think|shared_eligible` | `interaction_2x2_location_x_iit` | **-1.25** | [-3.75, +0.00] | boot=0.7264 |
| `c1_dev|think|shared_eligible` | `dac_iit_vs_rnd1_iit` | **-1.25** | [-6.25, +2.50] | McN=1.0000 |
| `c1_dev|think|shared_eligible` | `dac_iit_vs_rnd2_iit` | **+1.25** | [+0.00, +3.75] | McN=1.0000 |
| `c1_dev|think|shared_eligible` | `dac_iit_vs_rnd_iit_pool` | **+0.00** | [-2.50, +3.12] | boot=1.0000 |
| `c1_dev|think|shared_eligible` | `dac_iit_vs_unif_base` | **-1.25** | [-6.25, +2.50] | McN=1.0000 |
| `c1_dev|think|shared_eligible` | `dac_iit_vs_unif_paired` | **+0.00** | [-3.75, +3.75] | McN=1.0000 |
