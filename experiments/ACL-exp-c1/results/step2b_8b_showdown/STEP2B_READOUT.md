# ACL-exp-c1 · Step 2B 因果对决统计汇总（Qwen3-8B, v4 审计版）

- **唯一预注册主终点**：`c1_step2b_confirm_ce` (`no_think`, M0 共享合格分母 `n=1129`, 门槛 `≥ 800`)
- **`think` 非劣性对照集**：`c1_dev` (95% CI 下界 `≥ -2.0 pp`)
- **14B 最终四态判定**：**STOP → 两档在大样本下均已排除 ≥+5 pp 随机掩码增益，按预注册方案转向 Pareto/部署保真主叙事** (`overall_go_14b=False`)

## 一、通用能力与多切片双制度健康门禁表（取全部必需切片最严值）

| 组别 | 逐组保真校验 | IFEval (≥430) | GSM8K v1 (≥470) | KL (≤0.005) | `no_think` V/V_hes/V_auth / T1 | `think` V/V_hes/V_auth / T1 / 失败 | 全部健康 |
|---|---|---:|---:|---:|---|---|---|
| `M0` | 基线 | 基座参照 | 基座参照 | 0.000000 | — | — | 基线 |
| `std_targeted_dual` | arm_parity_r128 ✅ | 439/541 ✅ | 507/1319 ✅ | 0.00095 ✅ | 77.1/69.7/75.2% / T1=100.0% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |
| `std_random_lm` | arm_parity_r128 ✅ | 437/541 ✅ | 514/1319 ✅ | 0.000964 ✅ | 77.5/69.4/74.2% / T1=100.0% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |
| `std_random_lm_seed2` | arm_parity_r128 ✅ | 446/541 ✅ | 535/1319 ✅ | 0.000973 ✅ | 76.3/69.4/74.1% / T1=100.0% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |
| `std_uniform_lora` | arm_parity_r8 ✅ | 441/541 ✅ | 502/1319 ✅ | 0.001358 ✅ | 75.6/71.4/73.4% / T1=100.0% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |
| `tight_targeted_dual` | arm_parity_r32 ✅ | 444/541 ✅ | 521/1319 ✅ | 0.000928 ✅ | 92.1/71.1/82.8% / T1=100.0% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |
| `tight_random_lm` | arm_parity_r32 ✅ | 440/541 ✅ | 509/1319 ✅ | 0.000944 ✅ | 92.9/67.6/81.1% / T1=99.9% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |
| `tight_random_lm_seed2` | arm_parity_r32 ✅ | 447/541 ✅ | 549/1319 ✅ | 0.00094 ✅ | 91.2/68.5/79.4% / T1=99.9% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |
| `tight_uniform_lora` | arm_parity_r8 ✅ | 444/541 ✅ | 537/1319 ✅ | 0.001247 ✅ | 78.6/70.5/74.8% / T1=100.0% ❌ | 100.0/100.0/100.0% / T1=100.0% / 0.00% ✅ | **FAIL** |

## 二、预注册主终点 (`c1_step2b_confirm_ce`) 与 `think` 非劣性 (`c1_dev`) 配对因果对比

### standard_6.87M（单种子门禁：**HOLD ❌**）
- 检出力(`≥800`): `True` | 健康: `False` | 主终点胜双随机: `False` | 胜/Pareto Uniform: `True`

| 切片 / 制度 / 分母 | 对照组 | Dual 双对 | 对照双对 | 差值 Δ | 95% 配对 Bootstrap CI | p 值 | `u(V)-u(F)` Δ (95% CI) |
|---|---|---:|---:|---:|---|---:|---|
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_random_1` | 47.3% | 45.3% | **+1.9 pp** | [+1.1, +2.8] pp | McN=0.0000 | -0.452 [-0.528, -0.377] |
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_random_2` | 47.3% | 45.6% | **+1.7 pp** | [+1.0, +2.5] pp | McN=0.0000 | -0.339 [-0.404, -0.275] |
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_random_pool` | 47.3% | 45.5% | **+1.8 pp** | [+1.1, +2.6] pp | boot=0.0000 | -0.396 [-0.463, -0.328] |
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_uniform` | 47.3% | 45.4% | **+1.9 pp** | [+1.1, +2.7] pp | McN=0.0000 | +0.982 [+0.877, +1.087] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_random_1` | 50.1% | 48.7% | **+1.4 pp** | [+0.9, +2.1] pp | McN=0.0000 | -0.451 [-0.508, -0.393] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_random_2` | 50.1% | 48.8% | **+1.3 pp** | [+0.8, +1.9] pp | McN=0.0000 | -0.317 [-0.365, -0.268] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_random_pool` | 50.1% | 48.8% | **+1.4 pp** | [+0.8, +1.9] pp | boot=0.0000 | -0.384 [-0.435, -0.333] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_uniform` | 50.1% | 49.0% | **+1.2 pp** | [+0.6, +1.7] pp | McN=0.0001 | +1.040 [+0.958, +1.121] |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_random_1` | 95.0% | 96.2% | **-1.2 pp** | [-6.2, +2.5] pp | McN=1.0000 | — |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_random_2` | 95.0% | 93.8% | **+1.2 pp** | [+0.0, +3.8] pp | McN=1.0000 | — |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_random_pool` | 95.0% | 95.0% | **+0.0 pp** | [-2.5, +3.1] pp | boot=1.0000 | — |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_uniform` | 95.0% | 95.0% | **+0.0 pp** | [-5.0, +5.0] pp | McN=1.0000 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_random_1` | 95.0% | 96.2% | **-1.2 pp** | [-6.2, +2.5] pp | McN=1.0000 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_random_2` | 95.0% | 93.8% | **+1.2 pp** | [+0.0, +3.8] pp | McN=1.0000 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_random_pool` | 95.0% | 95.0% | **+0.0 pp** | [-2.5, +3.1] pp | boot=1.0000 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_uniform` | 95.0% | 95.0% | **+0.0 pp** | [-5.0, +5.0] pp | McN=1.0000 | — |

### tight_1.52M（单种子门禁：**HOLD ❌**）
- 检出力(`≥800`): `True` | 健康: `False` | 主终点胜双随机: `False` | 胜/Pareto Uniform: `True`

| 切片 / 制度 / 分母 | 对照组 | Dual 双对 | 对照双对 | 差值 Δ | 95% 配对 Bootstrap CI | p 值 | `u(V)-u(F)` Δ (95% CI) |
|---|---|---:|---:|---:|---|---:|---|
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_random_1` | 48.2% | 47.1% | **+1.1 pp** | [+0.4, +1.8] pp | McN=0.0042 | +0.886 [+0.830, +0.942] |
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_random_2` | 48.2% | 47.6% | **+0.6 pp** | [+0.1, +1.2] pp | McN=0.0391 | +0.650 [+0.598, +0.700] |
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_random_pool` | 48.2% | 47.3% | **+0.8 pp** | [+0.3, +1.4] pp | boot=0.0010 | +0.768 [+0.715, +0.819] |
| `c1_step2b_confirm_ce` / no_think / shared_eligible (n=1129 [cl=601]) | `dual_vs_uniform` | 48.2% | 45.7% | **+2.5 pp** | [+1.4, +3.5] pp | McN=0.0000 | -0.078 [-0.221, +0.069] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_random_1` | 50.5% | 49.7% | **+0.9 pp** | [+0.4, +1.4] pp | McN=0.0007 | +0.999 [+0.955, +1.044] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_random_2` | 50.5% | 50.1% | **+0.4 pp** | [+0.1, +0.8] pp | McN=0.0654 | +0.728 [+0.688, +0.770] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_random_pool` | 50.5% | 49.9% | **+0.6 pp** | [+0.3, +1.0] pp | boot=0.0004 | +0.864 [+0.822, +0.905] |
| `c1_step2b_confirm_ce` / no_think / all (n=1728 [cl=864]) | `dual_vs_uniform` | 50.5% | 49.2% | **+1.3 pp** | [+0.5, +2.2] pp | McN=0.0027 | -0.219 [-0.329, -0.113] |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_random_1` | 91.2% | 92.5% | **-1.2 pp** | [-6.2, +2.5] pp | McN=1.0000 | — |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_random_2` | 91.2% | 91.2% | **+0.0 pp** | [-3.8, +3.8] pp | McN=1.0000 | — |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_random_pool` | 91.2% | 91.9% | **-0.6 pp** | [-4.4, +3.1] pp | boot=0.8796 | — |
| `c1_dev` / think / shared_eligible (n=80) | `dual_vs_uniform` | 91.2% | 96.2% | **-5.0 pp** | [-10.0, -1.2] pp | McN=0.1250 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_random_1` | 91.2% | 92.5% | **-1.2 pp** | [-6.2, +2.5] pp | McN=1.0000 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_random_2` | 91.2% | 91.2% | **+0.0 pp** | [-3.8, +3.8] pp | McN=1.0000 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_random_pool` | 91.2% | 91.9% | **-0.6 pp** | [-4.4, +3.1] pp | boot=0.8796 | — |
| `c1_dev` / think / all (n=80) | `dual_vs_uniform` | 91.2% | 96.2% | **-5.0 pp** | [-10.0, -1.2] pp | McN=0.1250 | — |

